import os
import uuid
import logging
import re
from difflib import SequenceMatcher

from flask import Blueprint, request, redirect, render_template, flash, url_for

from auth import api_login_required, login_required
from db import get_cursor
from services import (
    ALLOWED_IMAGE_TYPES,
    upload_to_s3,
    detect_plate,
    infer_order_action_with_ai,
    infer_menu_correction_with_ai,
    generate_vehicle_chat_reply,
    generate_gate_transaction_message,
    is_checkout_message,
    is_recommendation_request,
)

logger = logging.getLogger(__name__)

entry_bp = Blueprint("entry_bp", __name__)

def _normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalize_match_text(text):
    """Normalize text for menu-name matching (strip punctuation and collapse spaces)."""
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_alias(cleaned_message, alias):
    if not alias:
        return False
    pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
    return bool(re.search(pattern, cleaned_message))


def _build_item_aliases(name):
    """Build a set of aliases so users can order by natural phrasing."""
    aliases = set()
    normalized_name = _normalize_match_text(name)
    if not normalized_name:
        return aliases

    aliases.add(normalized_name)

    # Remove brand prefix for phrases like "combo a" or "classic burger".
    if normalized_name.startswith("speed "):
        aliases.add(normalized_name[6:].strip())

    # Remove parenthetical suffixes: "chicken nuggets (6 pcs)" -> "chicken nuggets".
    without_parentheses = re.sub(r"\s*\([^)]*\)", "", normalized_name).strip()
    if without_parentheses:
        aliases.add(without_parentheses)

    words = normalized_name.split()

    # Add adjacent concatenations ("coca cola" -> "cocacola", "fish burger" -> "fishburger").
    aliases.add(normalized_name.replace(" ", ""))
    for i in range(len(words) - 1):
        aliases.add(words[i] + words[i + 1])

    # Common shorthand by menu family.
    if "fries" in words:
        # Keep fries aliases item-specific to avoid one phrase matching both
        # French Fries and Curly Fries at the same time.
        if "curly" in words:
            aliases.update({"curly fry", "curly fries"})
        elif "french" in words:
            aliases.update({"fries", "french fry", "french fries", "fry"})
        else:
            aliases.update({"fries", "fry"})
    if "nuggets" in words:
        aliases.update({"nuggets", "chicken nuggets"})
    if "onion" in words and "rings" in words:
        aliases.add("rings")
    if "mozzarella" in words and "sticks" in words:
        aliases.update({"mozzarella", "sticks", "mozzarella sticks"})
    if "coca" in words and "cola" in words:
        aliases.update(
            {
                "coke",
                "cola",
                "coca cola",
                "cocacola",
                "coco cola",
                "coka cola",
                "cocoa cola",
            }
        )
    if "sprite" in words:
        aliases.add("sprite")
    if "mineral" in words and "water" in words:
        aliases.update({"water", "mineral water"})
    if "milkshake" in words:
        aliases.add("shake")
    if "americano" in words:
        aliases.add("americano")
    if "latte" in words:
        aliases.add("latte")
    # Do not add a generic "combo" alias here: it causes
    # "combo a" to match every combo item (A/B/C). Keep combo
    # aliases specific via normalized names like "combo a".

    # Light singular/plural recovery for STT/typing variance.
    # Examples: "french fry" <-> "french fries", "nugget" <-> "nuggets".
    expanded_aliases = set(aliases)
    for alias in list(aliases):
        parts = alias.split()
        if not parts:
            continue
        last = parts[-1]

        if last.endswith("ies") and len(last) > 3:
            singular_last = last[:-3] + "y"
            expanded_aliases.add(" ".join(parts[:-1] + [singular_last]))
        elif last.endswith("s") and len(last) > 2:
            singular_last = last[:-1]
            expanded_aliases.add(" ".join(parts[:-1] + [singular_last]))
        elif last.endswith("y") and len(last) > 1:
            plural_last = last[:-1] + "ies"
            expanded_aliases.add(" ".join(parts[:-1] + [plural_last]))
        else:
            plural_last = last + "s"
            expanded_aliases.add(" ".join(parts[:-1] + [plural_last]))

    aliases = expanded_aliases

    return {alias for alias in aliases if alias}


WORD_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

QUANTITY_WORD_PATTERN = "|".join(sorted([*WORD_NUMBER_MAP.keys(), r"\d+"], key=len, reverse=True))

FOLLOW_UP_PHRASES = {
    "one",
    "ones",
    "it",
    "that",
    "this",
    "that one",
    "this one",
    "the one",
    "same",
    "same one",
    "another",
    "more",
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "please",
    "the usual",
}

AFFIRMATIVE_WORDS = {
    "yes",
    "y",
    "yeah",
    "yep",
    "ya",
    "ok",
    "okay",
    "sure",
    "confirm",
    "confirmed",
    "correct",
    "right",
}

REMOVE_WORDS = {
    "cancel",
    "remove",
    "delete",
    "drop",
    "take off",
    "dont want",
    "don't want",
    "no more",
}

SET_WORDS = {
    "change",
    "modify",
    "update",
    "make",
    "set",
    "switch",
}

ORDER_FILLER_WORDS = {
    "i",
    "me",
    "my",
    "want",
    "would",
    "like",
    "to",
    "order",
    "add",
    "get",
    "give",
    "have",
    "take",
    "please",
    "can",
    "could",
    "for",
    "a",
    "an",
    "the",
    "of",
}


def _quantity_token_to_int(token):
    if not token:
        return None

    token = str(token).strip().lower()
    if token.isdigit():
        return max(1, min(int(token), 20))
    if token in WORD_NUMBER_MAP:
        return WORD_NUMBER_MAP[token]
    return None


def _is_checkout_message(message):
    return is_checkout_message(message)


def _extract_quantity(message, alias):
    # Supports phrases like "2 veggie burger" and "veggie burger x2".
    pattern_before = rf"\b({QUANTITY_WORD_PATTERN})\s*(?:x\s*)?{re.escape(alias)}\b"
    match_before = re.search(pattern_before, message)
    if match_before:
        quantity = _quantity_token_to_int(match_before.group(1))
        if quantity is not None:
            return quantity

    pattern_after = rf"\b{re.escape(alias)}\b\s*(?:x\s*)?({QUANTITY_WORD_PATTERN})"
    match_after = re.search(pattern_after, message)
    if match_after:
        quantity = _quantity_token_to_int(match_after.group(1))
        if quantity is not None:
            return quantity

    pattern_to = rf"\b{re.escape(alias)}\b\s*(?:to|become|be|into)\s*({QUANTITY_WORD_PATTERN})\b"
    match_to = re.search(pattern_to, message)
    if match_to:
        quantity = _quantity_token_to_int(match_to.group(1))
        if quantity is not None:
            return quantity

    return 1


def _extract_bare_quantity(message):
    cleaned = _normalize_text(message)
    if not cleaned:
        return None

    for token in re.findall(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", cleaned):
        quantity = _quantity_token_to_int(token)
        if quantity is not None:
            return quantity

    if re.search(r"\b(?:a|an|another|the same|same)\b", cleaned):
        return 1

    return None


def _looks_like_follow_up_request(message):
    cleaned = _normalize_text(message)
    if not cleaned:
        return False

    if cleaned in FOLLOW_UP_PHRASES:
        return True

    if any(phrase in cleaned for phrase in {"one of", "that one", "this one", "same one", "the same"}):
        return True

    if _extract_bare_quantity(cleaned) is not None:
        return True

    follow_up_words = {"it", "that", "this", "same", "another", "more", "one", "ones", "please", "sure", "yes", "yeah", "yep", "ok", "okay"}
    words = set(cleaned.split())
    return bool(words & follow_up_words)


def _is_affirmative_reply(message):
    cleaned = _normalize_text(message)
    if not cleaned:
        return False
    if cleaned in AFFIRMATIVE_WORDS:
        return True
    return bool(set(cleaned.split()) & AFFIRMATIVE_WORDS)


def _is_combo_request(message):
    cleaned = _normalize_match_text(message)
    if not cleaned:
        return False
    return "combo" in cleaned or "set" in cleaned


def _detect_order_action(message, ai_order=None):
    cleaned = _normalize_text(message)
    if not cleaned:
        return "add"

    if any(phrase in cleaned for phrase in REMOVE_WORDS):
        return "remove"

    if any(phrase in cleaned for phrase in SET_WORDS):
        if re.search(r"\bto\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", cleaned):
            return "set"
        if _extract_bare_quantity(cleaned) is not None:
            return "set"

    if ai_order and str(ai_order.get("intent") or "").strip().lower() == "modify_order":
        if re.search(r"\bto\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", cleaned):
            return "set"
        if any(phrase in cleaned for phrase in SET_WORDS):
            return "set"
        return "remove"

    return "add"


def _extract_combo_label(message):
    cleaned = _normalize_match_text(message)
    if not cleaned:
        return ""
    match = re.search(r"\bcombo\s+([a-z0-9]+)\b", cleaned)
    if not match:
        return "combo"
    return f"combo {match.group(1)}"


def _combo_suggestions(menu_rows, limit=3):
    combos = []
    for row in menu_rows or []:
        category = str(row.get("category") or "").strip().lower()
        if category != "combo":
            continue
        combos.append(
            {
                "menu_id": int(row["menu_id"]),
                "name": str(row.get("name") or "").strip(),
                "category": str(row.get("category") or ""),
                "score": 1.0,
            }
        )
    combos.sort(key=lambda item: item["name"])
    return combos[:limit]


def _resolve_affirmed_suggestion_from_history(message, history, menu_rows):
    """Resolve a plain affirmative reply ('yes') to the last assistant suggestion."""
    if not _is_affirmative_reply(message):
        return []

    quantity = _extract_bare_quantity(message)

    for turn in reversed(history or []):
        role = str(turn.get("role") or "").strip().lower()
        if role != "assistant":
            continue

        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if "did you mean" not in _normalize_text(content):
            continue

        matched = _extract_requested_menu_items(content, menu_rows)
        if not matched:
            continue

        picked = matched[0]
        resolved_qty = quantity or _extract_bare_quantity(content) or int(picked.get("quantity") or 1)
        resolved = dict(picked)
        resolved["quantity"] = max(1, min(int(resolved_qty), 20))
        return [resolved]

    return []


def _is_checkout_confirmation_reply(message, history):
    if not _is_affirmative_reply(message):
        return False

    for turn in reversed(history or []):
        role = str(turn.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        content = _normalize_text(turn.get("content") or "")
        if not content:
            continue
        if "reply yes to confirm" in content and (
            "confirm your order" in content or "tell me what to change" in content
        ):
            return True
        return False

    return False


def _normalize_order_phrase_for_fuzzy(message):
    cleaned = _normalize_match_text(message)
    if not cleaned:
        return ""

    words = []
    for token in cleaned.split():
        if token.isdigit() or token in WORD_NUMBER_MAP:
            continue
        if token in ORDER_FILLER_WORDS:
            continue
        words.append(token)
    return " ".join(words).strip()


def _suggest_menu_candidates(message, menu_rows, limit=3):
    """Return best fuzzy menu candidates for unclear/mispronounced item requests."""
    normalized_message = _normalize_order_phrase_for_fuzzy(message)
    if not normalized_message:
        return []

    scored = []
    compact_message = normalized_message.replace(" ", "")

    for item in menu_rows or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue

        aliases = _build_item_aliases(name)
        aliases.add(_normalize_match_text(name))
        aliases = {alias for alias in aliases if alias}
        if not aliases:
            continue

        best_score = 0.0
        for alias in aliases:
            alias_compact = alias.replace(" ", "")

            if normalized_message == alias or compact_message == alias_compact:
                best_score = max(best_score, 1.0)
                continue

            if normalized_message in alias or alias in normalized_message:
                best_score = max(best_score, 0.9)
                continue

            ratio = SequenceMatcher(None, normalized_message, alias).ratio()
            compact_ratio = SequenceMatcher(None, compact_message, alias_compact).ratio()
            best_score = max(best_score, ratio, compact_ratio)

        if best_score >= 0.62:
            scored.append(
                {
                    "menu_id": int(item["menu_id"]),
                    "name": name,
                    "category": str(item.get("category") or ""),
                    "score": float(best_score),
                }
            )

    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:limit]


def _extract_requested_menu_items(message, menu_rows):
    cleaned = _normalize_match_text(message)
    requested = []

    for item in menu_rows:
        name = str(item.get("name") or "").strip()
        if not name:
            continue

        aliases = sorted(_build_item_aliases(name), key=len, reverse=True)

        matched = False
        quantity = 1
        for alias in aliases:
            if _contains_alias(cleaned, alias):
                matched = True
                quantity = _extract_quantity(cleaned, alias)
                break

        if matched:
            requested.append(
                {
                    "menu_id": int(item["menu_id"]),
                    "name": name,
                    "category": str(item.get("category") or ""),
                    "quantity": quantity,
                    "unit_price": float(item["price"]),
                }
            )

    return requested


def _infer_requested_menu_items_from_history(message, history, menu_rows):
    if not _looks_like_follow_up_request(message):
        return []

    quantity = _extract_bare_quantity(message) or 1
    for item in reversed(history or []):
        content = str(item.get("content") or "").strip()
        if not content:
            continue

        matched = _extract_requested_menu_items(content, menu_rows)
        if len(matched) == 1:
            matched_item = dict(matched[0])
            matched_item["quantity"] = quantity
            return [matched_item]

        if len(matched) > 1:
            return []

    return []


def _is_contextual_decline(message, history):
    """True when the user declines after the assistant offered more items."""
    cleaned = _normalize_text(message)
    if not cleaned:
        return False

    neg_words = {"no", "nope", "nah", "not", "none", "nothing", "dont", "don't", "na"}
    if not (set(cleaned.split()) & neg_words):
        return False

    for turn in reversed(history or []):
        if str(turn.get("role") or "").strip().lower() != "assistant":
            continue
        content = _normalize_text(turn.get("content") or "")
        if any(phrase in content for phrase in [
            "anything else", "would you like", "what else", "like to add",
            "else today", "add a side", "add a drink",
        ]):
            return True
        return False

    return False


def _has_order_intent(message):
    """True when the message looks like an order attempt (quantity, order verbs, etc.)."""
    if not message:
        return False
    cleaned = _normalize_text(message)
    if is_recommendation_request(cleaned):
        return False
    if re.search(r'\d', cleaned):
        return True
    if _extract_bare_quantity(cleaned) is not None:
        return True
    if _looks_like_follow_up_request(cleaned):
        return True
    order_words = {"want", "order", "add", "give", "take", "like", "get", "have"}
    return bool(set(cleaned.split()) & order_words)


def _persist_order_from_chat(plate_number, user_message, history=None):
    """
    Upsert order lines from current chat message.
    - Adds requested menu items into the latest active order (Pending/Preparing).
    - If driver indicates checkout and order has items, mark Pending -> Preparing.
    """
    checkout_requested = _is_checkout_message(user_message)
    checkout_confirmed = _is_checkout_confirmation_reply(user_message, history)
    checkout_requested = checkout_requested or _is_contextual_decline(user_message, history)

    with get_cursor(dict_cursor=True, commit=True) as cursor:
        cursor.execute(
            "SELECT vehicle_id FROM vehicles WHERE plate_number = %s LIMIT 1",
            (plate_number,),
        )
        vehicle_row = cursor.fetchone()
        if not vehicle_row:
            return {"saved": False, "reason": "vehicle_not_found"}

        vehicle_id = int(vehicle_row["vehicle_id"])

        cursor.execute(
            """
            SELECT order_id, status
            FROM orders
            WHERE vehicle_id = %s AND status IN ('Pending', 'Preparing')
            ORDER BY order_id DESC
            LIMIT 1
            FOR UPDATE
            """,
            (vehicle_id,),
        )
        active_order = cursor.fetchone()

        cursor.execute(
            """
            SELECT menu_id, name, category, price
            FROM menu_items
            WHERE available = 1
            """
        )
        menu_rows = cursor.fetchall() or []

        # When user is confirming a previous checkout preview ("yes"), do NOT
        # run item inference — the affirmative is for the order as-is, not a
        # new add. Running AI here is what caused duplicate items.
        if checkout_confirmed:
            ai_order = None
            requested_items = []
            order_action = "add"
            checkout_requested = False
        else:
            ai_order = infer_order_action_with_ai(user_message, history, menu_rows)
            requested_items = _resolve_affirmed_suggestion_from_history(
                user_message, history, menu_rows
            )
            order_action = _detect_order_action(user_message, ai_order)
            checkout_requested = checkout_requested or False

        if ai_order and not requested_items:
            checkout_requested = bool(ai_order.get("checkout"))
            if ai_order.get("needs_clarification"):
                return {
                    "saved": False,
                    "reason": "needs_clarification",
                    "ai_order": ai_order,
                }

            for ai_item in ai_order.get("items") or []:
                name = str(ai_item.get("name") or "").strip()
                if not name:
                    continue

                matched_items = _extract_requested_menu_items(name, menu_rows)
                if not matched_items:
                    continue

                matched_item = dict(matched_items[0])
                try:
                    matched_item["quantity"] = max(1, min(int(ai_item.get("quantity") or 1), 20))
                except (TypeError, ValueError):
                    matched_item["quantity"] = 1
                requested_items.append(matched_item)

        combo_request = _is_combo_request(user_message)
        if combo_request and requested_items:
            combo_items = [
                item
                for item in requested_items
                if str(item.get("category") or "").strip().lower() == "combo"
            ]
            if combo_items:
                requested_items = combo_items
            else:
                requested_items = []

        if not checkout_confirmed:
            if not requested_items:
                requested_items = _extract_requested_menu_items(user_message, menu_rows)
            if not requested_items:
                requested_items = _infer_requested_menu_items_from_history(user_message, history, menu_rows)

        if ai_order and ai_order.get("intent") == "checkout" and not requested_items:
            checkout_requested = True

        if not requested_items and not checkout_requested and not checkout_confirmed:
            if _has_order_intent(user_message):
                if combo_request:
                    combo_choices = _combo_suggestions(menu_rows, limit=3)
                    if combo_choices:
                        return {
                            "saved": False,
                            "reason": "needs_clarification",
                            "clarification_reason": "unknown_combo",
                            "clarification_target": _extract_combo_label(user_message),
                            "suggestions": combo_choices,
                            "suggested_quantity": _extract_bare_quantity(user_message)
                            or 1,
                        }

                suggestions = []

                ai_correction = infer_menu_correction_with_ai(user_message, history, menu_rows)
                if ai_correction and ai_correction.get("suggestions"):
                    ai_names = ai_correction.get("suggestions") or []
                    by_name = {
                        str(item.get("name") or "").strip().lower(): item
                        for item in (menu_rows or [])
                    }
                    for name in ai_names:
                        key = str(name or "").strip().lower()
                        found = by_name.get(key)
                        if not found:
                            continue
                        suggestions.append(
                            {
                                "menu_id": int(found["menu_id"]),
                                "name": str(found.get("name") or "").strip(),
                                "category": str(found.get("category") or ""),
                                "score": float(ai_correction.get("confidence") or 0),
                            }
                        )

                if not suggestions:
                    suggestions = _suggest_menu_candidates(user_message, menu_rows, limit=3)

                if suggestions:
                    return {
                        "saved": False,
                        "reason": "needs_clarification",
                        "suggestions": suggestions,
                        "suggested_quantity": _extract_bare_quantity(user_message) or 1,
                    }
            return {"saved": False, "reason": "no_order_intent"}

        if (checkout_requested or checkout_confirmed) and not requested_items and not active_order:
            return {"saved": False, "reason": "nothing_to_checkout"}

        if order_action in {"remove", "set"} and not active_order:
            return {"saved": False, "reason": "nothing_to_modify"}

        if active_order:
            order_id = int(active_order["order_id"])
            current_status = str(active_order.get("status") or "Pending")
        else:
            cursor.execute(
                """
                INSERT INTO orders (vehicle_id, status, total_amount)
                VALUES (%s, 'Pending', 0.00)
                """,
                (vehicle_id,),
            )
            order_id = int(cursor.lastrowid)
            current_status = "Pending"

        items_added = []
        items_removed = []
        items_updated = []
        for item in requested_items:
            cursor.execute(
                """
                SELECT order_item_id, quantity
                FROM order_items
                WHERE order_id = %s AND menu_id = %s
                LIMIT 1
                """,
                (order_id, item["menu_id"]),
            )
            existing_line = cursor.fetchone()

            if order_action == "remove":
                if not existing_line:
                    continue

                existing_qty = int(existing_line["quantity"])
                remove_qty = max(1, min(existing_qty, int(item["quantity"])))
                new_qty = existing_qty - remove_qty
                if new_qty <= 0:
                    cursor.execute(
                        "DELETE FROM order_items WHERE order_item_id = %s",
                        (existing_line["order_item_id"],),
                    )
                else:
                    subtotal = new_qty * float(item["unit_price"])
                    cursor.execute(
                        """
                        UPDATE order_items
                        SET quantity = %s,
                            unit_price = %s,
                            subtotal = %s
                        WHERE order_item_id = %s
                        """,
                        (new_qty, item["unit_price"], subtotal, existing_line["order_item_id"]),
                    )

                items_removed.append(
                    {
                        "name": item["name"],
                        "category": item.get("category"),
                        "quantity": remove_qty,
                        "unit_price": item["unit_price"],
                    }
                )
                continue

            if order_action == "set":
                target_qty = max(0, min(int(item["quantity"]), 20))
                if existing_line:
                    if target_qty <= 0:
                        cursor.execute(
                            "DELETE FROM order_items WHERE order_item_id = %s",
                            (existing_line["order_item_id"],),
                        )
                    else:
                        subtotal = target_qty * float(item["unit_price"])
                        cursor.execute(
                            """
                            UPDATE order_items
                            SET quantity = %s,
                                unit_price = %s,
                                subtotal = %s
                            WHERE order_item_id = %s
                            """,
                            (target_qty, item["unit_price"], subtotal, existing_line["order_item_id"]),
                        )
                elif target_qty > 0:
                    subtotal = target_qty * float(item["unit_price"])
                    cursor.execute(
                        """
                        INSERT INTO order_items (order_id, menu_id, quantity, unit_price, subtotal)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (order_id, item["menu_id"], target_qty, item["unit_price"], subtotal),
                    )

                items_updated.append(
                    {
                        "name": item["name"],
                        "category": item.get("category"),
                        "quantity": target_qty,
                        "unit_price": item["unit_price"],
                    }
                )
                continue

            if existing_line:
                new_qty = int(existing_line["quantity"]) + int(item["quantity"])
                subtotal = new_qty * float(item["unit_price"])
                cursor.execute(
                    """
                    UPDATE order_items
                    SET quantity = %s,
                        unit_price = %s,
                        subtotal = %s
                    WHERE order_item_id = %s
                    """,
                    (new_qty, item["unit_price"], subtotal, existing_line["order_item_id"]),
                )
                added_qty = int(item["quantity"])
            else:
                subtotal = int(item["quantity"]) * float(item["unit_price"])
                cursor.execute(
                    """
                    INSERT INTO order_items (order_id, menu_id, quantity, unit_price, subtotal)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (order_id, item["menu_id"], item["quantity"], item["unit_price"], subtotal),
                )
                added_qty = int(item["quantity"])

            items_added.append(
                {
                    "name": item["name"],
                    "category": item.get("category"),
                    "quantity": added_qty,
                    "unit_price": item["unit_price"],
                }
            )

        cursor.execute(
            """
            SELECT DISTINCT LOWER(mi.category) AS category
            FROM order_items oi
            INNER JOIN menu_items mi ON mi.menu_id = oi.menu_id
            WHERE oi.order_id = %s
            """,
            (order_id,),
        )
        order_categories = {
            str(row.get("category") or "").strip().lower()
            for row in (cursor.fetchall() or [])
            if str(row.get("category") or "").strip()
        }
        order_category_flags = {
            "has_burger": "burger" in order_categories,
            "has_side": "side" in order_categories,
            "has_drink": "drink" in order_categories,
            "has_combo": "combo" in order_categories,
        }

        cursor.execute(
            "SELECT COALESCE(SUM(subtotal), 0) AS total_amount FROM order_items WHERE order_id = %s",
            (order_id,),
        )
        totals_row = cursor.fetchone() or {}
        total_amount = float(totals_row.get("total_amount") or 0)

        all_order_lines = []
        if checkout_requested or checkout_confirmed:
            cursor.execute(
                """
                SELECT mi.name, oi.quantity, oi.unit_price, oi.subtotal
                FROM order_items oi
                INNER JOIN menu_items mi ON mi.menu_id = oi.menu_id
                WHERE oi.order_id = %s
                ORDER BY oi.order_item_id ASC
                """,
                (order_id,),
            )
            all_order_lines = [
                {
                    "name": row["name"],
                    "quantity": int(row["quantity"]),
                    "unit_price": float(row["unit_price"]),
                    "subtotal": float(row["subtotal"]),
                }
                for row in (cursor.fetchall() or [])
            ]

        next_status = current_status
        if checkout_requested and not checkout_confirmed and total_amount > 0:
            return {
                "saved": False,
                "reason": "checkout_confirmation_required",
                "order_id": order_id,
                "status": current_status,
                "total_amount": total_amount,
                "order_lines": all_order_lines,
                "checkout": False,
                "requires_checkout_confirmation": True,
            }

        if checkout_confirmed and total_amount > 0 and current_status == "Pending":
            next_status = "Preparing"

        cursor.execute(
            "UPDATE orders SET status = %s, total_amount = %s WHERE order_id = %s",
            (next_status, total_amount, order_id),
        )

        return {
            "saved": True,
            "order_id": order_id,
            "status": next_status,
            "total_amount": total_amount,
            "order_action": order_action,
            "items_added": items_added,
            "items_removed": items_removed,
            "items_updated": items_updated,
            "order_category_flags": order_category_flags,
            "checkout": checkout_confirmed,
            "order_lines": all_order_lines,
        }


def _record_entry_transaction(plate_number):
    plate_number = (plate_number or "").strip().upper()
    if not plate_number:
        return None

    with get_cursor(commit=True) as cursor:
        # Lock last row for this plate to avoid duplicate IN/OUT race conditions.
        cursor.execute(
            """
                SELECT status
                FROM entry_logs
                WHERE plate_number = %s
                ORDER BY entry_time DESC
                LIMIT 1
                FOR UPDATE
            """,
            (plate_number,),
        )

        last_log = cursor.fetchone()
        status = "OUT" if last_log and last_log[0] == "IN" else "IN"

        cursor.execute(
            """
                INSERT INTO entry_logs
                (plate_number, status)
                VALUES (%s, %s)
            """,
            (plate_number, status),
        )

        if status == "IN":
            cursor.execute(
                """
                    INSERT INTO vehicles (plate_number, total_visits, last_visit)
                    VALUES (%s, 1, NOW())
                    ON DUPLICATE KEY UPDATE
                        total_visits = total_visits + 1,
                        last_visit   = NOW()
                """,
                (plate_number,),
            )

            # Cancel any stale orders from a previous drive-thru session.
            cursor.execute(
                """
                    UPDATE orders o
                    INNER JOIN vehicles v ON v.vehicle_id = o.vehicle_id
                    SET o.status = 'Cancelled'
                    WHERE v.plate_number = %s AND o.status IN ('Pending', 'Preparing')
                """,
                (plate_number,),
            )

    transaction_message = f"Vehicle {plate_number} recorded as {status}."
    welcome_message = generate_gate_transaction_message(
        plate_number=plate_number,
        direction=status,
    )

    return {
        "success": True,
        "plate_number": plate_number,
        "direction": status,
        "transaction_message": transaction_message,
        "welcome_message": welcome_message,
    }


@entry_bp.route("/entry", methods=["GET", "POST"])
@login_required
def entry():

    if request.method == "POST":
        result = _record_entry_transaction(request.form.get("plate_number"))
        if not result:
            flash("Plate number is required.", "error")
            return redirect(url_for("entry_bp.entry"))

        # If AJAX request, return JSON with transaction details for chatbot
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return result

        flash(result["transaction_message"], "success")

        return redirect(url_for("entry_bp.entry"))

    return render_template("entry.html")


@entry_bp.route("/api/entry", methods=["POST"])
@api_login_required
def submit_entry_api():
    payload = request.get_json(silent=True) or {}
    plate_number = payload.get("plate_number")
    result = _record_entry_transaction(plate_number)

    if not result:
        return {"error": "Missing plate_number"}, 400

    return result


def _scan_plate_impl():
    """Accept an image upload, run YOLO + PaddleOCR, return detected plate."""
    if "image" not in request.files:
        return {"error": "No image provided"}, 400

    file = request.files["image"]

    if not file or file.filename == "":
        return {"error": "No file selected"}, 400

    raw_content_type = (file.content_type or "").strip().lower()
    normalized_content_type = raw_content_type.split(";", 1)[0]
    filename = (file.filename or "").strip().lower()
    extension_allowed = filename.endswith((".jpg", ".jpeg", ".png", ".webp"))

    # Browser camera uploads can arrive as application/octet-stream even when
    # they are valid images. Allow those when the filename extension is known.
    if normalized_content_type == "image/jpg":
        normalized_content_type = "image/jpeg"

    is_valid_type = normalized_content_type in ALLOWED_IMAGE_TYPES
    is_octet_stream_image = (
        normalized_content_type == "application/octet-stream" and extension_allowed
    )

    if not is_valid_type and not is_octet_stream_image and not (not normalized_content_type and extension_allowed):
        return {
            "error": "Invalid file type. Use JPEG, PNG or WebP.",
            "received_content_type": raw_content_type or None,
            "filename": file.filename,
        }, 400

    # Save with a random name to avoid collisions / path traversal
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join("upload", filename)
    file.save(filepath)

    try:
        # Upload original photo to S3
        s3_original_url = upload_to_s3(filepath, f"upload/{filename}")

        try:
            plate_text = detect_plate(filepath)
        except Exception:
            logger.exception("Plate detection failed for %s", filepath)
            return {"error": "Could not process the image. Please try again."}, 500
    finally:
        # Original has been uploaded (or the upload failed and was logged);
        # don't let uploaded photos accumulate on local disk
        if os.path.exists(filepath):
            os.remove(filepath)

    if not plate_text:
        return {"error": "No license plate detected in the image."}, 422

    result = {
        "plate": plate_text,
        "plate_number": plate_text,
    }
    if s3_original_url:
        result["s3_url"] = s3_original_url
    return result


@entry_bp.route("/scan-plate", methods=["POST"])
@login_required
def scan_plate():
    return _scan_plate_impl()


@entry_bp.route("/api/detect-plate", methods=["POST"])
@api_login_required
def detect_plate_api():
    return _scan_plate_impl()


@entry_bp.route("/api/assistant-chat", methods=["POST"])
@api_login_required
def assistant_chat():
    """Chat endpoint for follow-up driver questions after a plate scan."""
    payload = request.get_json(silent=True) or {}
    plate_number = (payload.get("plate_number") or "").strip().upper()
    direction = (payload.get("direction") or "").strip().upper()
    message = payload.get("message")
    history = payload.get("history")
    menu_items = payload.get("menu_items")  # optional list of {name, price, description, available}

    if not plate_number:
        return {"error": "Missing plate_number"}, 400

    if history is not None and not isinstance(history, list):
        return {"error": "history must be a list"}, 400

    if menu_items is not None and not isinstance(menu_items, list):
        menu_items = None  # ignore malformed input

    # Persist order FIRST so AI reads the updated DB ground truth
    order_update = None
    items_just_added = []
    items_removed = []
    items_updated = []
    order_action = "add"
    order_category_flags = None
    checkout_requested = False
    checkout_order_lines = []
    order_not_found = False
    needs_clarification = False
    clarification_suggestions = []
    clarification_quantity = 1
    clarification_reason = ""
    clarification_target = ""
    checkout_confirmation_required = False
    if direction != "OUT":
        try:
            order_update = _persist_order_from_chat(plate_number, message, history)
            if order_update and order_update.get("saved"):
                order_action = str(order_update.get("order_action") or "add")
                items_just_added = order_update.get("items_added") or []
                items_removed = order_update.get("items_removed") or []
                items_updated = order_update.get("items_updated") or []
                order_category_flags = order_update.get("order_category_flags")
                checkout_requested = bool(order_update.get("checkout"))
                checkout_order_lines = order_update.get("order_lines") or []
            elif order_update and order_update.get("reason") == "checkout_confirmation_required":
                checkout_confirmation_required = True
                checkout_order_lines = order_update.get("order_lines") or []
            elif order_update and order_update.get("reason") == "needs_clarification":
                needs_clarification = True
                clarification_suggestions = order_update.get("suggestions") or []
                clarification_reason = str(order_update.get("clarification_reason") or "")
                clarification_target = str(order_update.get("clarification_target") or "")
                try:
                    clarification_quantity = max(1, min(int(order_update.get("suggested_quantity") or 1), 20))
                except (TypeError, ValueError):
                    clarification_quantity = 1
            elif (
                order_update
                and not order_update.get("saved")
                and order_update.get("reason") == "no_order_intent"
                and _has_order_intent(message)
            ):
                needs_clarification = _looks_like_follow_up_request(message)
                order_not_found = not needs_clarification
        except Exception:
            logger.exception("Failed to persist chat order for plate %s", plate_number)

    reply = generate_vehicle_chat_reply(
        plate_number=plate_number,
        user_message=message,
        history=history,
        direction=direction,
        menu_items=menu_items,
        items_just_added=items_just_added,
        items_removed=items_removed,
        items_updated=items_updated,
        order_action=order_action,
        order_category_flags=order_category_flags,
        checkout_requested=checkout_requested,
        checkout_order_lines=checkout_order_lines,
        checkout_confirmation_required=checkout_confirmation_required,
        order_not_found=order_not_found,
        needs_clarification=needs_clarification,
        clarification_suggestions=clarification_suggestions,
        clarification_quantity=clarification_quantity,
        clarification_reason=clarification_reason,
        clarification_target=clarification_target,
    )

    if order_update and order_update.get("saved") and checkout_requested and order_update.get("order_id"):
        reply = f"{reply} Your order number is #{order_update['order_id']}."

    return {
        "reply": reply,
        "plate_number": plate_number,
        "direction": direction,
        "order": order_update,
    }