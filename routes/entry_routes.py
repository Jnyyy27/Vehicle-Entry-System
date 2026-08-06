import os
import uuid
import logging
import re

from flask import Blueprint, request, redirect, render_template, flash, url_for

from auth import login_required
from db import get_cursor
from services import (
    ALLOWED_IMAGE_TYPES,
    upload_to_s3,
    detect_plate,
    generate_vehicle_chat_reply,
    generate_gate_transaction_message,
)

logger = logging.getLogger(__name__)

entry_bp = Blueprint("entry_bp", __name__)


CHECKOUT_PHRASES = {
    "that is all",
    "thats all",
    "that's all",
    "nothing else",
    "no more",
    "that will be all",
    "checkout",
    "pay",
    "done",
    "finish",
    "finished",
}


def _normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_checkout_message(message):
    cleaned = _normalize_text(message)
    return any(phrase in cleaned for phrase in CHECKOUT_PHRASES)


def _extract_quantity(message, alias):
    # Supports phrases like "2 veggie burger" and "veggie burger x2".
    pattern_before = rf"(\d+)\s*(?:x\s*)?{re.escape(alias)}"
    match_before = re.search(pattern_before, message)
    if match_before:
        return max(1, min(int(match_before.group(1)), 20))

    pattern_after = rf"{re.escape(alias)}\s*(?:x\s*)?(\d+)"
    match_after = re.search(pattern_after, message)
    if match_after:
        return max(1, min(int(match_after.group(1)), 20))

    return 1


def _extract_requested_menu_items(message, menu_rows):
    cleaned = _normalize_text(message)
    requested = []

    for item in menu_rows:
        name = str(item.get("name") or "").strip()
        if not name:
            continue

        aliases = [name.lower()]
        if aliases[0].startswith("speed "):
            aliases.append(aliases[0][6:])
        # Also match space-stripped form so "fishburger" matches "Fish Burger"
        no_space = aliases[0].replace(" ", "")
        if no_space != aliases[0] and no_space not in aliases:
            aliases.append(no_space)
        # Adjacent word-pair concatenations: "cheeseburger" → "Speed Cheese Burger",
        # "chickenburger" → "Crispy Chicken Burger", etc.
        words_list = aliases[0].split()
        for i in range(len(words_list) - 1):
            pair = words_list[i] + words_list[i + 1]
            if pair not in aliases:
                aliases.append(pair)

        matched = False
        quantity = 1
        for alias in aliases:
            if alias and alias in cleaned:
                matched = True
                quantity = _extract_quantity(cleaned, alias)
                break

        if matched:
            requested.append(
                {
                    "menu_id": int(item["menu_id"]),
                    "name": name,
                    "quantity": quantity,
                    "unit_price": float(item["price"]),
                }
            )

    return requested


def _has_order_intent(message):
    """True when the message looks like an order attempt (quantity, order verbs, etc.)."""
    if not message:
        return False
    cleaned = _normalize_text(message)
    if re.search(r'\d', cleaned):
        return True
    order_words = {"want", "order", "add", "give", "take", "like", "get", "have"}
    return bool(set(cleaned.split()) & order_words)


def _persist_order_from_chat(plate_number, user_message):
    """
    Upsert order lines from current chat message.
    - Adds requested menu items into the latest active order (Pending/Preparing).
    - If driver indicates checkout and order has items, mark Pending -> Preparing.
    """
    checkout_requested = _is_checkout_message(user_message)

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
            SELECT menu_id, name, price
            FROM menu_items
            WHERE available = 1
            """
        )
        menu_rows = cursor.fetchall() or []
        requested_items = _extract_requested_menu_items(user_message, menu_rows)

        if not requested_items and not checkout_requested:
            return {"saved": False, "reason": "no_order_intent"}

        if checkout_requested and not requested_items and not active_order:
            return {"saved": False, "reason": "nothing_to_checkout"}

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
                    "quantity": added_qty,
                    "unit_price": item["unit_price"],
                }
            )

        cursor.execute(
            "SELECT COALESCE(SUM(subtotal), 0) AS total_amount FROM order_items WHERE order_id = %s",
            (order_id,),
        )
        totals_row = cursor.fetchone() or {}
        total_amount = float(totals_row.get("total_amount") or 0)

        next_status = current_status
        if checkout_requested and total_amount > 0 and current_status == "Pending":
            next_status = "Preparing"

        cursor.execute(
            "UPDATE orders SET status = %s, total_amount = %s WHERE order_id = %s",
            (next_status, total_amount, order_id),
        )

        # At checkout, return the complete order lines so the frontend can show a summary popup.
        all_order_lines = []
        if checkout_requested:
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

        return {
            "saved": True,
            "order_id": order_id,
            "status": next_status,
            "total_amount": total_amount,
            "items_added": items_added,
            "checkout": checkout_requested,
            "order_lines": all_order_lines,
        }


@entry_bp.route("/entry", methods=["GET", "POST"])
@login_required
def entry():

    if request.method == "POST":
        plate_number = request.form["plate_number"]

        with get_cursor(commit=True) as cursor:
            # FOR UPDATE locks the matching row(s) for the duration of this
            # transaction so two near-simultaneous scans of the same plate
            # can't both read "no prior log" / "last was IN" and both insert
            # "IN". Requires the entry_logs table to use InnoDB.
            cursor.execute("""
                SELECT status
                FROM entry_logs
                WHERE plate_number = %s
                ORDER BY entry_time DESC
                LIMIT 1
                FOR UPDATE
            """, (plate_number,))

            last_log = cursor.fetchone()
            status = "OUT" if last_log and last_log[0] == "IN" else "IN"

            cursor.execute("""
                INSERT INTO entry_logs
                (plate_number, status)
                VALUES (%s, %s)
            """, (plate_number, status))

            # Auto-upsert into vehicles: create on first visit, increment
            # total_visits and update last_visit on every IN scan.
            if status == "IN":
                cursor.execute("""
                    INSERT INTO vehicles (plate_number, total_visits, last_visit)
                    VALUES (%s, 1, NOW())
                    ON DUPLICATE KEY UPDATE
                        total_visits = total_visits + 1,
                        last_visit   = NOW()
                """, (plate_number,))

                # Cancel any stale Pending/Preparing orders from a previous session
                # so the new drive-thru session always starts with a clean slate.
                cursor.execute("""
                    UPDATE orders o
                    INNER JOIN vehicles v ON v.vehicle_id = o.vehicle_id
                    SET o.status = 'Cancelled'
                    WHERE v.plate_number = %s AND o.status IN ('Pending', 'Preparing')
                """, (plate_number,))

        transaction_message = f"Vehicle {plate_number} recorded as {status}."

        # If AJAX request, return JSON with transaction details for chatbot
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
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

        flash(transaction_message, "success")

        return redirect(url_for("entry_bp.entry"))

    return render_template("entry.html")


@entry_bp.route("/scan-plate", methods=["POST"])
@entry_bp.route("/api/detect-plate", methods=["POST"])
@login_required
def scan_plate():
    """Accept an image upload, run YOLO + PaddleOCR, return detected plate."""
    if "image" not in request.files:
        return {"error": "No image provided"}, 400

    file = request.files["image"]

    if not file or file.filename == "":
        return {"error": "No file selected"}, 400

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return {"error": "Invalid file type. Use JPEG, PNG or WebP."}, 400

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


@entry_bp.route("/api/assistant-chat", methods=["POST"])
@login_required
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
    checkout_requested = False
    checkout_order_lines = []
    order_not_found = False
    if direction != "OUT":
        try:
            order_update = _persist_order_from_chat(plate_number, message)
            if order_update and order_update.get("saved"):
                items_just_added = order_update.get("items_added") or []
                checkout_requested = bool(order_update.get("checkout"))
                checkout_order_lines = order_update.get("order_lines") or []
            elif (
                order_update
                and not order_update.get("saved")
                and order_update.get("reason") == "no_order_intent"
                and _has_order_intent(message)
            ):
                order_not_found = True
        except Exception:
            logger.exception("Failed to persist chat order for plate %s", plate_number)

    reply = generate_vehicle_chat_reply(
        plate_number=plate_number,
        user_message=message,
        history=history,
        direction=direction,
        menu_items=menu_items,
        items_just_added=items_just_added,
        checkout_requested=checkout_requested,
        checkout_order_lines=checkout_order_lines,
        order_not_found=order_not_found,
    )

    if order_update and order_update.get("saved") and checkout_requested and order_update.get("order_id"):
        reply = f"{reply} Your order number is #{order_update['order_id']}."

    return {
        "reply": reply,
        "plate_number": plate_number,
        "direction": direction,
        "order": order_update,
    }