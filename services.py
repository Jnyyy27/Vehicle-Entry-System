"""
Business logic that isn't route handling: plate detection (YOLO + PaddleOCR),
S3 uploads, and drive-thru ordering. Kept separate from routes so
route handlers stay thin and this logic is independently testable.
"""

import os
import logging
import uuid
import re
import json

import boto3
from botocore.exceptions import ClientError
import cv2
from dotenv import load_dotenv
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR

from db import get_cursor

load_dotenv()

logger = logging.getLogger(__name__)

# Make sure upload/cropped directories exist before anything tries to write to them
os.makedirs("upload", exist_ok=True)
os.makedirs("cropped", exist_ok=True)

plate_model = YOLO("models/license_plate_detector.pt")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "ap-southeast-1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

s3_client = boto3.client("s3", region_name=S3_REGION)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


# Kept as a last-resort alias table when no DB menu is available
ORDER_KEYWORDS = {
    "burger": "burger",
    "cheeseburger": "cheeseburger",
    "fries": "fries",
    "nuggets": "chicken nuggets",
    "nugget": "chicken nuggets",
    "chicken": "chicken",
    "drink": "drink",
    "coke": "coke",
    "sprite": "sprite",
    "water": "water",
    "coffee": "coffee",
    "tea": "tea",
    "shake": "shake",
    "sundae": "sundae",
    "ice cream": "ice cream",
    "combo": "combo meal",
    "meal": "combo meal",
}

CHECKOUT_EXACT_MESSAGES = {
    "that is all",
    "thats all",
    "that's all",
    "nothing",
    "nothing else",
    "no",
    "no no",
    "no la",
    "no lah",
    "no need",
    "no thank you",
    "no more",
    "no thanks",
    "nope",
    "nah",
    "that will be all",
    "i'm done",
    "im done",
    "done",
    "i'm good",
    "im good",
    "all good",
    "that's enough",
    "thats enough",
    "that's it",
    "thats it",
    "checkout",
    "pay",
    "finish",
    "finished",
}

CHECKOUT_PATTERN = re.compile(
    r"\b(?:that(?:'s| is| will be)? all|nothing else|no more|i(?:'m|m) done)\b"
)


def _normalize_message(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_checkout_message(text):
    cleaned = _normalize_message(text)
    if not cleaned:
        return False
    return cleaned in CHECKOUT_EXACT_MESSAGES or bool(CHECKOUT_PATTERN.search(cleaned))


def is_recommendation_request(text):
    cleaned = _normalize_message(text)
    if not cleaned:
        return False

    recommendation_phrases = (
        "what do you recommend",
        "what would you recommend",
        "recommend",
        "suggest",
        "popular",
        "best",
        "what should i get",
        "what should i order",
        "which one should i get",
        "which one do you recommend",
        "if i like",
        "if i want",
        "i feel like",
        "i'm feeling like",
        "im feeling like",
        "what is good",
        "any suggestion",
    )

    if any(phrase in cleaned for phrase in recommendation_phrases):
        return True

    preference_words = {"spicy", "beef", "chicken", "fish", "vegetarian", "veggie", "sweet", "salty", "crispy", "classic", "healthy", "light"}
    question_words = {"what", "which", "who", "how", "best", "good", "recommend", "suggest"}
    return "?" in cleaned and bool(set(cleaned.split()) & question_words) and bool(set(cleaned.split()) & preference_words)


def _extract_order_items(messages):
    found = []
    seen = set()
    for msg in messages:
        cleaned = _normalize_message(msg)
        for keyword, label in ORDER_KEYWORDS.items():
            if keyword in cleaned and label not in seen:
                found.append(label)
                seen.add(label)
    return found


def _menu_names_by_category(menu_items):
    grouped = {}
    for item in menu_items or []:
        if not item or not item.get("available"):
            continue
        category = str(item.get("category") or "Other").strip()
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        grouped.setdefault(category, []).append(name)
    return grouped


def generate_local_order_reply(user_message, history, menu_items=None):
    """Local fallback that still performs a useful order-taking conversation."""
    cleaned = _normalize_message(user_message)
    cleaned_words = set(cleaned.split())

    prior_user_messages = [
        str(item.get("content", ""))
        for item in (history or [])
        if str(item.get("role", "")).strip().lower() == "user"
    ]

    # Scan ALL turns (user + assistant) so items confirmed by the assistant
    # (e.g. "Got it, Veggie Burger RM10.90") are counted in the running order.
    all_turns_text = " ".join(
        str(item.get("content", "")) for item in (history or [])
    ) + " " + (user_message or "")
    all_turns_text = all_turns_text.lower()

    # --- Build price lookup ---------------------------------------------------
    prices = {}
    if menu_items:
        for mi in menu_items:
            prices[mi["name"]] = float(mi["price"])

    # --- Detect accumulated order (full conversation) -------------------------
    if menu_items:
        items = [
            mi["name"]
            for mi in menu_items
            if mi.get("available") and mi["name"].lower() in all_turns_text
        ]
    else:
        items = _extract_order_items(prior_user_messages + [user_message or ""])

    # --- Menu hint string -----------------------------------------------------
    if menu_items:
        by_category = _menu_names_by_category(menu_items)
        category_parts = []
        for category in ("Burger", "Side", "Drink", "Combo"):
            names = by_category.get(category) or []
            if names:
                category_parts.append(f"{category}s")
        if category_parts:
            menu_hint = ", ".join(category_parts)
        else:
            avail_names = [mi["name"] for mi in menu_items if mi.get("available")]
            menu_hint = ", ".join(avail_names[:4]) + (" and more" if len(avail_names) > 4 else "")
    else:
        menu_hint = "burgers and sides"

    # --- Intent detection (whole-word checks) ---------------------------------
    if not cleaned:
        return "Welcome to Speed Burger! What would you like to order today?"

    # Greeting
    if cleaned_words & {"hi", "hello", "hey", "helo"}:
        return f"Welcome to Speed Burger! We have {menu_hint}. What can I get for you today?"

    if any(phrase in cleaned for phrase in ["good morning", "good afternoon", "good evening"]):
        return f"Welcome to Speed Burger! We have {menu_hint}. What can I get for you today?"

    # Dietary / recommendation queries
    if any(word in cleaned for word in ["recommend", "suggest", "popular", "best", "what do you have"]) \
            or re.search(r'\bmenu\b', cleaned):
        if menu_items:
            # Signature burger query
            if any(w in cleaned for w in ["signature", "special", "original", "classic"]):
                match = next((mi for mi in menu_items if mi.get("available") and "classic" in mi["name"].lower()), None)
                if match:
                    return (
                        f"Our signature burger is the {match['name']} at RM{prices[match['name']]:.2f}. "
                        "A perfectly seared beef patty with fresh toppings on a toasted bun. Would you like to order that?"
                    )

            # General burger recommendation
            if any(w in cleaned for w in ["burger", "burgers"]):
                burger_picks = [
                    mi for mi in menu_items
                    if mi.get("available") and "burger" in mi["name"].lower()
                ]
                if burger_picks:
                    # Always lead with Speed Classic Burger, then add up to 2 more
                    classic = next((mi for mi in burger_picks if "classic" in mi["name"].lower()), None)
                    others = [mi for mi in burger_picks if mi is not classic][:2]
                    picks = ([classic] if classic else []) + others
                    pick_str = ", ".join(
                        f"{mi['name']} RM{prices[mi['name']]:.2f}" for mi in picks
                    )
                    return (
                        f"For burgers I recommend the {pick_str}. "
                        "Which one would you like?"
                    )

            if any(w in cleaned for w in ["side", "sides", "fries", "nuggets", "rings"]):
                side_picks = [
                    mi for mi in menu_items
                    if mi.get("available") and str(mi.get("category") or "").lower() == "side"
                ]
                if side_picks:
                    picks = side_picks[:3]
                    pick_str = ", ".join(
                        f"{mi['name']} RM{prices[mi['name']]:.2f}" for mi in picks
                    )
                    return f"For sides I recommend {pick_str}. Which one would you like?"

            if any(w in cleaned for w in ["drink", "drinks", "coffee", "tea", "coke", "sprite", "water", "latte", "americano", "milkshake"]):
                drink_picks = [
                    mi for mi in menu_items
                    if mi.get("available") and str(mi.get("category") or "").lower() == "drink"
                ]
                if drink_picks:
                    picks = drink_picks[:3]
                    pick_str = ", ".join(
                        f"{mi['name']} RM{prices[mi['name']]:.2f}" for mi in picks
                    )
                    return f"For drinks I recommend {pick_str}. What sounds good?"

            if any(w in cleaned for w in ["combo", "set", "meal"]):
                combo_picks = [
                    mi for mi in menu_items
                    if mi.get("available") and str(mi.get("category") or "").lower() == "combo"
                ]
                if combo_picks:
                    picks = combo_picks[:3]
                    pick_str = ", ".join(
                        f"{mi['name']} RM{prices[mi['name']]:.2f}" for mi in picks
                    )
                    return f"For combo sets I recommend {pick_str}. Which combo do you want?"

            if any(w in cleaned for w in ["vegetarian", "veggie", "vegan", "no meat"]):
                match = next((mi for mi in menu_items if mi.get("available") and "veggie" in mi["name"].lower()), None)
                if match:
                    return f"For vegetarians I recommend the {match['name']} at RM{prices[match['name']]:.2f}. Would you like to order that?"
            if any(w in cleaned for w in ["spicy", "hot"]):
                match = next((mi for mi in menu_items if mi.get("available") and "spicy" in mi["name"].lower()), None)
                if match:
                    return f"If you like spicy, try the {match['name']} at RM{prices[match['name']]:.2f}. Would you like that?"
            if any(w in cleaned for w in ["fish", "seafood"]):
                match = next((mi for mi in menu_items if mi.get("available") and "fish" in mi["name"].lower()), None)
                if match:
                    return f"We have the {match['name']} at RM{prices[match['name']]:.2f}. Would you like that?"

            # Generic recommendation — lead with Speed Classic Burger
            classic = next((mi for mi in menu_items if mi.get("available") and "classic" in mi["name"].lower()), None)
            if classic:
                return (
                    f"I recommend starting with our {classic['name']} at RM{prices[classic['name']]:.2f}. "
                    f"We also have {menu_hint}. What would you like?"
                )
        return f"Our menu includes {menu_hint}. Which one would you like to order?"

    # Modification
    if any(word in cleaned for word in ["remove", "cancel", "change", "edit"]):
        if items:
            return f"No problem. Your current order has {', '.join(items)}. What would you like to change?"
        return "No problem. Tell me what you would like to order."

    # Checkout
    if is_checkout_message(cleaned):
        if items:
            summary = ", ".join(
                f"{name} RM{prices[name]:.2f}" if name in prices else name
                for name in items
            )
            return f"Got it! Your order is: {summary}. Thank you for choosing Speed Burger, have a great day!"
        return "Please tell me your order items first before I can confirm."

    # Check what items are in the CURRENT message specifically (new additions)
    if menu_items:
        current_items = [
            mi["name"]
            for mi in menu_items
            if mi.get("available") and mi["name"].lower() in cleaned
        ]
    else:
        current_items = _extract_order_items([user_message or ""])

    if current_items:
        confirmations = [
            f"{name} RM{prices[name]:.2f}" if name in prices else name
            for name in current_items
        ]
        return f"Got it, {', '.join(confirmations)}. Would you like anything else?"

    # Order intent but item not found on menu
    order_intent = {"want", "order", "get", "have", "add", "give", "take", "like"}
    if cleaned_words & order_intent and menu_items:
        return f"Sorry, we do not have that on our menu. We currently offer {menu_hint}. What would you like?"

    if menu_items:
        return f"I am not sure about that. We currently have {menu_hint}. What would you like?"
    return "Please tell me your order item from the menu."


def upload_to_s3(local_path, s3_key):
    """
    Upload a local file to S3. Returns the public HTTPS URL, or None if the
    bucket name is not configured / the upload fails.
    """
    if not S3_BUCKET:
        return None
    try:
        s3_client.upload_file(
            local_path,
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": "image/jpeg"}
        )
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    except ClientError:
        logger.exception("S3 upload failed for key %s", s3_key)
        return None


def detect_plate(image_path):
    """
    Detect a license plate in an image using YOLO, then read its text with
    PaddleOCR. Returns the plate string in upper-case, or None.
    Cleans up the cropped-plate temp file after uploading it to S3.
    """
    results = plate_model(image_path, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes

    # Pick the highest-confidence detection
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())

    # Crop the plate region with a small padding border
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    pad = 8
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    cropped = img[y1:y2, x1:x2]

    # Persist the cropped plate for auditing
    crop_name = f"{uuid.uuid4().hex}.jpg"
    crop_path = os.path.join("cropped", crop_name)
    cv2.imwrite(crop_path, cropped)

    try:
        # Upload cropped plate to S3 (non-blocking — failures are logged, not raised)
        upload_to_s3(crop_path, f"cropped/{crop_name}")

        # Run OCR on the cropped region
        try:
            ocr_result = ocr.ocr(crop_path, cls=True)
        except TypeError:
            ocr_result = ocr.ocr(crop_path)
    finally:
        # Local copy has been uploaded (or the upload failed and was logged);
        # either way don't let temp crops accumulate on disk
        if os.path.exists(crop_path):
            os.remove(crop_path)

    if not ocr_result or not ocr_result[0]:
        return None

    # Collect text lines with acceptable confidence
    texts = [
        line[1][0]
        for line in ocr_result[0]
        if line[1][1] > 0.3
    ]

    # Fall back to all lines if nothing passed the threshold
    if not texts:
        texts = [line[1][0] for line in ocr_result[0]]

    plate_text = " ".join(texts).upper().strip()
    return plate_text if plate_text else None


def generate_gate_transaction_message(plate_number, direction):
    """
    Generate a direction-aware welcome after gate transaction (IN or OUT).
    Falls back to deterministic messages if Ollama unavailable.
    """
    if direction == "IN":
        fallback = "Welcome to Speed Burger! What would you like to order today?"
        instruction = "Welcome the driver warmly and ask what they would like to order today."
    else:
        fallback = "Thank you for visiting Speed Burger. Drive safe and come again next time!"
        instruction = "Thank the driver for visiting and invite them to come again next time."

    prompt = (
        "You are a Speed Burger drive-thru assistant speaking directly to drivers. "
        f"{instruction} "
        f"Vehicle plate: {plate_number}. "
        "Write exactly one short friendly sentence. "
        "Keep it under 20 words. No markdown, no emojis, no extra sentences."
    )

    message = call_ollama_text(prompt, temperature=0.4)
    return message or fallback


def call_ollama_text(prompt, temperature=0.3, timeout=20, stop=None):
    """Return plain text from Ollama generate API, or None on failure."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_predict": 120,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        text = (body.get("response") or "").strip()
        # Strip any leaked role prefixes the model may have continued writing
        for prefix in ("Driver:", "Assistant:", "User:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text or None
    except requests.RequestException:
        logger.warning("Ollama request failed", exc_info=True)
        return None


def call_ollama_json(prompt, temperature=0.1, timeout=20):
    """Return parsed JSON from Ollama generate API, or None on failure."""
    text = call_ollama_text(prompt, temperature=temperature, timeout=timeout)
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.debug("Ollama JSON parse failed: %s", cleaned, exc_info=True)
        return None


def infer_order_action_with_ai(user_message, history, menu_items):
    """Use Ollama to infer the user's intended ordering action from chat context."""
    if not user_message:
        return None

    menu_lines = []
    for item in menu_items or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        price = float(item.get("price") or 0)
        available = bool(item.get("available", True))
        menu_lines.append(
            f"- {name} | RM{price:.2f} | {'available' if available else 'sold_out'}"
        )

    history_lines = []
    for turn in (history or [])[-8:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            speaker = "Driver" if role == "user" else "Assistant"
            history_lines.append(f"{speaker}: {content[:400]}")

    prompt = (
        "You are an order parser for a drive-thru chatbot. "
        "Convert the driver's latest message into a single JSON object only. "
        "Use the conversation history and menu to infer what they mean, even for short replies like 'one', 'yes', 'that one', or 'give me the same'. "
        "Match items only to the provided menu. If the message is too vague to identify a specific item, set needs_clarification true. "
        "If the driver is finishing their order, set checkout true. "
        "Return JSON with exactly these keys: intent, checkout, needs_clarification, items. "
        "intent must be one of: add_item, modify_order, checkout, question, greeting, unknown. "
        "items must be an array of objects with name and quantity. Quantity must be an integer >= 1. "
        "Do not include markdown, code fences, or extra text.\n\n"
        "Menu:\n"
        + ("\n".join(menu_lines) if menu_lines else "(no menu provided)")
        + "\n\nConversation:\n"
        + ("\n".join(history_lines) if history_lines else "No prior conversation.")
        + "\nDriver: "
        + str(user_message).strip()
        + "\nJSON:"
    )

    data = call_ollama_json(prompt, temperature=0.05)
    if not isinstance(data, dict):
        return None

    intent = str(data.get("intent") or "unknown").strip().lower()
    checkout = bool(data.get("checkout"))
    needs_clarification = bool(data.get("needs_clarification"))

    items = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        quantity = item.get("quantity")
        if not name:
            continue
        try:
            quantity = max(1, min(int(quantity), 20))
        except (TypeError, ValueError):
            quantity = 1
        items.append({"name": name, "quantity": quantity})

    return {
        "intent": intent,
        "checkout": checkout,
        "needs_clarification": needs_clarification,
        "items": items,
    }


def infer_menu_correction_with_ai(user_message, history, menu_items):
    """Use Ollama to suggest likely menu items for unclear/mispronounced requests."""
    if not user_message or not menu_items:
        return None

    menu_lines = []
    for item in menu_items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        price = float(item.get("price") or 0)
        category = str(item.get("category") or "").strip() or "Other"
        menu_lines.append(f"- {name} | {category} | RM{price:.2f}")

    if not menu_lines:
        return None

    history_lines = []
    for turn in (history or [])[-8:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            speaker = "Driver" if role == "user" else "Assistant"
            history_lines.append(f"{speaker}: {content[:300]}")

    prompt = (
        "You map unclear drive-thru requests to likely menu items. "
        "Given the latest driver message, return likely intended menu item names from the provided menu only. "
        "This includes handling mispronunciations and spelling mistakes. "
        "Return JSON with exactly these keys: understood, confidence, suggestions. "
        "understood must be true only when you can confidently infer intended item(s). "
        "confidence must be a number 0.0 to 1.0 for the top suggestion. "
        "suggestions must be an array of up to 3 exact menu names, most likely first. "
        "No markdown, no extra keys, no extra text.\n\n"
        "Menu:\n"
        + "\n".join(menu_lines)
        + "\n\nConversation:\n"
        + ("\n".join(history_lines) if history_lines else "No prior conversation.")
        + "\nDriver: "
        + str(user_message).strip()
        + "\nJSON:"
    )

    data = call_ollama_json(prompt, temperature=0.05)
    if not isinstance(data, dict):
        return None

    understood = bool(data.get("understood"))
    try:
        confidence = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    raw_suggestions = data.get("suggestions") or []
    if not isinstance(raw_suggestions, list):
        raw_suggestions = []

    menu_names = {
        str(item.get("name") or "").strip().lower(): str(item.get("name") or "").strip()
        for item in menu_items
        if str(item.get("name") or "").strip()
    }

    suggestions = []
    seen = set()
    for name in raw_suggestions:
        key = str(name or "").strip().lower()
        canonical = menu_names.get(key)
        if not canonical:
            continue
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        suggestions.append(canonical)
        if len(suggestions) >= 3:
            break

    if not suggestions:
        return None

    return {
        "understood": understood,
        "confidence": confidence,
        "suggestions": suggestions,
    }


def _get_current_order_summary(plate_number):
    """
    Return the confirmed order lines for this plate as a plain-text string,
    e.g. "2x Speed Classic Burger RM9.90, 1x Spicy Chicken Burger RM13.90".
    Returns an empty string when no active order exists or on any DB error.
    """
    try:
        with get_cursor(dict_cursor=True) as cursor:
            cursor.execute(
                """
                SELECT o.order_id
                FROM orders o
                INNER JOIN vehicles v ON v.vehicle_id = o.vehicle_id
                WHERE v.plate_number = %s AND o.status IN ('Pending', 'Preparing')
                ORDER BY o.order_id DESC LIMIT 1
                """,
                (plate_number,),
            )
            order_row = cursor.fetchone()
            if not order_row:
                return ""

            cursor.execute(
                """
                SELECT mi.name, oi.quantity, oi.unit_price
                FROM order_items oi
                INNER JOIN menu_items mi ON mi.menu_id = oi.menu_id
                WHERE oi.order_id = %s
                ORDER BY oi.order_item_id ASC
                """,
                (int(order_row["order_id"]),),
            )
            lines = cursor.fetchall() or []
            if not lines:
                return ""

            parts = [
                f"{int(row['quantity'])}x {row['name']} RM{float(row['unit_price']):.2f}"
                for row in lines
            ]
            return ", ".join(parts)
    except Exception:
        logger.debug("Could not fetch current order for plate %s", plate_number, exc_info=True)
        return ""


def _category_recommendations(menu_items, category, limit=2):
    picks = []
    for item in menu_items or []:
        if not item or not item.get("available"):
            continue
        if str(item.get("category") or "").strip().lower() != category:
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        price = float(item.get("price") or 0)
        picks.append(f"{name} RM{price:.2f}")
        if len(picks) >= limit:
            break
    return picks


def _recommendation_reply_for_side_drink(cleaned_message, menu_items):
    if not menu_items:
        return None

    lowered = _normalize_message(cleaned_message)
    asks_side = any(
        key in lowered for key in ["side", "sides", "fries", "nuggets", "rings", "site"]
    )
    asks_drink = any(
        key in lowered
        for key in [
            "drink",
            "drinks",
            "coke",
            "sprite",
            "water",
            "coffee",
            "tea",
            "latte",
            "americano",
            "milkshake",
        ]
    )

    if not asks_side and not asks_drink:
        return None

    side_picks = _category_recommendations(menu_items, "side") if asks_side else []
    drink_picks = _category_recommendations(menu_items, "drink") if asks_drink else []

    if side_picks and drink_picks:
        return (
            f"For sides I recommend {', '.join(side_picks)}. "
            f"For drinks I recommend {', '.join(drink_picks)}."
        )
    if side_picks:
        return f"For sides I recommend {', '.join(side_picks)}. Which side would you like?"
    if drink_picks:
        return f"For drinks I recommend {', '.join(drink_picks)}. What sounds good?"
    return None


def _post_add_upsell_reply(items_just_added, order_category_flags, menu_items):
    added_categories = {
        str(item.get("category") or "").strip().lower()
        for item in (items_just_added or [])
        if str(item.get("category") or "").strip()
    }
    if "combo" in added_categories:
        return "Would you like anything else?"
    if "burger" not in added_categories:
        return "Would you like anything else?"

    flags = order_category_flags or {}
    has_side = bool(flags.get("has_side"))
    has_drink = bool(flags.get("has_drink"))
    has_combo = bool(flags.get("has_combo"))
    if has_combo or (has_side and has_drink):
        return "Would you like anything else?"

    side_picks = _category_recommendations(menu_items, "side", limit=2)
    drink_picks = _category_recommendations(menu_items, "drink", limit=2)

    if not has_side and not has_drink and side_picks and drink_picks:
        return (
            f"Would you like to add a side and a drink? "
            f"Popular picks are {', '.join(side_picks)} and {', '.join(drink_picks)}."
        )
    if not has_side and side_picks:
        return f"Would you like to add a side as well? Popular picks are {', '.join(side_picks)}."
    if not has_drink and drink_picks:
        return f"Would you like to add a drink as well? Popular picks are {', '.join(drink_picks)}."
    return "Would you like anything else?"


def _post_modify_reply(order_action, items_removed, items_updated):
    if order_action == "remove":
        if not items_removed:
            return "I could not find that item in your current order. What would you like to change?"
        parts = [
            f"removed {int(it['quantity'])}x {it['name']}"
            for it in items_removed
        ]
        return ", ".join(parts).capitalize() + ". What would you like next?"

    if order_action == "set":
        if not items_updated:
            return "I could not update that item yet. Tell me the exact quantity you want."
        parts = []
        for it in items_updated:
            quantity = int(it.get("quantity") or 0)
            if quantity <= 0:
                parts.append(f"removed {it['name']}")
            else:
                parts.append(f"updated {it['name']} to {quantity}")
        return ", ".join(parts).capitalize() + ". Anything else?"

    return None


_NATURALIZE_EVENT_DESCRIPTIONS = {
    "item_added": "The customer just added items — confirm what was added",
    "checkout_preview": "The customer is done ordering — read back their full order and ask them to confirm before sending to kitchen",
    "checkout_confirmed": "The customer confirmed their order — thank them and let them know it is being prepared",
    "order_modified": "The customer changed or removed items — confirm what changed",
}


def _naturalize_order_reply(event_type, facts_text, fallback, user_message="", history=None, extra_context=""):
    """Let the model phrase deterministic order facts more naturally without changing them."""
    if not facts_text:
        return fallback

    history_lines = []
    for turn in (history or [])[-4:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            speaker = "Customer" if role == "user" else "You"
            history_lines.append(f"{speaker}: {content[:200]}")

    event_desc = _NATURALIZE_EVENT_DESCRIPTIONS.get(event_type, event_type.replace("_", " "))

    prompt = (
        "You are a warm, natural-sounding AI assistant working the drive-thru at Speed Burger. "
        "You sound like a real person — varied, friendly, never robotic or scripted.\n\n"
        f"Situation: {event_desc}\n"
        f"Order facts (reproduce these exactly — never change item names, quantities, or prices): {facts_text}\n"
        + (f"Upsell or context hint: {extra_context}\n" if extra_context else "")
        + ("Recent conversation:\n" + "\n".join(history_lines) + "\n" if history_lines else "")
        + f"Customer said: {user_message}\n"
        "Write a single natural reply that:\n"
        "  - Includes the exact order facts above\n"
        "  - Varies your opener — never use the same greeting twice (avoid always starting with 'Got it' or 'Perfect')\n"
        "  - Sounds like a human drive-thru cashier, not a bot\n"
        "  - Is at most 2 short sentences\n"
        "  - Uses no markdown or bullet points\n"
        "Reply:"
    )

    phrased = call_ollama_text(
        prompt,
        temperature=0.55,
        stop=["Customer:", "\nCustomer", "Driver:", "\n\n"],
    )
    return phrased or fallback


def generate_vehicle_chat_reply(plate_number, user_message, history, direction="", menu_items=None,
                                items_just_added=None, items_removed=None,
                                items_updated=None, order_action="add", order_category_flags=None,
                                checkout_requested=False, checkout_order_lines=None,
                                checkout_confirmation_required=False,
                                order_not_found=False, needs_clarification=False,
                                clarification_suggestions=None, clarification_quantity=1,
                                clarification_reason="", clarification_target=""):
    """
    Generate a drive-thru ordering reply.
    - Item confirmations and checkout summaries are built deterministically (no LLM).
    - Ollama is called only for Q&A turns: greetings, recommendations, unknown queries.
    Falls back to generate_local_order_reply when Ollama is unavailable.
    """
    cleaned_message = (user_message or "").strip()
    if not cleaned_message:
        return "Please tell me what you would like to order."

    direction = (direction or "").strip().upper()
    goodbye_words = {
        "bye", "bye bye", "goodbye", "see you", "see you later", "thanks", "thank you", "thankyou"
    }

    if direction == "OUT":
        if any(word in _normalize_message(cleaned_message) for word in goodbye_words):
            return (
                f"Thank you for choosing Speed Burger, {plate_number}. "
                "Come back and see us soon!"
            )
        return (
            f"Thank you for choosing Speed Burger, {plate_number}. "
            "Come back and see us soon!"
        )

    safe_history = []
    for item in (history or [])[-6:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            safe_history.append({"role": role, "content": content[:400]})

    history_lines = []
    for turn in safe_history:
        speaker = "Driver" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{speaker}: {turn['content']}")

    conversation = "\n".join(history_lines)

    # ------------------------------------------------------------------ #
    # DETERMINISTIC PATH 1: item confirmation                            #
    # Always accurate — never let the LLM recount quantities.            #
    # ------------------------------------------------------------------ #
    if items_just_added and not checkout_requested:
        parts = [
            f"{int(it['quantity'])}x {it['name']} RM{float(it['unit_price']):.2f}"
            for it in items_just_added
        ]
        upsell = _post_add_upsell_reply(items_just_added, order_category_flags, menu_items)
        facts = ", ".join(parts)
        # Pass the upsell as natural language context so the AI decides how to phrase it
        upsell_hint = upsell if upsell != "Would you like anything else?" else ""
        return _naturalize_order_reply(
            event_type="item_added",
            facts_text=facts,
            fallback=facts + f". {upsell}",
            user_message=cleaned_message,
            history=safe_history,
            extra_context=upsell_hint,
        )

    modification_reply = _post_modify_reply(order_action, items_removed, items_updated)
    if modification_reply and not checkout_requested:
        facts_parts = []
        for it in items_removed or []:
            facts_parts.append(f"removed {int(it['quantity'])}x {it['name']}")
        for it in items_updated or []:
            quantity = int(it.get("quantity") or 0)
            if quantity <= 0:
                facts_parts.append(f"removed {it['name']}")
            else:
                facts_parts.append(f"updated {it['name']} to {quantity}")
        return _naturalize_order_reply(
            event_type="order_modified",
            facts_text=", ".join(facts_parts),
            fallback=modification_reply,
            user_message=cleaned_message,
            history=safe_history,
        )

    if checkout_confirmation_required:
        lines = checkout_order_lines or []
        if lines:
            parts = [
                f"{int(row['quantity'])}x {row['name']} RM{float(row['unit_price']):.2f}"
                for row in lines
            ]
            total = sum(float(r.get("unit_price", 0)) * int(r.get("quantity", 1)) for r in lines)
            facts = ", ".join(parts)
            fallback = (
                "Please confirm your order: "
                + facts
                + f" — Total RM{total:.2f}. Reply yes to confirm, or tell me what to change."
            )
            return _naturalize_order_reply(
                event_type="checkout_preview",
                facts_text=facts,
                fallback=fallback,
                user_message=cleaned_message,
                history=safe_history,
                extra_context=f"Total comes to RM{total:.2f}.",
            )
        return "Please confirm your order by replying yes, or tell me what to change."

    # ------------------------------------------------------------------ #
    # DETERMINISTIC PATH 2: checkout summary                             #
    # Read from DB-confirmed order_lines so the summary is always exact. #
    # ------------------------------------------------------------------ #
    if checkout_requested:
        lines = checkout_order_lines or []
        if lines:
            parts = [
                f"{int(row['quantity'])}x {row['name']} RM{float(row['unit_price']):.2f}"
                for row in lines
            ]
            total = sum(float(r.get("unit_price", 0)) * int(r.get("quantity", 1)) for r in lines)
            facts = ", ".join(parts)
            fallback = "Your order: " + facts + f" — Total RM{total:.2f}. Thank you for choosing Speed Burger!"
            return _naturalize_order_reply(
                event_type="checkout_confirmed",
                facts_text=facts,
                fallback=fallback,
                user_message=cleaned_message,
                history=safe_history,
                extra_context=f"Total RM{total:.2f}.",
            )
        return "Thank you for choosing Speed Burger! Have a great day."

    # ------------------------------------------------------------------ #
    # DETERMINISTIC PATH 3: item not found on menu                       #
    # Prevents Ollama from faking an order confirmation.                 #
    # ------------------------------------------------------------------ #
    if needs_clarification:
        suggestions = clarification_suggestions or []
        if suggestions:
            names = [str(item.get("name") or "").strip() for item in suggestions]
            names = [name for name in names if name]
            quantity = max(1, int(clarification_quantity or 1))
            if names:
                if clarification_reason == "unknown_combo":
                    combo_label = (clarification_target or "that combo").strip()
                    return (
                        f"We do not have {combo_label}. "
                        f"We have {', '.join(names)}. Which one would you like?"
                    )

                if len(names) == 1:
                    return (
                        f"Did you mean {quantity}x {names[0]}? "
                        "Please reply yes to confirm, or tell me the exact item name."
                    )

                top = names[0]
                alternatives = ", ".join(names[1:3])
                if alternatives:
                    return (
                        f"Did you mean {quantity}x {top}? "
                        f"If not, choose one: {alternatives}."
                    )
                return (
                    f"Did you mean {quantity}x {top}? "
                    "Please reply yes to confirm, or tell me the exact item name."
                )

        if menu_items:
            avail = [mi["name"] for mi in menu_items if mi.get("available")]
            hint = ", ".join(avail[:3]) + (" and more" if len(avail) > 3 else "")
            return f"Which item would you like one of? We currently have {hint}."
        return "Which item would you like one of?"

    if order_not_found:
        if menu_items:
            avail = [mi["name"] for mi in menu_items if mi.get("available")]
            hint = ", ".join(avail[:3]) + (" and more" if len(avail) > 3 else "")
            return f"Sorry, I don't have that on our menu. We have {hint}. What would you like?"
        return "Sorry, I don't have that item on our menu. What would you like to order?"

    # ------------------------------------------------------------------ #
    # OLLAMA PATH: greetings, recommendations, Q&A, anything unhandled   #
    # ------------------------------------------------------------------ #

    recommendation_request = is_recommendation_request(cleaned_message)

    # Build menu block for the prompt
    if menu_items:
        menu_lines = [
            f"- {mi['name']} RM{float(mi['price']):.2f}"
            + (" [SOLD OUT]" if not mi.get("available") else "")
            + ((" — " + str(mi.get("description") or "").strip()) if mi.get("description") else "")
            for mi in menu_items
        ]
        menu_block = "Speed Burger menu:\n" + "\n".join(menu_lines)
    else:
        menu_block = "(menu not available)"

    confirmed_order = _get_current_order_summary(plate_number)
    order_context = (
        f"Customer's order so far: {confirmed_order}"
        if confirmed_order
        else "Customer's order so far: nothing ordered yet."
    )

    prompt = (
        "You are an intelligent, friendly AI assistant running the drive-thru at Speed Burger. "
        "You sound like a real person — natural, warm, varied. Never robotic or scripted.\n"
        "Your job:\n"
        "- Help customers decide what to order, give honest recommendations based on their preference\n"
        "- If they mention a food preference (spicy, chicken, light, etc.), suggest the best match with a short reason\n"
        "- If they ask about the menu, highlight 1-2 standout items naturally\n"
        "- If their request is unclear, ask ONE short clarifying question\n"
        "- Never confirm or add an order — that is handled separately\n"
        "- Never invent items or prices not on the menu\n"
        "- Max 2 sentences. No lists, no markdown, no bullet points.\n\n"
        f"{menu_block}\n\n"
        f"{order_context}\n\n"
        "Conversation:\n"
        f"{conversation if conversation else 'Customer just pulled up.'}\n"
        f"Customer: {cleaned_message}\n"
        "Assistant:"
    )

    temperature = 0.45 if recommendation_request else 0.3
    model_reply = call_ollama_text(prompt, temperature=temperature, stop=["Customer:", "\nCustomer", "Driver:"])
    if model_reply:
        return model_reply
    return generate_local_order_reply(cleaned_message, safe_history, menu_items)