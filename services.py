"""
Business logic that isn't route handling: plate detection (YOLO + PaddleOCR),
S3 uploads, and drive-thru ordering. Kept separate from routes so
route handlers stay thin and this logic is independently testable.
"""

import os
import logging
import uuid
import re

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


def _normalize_message(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


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
    checkout_phrases = ["that is all", "thats all", "that's all", "nothing else",
                        "no more", "that will be all", "i'm done", "im done", "no thanks"]
    checkout_words = {"done", "checkout", "pay", "finish", "finished"}
    if any(p in cleaned for p in checkout_phrases) or (cleaned_words & checkout_words):
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


def generate_vehicle_chat_reply(plate_number, user_message, history, direction="", menu_items=None,
                                items_just_added=None, checkout_requested=False, checkout_order_lines=None,
                                order_not_found=False):
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
        return ", ".join(parts) + ". Would you like anything else?"

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
            return "Your order: " + ", ".join(parts) + ". Thank you for choosing Speed Burger!"
        return "Thank you for choosing Speed Burger! Have a great day."

    # ------------------------------------------------------------------ #
    # DETERMINISTIC PATH 3: item not found on menu                       #
    # Prevents Ollama from faking an order confirmation.                 #
    # ------------------------------------------------------------------ #
    if order_not_found:
        if menu_items:
            avail = [mi["name"] for mi in menu_items if mi.get("available")]
            hint = ", ".join(avail[:3]) + (" and more" if len(avail) > 3 else "")
            return f"Sorry, I don't have that on our menu. We have {hint}. What would you like?"
        return "Sorry, I don't have that item on our menu. What would you like to order?"

    # ------------------------------------------------------------------ #
    # OLLAMA PATH: Q&A turns (greetings, recommendations, queries)       #
    # ------------------------------------------------------------------ #

    # Build menu block for the prompt
    if menu_items:
        menu_lines = [
            f"- {mi['name']} RM{float(mi['price']):.2f}{' [SOLD OUT]' if not mi.get('available') else ''}"
            for mi in menu_items
        ]
        menu_block = "Speed Burger menu (name and unit price):\n" + "\n".join(menu_lines)
    else:
        menu_block = "(menu not available)"

    confirmed_order = _get_current_order_summary(plate_number)
    order_context = (
        f"Confirmed order so far: {confirmed_order}"
        if confirmed_order
        else "Confirmed order so far: nothing ordered yet."
    )

    prompt = (
        "You are a Speed Burger drive-thru order-taking assistant. "
        "Your ONLY job is to answer the driver's question or make a recommendation.\n"
        "RULES:\n"
        "1. Do NOT take orders or confirm quantities in this reply — order saving is handled separately.\n"
        "2. For menu/recommendation questions, suggest 1-2 items by name and unit price.\n"
        "3. For signature/classic queries, recommend Speed Classic Burger with a brief description.\n"
        "4. Only recommend items on the menu. If not listed, apologise and suggest the closest match.\n"
        "5. Reply in 1-2 short sentences. No markdown, no emojis, no plate numbers.\n\n"
        f"{menu_block}\n\n"
        f"{order_context}\n\n"
        "Conversation so far:\n"
        f"{conversation if conversation else 'No prior exchanges.'}\n"
        f"Driver: {cleaned_message}\n"
        "Assistant:"
    )

    model_reply = call_ollama_text(prompt, temperature=0.15, stop=["Driver:", "\nDriver", "User:"])
    if model_reply:
        return model_reply
    return generate_local_order_reply(cleaned_message, safe_history, menu_items)