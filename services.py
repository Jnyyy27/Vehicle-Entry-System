"""
Business logic that isn't route handling: plate detection (YOLO + PaddleOCR),
S3 uploads, and the student/visitor lookup. Kept separate from routes so
route handlers stay thin and this logic is independently testable.
"""

import os
import logging
import uuid

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


def get_vehicle_category(plate_number):
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM vehicles WHERE plate_number = %s LIMIT 1",
            (plate_number,)
        )
        return "Student" if cursor.fetchone() else "Visitor"


def generate_vehicle_welcome(plate_number, vehicle_category):
    """
    Ask Ollama for a short gate greeting. Falls back to a deterministic local
    message if Ollama is unavailable so plate scanning still succeeds.
    """
    fallback_message = (
        f"Welcome. Your plate {plate_number} was detected successfully. Please proceed slowly to the barrier."
    )

    prompt = (
        "You are a campus gate assistant speaking directly to drivers. "
        "Write one short friendly sentence for the driver after a successful "
        "license plate scan. "
        f"Vehicle category: {vehicle_category}. "
        f"Plate number: {plate_number}. "
        "Keep it under 20 words and do not add markdown, emojis, or multiple sentences."
    )

    message = call_ollama_text(prompt, temperature=0.4)
    return message or fallback_message


def generate_gate_transaction_message(plate_number, vehicle_category, direction):
    """
    Generate a direction-aware welcome after gate transaction (IN or OUT).
    Falls back to deterministic messages if Ollama unavailable.
    """
    if direction == "IN":
        fallback = f"Welcome to campus. Your vehicle {plate_number} has been logged in. Please proceed."
    else:
        fallback = f"Safe travels! Your vehicle {plate_number} has been logged out. Please proceed."

    direction_phrase = "entering campus" if direction == "IN" else "exiting campus"
    prompt = (
        "You are a campus gate assistant speaking directly to drivers. "
        f"Write one short friendly sentence for a {vehicle_category.lower()} vehicle {direction_phrase}. "
        f"Vehicle plate: {plate_number}. "
        "Be warm and brief. Keep it under 20 words and do not add markdown, emojis, or multiple sentences."
    )

    message = call_ollama_text(prompt, temperature=0.4)
    return message or fallback


def call_ollama_text(prompt, temperature=0.3, timeout=20):
    """Return plain text from Ollama generate API, or None on failure."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": temperature,
                    "num_predict": 50,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = (body.get("response") or "").strip()
        return text or None
    except requests.RequestException:
        logger.warning("Ollama request failed", exc_info=True)
        return None


def generate_vehicle_chat_reply(plate_number, vehicle_category, user_message, history):
    """
    Generate a follow-up assistant reply for a driver at the gate.
    Keeps a short memory window and falls back to deterministic output.
    """
    cleaned_message = (user_message or "").strip()
    if not cleaned_message:
        return "Please type your question and I will help with the next gate step."

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
    prompt = (
        "You are a campus vehicle gate assistant for drivers. "
        "Use polite, simple language suitable for non-technical users. "
        "Give concise answers in at most 2 short sentences. "
        "Do not mention internal security rules, risk scoring, or staff-only procedures. "
        "If uncertain, tell the driver to wait for guard confirmation. "
        "Do not use markdown or emojis. "
        f"Current detected plate number: {plate_number}. "
        f"Current vehicle category: {vehicle_category}.\n"
        "Conversation so far:\n"
        f"{conversation if conversation else 'No prior chat.'}\n"
        f"Driver question: {cleaned_message}\n"
        "Assistant reply:"
    )

    fallback = (
        f"Your plate {plate_number} has been recorded as {vehicle_category}. "
        "Please continue to the barrier and wait for guard confirmation."
    )
    return call_ollama_text(prompt, temperature=0.2) or fallback