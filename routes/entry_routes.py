import os
import uuid
import logging

from flask import Blueprint, request, redirect, render_template, flash, url_for

from auth import login_required
from db import get_cursor
from services import (
    ALLOWED_IMAGE_TYPES,
    upload_to_s3,
    detect_plate,
    get_vehicle_category,
)

logger = logging.getLogger(__name__)

entry_bp = Blueprint("entry_bp", __name__)


@entry_bp.route("/entry", methods=["GET", "POST"])
@login_required
def entry():

    if request.method == "POST":
        plate_number = request.form["plate_number"]

        # Student or Visitor
        vehicle_category = get_vehicle_category(plate_number)

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
                (plate_number, vehicle_category, status)
                VALUES (%s, %s, %s)
            """, (plate_number, vehicle_category, status))

        flash(
            f"{vehicle_category} vehicle {plate_number} recorded as {status}.",
            "success"
        )

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