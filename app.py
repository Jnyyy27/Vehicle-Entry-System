import os
import uuid
import boto3
from botocore.exceptions import ClientError
from urllib import response
import pymysql
import requests
import jwt
import base64
import cv2
import numpy as np
from functools import wraps
from jwt import PyJWKClient
from dotenv import load_dotenv
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
from flask import Flask, request, redirect, render_template, session, flash, url_for
from ultralytics import YOLO
from paddleocr import PaddleOCR

load_dotenv()

plate_model = YOLO("models/license_plate_detector.pt")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

# Limit upload size to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# AWS S3 Configuration
S3_BUCKET   = os.getenv("S3_BUCKET")
S3_REGION   = os.getenv("S3_REGION", "ap-southeast-1")

s3_client = boto3.client("s3", region_name=S3_REGION)

COGNITO_DOMAIN = os.getenv("COGNITO_DOMAIN")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
REDIRECT_URI = os.getenv("REDIRECT_URI")

# Region is just the prefix of the user pool ID (e.g. "ap-southeast-1_AbCdEfGhI")
COGNITO_REGION = COGNITO_USER_POOL_ID.split("_")[0]
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"

# Fetches and caches Cognito's public signing keys. Refreshes automatically
# if it sees a key id (kid) it doesn't recognize yet (e.g. after key rotation).
jwks_client = PyJWKClient(JWKS_URL)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        token = session.get("id_token")

        if not token:
            return redirect("/login")

        try:
            user = verify_id_token(token)

            # IMPORTANT: rebuild session user every request
            groups = user.get("cognito:groups", [])

            if "Admin" in groups:
                role = "Admin"
            elif "SecurityGuard" in groups:
                role = "SecurityGuard"
            else:
                role = "User"

            session["user"] = {
                "email": user["email"],
                "sub": user["sub"],
                "role": role
            }

        except jwt.InvalidTokenError:
            session.clear()
            return redirect("/login")

        return f(*args, **kwargs)

    return decorated

def verify_id_token(token):
    """
    Verifies a Cognito ID token's signature, issuer, audience and expiry.
    Raises jwt.PyJWTError (or a subclass) if anything is invalid.
    Returns the decoded claims dict if the token is genuine.
    """
    signing_key = jwks_client.get_signing_key_from_jwt(token)

    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=CLIENT_ID,
        issuer=COGNITO_ISSUER,
    )

    # Cognito issues both ID tokens and access tokens; make sure we were
    # handed the right one (access tokens don't carry email/sub the same way).
    if claims.get("token_use") != "id":
        raise jwt.InvalidTokenError("Expected an ID token")

    return claims

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def upload_to_s3(local_path, s3_key):
    """
    Upload a local file to S3.  Returns the public HTTPS URL, or None if the
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
    except ClientError as exc:
        print(f"[S3] Upload failed for {s3_key}: {exc}")
        return None


def detect_plate(image_path):
    """
    Detect a license plate in an image using YOLO, then read its text with
    PaddleOCR.  Returns the plate string in upper-case, or None.
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

    # Upload cropped plate to S3 (non-blocking — failures are logged, not raised)
    upload_to_s3(crop_path, f"cropped/{crop_name}")

    # Run OCR on the cropped region
    try:
        ocr_result = ocr.ocr(crop_path, cls=True)
    except TypeError:
        ocr_result = ocr.ocr(crop_path)

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

    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM vehicles
        WHERE plate_number = %s
    """, (plate_number,))

    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        return "Student"
    else:
        return "Visitor"

@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login")
def login():
    return redirect(
        f"{COGNITO_DOMAIN}/login?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=email+openid+phone"
    )

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        user=session.get("user", {})
)

@app.route("/vehicles", methods=["GET", "POST"])
@login_required
def vehicles():

    # Only Admin can access this page
    if session.get("user", {}).get("role") != "Admin":
        return render_template("403.html"), 403

    if request.method == "POST":

        student_name = request.form["student_name"]
        student_id = request.form["student_id"]
        plate_number = request.form["plate_number"]
        vehicle_type = request.form["vehicle_type"]

        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )

        cursor = conn.cursor()

        sql = """
        INSERT INTO vehicles
        (student_name, student_id, plate_number, vehicle_type)
        VALUES (%s,%s,%s,%s)
        """

        cursor.execute(sql, (
            student_name,
            student_id,
            plate_number,
            vehicle_type
        ))

        conn.commit()

        cursor.close()
        conn.close()

        flash(f"Vehicle {plate_number} registered successfully!", "success")
        return redirect(url_for("vehicles"))

    return render_template("vehicles.html")

@app.route("/logs")
@login_required
def logs():

    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, plate_number, vehicle_category, status, entry_time
        FROM entry_logs
        ORDER BY id DESC
    """)

    logs = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("logs.html", logs=logs)

@app.route("/entry", methods=["GET", "POST"])
@login_required
def entry():

    if request.method == "POST":
        plate_number = request.form["plate_number"]

        # Student or Visitor
        vehicle_category = get_vehicle_category(plate_number)

        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )

        cursor = conn.cursor()

        # Check the latest status
        cursor.execute("""
            SELECT status
            FROM entry_logs
            WHERE plate_number = %s
            ORDER BY entry_time DESC
            LIMIT 1
        """, (plate_number,))

        last_log = cursor.fetchone()

        # Determine IN / OUT
        if last_log is None:
            status = "IN"
        elif last_log[0] == "IN":
            status = "OUT"
        else:
            status = "IN"

        # Save the new log
        cursor.execute("""
            INSERT INTO entry_logs
            (plate_number, vehicle_category, status)
            VALUES (%s, %s, %s)
        """, (
            plate_number,
            vehicle_category,
            status
        ))

        conn.commit()
        cursor.close()
        conn.close()

        flash(
            f"{vehicle_category} vehicle {plate_number} recorded as {status}.",
            "success"
        )

        return redirect(url_for("entry"))

    return render_template("entry.html")

@app.route("/scan-plate", methods=["POST"])
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

    # Upload original photo to S3
    s3_original_url = upload_to_s3(filepath, f"upload/{filename}")

    try:
        plate_text = detect_plate(filepath)
    except Exception as exc:
        return {"error": f"Detection error: {exc}"}, 500

    if not plate_text:
        return {"error": "No license plate detected in the image."}, 422

    result = {"plate_number": plate_text}
    if s3_original_url:
        result["s3_url"] = s3_original_url
    return result

@app.route("/callback")
def callback():
    code = request.args.get("code")

    token_url = f"{COGNITO_DOMAIN}/oauth2/token"

    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {auth_b64}"
    }

    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }

    response = requests.post(token_url, data=data, headers=headers)
    print(response.status_code)
    print(response.text)
    tokens = response.json()

    print("TOKEN RESPONSE:")
    print(tokens)
    id_token = tokens.get("id_token")
    print("ID TOKEN:")
    print(id_token)
    print(type(id_token))

    try:
        # Verify JWT and get user information
        user = verify_id_token(id_token)
        print(user)

        # Get Cognito Group
        groups = user.get("cognito:groups", [])

        if "Admin" in groups:
            role = "Admin"
        elif "SecurityGuard" in groups:
            role = "SecurityGuard"
        else:
            role = "User"

        # Store everything in session
        session["id_token"] = id_token
        session["access_token"] = tokens.get("access_token")

        session["user"] = {
            "email": user["email"],
            "sub": user["sub"],
            "role": role
        }

    except jwt.PyJWTError as e:
        print("TOKEN VERIFICATION FAILED:", e)
        return redirect("/login")

    return redirect("/dashboard")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)