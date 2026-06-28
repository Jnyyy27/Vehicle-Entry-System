import os
from urllib import response
import pymysql
import requests
import jwt
import base64
from functools import wraps
from jwt import PyJWKClient
from dotenv import load_dotenv
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
from flask import Flask, request, redirect, render_template, session

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

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
            session["user"] = user
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            session.clear()
            return redirect("/login")
        except jwt.InvalidTokenError:
            print("Invalid token")
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
        user=session["user"]
)

@app.route("/vehicles", methods=["GET", "POST"])
@login_required
def vehicles():

    # Only Admin can access this page
    if session["user"]["role"] != "Admin":
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

        return "Vehicle Registered Successfully!"

    return render_template("vehicles.html")

@app.route("/logs")
@login_required
def logs():

    if "user" not in session:
        return redirect("/login")

    # Both Admin and Security Guard
    return render_template("logs.html")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)