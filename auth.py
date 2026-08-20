"""
Local authentication: bcrypt passwords stored in the DB, sessions for the
web front-end, and HS256-signed JWTs for the API / Flutter clients.

No external identity provider.
"""

import datetime
import os
from functools import wraps

import bcrypt
import jwt
from dotenv import load_dotenv
from flask import g, render_template, request, redirect, session

load_dotenv()

_SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
_TOKEN_ALGORITHM = "HS256"
_TOKEN_EXPIRY_HOURS = 24


# ─────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt. Store the result in the DB."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def check_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches hashed (bcrypt, timing-attack resistant)."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ─────────────────────────────────────────────
# Token helpers  (used by /api/login -> Flutter)
# ─────────────────────────────────────────────

def create_token(user: dict) -> str:
    """Return a signed HS256 JWT for the given user dict."""
    payload = {
        "sub": user["email"],
        "email": user["email"],
        "role": user.get("role", "User"),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=_TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, _SECRET_KEY, algorithm=_TOKEN_ALGORITHM)


def verify_token(token: str):
    """Decode and verify a locally-signed JWT. Returns claims dict or None."""
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_TOKEN_ALGORITHM])
    except jwt.PyJWTError:
        return None


# ─────────────────────────────────────────────
# User resolution
# ─────────────────────────────────────────────

def _bearer_token_from_request():
    auth_header = (request.headers.get("Authorization") or "").strip()
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def get_api_user():
    """
    Resolve an authenticated user for any call (web session or API bearer).

    Priority:
    1) Flask session["user"]  -- set by the web /login form
    2) Authorization: Bearer <token>  -- set by Flutter / API clients

    Returns a user dict {email, role} or None.
    """
    user = session.get("user")
    if user:
        return user

    token = _bearer_token_from_request()
    if not token:
        return None

    claims = verify_token(token)
    if not claims:
        return None

    return {"email": claims["email"], "role": claims.get("role", "User")}


# ─────────────────────────────────────────────
# Route decorators
# ─────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_api_user():
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("user", {}).get("role") != "Admin":
            return render_template("403.html"), 403
        return f(*args, **kwargs)
    return decorated


def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_api_user()
        if not user:
            return {"error": "unauthorized"}, 401
        g.api_user = user
        return f(*args, **kwargs)
    return decorated


def api_admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = getattr(g, "api_user", None)
        if not user:
            user = get_api_user()
            if not user:
                return {"error": "unauthorized"}, 401
            g.api_user = user
        if user.get("role") != "Admin":
            return {"error": "forbidden"}, 403
        return f(*args, **kwargs)
    return decorated