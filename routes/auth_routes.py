import base64
import logging

import jwt
import requests
from flask import Blueprint, request, redirect, render_template, session

from auth import (
    COGNITO_DOMAIN,
    CLIENT_ID,
    CLIENT_SECRET,
    REDIRECT_URI,
    verify_id_token,
    role_from_groups,
)

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route("/")
def home():
    return render_template("login.html")


@auth_bp.route("/login")
def login():
    return redirect(
        f"{COGNITO_DOMAIN}/login?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=email+openid+phone"
    )


@auth_bp.route("/callback")
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

    token_response = requests.post(token_url, data=data, headers=headers)
    tokens = token_response.json()

    id_token = tokens.get("id_token")

    if not id_token:
        logger.error("Cognito token exchange failed: status=%s", token_response.status_code)
        return redirect("/login")

    try:
        # Verify JWT and get user information
        user = verify_id_token(id_token)
        groups = user.get("cognito:groups", [])
        role = role_from_groups(groups)

        # Store everything in session
        # NOTE: never log id_token / access_token — they are live credentials
        session["id_token"] = id_token
        session["access_token"] = tokens.get("access_token")

        session["user"] = {
            "email": user["email"],
            "sub": user["sub"],
            "role": role
        }

        logger.info("User %s logged in with role %s", user["email"], role)

    except jwt.PyJWTError:
        logger.exception("Token verification failed during callback")
        return redirect("/login")

    return redirect("/dashboard")


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")