import base64
import logging
import os
from urllib.parse import urlencode, urlsplit, urlunsplit

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


def _allowed_redirect_targets():
    targets = {REDIRECT_URI}

    frontend_origins = os.getenv("FRONTEND_ORIGIN", "")
    for origin in frontend_origins.split(","):
        origin = origin.strip()
        if origin:
            targets.add(origin)

    return targets


def _normalize_redirect_origin(target):
    parsed_target = urlsplit(target)
    if not parsed_target.scheme or not parsed_target.netloc:
        return None

    return urlunsplit((parsed_target.scheme, parsed_target.netloc, "", "", ""))


def _is_safe_redirect_target(target):
    if not target:
        return False

    parsed_target = urlsplit(target)
    if not parsed_target.scheme or not parsed_target.netloc:
        return target.startswith("/") and not target.startswith("//")

    normalized_target = _normalize_redirect_origin(target)
    allowed_origins = {
        _normalize_redirect_origin(candidate) or candidate
        for candidate in _allowed_redirect_targets()
    }
    return normalized_target in allowed_origins


def _get_post_login_target():
    target = (request.args.get("next") or "").strip()
    if not _is_safe_redirect_target(target):
        return "/dashboard"
    return target


def _build_callback_redirect(target, tokens=None):
    if not target or not _is_safe_redirect_target(target):
        return "/dashboard"

    parsed_target = urlsplit(target)
    if not parsed_target.scheme or not parsed_target.netloc:
        return target

    if not tokens:
        return target

    fragment = urlencode(
        {
            "id_token": tokens.get("id_token", ""),
        }
    )
    return urlunsplit(
        (
            parsed_target.scheme,
            parsed_target.netloc,
            parsed_target.path,
            parsed_target.query,
            fragment,
        )
    )


def _clear_post_login_target():
    session.pop("post_login_redirect", None)


@auth_bp.route("/")
def home():
    return render_template("login.html")


@auth_bp.route("/login")
def login():
    next_target = _get_post_login_target()
    session["post_login_redirect"] = next_target

    return redirect(
        f"{COGNITO_DOMAIN}/login?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope=email+openid+phone"
    )


@auth_bp.route("/callback")
def callback():
    redirect_target = session.get("post_login_redirect", "/dashboard")
    code = request.args.get("code")

    if not code:
        logger.error("Cognito callback missing authorization code")
        _clear_post_login_target()
        return redirect(_build_callback_redirect(redirect_target))

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
        _clear_post_login_target()
        return redirect(_build_callback_redirect(redirect_target))

    try:
        # Verify JWT and get user information
        user = verify_id_token(id_token)
        groups = user.get("cognito:groups", [])
        role = role_from_groups(groups)

        is_external_redirect = bool(urlsplit(redirect_target).scheme and urlsplit(redirect_target).netloc)

        # For legacy server-rendered pages, keep session-based login behavior.
        # For Flutter/external redirects, avoid placing large JWTs in the session
        # cookie because that can exceed header limits behind Nginx and trigger 502.
        session["user"] = {
            "email": user["email"],
            "sub": user["sub"],
            "role": role
        }
        if not is_external_redirect:
            # NOTE: never log id_token / access_token — they are live credentials
            session["id_token"] = id_token
            session["access_token"] = tokens.get("access_token")
        else:
            session.pop("id_token", None)
            session.pop("access_token", None)

        logger.info("User %s logged in with role %s", user["email"], role)

    except jwt.PyJWTError:
        logger.exception("Token verification failed during callback")
        _clear_post_login_target()
        return redirect(_build_callback_redirect(redirect_target))

    final_redirect = _build_callback_redirect(redirect_target, tokens=tokens)
    _clear_post_login_target()
    return redirect(final_redirect)


@auth_bp.route("/logout")
def logout():
    next_target = _get_post_login_target()
    session.clear()
    return redirect(next_target if next_target != "/dashboard" else "/")