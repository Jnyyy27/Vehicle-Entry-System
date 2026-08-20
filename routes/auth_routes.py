import logging
from urllib.parse import urlsplit

from flask import Blueprint, request, redirect, render_template, session

from auth import check_password
from db import get_cursor

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth_bp", __name__)


def _is_safe_redirect(target: str) -> bool:
    """Only allow local paths (no scheme / netloc) to prevent open redirects."""
    if not target:
        return False
    parsed = urlsplit(target)
    return (not parsed.scheme and not parsed.netloc and target.startswith("/"))


def _next_target() -> str:
    target = (request.args.get("next") or "").strip()
    return target if _is_safe_redirect(target) else "/dashboard"


@auth_bp.route("/")
def home():
    return render_template("login.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        next_url = (request.form.get("next") or "/dashboard").strip()
        if not _is_safe_redirect(next_url):
            next_url = "/dashboard"

        if not email or not password:
            error = "Email and password are required."
        else:
            with get_cursor(dict_cursor=True) as cursor:
                cursor.execute(
                    "SELECT email, password_hash, role FROM users WHERE email = %s LIMIT 1",
                    (email,),
                )
                row = cursor.fetchone()

            if row and check_password(password, row["password_hash"]):
                session["user"] = {"email": row["email"], "role": row["role"]}
                logger.info("User %s logged in with role %s", row["email"], row["role"])
                return redirect(next_url)
            else:
                error = "Invalid email or password."

    next_url = _next_target()
    return render_template("login.html", error=error, next=next_url)


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")