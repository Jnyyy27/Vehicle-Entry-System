"""
Cognito ID-token verification and the login_required / admin_required
decorators used by every protected route.
"""

import os
from functools import wraps

import jwt
from jwt import PyJWKClient
from dotenv import load_dotenv
from flask import session, redirect, render_template

load_dotenv()

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


def role_from_groups(groups):
    if "Admin" in groups:
        return "Admin"
    if "SecurityGuard" in groups:
        return "SecurityGuard"
    return "User"


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

            session["user"] = {
                "email": user["email"],
                "sub": user["sub"],
                "role": role_from_groups(groups),
            }

        except jwt.InvalidTokenError:
            session.clear()
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