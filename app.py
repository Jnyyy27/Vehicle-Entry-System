import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

from dotenv import load_dotenv
from flask import Flask, request
from flask_wtf.csrf import CSRFProtect

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Import routes AFTER load_dotenv() so any module-level env-dependent setup
# (Cognito JWKS client, S3 client, model loading) sees the right values.
from routes.auth_routes import auth_bp
from routes.dashboard_routes import dashboard_bp
from routes.vehicles_routes import vehicles_bp
from routes.logs_routes import logs_bp
from routes.entry_routes import entry_bp, detect_plate_api, assistant_chat, submit_entry_api
from routes.menu_routes import menu_bp
from routes.orders_routes import orders_bp
from routes.api_routes import api_bp


app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

# Limit upload size to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Harden session cookies
app.config.update(
    SESSION_COOKIE_SECURE=True,     # only sent over HTTPS
    SESSION_COOKIE_HTTPONLY=True,   # not accessible to JS
    SESSION_COOKIE_SAMESITE="Lax",
)

# CSRF protection for all POST forms (entry.html / delete forms
# need {{ csrf_token() }} added as a hidden field)
csrf = CSRFProtect(app)

# API clients (Flutter mobile/web) are authenticated via bearer/session
# and do not submit Flask form CSRF tokens.
csrf.exempt(api_bp)
csrf.exempt(detect_plate_api)
csrf.exempt(assistant_chat)
csrf.exempt(submit_entry_api)

app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(vehicles_bp)
app.register_blueprint(logs_bp)
app.register_blueprint(entry_bp)
app.register_blueprint(menu_bp)
app.register_blueprint(orders_bp)
app.register_blueprint(api_bp)


def _allowed_frontend_origins():
    raw_value = os.getenv("FRONTEND_ORIGIN", "*")
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or ["*"]


@app.before_request
def handle_api_preflight():
    if request.path.startswith("/api/") and request.method == "OPTIONS":
        return ("", 204)


@app.after_request
def add_cors_headers(response):
    if request.path.startswith("/api/"):
        allowed_origins = _allowed_frontend_origins()
        request_origin = request.headers.get("Origin")

        if "*" in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif request_origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = request_origin
            response.headers["Vary"] = "Origin"

        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)