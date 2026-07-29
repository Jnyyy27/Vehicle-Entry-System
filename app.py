import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

from dotenv import load_dotenv
from flask import Flask
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
from routes.entry_routes import entry_bp
from routes.menu_routes import menu_bp

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

# CSRF protection for all POST forms (vehicles.html / entry.html / edit-delete
# forms need {{ csrf_token() }} added as a hidden field)
csrf = CSRFProtect(app)

app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(vehicles_bp)
app.register_blueprint(logs_bp)
app.register_blueprint(entry_bp)
app.register_blueprint(menu_bp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)