from flask import Blueprint, render_template, session

from auth import login_required

dashboard_bp = Blueprint("dashboard_bp", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        user=session.get("user", {})
    )