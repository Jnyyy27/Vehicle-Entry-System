from flask import Blueprint, render_template

from auth import login_required
from db import get_cursor

logs_bp = Blueprint("logs_bp", __name__)


@logs_bp.route("/logs")
@login_required
def logs():
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT id, plate_number, vehicle_category, status, entry_time
            FROM entry_logs
            ORDER BY id DESC
        """)
        logs = cursor.fetchall()

    return render_template("logs.html", logs=logs)