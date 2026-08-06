from flask import Blueprint, redirect, render_template, flash, url_for

from auth import login_required, admin_required
from db import get_cursor

vehicles_bp = Blueprint("vehicles_bp", __name__)


@vehicles_bp.route("/vehicles/list", methods=["GET"])
@login_required
@admin_required
def vehicles_list():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute("""
            SELECT vehicle_id, plate_number, owner_name, phone, total_visits, last_visit, created_at
            FROM vehicles
            ORDER BY last_visit DESC, vehicle_id DESC
        """)
        vehicles = cursor.fetchall()

    return render_template("vehicles_list.html", vehicles=vehicles)


@vehicles_bp.route("/vehicles/delete/<int:vehicle_id>", methods=["POST"])
@login_required
@admin_required
def vehicles_delete(vehicle_id):
    with get_cursor(commit=True) as cursor:
        cursor.execute("DELETE FROM vehicles WHERE vehicle_id=%s", (vehicle_id,))

    flash("Vehicle record deleted.", "success")
    return redirect(url_for("vehicles_bp.vehicles_list"))