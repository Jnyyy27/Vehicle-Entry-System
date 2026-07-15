from flask import Blueprint, request, redirect, render_template, flash, url_for

from auth import login_required, admin_required
from db import get_cursor

vehicles_bp = Blueprint("vehicles_bp", __name__)


@vehicles_bp.route("/vehicles", methods=["GET", "POST"])
@login_required
@admin_required
def vehicles():

    if request.method == "POST":

        student_name = request.form["student_name"]
        student_id = request.form["student_id"]
        plate_number = request.form["plate_number"]
        vehicle_type = request.form["vehicle_type"]

        with get_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO vehicles
                (student_name, student_id, plate_number, vehicle_type)
                VALUES (%s, %s, %s, %s)
            """, (student_name, student_id, plate_number, vehicle_type))

        flash(f"Vehicle {plate_number} registered successfully!", "success")
        return redirect(url_for("vehicles_bp.vehicles"))

    return render_template("vehicles.html")


@vehicles_bp.route("/vehicles/list", methods=["GET"])
@login_required
@admin_required
def vehicles_list():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute("""
            SELECT vehicle_id, student_name, student_id, plate_number, vehicle_type
            FROM vehicles
            ORDER BY vehicle_id DESC
        """)
        vehicles = cursor.fetchall()

    return render_template("vehicles_list.html", vehicles=vehicles)


@vehicles_bp.route("/vehicles/edit/<int:vehicle_id>", methods=["POST"])
@login_required
@admin_required
def vehicles_edit(vehicle_id):
    student_name = request.form["student_name"]
    student_id = request.form["student_id"]
    plate_number = request.form["plate_number"]
    vehicle_type = request.form["vehicle_type"]

    with get_cursor(commit=True) as cursor:
        cursor.execute("""
            UPDATE vehicles
            SET student_name=%s, student_id=%s, plate_number=%s, vehicle_type=%s
            WHERE vehicle_id=%s
        """, (student_name, student_id, plate_number, vehicle_type, vehicle_id))

    flash(f"Vehicle {plate_number} updated successfully!", "success")
    return redirect(url_for("vehicles_bp.vehicles_list"))


@vehicles_bp.route("/vehicles/delete/<int:vehicle_id>", methods=["POST"])
@login_required
@admin_required
def vehicles_delete(vehicle_id):
    with get_cursor(commit=True) as cursor:
        cursor.execute("DELETE FROM vehicles WHERE vehicle_id=%s", (vehicle_id,))

    flash("Vehicle deleted.", "success")
    return redirect(url_for("vehicles_bp.vehicles_list"))