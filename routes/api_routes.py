import logging
from flask import Blueprint, g

from auth import api_login_required, api_admin_required
from db import get_cursor
from routes.menu_routes import get_menu_image_url

logger = logging.getLogger(__name__)

api_bp = Blueprint("api_bp", __name__)


@api_bp.route("/api/health", methods=["GET"])
def health():
    return {"ok": True}


@api_bp.route("/api/me", methods=["GET"])
@api_login_required
def me():
    return {"user": getattr(g, "api_user", {})}


@api_bp.route("/api/dashboard", methods=["GET"])
@api_login_required
def dashboard_summary():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute("SELECT COUNT(*) AS count FROM vehicles")
        vehicles_count = int((cursor.fetchone() or {}).get("count") or 0)

        cursor.execute("SELECT COUNT(*) AS count FROM entry_logs")
        logs_count = int((cursor.fetchone() or {}).get("count") or 0)

        cursor.execute("SELECT COUNT(*) AS count FROM orders")
        orders_count = int((cursor.fetchone() or {}).get("count") or 0)

        cursor.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM orders
            GROUP BY status
            """
        )
        order_status_rows = cursor.fetchall() or []

    order_status_counts = {
        "Pending": 0,
        "Preparing": 0,
        "Ready": 0,
        "Completed": 0,
        "Cancelled": 0,
    }
    for row in order_status_rows:
        status = str(row.get("status") or "")
        if status in order_status_counts:
            order_status_counts[status] = int(row.get("count") or 0)

    return {
        "vehicles_count": vehicles_count,
        "logs_count": logs_count,
        "orders_count": orders_count,
        "order_status_counts": order_status_counts,
    }


@api_bp.route("/api/menu", methods=["GET"])
@api_login_required
def menu_items():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(
            """
            SELECT menu_id, name, category, price, description, image_key, available
            FROM menu_items
            ORDER BY category, name
            """
        )
        items = cursor.fetchall() or []

    categories = []
    for item in items:
        item["image_url"] = get_menu_image_url(item.get("image_key"))
        category = item.get("category")
        if category and category not in categories:
            categories.append(category)

    return {"items": items, "categories": categories}


@api_bp.route("/api/logs", methods=["GET"])
@api_login_required
def logs():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(
            """
            SELECT id, plate_number, status, entry_time
            FROM entry_logs
            ORDER BY id DESC
            """
        )
        logs_rows = cursor.fetchall() or []
    return {"logs": logs_rows}


@api_bp.route("/api/vehicles", methods=["GET"])
@api_login_required
@api_admin_required
def vehicles():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(
            """
            SELECT vehicle_id, plate_number, owner_name, phone, total_visits, last_visit, created_at
            FROM vehicles
            ORDER BY last_visit DESC, vehicle_id DESC
            """
        )
        vehicles_rows = cursor.fetchall() or []
    return {"vehicles": vehicles_rows}


@api_bp.route("/api/vehicles/<int:vehicle_id>", methods=["DELETE"])
@api_login_required
@api_admin_required
def delete_vehicle(vehicle_id):
    with get_cursor(commit=True) as cursor:
        cursor.execute("DELETE FROM vehicles WHERE vehicle_id = %s", (vehicle_id,))
    return {"success": True, "vehicle_id": vehicle_id}


@api_bp.route("/api/orders/display", methods=["GET"])
@api_login_required
def orders_display():
    order_rows = []
    status_counts = {
        "Pending": 0,
        "Preparing": 0,
        "Ready": 0,
        "Completed": 0,
        "Cancelled": 0,
    }

    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(
            """
            SELECT
                o.order_id,
                o.status,
                o.total_amount,
                o.created_at,
                v.plate_number
            FROM orders o
            INNER JOIN vehicles v ON v.vehicle_id = o.vehicle_id
            ORDER BY
                FIELD(o.status, 'Pending', 'Preparing', 'Ready', 'Completed', 'Cancelled'),
                o.created_at DESC
            LIMIT 60
            """
        )
        order_rows = cursor.fetchall() or []

        if order_rows:
            order_ids = [str(int(row["order_id"])) for row in order_rows]
            in_clause = ",".join(order_ids)
            cursor.execute(
                f"""
                SELECT
                    oi.order_id,
                    mi.name,
                    oi.quantity,
                    oi.unit_price,
                    oi.subtotal
                FROM order_items oi
                INNER JOIN menu_items mi ON mi.menu_id = oi.menu_id
                WHERE oi.order_id IN ({in_clause})
                ORDER BY oi.order_id DESC, oi.order_item_id ASC
                """
            )
            line_rows = cursor.fetchall() or []
        else:
            line_rows = []

    lines_by_order = {}
    for line in line_rows:
        order_id = int(line["order_id"])
        lines_by_order.setdefault(order_id, []).append(
            {
                "name": line["name"],
                "quantity": int(line["quantity"]),
                "unit_price": float(line["unit_price"]),
                "subtotal": float(line["subtotal"]),
            }
        )

    orders = []
    for row in order_rows:
        status = str(row.get("status") or "Pending")
        if status in status_counts:
            status_counts[status] += 1

        order_id = int(row["order_id"])
        orders.append(
            {
                "order_id": order_id,
                "plate_number": row.get("plate_number") or "-",
                "status": status,
                "total_amount": float(row.get("total_amount") or 0),
                "created_at": row.get("created_at"),
                "lines": lines_by_order.get(order_id, []),
            }
        )

    active_count = status_counts["Pending"] + status_counts["Preparing"] + status_counts["Ready"]

    return {
        "orders": orders,
        "total_orders": len(orders),
        "active_count": active_count,
        "status_counts": status_counts,
    }