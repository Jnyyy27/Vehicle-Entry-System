import logging
from flask import Blueprint, g, request

from auth import api_login_required, api_admin_required
from db import get_cursor
from routes.menu_routes import get_menu_image_url

logger = logging.getLogger(__name__)

api_bp = Blueprint("api_bp", __name__)


def _get_vehicle_id(cursor, plate_number):
    cursor.execute(
        "SELECT vehicle_id FROM vehicles WHERE plate_number = %s LIMIT 1",
        (plate_number,),
    )
    row = cursor.fetchone() or {}
    vehicle_id = row.get("vehicle_id")
    return int(vehicle_id) if vehicle_id else None


def _get_latest_active_order(cursor, vehicle_id):
    cursor.execute(
        """
        SELECT order_id, status, total_amount
        FROM orders
        WHERE vehicle_id = %s AND status IN ('Pending', 'Preparing')
        ORDER BY order_id DESC
        LIMIT 1
        """,
        (vehicle_id,),
    )
    return cursor.fetchone()


def _get_latest_pending_order(cursor, vehicle_id):
    cursor.execute(
        """
        SELECT order_id, status, total_amount
        FROM orders
        WHERE vehicle_id = %s AND status = 'Pending'
        ORDER BY order_id DESC
        LIMIT 1
        """,
        (vehicle_id,),
    )
    return cursor.fetchone()


def _load_order_snapshot(cursor, plate_number, order_row):
    if not order_row:
        return {
            "plate_number": plate_number,
            "order": None,
            "lines": [],
            "can_edit": False,
        }

    order_id = int(order_row["order_id"])
    status = str(order_row.get("status") or "Pending")

    cursor.execute(
        """
        SELECT
            oi.menu_id,
            mi.name,
            mi.category,
            oi.quantity,
            oi.unit_price,
            oi.subtotal
        FROM order_items oi
        INNER JOIN menu_items mi ON mi.menu_id = oi.menu_id
        WHERE oi.order_id = %s
        ORDER BY oi.order_item_id ASC
        """,
        (order_id,),
    )
    lines_raw = cursor.fetchall() or []
    lines = [
        {
            "menu_id": int(row["menu_id"]),
            "name": row["name"],
            "category": row.get("category") or "",
            "quantity": int(row["quantity"]),
            "unit_price": float(row["unit_price"]),
            "subtotal": float(row["subtotal"]),
        }
        for row in lines_raw
    ]

    total_amount = float(sum(line["subtotal"] for line in lines))
    cursor.execute(
        "UPDATE orders SET total_amount = %s WHERE order_id = %s",
        (total_amount, order_id),
    )

    return {
        "plate_number": plate_number,
        "order": {
            "order_id": order_id,
            "status": status,
            "total_amount": total_amount,
        },
        "lines": lines,
        "can_edit": status == "Pending",
    }


def _get_or_create_pending_order(cursor, plate_number):
    vehicle_id = _get_vehicle_id(cursor, plate_number)
    if not vehicle_id:
        return None, "vehicle_not_found"

    order_row = _get_latest_pending_order(cursor, vehicle_id)
    if order_row:
        return order_row, None

    cursor.execute(
        """
        INSERT INTO orders (vehicle_id, status, total_amount)
        VALUES (%s, 'Pending', 0.00)
        """,
        (vehicle_id,),
    )
    order_id = int(cursor.lastrowid)
    return {"order_id": order_id, "status": "Pending", "total_amount": 0}, None


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


@api_bp.route("/api/orders/cart", methods=["GET"])
@api_login_required
def order_cart():
    plate_number = (request.args.get("plate_number") or "").strip().upper()
    if not plate_number:
        return {"error": "Missing plate_number"}, 400

    with get_cursor(dict_cursor=True, commit=True) as cursor:
        vehicle_id = _get_vehicle_id(cursor, plate_number)
        if not vehicle_id:
            return {
                "plate_number": plate_number,
                "order": None,
                "lines": [],
                "can_edit": False,
            }

        # Cart represents only editable draft orders.
        order_row = _get_latest_pending_order(cursor, vehicle_id)
        return _load_order_snapshot(cursor, plate_number, order_row)


@api_bp.route("/api/orders/cart/items", methods=["POST"])
@api_login_required
def order_cart_add_item():
    payload = request.get_json(silent=True) or {}
    plate_number = (payload.get("plate_number") or "").strip().upper()
    menu_id = payload.get("menu_id")
    quantity = payload.get("quantity", 1)

    if not plate_number:
        return {"error": "Missing plate_number"}, 400

    try:
        menu_id = int(menu_id)
        quantity = int(quantity)
    except (TypeError, ValueError):
        return {"error": "menu_id and quantity must be integers"}, 400

    if quantity < 1 or quantity > 20:
        return {"error": "quantity must be between 1 and 20"}, 400

    with get_cursor(dict_cursor=True, commit=True) as cursor:
        order_row, error = _get_or_create_pending_order(cursor, plate_number)
        if error == "vehicle_not_found":
            return {
                "error": "No active plate session found. Submit entry first."
            }, 404

        order_id = int(order_row["order_id"])
        status = str(order_row.get("status") or "Pending")
        if status != "Pending":
            return {"error": "Order can no longer be edited."}, 409

        cursor.execute(
            """
            SELECT menu_id, price, available
            FROM menu_items
            WHERE menu_id = %s
            LIMIT 1
            """,
            (menu_id,),
        )
        menu_row = cursor.fetchone() or {}
        if not menu_row:
            return {"error": "Menu item not found"}, 404
        if not bool(menu_row.get("available")):
            return {"error": "Menu item is unavailable"}, 409

        unit_price = float(menu_row.get("price") or 0)

        cursor.execute(
            """
            SELECT order_item_id, quantity
            FROM order_items
            WHERE order_id = %s AND menu_id = %s
            LIMIT 1
            """,
            (order_id, menu_id),
        )
        line_row = cursor.fetchone()

        if line_row:
            new_qty = int(line_row["quantity"]) + quantity
            subtotal = float(new_qty * unit_price)
            cursor.execute(
                """
                UPDATE order_items
                SET quantity = %s, unit_price = %s, subtotal = %s
                WHERE order_item_id = %s
                """,
                (new_qty, unit_price, subtotal, int(line_row["order_item_id"])),
            )
        else:
            subtotal = float(quantity * unit_price)
            cursor.execute(
                """
                INSERT INTO order_items (order_id, menu_id, quantity, unit_price, subtotal)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (order_id, menu_id, quantity, unit_price, subtotal),
            )

        latest = {"order_id": order_id, "status": status, "total_amount": 0}
        return _load_order_snapshot(cursor, plate_number, latest)


@api_bp.route("/api/orders/cart/items", methods=["PATCH"])
@api_login_required
def order_cart_update_item():
    payload = request.get_json(silent=True) or {}
    plate_number = (payload.get("plate_number") or "").strip().upper()
    menu_id = payload.get("menu_id")
    quantity = payload.get("quantity")

    if not plate_number:
        return {"error": "Missing plate_number"}, 400

    try:
        menu_id = int(menu_id)
        quantity = int(quantity)
    except (TypeError, ValueError):
        return {"error": "menu_id and quantity must be integers"}, 400

    if quantity < 0 or quantity > 20:
        return {"error": "quantity must be between 0 and 20"}, 400

    with get_cursor(dict_cursor=True, commit=True) as cursor:
        vehicle_id = _get_vehicle_id(cursor, plate_number)
        if not vehicle_id:
            return {"error": "No active plate session found."}, 404

        order_row = _get_latest_pending_order(cursor, vehicle_id)
        if not order_row:
            return {"error": "No draft order found."}, 404

        order_id = int(order_row["order_id"])
        status = str(order_row.get("status") or "Pending")
        if status != "Pending":
            return {"error": "Order can no longer be edited."}, 409

        cursor.execute(
            """
            SELECT order_item_id, unit_price
            FROM order_items
            WHERE order_id = %s AND menu_id = %s
            LIMIT 1
            """,
            (order_id, menu_id),
        )
        line_row = cursor.fetchone()
        if not line_row:
            return {"error": "Order item not found"}, 404

        if quantity == 0:
            cursor.execute(
                "DELETE FROM order_items WHERE order_item_id = %s",
                (int(line_row["order_item_id"]),),
            )
        else:
            unit_price = float(line_row.get("unit_price") or 0)
            subtotal = float(quantity * unit_price)
            cursor.execute(
                """
                UPDATE order_items
                SET quantity = %s, subtotal = %s
                WHERE order_item_id = %s
                """,
                (quantity, subtotal, int(line_row["order_item_id"])),
            )

        latest = {"order_id": order_id, "status": status, "total_amount": 0}
        return _load_order_snapshot(cursor, plate_number, latest)


@api_bp.route("/api/orders/cart/confirm", methods=["POST"])
@api_login_required
def order_cart_confirm():
    payload = request.get_json(silent=True) or {}
    plate_number = (payload.get("plate_number") or "").strip().upper()

    if not plate_number:
        return {"error": "Missing plate_number"}, 400

    with get_cursor(dict_cursor=True, commit=True) as cursor:
        vehicle_id = _get_vehicle_id(cursor, plate_number)
        if not vehicle_id:
            return {"error": "No active plate session found."}, 404

        order_row = _get_latest_pending_order(cursor, vehicle_id)
        if not order_row:
            return {"error": "No draft order found."}, 404

        order_id = int(order_row["order_id"])
        status = str(order_row.get("status") or "Pending")

        cursor.execute(
            "SELECT COUNT(*) AS count FROM order_items WHERE order_id = %s",
            (order_id,),
        )
        line_count = int((cursor.fetchone() or {}).get("count") or 0)
        if line_count == 0:
            return {"error": "Add items before confirming the order."}, 409

        if status == "Pending":
            cursor.execute(
                "UPDATE orders SET status = 'Preparing' WHERE order_id = %s",
                (order_id,),
            )
            status = "Preparing"

        latest = {"order_id": order_id, "status": status, "total_amount": 0}
        confirmed = _load_order_snapshot(cursor, plate_number, latest)
        return {
            "plate_number": plate_number,
            "message": "Order confirmed and sent to kitchen.",
            "confirmed_order": confirmed.get("order"),
            "confirmed_lines": confirmed.get("lines") or [],
            # Cart clears immediately after confirm so a new draft can start.
            "order": None,
            "lines": [],
            "can_edit": True,
        }


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