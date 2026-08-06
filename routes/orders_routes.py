from flask import Blueprint, render_template

from auth import login_required
from db import get_cursor

orders_bp = Blueprint("orders_bp", __name__)


@orders_bp.route("/orders/display")
@login_required
def order_display():
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

    return render_template(
        "orders_display.html",
        orders=orders,
        total_orders=len(orders),
        active_count=active_count,
        status_counts=status_counts,
    )