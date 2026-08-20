"""
CLI helper: create or reset a user in the local users table.

Usage:
    python manage_users.py add --email alice@example.com --password secret123 --role Admin
    python manage_users.py reset-password --email alice@example.com --password newpass
    python manage_users.py list
"""

import argparse
import sys

from auth import hash_password
from db import get_cursor


def cmd_add(args):
    pw_hash = hash_password(args.password)
    with get_cursor(commit=True) as cursor:
        cursor.execute(
            """
            INSERT INTO users (email, password_hash, role)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash), role = VALUES(role)
            """,
            (args.email.strip().lower(), pw_hash, args.role),
        )
    print(f"User '{args.email}' saved with role '{args.role}'.")


def cmd_reset(args):
    pw_hash = hash_password(args.password)
    with get_cursor(commit=True) as cursor:
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE email = %s",
            (pw_hash, args.email.strip().lower()),
        )
        if cursor.rowcount == 0:
            print(f"No user found with email '{args.email}'.", file=sys.stderr)
            sys.exit(1)
    print(f"Password updated for '{args.email}'.")


def cmd_list(_args):
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute("SELECT email, role, created_at FROM users ORDER BY created_at")
        rows = cursor.fetchall() or []
    if not rows:
        print("No users found.")
        return
    for row in rows:
        print(f"  {row['email']:<40}  {row['role']:<15}  {row['created_at']}")


def main():
    parser = argparse.ArgumentParser(description="Manage local users")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add or update a user")
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--password", required=True)
    p_add.add_argument("--role", choices=["Admin", "SecurityGuard", "User"], default="User")

    p_reset = sub.add_parser("reset-password", help="Reset a user's password")
    p_reset.add_argument("--email", required=True)
    p_reset.add_argument("--password", required=True)

    sub.add_parser("list", help="List all users")

    args = parser.parse_args()
    {"add": cmd_add, "reset-password": cmd_reset, "list": cmd_list}[args.command](args)


if __name__ == "__main__":
    main()