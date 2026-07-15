"""
Database access layer: a shared connection pool + a context manager
that guarantees connections are always returned/closed, even on error.

Requires: pip install DBUtils
"""

import logging
from contextlib import contextmanager

import pymysql
from dbutils.pooled_db import PooledDB

from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

logger = logging.getLogger(__name__)

# One pool, created once at import time, shared across all requests.
# - mincached/maxcached: idle connections kept ready
# - maxconnections: hard ceiling so you never exhaust MySQL's own limit
# - ping=1: check the connection is alive before handing it out
#   (avoids "MySQL server has gone away" after idle periods)
pool = PooledDB(
    creator=pymysql,
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    mincached=2,
    maxcached=5,
    maxconnections=20,
    blocking=True,
    ping=1,
)


@contextmanager
def get_cursor(dict_cursor=False, commit=False):
    """
    Usage:
        with get_cursor() as cursor:
            cursor.execute("SELECT * FROM vehicles WHERE plate_number = %s", (plate,))
            row = cursor.fetchone()

        with get_cursor(commit=True) as cursor:
            cursor.execute("INSERT INTO vehicles (...) VALUES (...)", (...))

        with get_cursor(dict_cursor=True) as cursor:
            cursor.execute("SELECT * FROM vehicles")
            rows = cursor.fetchall()  # list[dict]

    - Always returns the connection to the pool (or closes it), even if
      cursor.execute() raises.
    - Rolls back on exception so a half-finished multi-statement write
      doesn't linger uncommitted.
    - Set commit=True for INSERT/UPDATE/DELETE; leave False for SELECTs.
    """
    conn = pool.connection()
    cursor_cls = pymysql.cursors.DictCursor if dict_cursor else None
    cursor = conn.cursor(cursor_cls) if cursor_cls else conn.cursor()
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database operation failed, rolled back")
        raise
    finally:
        cursor.close()
        conn.close()  # returns the connection to the pool, doesn't actually close it