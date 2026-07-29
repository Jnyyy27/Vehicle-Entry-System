import logging

import boto3
from botocore.exceptions import ClientError
from flask import Blueprint, render_template

from db import get_cursor

# TODO: confirm these two names against whatever your config.py already
# exposes for the S3 client used elsewhere in the app (e.g. for vehicle
# photos). Rename the import if config.py uses different variable names.
from config import S3_BUCKET_NAME, AWS_REGION

logger = logging.getLogger(__name__)

menu_bp = Blueprint('menu', __name__)

# Reuses the standard AWS credential chain (env vars / instance role / etc.),
# same as however your existing S3 client elsewhere in the app is set up.
s3_client = boto3.client('s3', region_name=AWS_REGION)

PRESIGNED_URL_TTL_SECONDS = 3600  # 1 hour


def get_menu_image_url(image_key):
    """Returns a time-limited signed URL for an S3 object, or None on failure."""
    if not image_key:
        return None
    try:
        return s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': image_key},
            ExpiresIn=PRESIGNED_URL_TTL_SECONDS,
        )
    except ClientError:
        logger.exception("Failed to presign S3 URL for key %s", image_key)
        return None


@menu_bp.route('/menu')
def menu():
    with get_cursor(dict_cursor=True) as cursor:
        cursor.execute(
            """
            SELECT menu_id, name, category, price, description, image_key, available
            FROM menu_items
            ORDER BY category, name
            """
        )
        items = cursor.fetchall()

    for item in items:
        item['image_url'] = get_menu_image_url(item['image_key'])

    # Preserve DB ordering (category, name) rather than re-sorting alphabetically
    seen = []
    for item in items:
        if item['category'] not in seen:
            seen.append(item['category'])
    categories = seen

    return render_template('menu.html', items=items, categories=categories)