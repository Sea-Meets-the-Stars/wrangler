""" Reproduce the two Cursor Bugbot findings on PR #4 against the live bucket.

1. ``open_range`` with a start that is not on the hour mislabels the ``time``
   axis and drops the last snapshot (``llc_v2_access.hours_between``).
2. ``consolidate_store`` writes consolidated metadata before its zero-member
   guard; ``SURFACE/20231228T23.zarr`` (an empty store) was left with an empty
   consolidated document, which ``open_range`` then treats as a real hour.

Both are fixed as of 2026-10-27 (and the empty ``zarr.json`` deleted), so the
expected output is now ``time = 07:00, 08:00, 09:00``, the store absent, and
both ranges returning 22:00 and 00:00 only.  Read-only.  Run with::

    PYTHONPATH=. conda run -n ocean14 python wrangler/scripts/llc_v2_bugbot_check.py
"""

import json

import boto3
from botocore.config import Config

from wrangler.ogcm import llc_v2_access as acc
from wrangler.ogcm.llc_v2 import s3_endpoint


def check_time_axis():
    """Finding 1: half-hour start."""
    ds = acc.open_range('2023-03-01 06:30', '2023-03-01 09:00', variables=['Theta'])
    print("open_range 06:30 .. 09:00 -> time =",
          [str(t)[:16] for t in ds.time.values], "(expected 07:00, 08:00, 09:00)")


def check_empty_consolidated(name='20231228T23.zarr'):
    """Finding 2: consolidated-metadata member count of an empty store, and its effect."""
    s3 = boto3.client('s3', endpoint_url=s3_endpoint(),
                      config=Config(s3={'addressing_style': 'path'}))
    try:
        obj = s3.get_object(Bucket='llc4320-v2', Key=f'SURFACE/{name}/zarr.json')
    except s3.exceptions.NoSuchKey:
        # Deleted 2026-10-27, so the hour reads as missing like the other gaps.
        print(f"{name}: no zarr.json (store absent)")
    else:
        meta = json.loads(obj['Body'].read())
        cm = meta.get('consolidated_metadata')
        print(f"{name}: zarr.json written {obj['LastModified']}, consolidated members:",
              None if cm is None else len(cm.get('metadata', {})))
    for kw in ({}, {'variables': ['Theta']}):
        try:
            ds = acc.open_range('2023-12-28 22', '2023-12-29 00', **kw)
            print(f"open_range 22 .. 00 {kw}: time =",
                  [str(t)[:13] for t in ds.time.values])
        except Exception as e:
            print(f"open_range 22 .. 00 {kw}: FAILED {type(e).__name__}: {e}")


if __name__ == '__main__':
    check_time_axis()
    check_empty_consolidated()
