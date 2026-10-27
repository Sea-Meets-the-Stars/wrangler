""" Monitor the LLC4320 v2 surface extraction directly from Nautilus S3.

Lists the hourly timestep stores under ``s3://llc4320-v2/SURFACE/`` with cheap
delimiter listings (one page per 1,000 stores, never the chunk objects), then
reports:

* how many stores exist, out of the expected 9,503, and whether the hours
  they cover are contiguous (lists any gaps);
* the write rate, from the ``LastModified`` time of each store's group
  ``zarr.json`` (overall and over the most recent stores), and the projected
  finish time;
* ``complete`` / date attributes of the newest few stores, and the size and
  object count of a recent complete one;
* the state of ``grid.zarr``.

Run with::

    conda run -n ocean14 python wrangler/scripts/llc_v2_progress.py
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import boto3
from botocore.config import Config

from wrangler.ogcm.llc_v2 import s3_endpoint

N_EXPECTED = 9503
FIRST_HOUR = datetime(2023, 1, 1, 1)


def client(endpoint=None):
    """boto3 S3 client for Nautilus (path-style addressing)."""
    return boto3.client('s3', endpoint_url=s3_endpoint(endpoint),
                        config=Config(s3={'addressing_style': 'path'}))


def list_stores(s3, bucket, prefix):
    """Return the sorted store names (``YYYYMMDDTHH.zarr``) under ``prefix``."""
    names = []
    for page in s3.get_paginator('list_objects_v2').paginate(
            Bucket=bucket, Prefix=prefix, Delimiter='/'):
        for cp in page.get('CommonPrefixes', []):
            names.append(cp['Prefix'][len(prefix):].rstrip('/'))
    return sorted(names)


def store_hour(name):
    """Parse ``20230101T01.zarr`` into a datetime."""
    return datetime.strptime(name.split('.')[0], '%Y%m%dT%H')


def group_meta(s3, bucket, key):
    """Return (attributes, LastModified) of a store's group ``zarr.json``."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    meta = json.loads(obj['Body'].read())
    return meta.get('attributes', {}), obj['LastModified']


def store_size(s3, bucket, prefix):
    """Total bytes and object count of everything under ``prefix``."""
    nbytes = nobj = 0
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get('Contents', []):
            nbytes += o['Size']
            nobj += 1
    return nbytes, nobj


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--bucket', default='llc4320-v2')
    p.add_argument('--prefix', default='SURFACE/')
    p.add_argument('--recent', type=int, default=24,
                   help='number of newest stores for the recent rate')
    args = p.parse_args()
    s3 = client()
    now = datetime.now(timezone.utc)

    names = list_stores(s3, args.bucket, args.prefix)
    grid = [n for n in names if n == 'grid.zarr']
    steps = [n for n in names if n != 'grid.zarr']
    others = [n for n in steps if not n.endswith('.zarr')]
    steps = [n for n in steps if n.endswith('.zarr')]
    print(f"checked {now:%Y-%m-%d %H:%M} UTC")
    print(f"timestep stores: {len(steps)} / {N_EXPECTED} "
          f"({100 * len(steps) / N_EXPECTED:.2f} %)")
    if others:
        print(f"other prefixes: {others}")
    if not steps:
        return

    hours = [store_hour(n) for n in steps]
    span = int((hours[-1] - hours[0]).total_seconds() // 3600) + 1
    print(f"span: {steps[0]} .. {steps[-1]}  ({span} hours spanned)")
    have = set(hours)
    gaps = [h for h in (hours[0] + timedelta(hours=i) for i in range(span)) if h not in have]
    print(f"gaps: {len(gaps)}" + (f"  first: {[g.strftime('%Y%m%dT%H') for g in gaps[:10]]}"
                                   if gaps else ''))

    # Timing from the group zarr.json of the first store and the newest ones.
    # The newest store's zarr.json is rewritten when it finishes, so for the
    # in-flight store it reflects creation time.
    _, t_first = group_meta(s3, args.bucket, f"{args.prefix}{steps[0]}/zarr.json")
    recent = steps[-(args.recent + 1):]
    recent_meta = [group_meta(s3, args.bucket, f"{args.prefix}{n}/zarr.json") for n in recent]
    t_last = recent_meta[-1][1]
    overall = (t_last - t_first).total_seconds() / max(len(steps) - 1, 1)
    rec = (recent_meta[-1][1] - recent_meta[0][1]).total_seconds() / max(len(recent) - 1, 1)
    remaining = N_EXPECTED - len(steps)
    print(f"first store written: {t_first:%Y-%m-%d %H:%M} UTC; newest: {t_last:%Y-%m-%d %H:%M} UTC "
          f"({(now - t_last).total_seconds() / 60:.1f} min ago)")
    print(f"rate: {overall:.1f} s/store overall, {rec:.1f} s/store over last {len(recent) - 1}")
    for label, r in (('overall', overall), ('recent', rec)):
        eta = t_last + timedelta(seconds=r * remaining)
        print(f"  finish at {label} rate: {eta:%Y-%m-%d %H:%M} UTC "
              f"({r * remaining / 86400:.1f} days from newest)")

    # Per-store gaps in write time -- large pauses mean the job stopped.
    dts = [(b[1] - a[1]).total_seconds() for a, b in zip(recent_meta, recent_meta[1:])]
    print(f"recent inter-store intervals (s): min {min(dts):.0f}, max {max(dts):.0f}")

    print("newest stores:")
    for n, (attrs, t) in list(zip(recent, recent_meta))[-4:]:
        print(f"  {n}  complete={attrs.get('complete')}  "
              f"iter={attrs.get('selected_iteration')}  date={attrs.get('selected_date_utc')}  "
              f"written {t:%m-%d %H:%M}")

    # Size of the newest complete store.
    for n, (attrs, _) in zip(reversed(recent), reversed(recent_meta)):
        if attrs.get('complete'):
            nb, no = store_size(s3, args.bucket, f"{args.prefix}{n}/")
            print(f"size of {n}: {nb / 1e6:.1f} MB, {no} objects; "
                  f"projected total {nb * N_EXPECTED / 1e12:.2f} TB")
            break

    if grid:
        attrs, t = group_meta(s3, args.bucket, f"{args.prefix}grid.zarr/zarr.json")
        print(f"grid.zarr: complete={attrs.get('complete')} skipped={attrs.get('skipped')}")
    else:
        print("grid.zarr: MISSING")


if __name__ == '__main__':
    main()
