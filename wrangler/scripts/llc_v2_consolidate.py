""" Backfill consolidated metadata into LLC4320 v2 stores written before it was added.

Without consolidated metadata every open walks the group and each array's
``zarr.json`` separately, several round trips per store; with it a reader
fetches one document.  Over a year-long date range that is the difference
between tens of thousands of requests and ten thousand.

Chunks are never touched -- this only rewrites each store's group metadata::

    wr_llc_v2_consolidate s3://llc4320-v2/SURFACE --dry-run
    wr_llc_v2_consolidate s3://llc4320-v2/SURFACE --workers 8

Safe to re-run: consolidating an already-consolidated store is a no-op.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def parser(options=None):
    p = argparse.ArgumentParser(
        description='Write consolidated metadata into LLC4320 v2 Zarr stores.')
    p.add_argument('root', help='directory or s3:// prefix holding YYYYMMDDTHH.zarr and grid.zarr')
    p.add_argument('--start', help='first store to consider, YYYYMMDDTHH (inclusive)')
    p.add_argument('--end', help='last store to consider, YYYYMMDDTHH (inclusive)')
    p.add_argument('--limit', type=int, help='stop after this many stores')
    p.add_argument('--workers', type=int, default=1, help='parallel processes (default 1)')
    p.add_argument('--dry-run', action='store_true', help='list what would be done')
    p.add_argument('--endpoint', help='S3 endpoint (default $ENDPOINT_URL or Nautilus west)')
    p.add_argument('--profile', help='AWS credentials profile (default $AWS_PROFILE)')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args() if options is None else p.parse_args(options)


def _one(args):
    """Consolidate a single store; returns (name, ok, seconds)."""
    root, name, endpoint, profile = args
    from wrangler.ogcm import llc_v2
    t0 = time.time()
    ok = llc_v2.consolidate_store(f'{root}/{name}', endpoint, profile)
    return name, ok, time.time() - t0


def main(args):
    from wrangler.ogcm import llc_v2
    from wrangler.scripts.llc_v2_fix_faces import list_stores

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    root = args.root.rstrip('/')
    names = list_stores(root, args.endpoint, args.profile)
    if args.start:
        names = [n for n in names if n[:11] >= args.start]
    if args.end:
        names = [n for n in names if n[:11] <= args.end]
    if args.limit:
        names = names[:args.limit]
    names = ['grid.zarr'] + names          # the grid first; readers need it most

    print(f'{len(names)} store(s) under {root}', flush=True)
    if args.dry_run:
        for n in names[:10]:
            print(f'  would consolidate {n}')
        if len(names) > 10:
            print(f'  ... and {len(names) - 10} more')
        return {'total': len(names), 'ok': 0, 'failed': 0}

    items = [(root, n, args.endpoint, args.profile) for n in names]
    ok = failed = 0
    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, it): it[1] for it in items}
            for i, fut in enumerate(as_completed(futs), 1):
                name, good, dt = fut.result()
                ok, failed = ok + bool(good), failed + (not good)
                if i % 200 == 0 or not good:
                    print(f'[{i}/{len(items)}] {name}: '
                          f'{"ok" if good else "FAILED"} ({dt:.1f} s)', flush=True)
    else:
        for i, it in enumerate(items, 1):
            name, good, dt = _one(it)
            ok, failed = ok + bool(good), failed + (not good)
            if i % 200 == 0 or not good:
                print(f'[{i}/{len(items)}] {name}: '
                      f'{"ok" if good else "FAILED"} ({dt:.1f} s)', flush=True)
    print(f'summary: ok={ok} failed={failed}  elapsed={(time.time()-t0)/3600:.2f} h', flush=True)
    return {'total': len(items), 'ok': ok, 'failed': failed}


if __name__ == '__main__':
    main(parser())
