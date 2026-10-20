""" Command-line driver: repair faces 7-12 of LLC4320 v2 stores written before 2026-10-02.

Stores written before the face-order fix hold faces 7-12 in MITgcm compact
order (see ``llc_v2.compact_to_faces``).  This rewrites them in place, one
store at a time, and marks each with ``face_layout = "llc_faces"``::

    wr_llc_v2_fix_faces s3://llc4320-v2/SURFACE --dry-run --limit 3 -v
    wr_llc_v2_fix_faces s3://llc4320-v2/SURFACE --limit 1 -v
    wr_llc_v2_fix_faces s3://llc4320-v2/SURFACE --workers 6

grid.zarr is checked (and repaired if needed) first, since the hourly
stores' layout check uses its land mask.  Each store's layout is inferred
from its own NaN pattern before anything is written, so re-running is
safe: repaired stores are skipped and nothing is permuted twice.  The
original faces of the store being rewritten are kept in ``--backup-dir``
until the rewrite is verified; an interrupted store is finished from its
backup on the next run.
"""

import argparse
import logging
import os
import time


def parser(options=None):
    p = argparse.ArgumentParser(description='Repair faces 7-12 of LLC4320 v2 Zarr stores in place.')
    p.add_argument('root', help='directory or s3:// prefix holding YYYYMMDDTHH.zarr and grid.zarr')
    p.add_argument('--start', help='first store to consider, YYYYMMDDTHH (inclusive)')
    p.add_argument('--end', help='last store to consider, YYYYMMDDTHH (inclusive)')
    p.add_argument('--limit', type=int, help='stop after this many hourly stores')
    p.add_argument('--dry-run', action='store_true', help='classify only; write nothing')
    p.add_argument('--workers', type=int, default=1, help='parallel processes (default 1)')
    p.add_argument('--backup-dir', default=os.path.expanduser('~/llc_v2_face_backups'),
                   help='local dir for originals of in-flight stores (~0.45 GB each)')
    p.add_argument('--endpoint', help='S3 endpoint (default $ENDPOINT_URL or Nautilus west)')
    p.add_argument('--profile', help='AWS credentials profile (default $AWS_PROFILE)')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args() if options is None else p.parse_args(options)


def list_stores(root, endpoint=None, profile=None):
    """Sorted hourly store names (YYYYMMDDTHH.zarr) under *root*."""
    from wrangler.ogcm import llc_v2
    if llc_v2.is_s3(root):
        fs = llc_v2.s3_filesystem(endpoint, profile, asynchronous=False)
        entries = [os.path.basename(p.rstrip('/')) for p in fs.ls(root.rstrip('/'), detail=False)]
    else:
        entries = os.listdir(root)
    return sorted(e for e in entries if e.endswith('.zarr') and e[:8].isdigit())


_MASKS = {}


def _init_worker(grid_url, endpoint, profile, verbose):
    from wrangler.ogcm import llc_v2
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')
    _MASKS.update(llc_v2.wet_masks_for_repair(grid_url, endpoint=endpoint, profile=profile))


def _repair_one(url, backup_dir, dry_run, endpoint, profile):
    from wrangler.ogcm import llc_v2
    t0 = time.time()
    try:
        status = llc_v2.repair_store_faces(url, _MASKS, backup_dir=backup_dir, dry_run=dry_run,
                                           endpoint=endpoint, profile=profile)
    except Exception as e:                      # report, keep going with the other stores
        status = f'error: {e}'
    return url, status, time.time() - t0


def main(args):
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from wrangler.ogcm import llc_v2

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')
    root = args.root.rstrip('/')
    grid_url = f'{root}/grid.zarr'
    kw = dict(endpoint=args.endpoint, profile=args.profile)

    status = llc_v2.repair_store_faces(grid_url, backup_dir=args.backup_dir,
                                       dry_run=args.dry_run, **kw)
    print(f'grid.zarr: {status}', flush=True)
    if status.startswith('unknown'):
        raise SystemExit('grid.zarr layout could not be determined; stopping')

    names = list_stores(root, **kw)
    if args.start:
        names = [n for n in names if n[:11] >= args.start]
    if args.end:
        names = [n for n in names if n[:11] <= args.end]
    if args.limit:
        names = names[:args.limit]
    urls = [f'{root}/{n}' for n in names]
    print(f'{len(urls)} hourly store(s) to check', flush=True)

    counts = Counter()
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(grid_url, args.endpoint, args.profile, args.verbose)) as ex:
        futs = [ex.submit(_repair_one, u, args.backup_dir, args.dry_run, args.endpoint,
                          args.profile) for u in urls]
        for n, fut in enumerate(futs, 1):
            url, st, dt_s = fut.result()
            counts[st.split(':')[0]] += 1
            print(f'[{n}/{len(urls)}] {os.path.basename(url)}: {st}  ({dt_s:.0f} s)', flush=True)
    el = time.time() - t0
    print('summary: ' + ' '.join(f'{k}={v}' for k, v in sorted(counts.items()))
          + f'  elapsed={el / 3600:.2f} h')
    return dict(counts)


if __name__ == '__main__':
    main(parser())
