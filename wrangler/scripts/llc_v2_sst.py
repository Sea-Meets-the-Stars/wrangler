""" Command-line driver: extract global LLC4320 v2 SST to Zarr (local or Nautilus S3).

Intended to be run on the Pleiades login node, where the raw output lives::

    wr_llc_v2_sst /nobackupp27/dbwhitt/llc_4320/OUT <maskDir> \
        s3://llc4320-v2/SURFACE --dry-run
    wr_llc_v2_sst /nobackupp27/dbwhitt/llc_4320/OUT <maskDir> \
        s3://llc4320-v2/SURFACE --limit 1          # first real store
    wr_llc_v2_sst /nobackupp27/dbwhitt/llc_4320/OUT <maskDir> \
        s3://llc4320-v2/SURFACE                    # everything on disk

Credentials (PAB convention): ``ENDPOINT_URL`` (default
https://s3-west.nrp-nautilus.io) and ``AWS_PROFILE`` (default "default") in
``~/.aws/credentials``, or ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``.
"""

import argparse
import logging
from datetime import datetime, timezone


def _parse_date(s: str) -> datetime:
    """'2023-01-01' or '2023-01-01T06' -> UTC datetime."""
    for fmt in ('%Y-%m-%dT%H', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"bad date {s!r}; use YYYY-MM-DD[THH]")


def parser(options=None):
    p = argparse.ArgumentParser(
        description='Extract global LLC4320 v2 SST (Theta, k=0) to per-hour Zarr stores.')
    p.add_argument('out_dir', help='raw-output parent, e.g. /nobackupp27/dbwhitt/llc_4320/OUT')
    p.add_argument('mask_dir', help='directory holding hFacC.bits')
    p.add_argument('dest', help='s3://llc4320-v2/SURFACE or a local directory')
    p.add_argument('--FS', type=int, default=4320, help='facet side (default 4320)')
    p.add_argument('--start', type=_parse_date, help='first date to include, YYYY-MM-DD[THH]')
    p.add_argument('--end', type=_parse_date, help='exclusive end date, YYYY-MM-DD[THH]')
    p.add_argument('--limit', type=int, help='process at most N timesteps')
    p.add_argument('--dry-run', action='store_true', help='list what would be written')
    p.add_argument('--no-skip-existing', action='store_true',
                   help='rewrite stores even if already complete')
    p.add_argument('--no-grid', action='store_true', help='do not write grid.zarr')
    p.add_argument('--endpoint', help='S3 endpoint (default $ENDPOINT_URL or Nautilus west)')
    p.add_argument('--profile', help='AWS credentials profile (default $AWS_PROFILE)')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args() if options is None else p.parse_args(options)


def main(args):
    from wrangler.ogcm import llc_v2
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    stats = llc_v2.extract_sst(
        args.out_dir, args.mask_dir, args.dest, FS=args.FS,
        start=args.start, end=args.end, limit=args.limit,
        skip_existing=not args.no_skip_existing, dry_run=args.dry_run,
        write_grid=not args.no_grid, endpoint=args.endpoint, profile=args.profile)
    print(f"discovered={stats['discovered']} written={stats['written']} "
          f"skipped={stats['skipped']} inferred_dt_s={stats['dt_seconds']}")
    return stats
