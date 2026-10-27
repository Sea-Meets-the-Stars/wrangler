""" Relabel a store's horizontal dimensions to the true Arakawa C-grid staggering.

Stores written before 2026-10-20 declared ``(face, j, i)`` for every variable,
so the staggered ones claimed to sit at cell centres and ``xgcm`` could not
build a Grid from them (reported by Lauren, 2026-10-20)::

    wr_llc_v2_fix_grid_dims s3://llc4320-v2/SURFACE/grid.zarr --dry-run
    wr_llc_v2_fix_grid_dims s3://llc4320-v2/SURFACE/grid.zarr

Metadata only: the values are already correct and in the correct positions,
so no chunk is read or rewritten.  Safe to re-run -- a store already carrying
the right names is reported as ``already`` and left alone.

Only ``grid.zarr`` is affected in practice; the hourly stores hold ``Theta``
alone, a tracer, which genuinely belongs at cell centres.
"""

import argparse
import logging


def parser(options=None):
    p = argparse.ArgumentParser(
        description='Fix C-grid dimension labels on an LLC4320 v2 Zarr store.')
    p.add_argument('store', nargs='?', default='s3://llc4320-v2/SURFACE/grid.zarr',
                   help='store URL (default the public grid.zarr)')
    p.add_argument('--dry-run', action='store_true', help='report without writing')
    p.add_argument('--endpoint', help='S3 endpoint (default $ENDPOINT_URL or Nautilus west)')
    p.add_argument('--profile', help='AWS credentials profile (default $AWS_PROFILE)')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args() if options is None else p.parse_args(options)


def main(args):
    from wrangler.ogcm import llc_v2
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    rep = llc_v2.repair_store_dims(args.store, dry_run=args.dry_run,
                                   endpoint=args.endpoint, profile=args.profile)
    verb = 'would relabel' if args.dry_run else 'relabelled'
    print(f"{args.store}")
    print(f"  {verb}: {len(rep['fixed'])}" + (f"  {rep['fixed']}" if rep['fixed'] else ""))
    print(f"  already correct: {len(rep['already'])}")
    if rep['coords_added']:
        print(f"  coordinates {'to add' if args.dry_run else 'added'}: {rep['coords_added']}")
    if rep['attrs_set']:
        print(f"  COMODO attrs set on: {rep['attrs_set']}")
    return rep


if __name__ == '__main__':
    main(parser())
