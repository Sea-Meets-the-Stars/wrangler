""" Inventory the LLC4320 v2 raw-output tree: which fields, in which file type, per folder.

One compact report instead of ad-hoc ``ls`` commands::

    wr_llc_v2_inventory /nobackupp27/dbwhitt/llc_4320/OUT
    wr_llc_v2_inventory /nobackupp27/dbwhitt/llc_4320/OUT --max-folders 5
    wr_llc_v2_inventory /nobackupp27/dbwhitt/llc_4320/OUT --find /nobackupp29/bcnelson/MITgcm --depth 4

Per folder it prints the file count, the fields present as ``.shrunk`` and as
``.data``, the iteration range, and any non-field entries (namelists, subdirs).
It ends with a summary: how many folders hold ``<field>.*.shrunk`` and their
date range, plus every (field, extension) combination seen.  ``--find`` runs a
bounded search for ``*.shrunk`` files under another root.
"""

import argparse
import os


def parser(options=None):
    p = argparse.ArgumentParser(description='Inventory LLC4320 v2 raw-output folders.')
    p.add_argument('out_dir', help='raw-output parent, e.g. /nobackupp27/dbwhitt/llc_4320/OUT')
    p.add_argument('--field', default='SST', help='field to look for (default SST)')
    p.add_argument('--ext', default='data', help='extension to look for (default data)')
    p.add_argument('--max-folders', type=int, help='only inventory the first N folders')
    p.add_argument('--find', metavar='ROOT', help='also search ROOT for *.shrunk files')
    p.add_argument('--depth', type=int, default=3, help='max depth for --find (default 3)')
    p.add_argument('--quiet', action='store_true', help='summary only, no per-folder lines')
    return p.parse_args() if options is None else p.parse_args(options)


def _fmt_fields(inv, ext):
    names = inv.fields_with_ext(ext)
    return ','.join(names) if names else '-'


def main(args):
    from wrangler.ogcm import llc_v2

    unreadable = []
    invs = llc_v2.inventory(args.out_dir, max_folders=args.max_folders, unreadable=unreadable)
    if not args.quiet:
        print(f"{'folder':42s} {'files':>5s}  iterations         shrunk: ...  data: ...")
        for inv in invs:
            its = [v for v in inv.fields.values()]
            rng = (f"{min(v[1] for v in its)}..{max(v[2] for v in its)}" if its else '-')
            print(f"{os.path.basename(inv.folder):42s} {inv.n_files:5d}  {rng:18s} "
                  f"shrunk:[{_fmt_fields(inv, 'shrunk')}]  data:[{_fmt_fields(inv, 'data')}]")
            others = [o for o in inv.other if not o.startswith('data')]
            if others:
                print(f"{'':42s}        other: {' '.join(others[:8])}"
                      f"{' ...' if len(others) > 8 else ''}")

    summ = llc_v2.summarize_inventory(invs, args.field, args.ext)
    print(f"\nfolders: {summ['n_folders']}   "
          f"with {args.field}.*.{args.ext}: {summ['with_field']}", end='')
    if summ['first_with'] is not None:
        print(f"   ({summ['first_with']:%Y-%m-%d %H:%M} to {summ['last_with']:%Y-%m-%d %H:%M})")
    else:
        print()
    if unreadable:
        print(f"unreadable (permission denied), skipped: {len(unreadable)}: "
              f"{' '.join(os.path.basename(u) for u in unreadable)}")
    summ['unreadable'] = [os.path.basename(u) for u in unreadable]
    if 0 < len(summ['without_field']) <= 12:
        print(f"folders lacking it: {' '.join(summ['without_field'])}")
    print("(field, ext) -> number of folders containing it:")
    for (field, ext), n in summ['combos'].items():
        print(f"  {field}.{ext:10s} {n}")

    if args.find:
        hits, total = llc_v2.find_files(args.find, '.shrunk', max_depth=args.depth)
        print(f"\n*.shrunk files under {args.find} (depth <= {args.depth}): {total}")
        for h in hits:
            print(f"  {h}")
        if total > len(hits):
            print(f"  ... ({total - len(hits)} more)")
    return summ


if __name__ == '__main__':
    main(parser())
