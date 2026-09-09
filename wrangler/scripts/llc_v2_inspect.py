""" Sanity-check an LLC4320 v2 Zarr store written by ``wr_llc_v2_sst``.

Meant for the first real decode on Pleiades, where Claude can't look:
prints store attributes, per-face statistics (wet fraction, min/max/mean),
flags physically implausible values, and -- if a plain ``Eta.<iter>.data``
file from the same folder is given -- checks that the SST land mask agrees
with Eta's zero-over-land pattern, which independently validates the mask
bit order and mask/data pairing.  Optionally saves a 13-panel PNG.

    wr_llc_v2_inspect /path/to/20230101T00.zarr
    wr_llc_v2_inspect /path/to/20230101T00.zarr --eta OUT/<folder>/Eta.0000000720.data
    wr_llc_v2_inspect s3://llc4320-v2/SURFACE/20230101T00.zarr --png sst.png
"""

import argparse

import numpy as np

# Plausible open-ocean SST range (degC); anything outside is flagged.
SST_MIN, SST_MAX = -3.0, 40.0


def parser(options=None):
    p = argparse.ArgumentParser(description='Inspect an LLC4320 v2 surface Zarr store.')
    p.add_argument('store', help='local path or s3:// URL of a YYYYMMDDTHH.zarr store')
    p.add_argument('--var', default='Theta', help='variable to inspect (default Theta)')
    p.add_argument('--eta', help='matching Eta.<iteration>.data file to cross-check the land mask')
    p.add_argument('--FS', type=int, default=4320, help='facet side for --eta (default 4320)')
    p.add_argument('--png', help='write a 13-panel image of the field here (needs matplotlib)')
    p.add_argument('--endpoint', help='S3 endpoint (default $ENDPOINT_URL or Nautilus west)')
    p.add_argument('--profile', help='AWS credentials profile (default $AWS_PROFILE)')
    return p.parse_args() if options is None else p.parse_args(options)


def face_stats(field: np.ndarray):
    """Per-face (n_wet, wet_fraction, min, max, mean) for a (13, FS, FS) field."""
    rows = []
    for f in range(field.shape[0]):
        a = field[f]
        wet = np.isfinite(a)
        n = int(wet.sum())
        if n:
            rows.append((f, n, n / a.size, float(np.nanmin(a)), float(np.nanmax(a)),
                         float(np.nanmean(a))))
        else:
            rows.append((f, 0, 0.0, np.nan, np.nan, np.nan))
    return rows


def compare_with_eta(field: np.ndarray, eta: np.ndarray):
    """Agreement between the field's finite (wet) points and Eta's non-zero points.

    Returns:
        dict: fractions of Eta-wet points that are NaN in *field* and of
            field-wet points that are exactly 0 in Eta (both ~0 if the mask
            is right; ~0.3-0.5 if the bit order or mask file is wrong).
    """
    eta_wet = eta != 0.0
    fld_wet = np.isfinite(field)
    n_eta = int(eta_wet.sum())
    n_fld = int(fld_wet.sum())
    return {
        'n_eta_wet': n_eta,
        'n_field_wet': n_fld,
        'eta_wet_but_field_nan': float((eta_wet & ~fld_wet).sum() / max(n_eta, 1)),
        'field_wet_but_eta_zero': float((fld_wet & ~eta_wet).sum() / max(n_fld, 1)),
    }


def main(args):
    from wrangler.ogcm import llc_v2

    root = llc_v2.open_zarr_group(args.store, mode='r', endpoint=args.endpoint,
                                  profile=args.profile)
    print(f"store: {args.store}")
    for k, v in sorted(root.attrs.items()):
        print(f"  {k}: {v}")
    print(f"  arrays: {sorted(root.array_keys())}")
    if args.var not in root:
        raise SystemExit(f"variable {args.var!r} not in store")
    z = root[args.var]
    print(f"{args.var}: shape={z.shape} chunks={z.chunks} dtype={z.dtype} "
          f"dims={tuple(z.metadata.dimension_names)} attrs={dict(z.attrs)}")
    field = z[:]

    print(f"\n{'face':>4} {'n_wet':>10} {'wet_frac':>8} {'min':>9} {'max':>9} {'mean':>9}")
    for f, n, frac, mn, mx, mean in face_stats(field):
        print(f"{f:>4} {n:>10} {frac:>8.3f} {mn:>9.3f} {mx:>9.3f} {mean:>9.3f}")
    wet = np.isfinite(field)
    n_wet = int(wet.sum())
    print(f"\nglobal: wet fraction {n_wet / field.size:.3f}, "
          f"min {np.nanmin(field):.3f}, max {np.nanmax(field):.3f}, "
          f"mean {np.nanmean(field):.3f}")
    zero = int((field == 0.0).sum())
    if args.var == 'Theta':
        bad = int(((field < SST_MIN) | (field > SST_MAX)).sum())
        print(f"values outside [{SST_MIN}, {SST_MAX}]: {bad}   exact zeros: {zero} "
              f"({zero / max(n_wet, 1):.4f} of wet points -- should be ~0)")
        verdict = 'OK' if bad == 0 and zero / max(n_wet, 1) < 1e-3 else 'SUSPICIOUS'
        print(f"value check: {verdict}")
    else:
        bad = 0
        print(f"exact zeros: {zero} ({zero / max(n_wet, 1):.4f} of finite points; "
              f"land is 0, not NaN, in grid variables)")
        print("value check: n/a (plausibility range is only defined for Theta); "
              "judge the per-face min/max above")

    result = {'n_wet': n_wet, 'n_bad': bad, 'n_zero': zero}
    if args.eta:
        eta = llc_v2.read_data_field(args.eta, args.FS)
        cmp = compare_with_eta(field, eta)
        result.update(cmp)
        print(f"\nmask check vs {args.eta}:")
        for k, v in cmp.items():
            print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")
        ok = cmp['eta_wet_but_field_nan'] < 0.01 and cmp['field_wet_but_eta_zero'] < 0.01
        print(f"mask check: {'OK' if ok else 'MISMATCH -- mask bit order / mask file / pairing is wrong'}")

    if args.png:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping --png")
        else:
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            vmin, vmax = np.nanpercentile(field, [1, 99])
            for f, ax in enumerate(axes.ravel()):
                ax.set_axis_off()
                if f < field.shape[0]:
                    im = ax.imshow(field[f][::-1], vmin=vmin, vmax=vmax, cmap='viridis')
                    ax.set_title(f"face {f}")
            fig.colorbar(im, ax=axes, shrink=0.6, label=args.var)
            fig.suptitle(f"{args.store}  {root.attrs.get('selected_date_utc', '')}")
            fig.savefig(args.png, dpi=80, bbox_inches='tight')
            print(f"wrote {args.png}")
    return result


if __name__ == '__main__':
    main(parser())
