""" Figures for the LLC4320 v2 slide deck (docs/slides/llc4320_v2.pptx).

Reads straight from the Nautilus bucket ``s3://llc4320-v2/SURFACE`` written by
``wr_llc_v2_surface``; needs the ``[default]`` (or ``$AWS_PROFILE``)
credentials with read access and the wrangler package installed.

    python docs/slides/llc4320_v2_figures.py                 # all figures
    python docs/slides/llc4320_v2_figures.py --only global   # one figure
    python docs/slides/llc4320_v2_figures.py --date 20230701T12 --outdir /tmp/figs

Figures (PNG, written to ``--outdir``, default ``docs/slides/figs``):

    global    global SST, all 13 faces subsampled and binned onto 0.25 deg
    zoom      Gulf Stream at native 1/48 deg resolution (no subsampling)
    series    hourly SST at three points for a week (diurnal cycle) and
              daily SST at the same points for every day written so far
    progress  stores written vs. wall-clock time (from S3 LastModified),
              plus progress.csv for the deck's native chart
    layout    faces 10-12 as stored before the face-order fix vs. after

Stores written before 2026-10-02 hold faces 7-12 in MITgcm compact order
(no ``face_layout`` attribute); the readers here fix that on the fly.

Downloads are cached as .npz under ``--cache`` (default
``docs/slides/figs/cache``) so re-styling a figure does not re-read S3.
A full-resolution store is ~470 MB and takes ~2 min to read.
"""

import argparse
import csv
import datetime as dt
import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from wrangler.ogcm import llc_v2
except ModuleNotFoundError:          # not pip-installed: use this checkout
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from wrangler.ogcm import llc_v2

BUCKET = 'llc4320-v2'
PREFIX = 'SURFACE'
ROOT = f's3://{BUCKET}/{PREFIX}'

# Gulf Stream box for the native-resolution zoom (lon_min, lon_max, lat_min, lat_max)
GULF_STREAM = (-76.0, -56.0, 33.0, 45.0)

# Points for the time series: (label, lon, lat)
POINTS = [('Gulf Stream (38N, 65W)', -65.0, 38.0),
          ('Eq. Pacific (0N, 140W)', -140.0, 0.0),
          ('Southern Ocean (55S, 120E)', 120.0, -55.0)]
COLORS = ['#c0392b', '#e08e0b', '#1f6f9f']

# Each figure is drawn at the size it occupies on its slide (inches), with all
# text at FONT_PT, so labels read at >= 20 pt in the deck.
FONT_PT = 20


def parser(options=None):
    p = argparse.ArgumentParser(description='Make the LLC4320 v2 slide figures from S3.')
    p.add_argument('--only', choices=['global', 'zoom', 'series', 'progress', 'layout'],
                   help='make just this figure')
    p.add_argument('--date', default='20230701T12', help='store for global/zoom (YYYYMMDDTHH)')
    p.add_argument('--stride', type=int, default=4, help='subsampling for the global map')
    p.add_argument('--outdir', default=os.path.join(HERE, 'figs'))
    p.add_argument('--cache', default=None, help='cache dir (default <outdir>/cache)')
    return p.parse_args() if options is None else p.parse_args(options)


# ---------------------------------------------------------------- data access

def _cached(cache, name, fn):
    """Return fn()'s dict of arrays, cached as <cache>/<name>.npz."""
    path = os.path.join(cache, name + '.npz')
    if os.path.exists(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    out = fn()
    os.makedirs(cache, exist_ok=True)
    np.savez(path, **out)
    return out


def open_store(name):
    """Open SURFACE/<name>.zarr read-only.

    Note: ``list(group.keys())`` comes back empty over s3fs for these
    stores; index variables by name instead (``g['Theta']``).
    """
    return llc_v2.open_zarr_group(f'{ROOT}/{name}.zarr', mode='r')


def s3_client():
    import boto3
    import botocore
    sess = boto3.session.Session(profile_name=os.environ.get('AWS_PROFILE', 'default'))
    return sess.client('s3', endpoint_url=llc_v2.s3_endpoint(),
                       config=botocore.config.Config(s3={'addressing_style': 'path'}))


def list_stores(client=None):
    """Sorted datetimes of every hourly store under SURFACE/ (grid.zarr excluded)."""
    client = client or s3_client()
    names = []
    for page in client.get_paginator('list_objects_v2').paginate(
            Bucket=BUCKET, Prefix=PREFIX + '/', Delimiter='/'):
        names += [c['Prefix'].split('/')[1] for c in page.get('CommonPrefixes', [])]
    return sorted(dt.datetime.strptime(n[:11], '%Y%m%dT%H') for n in names
                  if n[:8].isdigit())


def grid_lonlat(stride, cache):
    """Subsampled XC, YC (13, FS/stride, FS/stride)."""
    def fetch():
        g = open_store('grid')
        return {'XC': g['XC'][:, ::stride, ::stride], 'YC': g['YC'][:, ::stride, ::stride]}
    return _cached(cache, f'grid_s{stride}', fetch)


def is_legacy(group):
    """True for stores written before the face-order fix (faces 7-12 in compact order)."""
    return group.attrs.get('face_layout') != llc_v2.FACE_LAYOUT


def read_true_face(group, var, face, FS=4320):
    """One (FS, FS) face in the true layout; for legacy stores faces 7-12 are
    rebuilt from the three stored faces of their facet (see llc_v2.compact_to_faces)."""
    if face < 7 or not is_legacy(group):
        return group[var][face]
    f0 = 7 if face < 10 else 10
    facet = group[var][f0:f0 + 3].reshape(FS, 3 * FS)
    k = face - f0
    return facet[:, k * FS:(k + 1) * FS]


def nearest_index(lon, lat, cache, stride=8):
    """True-layout (face, j, i) of the grid cell nearest (lon, lat).

    Searched in the stored layout (any permutation works for a nearest
    search), refined at full resolution, then converted to true faces.
    """
    g = grid_lonlat(stride, cache)
    d = (np.cos(np.radians(lat)) * (((g['XC'] - lon + 180) % 360) - 180)) ** 2 + (g['YC'] - lat) ** 2
    f, j, i = np.unravel_index(np.argmin(d), d.shape)
    grid = open_store('grid')
    j0, i0 = max(j * stride - stride, 0), max(i * stride - stride, 0)
    sl = (f, slice(j0, j0 + 3 * stride), slice(i0, i0 + 3 * stride))
    X, Y = grid['XC'][sl], grid['YC'][sl]
    d = (np.cos(np.radians(lat)) * (((X - lon + 180) % 360) - 180)) ** 2 + (Y - lat) ** 2
    jj, ii = np.unravel_index(np.argmin(d), d.shape)
    fji = (int(f), int(j0 + jj), int(i0 + ii))
    return llc_v2.compact_index_to_face(*fji, FS=4320) if is_legacy(grid) else fji


# ---------------------------------------------------------------- figures

def fig_global(args, cache):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    s = args.stride
    lonlat = grid_lonlat(s, cache)
    sst = _cached(cache, f'sst_{args.date}_s{s}',
                  lambda: {'Theta': open_store(args.date)['Theta'][:, ::s, ::s]})['Theta']
    lon, lat, v = lonlat['XC'].ravel(), lonlat['YC'].ravel(), sst.ravel()
    ok = np.isfinite(v)
    # bin the native points onto a regular 0.25 deg grid (avoids LLC seams in pcolormesh)
    lon_e, lat_e = np.arange(-180, 180.01, 0.25), np.arange(-90, 90.01, 0.25)
    tot, _, _ = np.histogram2d(lat[ok], lon[ok], bins=[lat_e, lon_e], weights=v[ok])
    cnt, _, _ = np.histogram2d(lat[ok], lon[ok], bins=[lat_e, lon_e])
    with np.errstate(invalid='ignore'):
        img = tot / cnt

    fig = plt.figure(figsize=(8.9, 5.3))
    ax = plt.axes(projection=ccrs.Robinson())
    im = ax.imshow(img, origin='lower', extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(),
                   cmap='RdYlBu_r', vmin=-2, vmax=31, interpolation='nearest', zorder=2)
    ax.add_feature(cfeature.LAND, facecolor='0.82', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='0.82', zorder=0)   # NaN cells (e.g. ice shelves) grey
    ax.set_global()
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, fraction=0.05, aspect=35)
    cb.set_label('SST (°C)')
    out = os.path.join(args.outdir, 'llc4320_v2_sst_global.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_zoom(args, cache):
    import matplotlib.pyplot as plt

    lon0, lon1, lat0, lat1 = GULF_STREAM
    # true face holding the box centre, then the box at full resolution
    f, _, _ = nearest_index(0.5 * (lon0 + lon1), 0.5 * (lat0 + lat1), cache)

    def fetch():
        g, st = open_store('grid'), open_store(args.date)
        X, Y = read_true_face(g, 'XC', f), read_true_face(g, 'YC', f)
        inbox = (X >= lon0) & (X <= lon1) & (Y >= lat0) & (Y <= lat1)
        jj, ii = np.nonzero(inbox)
        sl = (slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1))
        return {'XC': X[sl], 'YC': Y[sl], 'Theta': read_true_face(st, 'Theta', f)[sl]}
    d = _cached(cache, f'zoom_{args.date}_true_f{f}', fetch)
    nj, ni = d['Theta'].shape

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    im = ax.pcolormesh(d['XC'], d['YC'], d['Theta'], cmap='RdYlBu_r', vmin=8, vmax=29,
                       shading='nearest', rasterized=True)
    ax.set_facecolor('0.82')
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.radians(0.5 * (lat0 + lat1))))
    ax.set_xticks([-75, -70, -65, -60])
    ax.set_xlabel('longitude (°E)')
    ax.set_ylabel('latitude (°N)')
    cb = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label('SST (°C)')
    print(f'  zoom: face {f}, {nj} x {ni} cells')
    out = os.path.join(args.outdir, 'llc4320_v2_sst_gulfstream.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_series(args, cache):
    import matplotlib.pyplot as plt

    idx = _cached(cache, 'point_index_true',
                  lambda: {'fji': np.array([nearest_index(lon, lat, cache) for _, lon, lat in POINTS])})['fji']
    stores = list_stores()

    def sample(dates):
        vals = np.full((len(dates), len(POINTS)), np.nan, dtype=np.float32)
        for n, t in enumerate(dates):
            try:
                st = open_store(t.strftime('%Y%m%dT%H'))
                th = st['Theta']
            except Exception:
                continue
            legacy = is_legacy(st)
            for p, fji in enumerate(idx):
                fji = tuple(int(x) for x in fji)
                if legacy:
                    fji = llc_v2.face_index_to_compact(*fji, FS=4320)
                vals[n, p] = th[fji]
        return vals

    # one week hourly (diurnal cycle) and every day at 12 UTC through the last store
    week = [t for t in stores if dt.datetime(2023, 7, 1) <= t < dt.datetime(2023, 7, 8)]
    daily = [t for t in stores if t.hour == 12]
    tag = f'{daily[-1]:%Y%m%d}'
    w = _cached(cache, 'series_week', lambda: {'t': np.array(week, dtype='datetime64[h]'), 'v': sample(week)})
    dly = _cached(cache, f'series_daily_{tag}',
                  lambda: {'t': np.array(daily, dtype='datetime64[h]'), 'v': sample(daily)})

    import matplotlib.dates as mdates

    # left: one week hourly as anomalies (diurnal cycle); right: daily SST
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.3, 5.0), gridspec_kw={'width_ratios': [1, 1.6]})
    for p, (label, _, _) in enumerate(POINTS):
        v = w['v'][:, p]
        a0.plot(w['t'], v - np.nanmean(v), color=COLORS[p], lw=2)
        a1.plot(dly['t'], dly['v'][:, p], color=COLORS[p], lw=1.6, label=label.split(' (')[0])
    a0.set_ylabel('SST − weekly mean (°C)')
    a1.set_ylabel('SST (°C)')
    a0.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    a0.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7]))
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for ax in (a0, a1):
        ax.grid(alpha=0.3)
    fig.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out = os.path.join(args.outdir, 'llc4320_v2_sst_series.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_progress(args, cache):
    """Stores written vs. wall clock; also writes progress.csv for a native chart."""
    import matplotlib.pyplot as plt

    client = s3_client()
    stores = list_stores(client)
    picks = sorted(set(list(range(0, len(stores), 60)) + [len(stores) - 1]))
    rows = []
    for n in picks:
        key = f'{PREFIX}/{stores[n]:%Y%m%dT%H}.zarr/zarr.json'
        lm = client.head_object(Bucket=BUCKET, Key=key)['LastModified']
        rows.append((lm.replace(tzinfo=None), n + 1, stores[n]))
    with open(os.path.join(args.outdir, 'progress.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['written_utc', 'n_stores', 'model_date'])
        for lm, n, t in rows:
            wr.writerow([f'{lm:%Y-%m-%d %H:%M}', n, f'{t:%Y-%m-%d %H:00}'])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], color='#1f6f9f', lw=2)
    ax.axhline(9503, color='0.5', ls='--', lw=1)
    ax.text(rows[0][0], 9503, '  9,503 hours on disk', va='bottom', color='0.4')
    ax.set_ylabel('stores written')
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out = os.path.join(args.outdir, 'llc4320_v2_progress.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_layout(args, cache):
    """Faces 10-12 SST as stored before the face-order fix vs. after llc_v2.compact_to_faces."""
    import matplotlib.pyplot as plt

    s = 8
    def fetch():
        st = open_store(args.date)
        raw = st['Theta'][10:13]
        fixed = np.stack(np.split(raw.reshape(4320, 3 * 4320), 3, axis=1))
        return {'stored': raw[:, ::s, ::s], 'fixed': fixed[:, ::s, ::s]}
    d = _cached(cache, f'layout_{args.date}_s{s}', fetch)

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.6))
    for r, key in enumerate(('stored', 'fixed')):
        for k in range(3):
            ax = axes[r, k]
            ax.imshow(d[key][k], origin='lower', cmap='RdYlBu_r', vmin=-2, vmax=31,
                      interpolation='nearest')
            ax.set_facecolor('0.82')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f'face {10 + k}')
        axes[r, 0].set_ylabel('as stored' if r == 0 else 'fixed')
    fig.tight_layout()
    out = os.path.join(args.outdir, 'llc4320_v2_face_layout.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


FIGURES = {'global': fig_global, 'zoom': fig_zoom, 'series': fig_series,
           'progress': fig_progress, 'layout': fig_layout}


def main(args):
    import matplotlib
    matplotlib.rcParams.update({'font.size': FONT_PT, 'axes.titlesize': FONT_PT,
                                'axes.labelsize': FONT_PT, 'xtick.labelsize': FONT_PT,
                                'ytick.labelsize': FONT_PT, 'legend.fontsize': FONT_PT})
    os.makedirs(args.outdir, exist_ok=True)
    cache = args.cache or os.path.join(args.outdir, 'cache')
    for name, fn in FIGURES.items():
        if args.only and name != args.only:
            continue
        print(f'{name}: {fn(args, cache)}', flush=True)


if __name__ == '__main__':
    main(parser())
