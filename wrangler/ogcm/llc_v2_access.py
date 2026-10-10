""" Reader helpers for the public LLC4320 v2 surface product on Nautilus S3.

The product is one Zarr store per hour (``{YYYYMMDD}T{HH}.zarr``) plus a
static ``grid.zarr``, all under ``s3://llc4320-v2/SURFACE``.  Each hourly
store holds its fields on the native 13-facet LLC grid with dimensions
``(face, j, i)`` and **no time dimension** -- the hour is the store's name.

That layout is good for writing and for single-time access, and awkward for
everything else, so this module supplies the missing pieces:

* `open_grid` / `open_store` -- one call each, with the right S3 options.
* `open_range` -- a date range as a single lazy dataset with a ``time`` axis.
* `nearest_index` -- latitude/longitude to ``(face, j, i)``, cached on disk.
* `extract_point` -- a time series at a location.
* `region_slices` -- the index ranges of a latitude/longitude box, per face.

Everything is read-only and works anonymously, so no credentials are needed::

    from wrangler.ogcm import llc_v2_access as llc

    ds = llc.open_range('2023-03-01', '2023-03-08')      # 169 hourly steps
    sst = llc.extract_point(36.8, -121.9, '2023-03-01', '2023-04-01')

``xarray`` is imported lazily, so importing this module is cheap.
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

DEFAULT_ENDPOINT = 'https://s3-west.nrp-nautilus.io'
DEFAULT_ROOT = 's3://llc4320-v2/SURFACE'
GRID_STORE = 'grid.zarr'
N_FACES = 13
CADENCE = timedelta(hours=1)
CACHE_DIR = os.path.expanduser('~/.cache/wrangler/llc4320_v2')

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Opening stores
# ---------------------------------------------------------------------------

def storage_options(endpoint: str = None, anon: bool = True, profile: str = None) -> dict:
    """``storage_options`` for ``xarray.open_zarr`` against Nautilus.

    Nautilus runs Ceph RGW, which needs path-style addressing; the bucket is
    public, so *anon* is True by default and no credentials are consulted.

    Args:
        endpoint (str, optional): S3 endpoint. Defaults to ``$ENDPOINT_URL``
            or the Nautilus west endpoint.
        anon (bool, optional): read anonymously. Defaults to True.
        profile (str, optional): credentials profile; implies ``anon=False``.
    """
    opts = {
        'client_kwargs': {'endpoint_url':
                          (endpoint or os.environ.get('ENDPOINT_URL') or DEFAULT_ENDPOINT)},
        'config_kwargs': {'s3': {'addressing_style': 'path'}},
    }
    if profile:
        opts['profile'] = profile
    else:
        opts['anon'] = anon
    return opts


def _is_remote(root) -> bool:
    """True for an ``s3://`` (or other fsspec) URL, False for a local path."""
    return '://' in str(root)


def _open_kwargs(root, **kw) -> dict:
    """``storage_options`` for a remote root, nothing for a local directory.

    Lets every helper here work unchanged against a local mirror of the
    product (or a test fixture), where fsspec options would be rejected.
    """
    return {'storage_options': storage_options(**kw)} if _is_remote(root) else {}


def _parse_date(value) -> datetime:
    """Accept a datetime, ``'YYYY-MM-DD'``, ``'YYYY-MM-DD HH'`` or ``'YYYYMMDDTHH'``."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H', '%Y-%m-%d %H',
                '%Y-%m-%d', '%Y%m%dT%H'):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"cannot parse date {value!r}; try 'YYYY-MM-DD' or 'YYYY-MM-DD HH'")


def store_name(date) -> str:
    """``'2023-03-01 06:00'`` -> ``'20230301T06.zarr'``."""
    return _parse_date(date).strftime('%Y%m%dT%H') + '.zarr'


def hours_between(start, end):
    """Hourly datetimes from *start* to *end*, both inclusive."""
    t, t1 = _parse_date(start), _parse_date(end)
    if t1 < t:
        raise ValueError(f"end ({t1}) precedes start ({t})")
    out = []
    while t <= t1:
        out.append(t)
        t += CADENCE
    return out


def open_grid(root: str = DEFAULT_ROOT, **kw):
    """Open ``grid.zarr`` (XC, YC, Depth, maskC, hFac*_k0, RC/RF, ...).

    Returns:
        xarray.Dataset: dims ``(face, j, i)`` plus the 1D vertical axes.
    """
    import xarray as xr
    return xr.open_zarr(f"{str(root).rstrip('/')}/{GRID_STORE}",
                        consolidated=None, **_open_kwargs(root, **kw))


def open_store(date, root: str = DEFAULT_ROOT, **kw):
    """Open the store for one hour, as written (no ``time`` dimension).

    Args:
        date: anything `_parse_date` accepts.
    """
    import xarray as xr
    return xr.open_zarr(f"{str(root).rstrip('/')}/{store_name(date)}",
                        consolidated=None, **_open_kwargs(root, **kw))


def open_range(start, end, root: str = DEFAULT_ROOT, variables=None,
               skip_missing: bool = True, **kw):
    """Open a date range as one lazy dataset with a ``time`` dimension.

    Store names are generated from the hourly cadence rather than listed, so
    this needs only ``GetObject`` and works even where listing is not
    permitted.  Nothing is read until you compute: the result is lazy, and
    its ``time`` axis is the only thing assembled eagerly.

    Opening N hours costs N metadata requests, so a week is quick and a year
    takes a few minutes. Subset in time first where you can.

    Args:
        start, end: first and last hour, both inclusive.
        root (str, optional): prefix holding the stores.
        variables (list, optional): keep only these (e.g. ``['Theta']``).
        skip_missing (bool, optional): warn and skip an hour that cannot be
            opened, rather than failing the whole range. Defaults to True.

    Returns:
        xarray.Dataset: dims ``(time, face, j, i)``.
    """
    import xarray as xr
    dates = hours_between(start, end)
    opts = _open_kwargs(root, **kw)
    root = str(root).rstrip('/')
    parts, stamps = [], []
    for d in dates:
        try:
            ds = xr.open_zarr(f"{root}/{store_name(d)}", consolidated=None, **opts)
        except Exception as e:
            if not skip_missing:
                raise
            logger.warning("skipping %s (%s: %s)", store_name(d), type(e).__name__, e)
            continue
        if variables is not None:
            ds = ds[list(variables)]
        parts.append(ds)
        stamps.append(np.datetime64(d.replace(tzinfo=None), 'ns'))
    if not parts:
        raise IOError(f"no stores could be opened between {start} and {end}")
    out = xr.concat(parts, dim='time', coords='minimal', compat='override',
                    combine_attrs='drop_conflicts')
    out = out.assign_coords(time=('time', np.array(stamps)))
    out['time'].attrs.update({'standard_name': 'time', 'long_name': 'UTC time of snapshot'})
    return out


# ---------------------------------------------------------------------------
# Locating a latitude/longitude on the native grid
# ---------------------------------------------------------------------------

def _coarse_grid(root: str = DEFAULT_ROOT, stride: int = 16, cache_dir: str = CACHE_DIR,
                 **kw):
    """Sub-sampled XC/YC for the whole globe, cached on disk after the first call.

    The full XC/YC are ~970 MB each, so the first call downloads them and
    saves the strided copy (a few MB) under *cache_dir*. Later calls are
    instant and offline.

    Returns:
        tuple: (xc, yc, stride), each ``(13, FS//stride, FS//stride)``.
    """
    tag = f"coarse_xcyc_stride{stride}.npz"
    path = os.path.join(cache_dir, tag) if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        return z['xc'], z['yc'], int(z['stride'])
    logger.info("building the coarse grid cache (downloads XC/YC once, ~2 GB)")
    g = open_grid(root, **kw)
    sl = slice(None, None, stride)
    xc = g['XC'].isel(j=sl, i=sl).values.astype(np.float32)
    yc = g['YC'].isel(j=sl, i=sl).values.astype(np.float32)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, xc=xc, yc=yc, stride=stride)
        logger.info("cached coarse grid at %s", path)
    return xc, yc, stride


def _angular_distance(lat, lon, lats, lons):
    """Great-circle central angle (radians) from one point to an array of points."""
    la0, lo0 = np.radians(lat), np.radians(lon)
    la, lo = np.radians(lats.astype(np.float64)), np.radians(lons.astype(np.float64))
    d = (np.sin((la - la0) / 2) ** 2
         + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2)
    return 2 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def nearest_index(lat: float, lon: float, root: str = DEFAULT_ROOT, stride: int = 16,
                  cache_dir: str = CACHE_DIR, grid=None, **kw):
    """Nearest native cell to a latitude/longitude.

    Two passes: find the closest point of a sub-sampled grid, then refine at
    full resolution inside the surrounding window.  The coarse grid is cached
    (see `_coarse_grid`), so only the first call downloads anything.

    Longitudes may be given in either -180..180 or 0..360.

    Args:
        lat, lon (float): degrees north / east.
        grid (xarray.Dataset, optional): an already-open `open_grid` result,
            reused for the refinement step.

    Returns:
        dict: ``{'face', 'j', 'i', 'lat', 'lon', 'distance_km'}`` where lat/lon
            are the chosen cell's own centre.
    """
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    xc, yc, stride = _coarse_grid(root, stride, cache_dir, **kw)
    d = _angular_distance(lat, lon, yc, xc)
    f, cj, ci = np.unravel_index(np.nanargmin(d), d.shape)

    if grid is None:
        grid = open_grid(root, **kw)
    FS = grid.sizes['j']
    half = stride                      # search +/- one coarse cell at full resolution
    j0, j1 = max(0, cj * stride - half), min(FS, cj * stride + half + 1)
    i0, i1 = max(0, ci * stride - half), min(FS, ci * stride + half + 1)
    sub = grid.isel(face=int(f), j=slice(j0, j1), i=slice(i0, i1))
    yf, xf = sub['YC'].values, sub['XC'].values
    df = _angular_distance(lat, lon, yf, xf)
    jj, ii = np.unravel_index(np.nanargmin(df), df.shape)
    return {'face': int(f), 'j': int(j0 + jj), 'i': int(i0 + ii),
            'lat': float(yf[jj, ii]), 'lon': float(xf[jj, ii]),
            'distance_km': float(df[jj, ii] * 6371.0)}


def extract_point(lat: float, lon: float, start, end, root: str = DEFAULT_ROOT,
                  variables=('Theta',), stride: int = 16, cache_dir: str = CACHE_DIR, **kw):
    """Time series at the native cell nearest a latitude/longitude.

    Args:
        lat, lon (float): degrees north / east.
        start, end: first and last hour, both inclusive.
        variables (iterable, optional): defaults to ``('Theta',)``.

    Returns:
        xarray.Dataset: dims ``(time,)``, carrying the chosen cell's
            ``face``/``j``/``i`` and its true ``lat``/``lon`` in the
            attributes and as scalar coordinates.
    """
    grid = open_grid(root, **kw)
    loc = nearest_index(lat, lon, root, stride, cache_dir, grid=grid, **kw)
    ds = open_range(start, end, root, variables=list(variables), **kw)
    out = ds.isel(face=loc['face'], j=loc['j'], i=loc['i'])
    out = out.assign_coords(lat=loc['lat'], lon=loc['lon'],
                            face=loc['face'], j=loc['j'], i=loc['i'])
    out.attrs.update({'requested_lat': float(lat), 'requested_lon': float(lon),
                      'distance_km': loc['distance_km']})
    return out


def region_slices(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                  root: str = DEFAULT_ROOT, stride: int = 16, cache_dir: str = CACHE_DIR,
                  **kw) -> dict:
    """Index ranges covering a latitude/longitude box, per face.

    A box on the sphere can straddle several of the 13 facets, and the facets
    are not aligned with latitude/longitude, so there is no single rectangular
    slice.  This returns, for each face that intersects the box, the smallest
    ``(j, i)`` rectangle containing its part of it -- found on the sub-sampled
    grid and padded by one coarse cell, so it is inclusive rather than exact.
    Mask the result with the true coordinates afterwards (see the HOWTO).

    Returns:
        dict: ``{face: (slice_j, slice_i)}`` for the intersecting faces only.
    """
    xc, yc, stride = _coarse_grid(root, stride, cache_dir, **kw)
    lon_min = ((lon_min + 180.0) % 360.0) - 180.0
    lon_max = ((lon_max + 180.0) % 360.0) - 180.0
    inside = (yc >= lat_min) & (yc <= lat_max)
    if lon_min <= lon_max:
        inside &= (xc >= lon_min) & (xc <= lon_max)
    else:                                    # box crosses the date line
        inside &= (xc >= lon_min) | (xc <= lon_max)
    FS = xc.shape[1] * stride
    out = {}
    for f in range(N_FACES):
        jj, ii = np.nonzero(inside[f])
        if jj.size == 0:
            continue
        j0 = max(0, (jj.min() - 1) * stride)
        j1 = min(FS, (jj.max() + 2) * stride)
        i0 = max(0, (ii.min() - 1) * stride)
        i1 = min(FS, (ii.max() + 2) * stride)
        out[f] = (slice(int(j0), int(j1)), slice(int(i0), int(i1)))
    return out
