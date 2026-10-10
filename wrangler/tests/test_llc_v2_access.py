""" Tests for the public reader helpers (`wrangler.ogcm.llc_v2_access`).

Everything runs against a tiny local product built in a tmp dir with the same
writers used for the real one, so no network and no credentials are involved.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from wrangler.ogcm import llc_v2
from wrangler.ogcm import llc_v2_access as llc

FS = 8                       # facet side; must be divisible by 8
N = llc_v2.N_FACETS * FS * FS


def _write(path, arr):
    with open(path, 'wb') as f:
        f.write(np.asarray(arr, dtype='>f4').tobytes())


def _synthetic_lonlat(FS=FS):
    """A crude but unambiguous global mesh: each face gets its own lon band.

    Face f spans longitudes [-180 + f*360/13, ...] and latitudes -80..80, so
    every (face, j, i) has a distinct position and a nearest-point search has
    a single right answer.
    """
    xc = np.zeros((llc_v2.N_FACETS, FS, FS), np.float32)
    yc = np.zeros_like(xc)
    width = 360.0 / llc_v2.N_FACETS
    for f in range(llc_v2.N_FACETS):
        lon0 = -180.0 + f * width
        xc[f] = lon0 + (np.arange(FS) + 0.5) * width / FS          # varies along i
        yc[f] = (-80.0 + (np.arange(FS) + 0.5) * 160.0 / FS)[:, None]  # varies along j
    return xc, yc


@pytest.fixture
def product(tmp_path):
    """A local 3-hour product: grid.zarr + 20230301T00/01/02.zarr."""
    pytest.importorskip('zarr')
    pytest.importorskip('xarray')
    grid_dir = tmp_path / 'griddir'
    grid_dir.mkdir()
    xc, yc = _synthetic_lonlat()
    # compact_to_faces is applied on read, so write the inverse to get these
    # exact values back out of grid.zarr.
    _write(grid_dir / 'XC.data', llc_v2.faces_to_compact(xc))
    _write(grid_dir / 'YC.data', llc_v2.faces_to_compact(yc))
    depth = np.full(N, 1000.0, np.float32)
    depth[:FS * FS] = 0.0                       # face 0 of the compact stream is land
    _write(grid_dir / 'Depth.data', depth)

    dest = tmp_path / 'SURFACE'
    llc_v2.write_grid_store(str(dest), grid_dir=str(grid_dir), FS=FS)

    truth = {}
    for h in range(3):
        date = datetime(2023, 3, 1, h, tzinfo=timezone.utc)
        step = llc_v2.Timestep(field='SST', folder=str(tmp_path), path=str(tmp_path / 'x'),
                               iteration=720 * (h + 1), n_in_folder=h, date=date)
        arr = np.full((llc_v2.N_FACETS, FS, FS), 10.0 + h, np.float32)
        arr[0] = np.nan                          # land
        truth[h] = arr
        llc_v2.write_timestep_store(str(dest), step,
                                    {'Theta': (arr, {'units': 'degC'})})
    return str(dest), xc, yc, truth


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------

def test_parse_date_and_store_name():
    for text in ('2023-03-01 06:00', '2023-03-01 06', '2023-03-01T06', '20230301T06'):
        assert llc.store_name(text) == '20230301T06.zarr'
    assert llc.store_name(datetime(2023, 3, 1, 6)) == '20230301T06.zarr'
    assert llc.store_name('2023-03-01') == '20230301T00.zarr'
    with pytest.raises(ValueError):
        llc.store_name('the first of March')


def test_hours_between():
    hrs = llc.hours_between('2023-03-01 00', '2023-03-01 03')
    assert [h.hour for h in hrs] == [0, 1, 2, 3]           # inclusive both ends
    assert len(llc.hours_between('2023-03-01', '2023-03-02')) == 25
    assert all(h.tzinfo is timezone.utc for h in hrs)
    with pytest.raises(ValueError):
        llc.hours_between('2023-03-02', '2023-03-01')


def test_storage_options():
    o = llc.storage_options()
    assert o['anon'] is True
    assert o['client_kwargs']['endpoint_url'] == llc.DEFAULT_ENDPOINT
    assert o['config_kwargs']['s3']['addressing_style'] == 'path'
    assert llc.storage_options(profile='me')['profile'] == 'me'
    assert 'anon' not in llc.storage_options(profile='me')
    assert llc.storage_options(endpoint='https://x')['client_kwargs']['endpoint_url'] == 'https://x'
    # Remote roots get storage_options; local ones must not.
    assert 'storage_options' in llc._open_kwargs('s3://b/p')
    assert llc._open_kwargs('/tmp/p') == {}


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

def test_open_grid_and_store(product):
    dest, xc, yc, truth = product
    g = llc.open_grid(dest)
    assert dict(g.sizes)['face'] == 13 and g.sizes['j'] == FS
    np.testing.assert_allclose(g['XC'].values, xc, rtol=1e-5)
    np.testing.assert_allclose(g['YC'].values, yc, rtol=1e-5)
    assert 'maskC' in g and 'Depth' in g

    ds = llc.open_store('2023-03-01 01', dest)
    assert 'time' not in ds.dims                     # a single store has no time axis
    np.testing.assert_allclose(ds['Theta'].values, truth[1], equal_nan=True)


def test_open_range(product):
    dest, _, _, truth = product
    ds = llc.open_range('2023-03-01 00', '2023-03-01 02', root=dest)
    assert dict(ds.sizes) == {'time': 3, 'face': 13, 'j': FS, 'i': FS}
    assert [str(t)[:13] for t in ds.time.values] == \
        ['2023-03-01T00', '2023-03-01T01', '2023-03-01T02']
    for h in range(3):
        np.testing.assert_allclose(ds['Theta'].isel(time=h).values, truth[h], equal_nan=True)
    assert ds['time'].attrs['standard_name'] == 'time'

    # Variable selection.
    assert list(llc.open_range('2023-03-01 00', '2023-03-01 01', root=dest,
                               variables=['Theta']).data_vars) == ['Theta']

    # A missing hour is skipped by default, and fatal when asked for.
    ds = llc.open_range('2023-03-01 00', '2023-03-01 05', root=dest)
    assert ds.sizes['time'] == 3
    with pytest.raises(Exception):
        llc.open_range('2023-03-01 00', '2023-03-01 05', root=dest, skip_missing=False)
    with pytest.raises(IOError):
        llc.open_range('2024-01-01 00', '2024-01-01 02', root=dest)


# ---------------------------------------------------------------------------
# Locating points and regions
# ---------------------------------------------------------------------------

def test_nearest_index(product, tmp_path):
    dest, xc, yc, _ = product
    cache = str(tmp_path / 'cache')
    g = llc.open_grid(dest)
    # Ask for the exact centre of a known cell and expect that cell back.
    for face, j, i in ((3, 2, 5), (7, 6, 1), (12, 0, 7)):
        r = llc.nearest_index(float(yc[face, j, i]), float(xc[face, j, i]),
                              root=dest, stride=2, cache_dir=cache, grid=g)
        assert (r['face'], r['j'], r['i']) == (face, j, i)
        assert r['distance_km'] < 1e-3
    # The cache is written once and reused.
    assert os.path.exists(os.path.join(cache, 'coarse_xcyc_stride2.npz'))
    # Longitudes beyond 180 wrap rather than failing.
    a = llc.nearest_index(0.0, -170.0, root=dest, stride=2, cache_dir=cache, grid=g)
    b = llc.nearest_index(0.0, 190.0, root=dest, stride=2, cache_dir=cache, grid=g)
    assert (a['face'], a['j'], a['i']) == (b['face'], b['j'], b['i'])


def test_extract_point(product, tmp_path):
    dest, xc, yc, truth = product
    face, j, i = 5, 3, 4
    out = llc.extract_point(float(yc[face, j, i]), float(xc[face, j, i]),
                            '2023-03-01 00', '2023-03-01 02', root=dest,
                            stride=2, cache_dir=str(tmp_path / 'c2'))
    assert dict(out.sizes) == {'time': 3}
    np.testing.assert_allclose(out['Theta'].values, [10.0, 11.0, 12.0])
    assert int(out['face']) == face and int(out['j']) == j and int(out['i']) == i
    assert out.attrs['requested_lat'] == pytest.approx(float(yc[face, j, i]))
    assert out.attrs['distance_km'] < 1e-3


def test_region_slices(product, tmp_path):
    dest, xc, yc, _ = product
    cache = str(tmp_path / 'c3')
    # A band that only face 4 covers in longitude.
    lo, hi = float(xc[4].min()), float(xc[4].max())
    sl = llc.region_slices(-10, 10, lo, hi, root=dest, stride=2, cache_dir=cache)
    assert 4 in sl
    jsl, isl = sl[4]
    sub = llc.open_grid(dest).isel(face=4, j=jsl, i=isl)
    assert float(sub['YC'].max()) >= 10 or float(sub['YC'].min()) <= -10  # padded, inclusive
    # Nothing matches an impossible latitude band.
    assert llc.region_slices(89, 90, -180, 180, root=dest, stride=2, cache_dir=cache) == {}
    # A box crossing the date line picks up both ends.
    sl = llc.region_slices(-10, 10, 170, -170, root=dest, stride=2, cache_dir=cache)
    assert sl, "date-line-crossing box should match the faces at both ends"
