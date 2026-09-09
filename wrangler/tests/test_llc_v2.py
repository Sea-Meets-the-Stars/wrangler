""" Tests for the pure-Python .shrunk (LLC4320 v2) decoder.

There's no real .shrunk/.bits sample data available locally (it only
lives on NASA Pleiades), so these build small synthetic files that
follow the exact same bit-packing / big-endian float32 scheme as the
real format (per matlab_v0/llc_shrunk_mex.c) and check the round trip.
"""

import os

import numpy as np
import pytest

from wrangler.ogcm import llc_v2


FS = 8  # facet side; must be divisible by 8. Small so tests are fast.


def _write_level(tmp_path, name_prefix, mask_bits, values, mode='wb'):
    """Write one level's mask bits and packed big-endian float32 values.

    Args:
        mask_bits (np.ndarray): boolean array, length 13*FS*FS.
        values (np.ndarray): float32 values for the True positions, in
            flat order.

    Returns:
        tuple: (mask_file, shrunk_file) paths.
    """
    mask_file = tmp_path / f"{name_prefix}.bits"
    shrunk_file = tmp_path / f"{name_prefix}.shrunk"

    packed = np.packbits(mask_bits.astype(np.uint8), bitorder='little')
    with open(mask_file, mode) as f:
        f.write(packed.tobytes())
    with open(shrunk_file, mode) as f:
        f.write(values.astype('>f4').tobytes())

    return str(mask_file), str(shrunk_file)


def test_mask_bytes_per_level():
    assert llc_v2.mask_bytes_per_level(FS) == 13 * FS * FS // 8
    with pytest.raises(ValueError):
        llc_v2.mask_bytes_per_level(7)  # not divisible by 8


def test_mask_file_for_field():
    assert llc_v2.mask_file_for_field('U') == 'hFacW.bits'
    assert llc_v2.mask_file_for_field('V') == 'hFacS.bits'
    for field in ['Theta', 'Salt', 'W', 'Eta', 'oceTAUX']:
        assert llc_v2.mask_file_for_field(field) == 'hFacC.bits'


def test_read_mask_level_and_decompress_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    n = 13 * FS * FS
    mask_bits = rng.random(n) < 0.6  # ~60% wet
    values = rng.uniform(-2, 35, size=int(mask_bits.sum())).astype(np.float32)

    mask_file, shrunk_file = _write_level(tmp_path, 'level0', mask_bits, values)

    # Mask round-trips exactly.
    mask = llc_v2.read_mask_level(mask_file, FS, level=0)
    np.testing.assert_array_equal(mask, mask_bits)

    # Decompressed field: wet points match input values, dry points are 0.
    field = llc_v2.decompress_level(shrunk_file, mask, byte_offset=0)
    assert field.shape == (13, FS, FS)
    np.testing.assert_allclose(field.reshape(-1)[mask], values, rtol=1e-6)
    assert np.all(field.reshape(-1)[~mask] == 0.0)


def test_detect_nz_and_multi_level_offsets(tmp_path):
    rng = np.random.default_rng(7)
    n = 13 * FS * FS
    n_levels = 3

    all_masks = [rng.random(n) < 0.5 for _ in range(n_levels)]
    all_values = [
        rng.uniform(-2, 35, size=int(m.sum())).astype(np.float32)
        for m in all_masks
    ]

    mask_file = tmp_path / 'hFacC.bits'
    shrunk_file = tmp_path / 'Theta.0000000000.shrunk'
    with open(mask_file, 'wb') as fm, open(shrunk_file, 'wb') as fd:
        for m in all_masks:
            fm.write(np.packbits(m.astype(np.uint8), bitorder='little').tobytes())
        for v in all_values:
            fd.write(v.astype('>f4').tobytes())

    assert llc_v2.detect_nz(str(mask_file), FS) == n_levels

    for lev in range(n_levels):
        expected_offset = sum(int(m.sum()) for m in all_masks[:lev]) * 4
        assert llc_v2.level_byte_offset(str(mask_file), FS, lev) == expected_offset

        mask = llc_v2.read_mask_level(str(mask_file), FS, level=lev)
        np.testing.assert_array_equal(mask, all_masks[lev])

        field = llc_v2.decompress_level(str(shrunk_file), mask, byte_offset=expected_offset)
        np.testing.assert_allclose(
            field.reshape(-1)[mask], all_values[lev], rtol=1e-6)


def test_read_shrunk_field_and_read_sst(tmp_path):
    data_dir = tmp_path / 'data'
    mask_dir = tmp_path / 'mask'
    data_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.default_rng(99)
    n = 13 * FS * FS
    mask_bits = rng.random(n) < 0.7
    sst_values = rng.uniform(-2, 32, size=int(mask_bits.sum())).astype(np.float32)

    packed = np.packbits(mask_bits.astype(np.uint8), bitorder='little')
    with open(mask_dir / 'hFacC.bits', 'wb') as f:
        f.write(packed.tobytes())
    with open(data_dir / 'Theta.0000021240.shrunk', 'wb') as f:
        f.write(sst_values.astype('>f4').tobytes())

    field = llc_v2.read_shrunk_field(str(data_dir), str(mask_dir), 'Theta',
                                     21240, FS, level=0)
    assert field.shape == (13, FS, FS)
    np.testing.assert_allclose(field.reshape(-1)[mask_bits], sst_values, rtol=1e-6)

    # read_sst is the same thing for Theta/k=0.
    sst = llc_v2.read_sst(str(data_dir), str(mask_dir), 21240, FS=FS)
    np.testing.assert_array_equal(sst, field)


def test_read_shrunk_field_missing_files(tmp_path):
    with pytest.raises(IOError):
        llc_v2.read_shrunk_field(str(tmp_path), str(tmp_path), 'Theta',
                                 123, FS, level=0)


# ---------------------------------------------------------------------------
# Discovery (folder naming + Dimitris' "start + n hours" dating rule)
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

STRIDE = 144  # iterations per hour used by the shrunk-format fixtures below

# Two run segments with different deltaT / nIter0 (Dan: both vary per folder).
#   folder 1: 00h -> 03h, deltaT=5 s  (720 it/h), nIter0=0     -> files at 720, 1440, 2160
#   folder 2: 03h -> 05h, deltaT=10 s (360 it/h), nIter0=2160  -> files at 2520, 2880
# Dating rule: first file = folder start + 1 h, last file = folder end.
SEG1 = ('2023_01_01_000000_to_2023_01_01_030000', 720, 0, [1, 2, 3])
SEG2 = ('2023_01_01_030000_to_2023_01_01_050000', 360, 2160, [4, 5])


def _write_data(path, arr):
    with open(path, 'wb') as f:
        f.write(np.asarray(arr, dtype='>f4').tobytes())


def _make_out_tree(tmp_path, FS=FS, seed=0, fields=('SST', 'SSS')):
    """Synthetic /nobackupp27/.../OUT with hourly uncompressed surface .data files.

    Land points are exactly 0 in every field (as MITgcm writes them); the
    matching hFacC.bits mask is written to a mask dir.  Folder 1 carries a
    `data` namelist, folder 2 a STDOUT.0000, each consistent with the file
    iteration numbers.  Distractors: Eta.*.data (another field), .meta
    companions, a non-output folder, a stray file.

    Returns:
        (out_dir, mask_dir, truth) with truth = {iteration: {field: flat float32 values
        over the whole grid (0 on land)}} plus truth['wet'] = flat bool mask,
        truth['hours'] = {iteration: hour of day}.
    """
    rng = np.random.default_rng(seed)
    n = 13 * FS * FS
    wet = rng.random(n) < 0.7
    out_dir = tmp_path / 'OUT'
    mask_dir = tmp_path / 'mask'
    out_dir.mkdir()
    mask_dir.mkdir()
    with open(mask_dir / 'hFacC.bits', 'wb') as f:
        f.write(np.packbits(wet.astype(np.uint8), bitorder='little').tobytes())
    # U/V masks (slightly different, as on a C-grid) for SSU/SSV tests.
    for name in ('hFacW.bits', 'hFacS.bits'):
        with open(mask_dir / name, 'wb') as f:
            f.write(np.packbits(wet.astype(np.uint8), bitorder='little').tobytes())

    truth = {'wet': wet, 'hours': {}}
    for name, stride, n_iter0, hours in (SEG1, SEG2):
        d = out_dir / name
        d.mkdir()
        t0_hour = int(name[11:13])
        for k, h in enumerate(hours):
            it = n_iter0 + (k + 1) * stride
            assert h == t0_hour + k + 1
            truth['hours'][it] = h
            truth[it] = {}
            for fld in fields:
                vals = np.zeros(n, dtype=np.float32)
                lo, hi = (-2, 32) if fld == 'SST' else (30, 38)
                vals[wet] = rng.uniform(lo, hi, size=int(wet.sum())).astype(np.float32)
                vals[wet & (vals == 0)] = 0.5
                _write_data(d / f'{fld}.{it:010d}.data', vals)
                (d / f'{fld}.{it:010d}.meta').write_text('nDims = [ 2 ];\n')
                truth[it][fld] = vals
            eta = np.zeros(n, dtype=np.float32)
            eta[wet] = 0.1
            _write_data(d / f'Eta.{it:010d}.data', eta)
        dt = 3600 // stride
        if name == SEG1[0]:
            (d / 'data').write_text(f" &PARM03\n nIter0={n_iter0},\n deltaT={dt}.,\n &\n")
        else:
            (d / 'STDOUT.0000').write_text(
                "(PID.TID 0000.0001) // Model parameters\n"
                f"(PID.TID 0000.0001) > nIter0={n_iter0},\n"
                f"(PID.TID 0000.0001) > deltaT={dt}.,\n"
                f"(PID.TID 0000.0001) deltaT = {dt:.15E} /* time step */\n")
    (out_dir / 'not_an_output_folder').mkdir()
    (out_dir / 'proc').mkdir()
    (out_dir / 'README.txt').write_text('ignore me')
    return str(out_dir), str(mask_dir), truth


def test_parse_folder_name():
    t0, t1 = llc_v2.parse_folder_name('2023_01_08_060000_to_2023_01_12_020000')
    assert t0 == datetime(2023, 1, 8, 6, tzinfo=timezone.utc)
    assert t1 == datetime(2023, 1, 12, 2, tzinfo=timezone.utc)
    # Full paths are fine too.
    t0b, _ = llc_v2.parse_folder_name('/nobackupp27/dbwhitt/llc_4320/OUT/'
                                      '2023_01_08_060000_to_2023_01_12_020000/')
    assert t0b == t0
    with pytest.raises(ValueError):
        llc_v2.parse_folder_name('matlab_v0')


def test_store_name_for_date():
    d = datetime(2023, 1, 1, 6, tzinfo=timezone.utc)
    assert llc_v2.store_name_for_date(d) == '20230101T06.zarr'


def test_discover_timesteps_dates_and_ordering(tmp_path, caplog):
    out_dir, _, truth = _make_out_tree(tmp_path)
    with caplog.at_level('INFO', logger='wrangler.ogcm.llc_v2'):
        steps = llc_v2.discover_timesteps(out_dir, 'SST')

    assert len(steps) == 5
    iters = sorted(truth['hours'])
    assert [s.iteration for s in steps] == iters
    # Dan's rule: first file = folder start + 1 h ... last = folder end.
    expected = [datetime(2023, 1, 1, truth['hours'][it], tzinfo=timezone.utc) for it in iters]
    assert [s.date for s in steps] == expected
    assert [s.date.hour for s in steps] == [1, 2, 3, 4, 5]
    assert [s.n_in_folder for s in steps] == [0, 1, 2, 0, 1]
    assert steps[3].folder.endswith(SEG2[0])
    assert all(os.path.isfile(s.path) and s.path.endswith('.data') for s in steps)
    assert steps[0].store_name == '20230101T01.zarr'
    assert steps[4].store_name == '20230101T05.zarr'

    # Distinct model timesteps per segment, from the within-folder strides.
    assert llc_v2.infer_timestep_seconds(steps) == [5.0, 10.0]

    # Namelist (folder 1) and STDOUT (folder 2) both agree with the rule:
    # info lines, no warnings.
    assert caplog.text.count('numbering convention: absolute') == 2
    assert 'disagrees' not in caplog.text
    assert 'spans' not in caplog.text

    # Other fields / extensions discover independently.
    assert len(llc_v2.discover_timesteps(out_dir, 'Eta')) == 5
    assert llc_v2.discover_timesteps(out_dir, 'Salt') == []
    assert llc_v2.discover_timesteps(out_dir, 'SST', ext='shrunk') == []


def test_discover_timesteps_date_window(tmp_path):
    out_dir, _, _ = _make_out_tree(tmp_path)
    start = datetime(2023, 1, 1, 2)             # naive -> treated as UTC
    end = datetime(2023, 1, 1, 5, tzinfo=timezone.utc)
    steps = llc_v2.discover_timesteps(out_dir, 'SST', start=start, end=end)
    assert [s.date.hour for s in steps] == [2, 3, 4]


def test_discover_timesteps_warns_on_gap(tmp_path, caplog):
    out_dir, _, _ = _make_out_tree(tmp_path)
    # A folder claiming 4 hours (00..04 on Jan 2) but holding hours 1, 3, 4
    # only: non-constant iteration stride + span/count mismatch -> warnings.
    gap = os.path.join(out_dir, '2023_01_02_000000_to_2023_01_02_040000')
    os.mkdir(gap)
    for h in (1, 3, 4):
        open(os.path.join(gap, f'SST.{h * 720:010d}.data'), 'wb').close()
    with caplog.at_level('WARNING', logger='wrangler.ogcm.llc_v2'):
        steps = llc_v2.discover_timesteps(out_dir, 'SST')
    assert len(steps) == 8
    assert 'stride is not constant' in caplog.text
    assert 'spans 4.00 h but holds 3 files -- expected exactly 4' in caplog.text
    # The gap folder's dates after the hole are (knowingly) position-based.
    gap_steps = [s for s in steps if s.folder == gap]
    assert [s.date.hour for s in gap_steps] == [1, 2, 3]
    # Gap folder strides 1440 and 720 add 2.5 s and 5 s to the segment values.
    assert llc_v2.infer_timestep_seconds(steps) == [2.5, 5.0, 10.0]
def test_decompress_dry_value(tmp_path):
    rng = np.random.default_rng(3)
    n = 13 * FS * FS
    mask_bits = rng.random(n) < 0.5
    values = rng.uniform(0, 1, size=int(mask_bits.sum())).astype(np.float32)
    _, shrunk_file = _write_level(tmp_path, 'lvl', mask_bits, values)
    field = llc_v2.decompress_level(shrunk_file, mask_bits, dry_value=np.nan)
    flat = field.reshape(-1)
    assert np.all(np.isnan(flat[~mask_bits]))
    np.testing.assert_allclose(flat[mask_bits], values, rtol=1e-6)


# ---------------------------------------------------------------------------
# Zarr output (needs zarr; skipped in envs without it, e.g. ocean14)
# ---------------------------------------------------------------------------

def test_extract_surface_end_to_end_local(tmp_path, caplog):
    zarr = pytest.importorskip('zarr')
    out_dir, mask_dir, truth = _make_out_tree(tmp_path)
    wet = truth['wet']
    dest = tmp_path / 'dest'

    # Dry run writes nothing.
    stats = llc_v2.extract_surface(out_dir, str(dest), fields=['SST', 'SSS'], FS=FS,
                                   dry_run=True)
    assert stats['discovered'] == 5 and stats['written'] == 0
    assert not dest.exists()

    stats = llc_v2.extract_surface(out_dir, str(dest), fields=['SST', 'SSS'],
                                   mask_dir=mask_dir, FS=FS)
    assert stats == {'discovered': 5, 'written': 5, 'skipped': 0, 'incomplete': 0,
                     'dt_seconds': [5.0, 10.0]}

    names = sorted(p.name for p in dest.iterdir())
    assert names == ['20230101T01.zarr', '20230101T02.zarr', '20230101T03.zarr',
                     '20230101T04.zarr', '20230101T05.zarr', 'grid.zarr']

    # grid.zarr from the mask dir alone: maskC only.
    g = zarr.open_group(str(dest / 'grid.zarr'), mode='r', use_consolidated=False)
    np.testing.assert_array_equal(g['maskC'][:].reshape(-1), wet)
    assert g.attrs['complete'] is True and g.attrs['variables'] == ['maskC']

    # One hourly store: dims/chunks/attrs/values, NaN over land, both variables.
    steps = llc_v2.discover_timesteps(out_dir, 'SST')
    s = steps[3]                                            # 04h, folder 2
    g = zarr.open_group(str(dest / s.store_name), mode='r', use_consolidated=False)
    assert sorted(g.array_keys()) == ['Salt', 'Theta', 'face', 'i', 'j']
    z = g['Theta']
    assert z.shape == (13, FS, FS)
    assert z.chunks == (1, FS, FS)            # (1, 720, 720) clipped to FS=8
    assert tuple(z.metadata.dimension_names) == ('face', 'j', 'i')
    assert g.attrs['selected_iteration'] == s.iteration == 2160 + 360
    assert g.attrs['selected_date_utc'] == '2023-01-01 04:00:00'
    assert g.attrs['source_folder'] == SEG2[0]
    assert g.attrs['variables'] == ['Salt', 'Theta']
    assert g.attrs['complete'] is True
    assert z.attrs['units'] == 'degC' and z.attrs['source_file_prefix'] == 'SST'
    assert z.attrs['source_file'] == f'SST.{s.iteration:010d}.data'
    for var, prefix in (('Theta', 'SST'), ('Salt', 'SSS')):
        flat = g[var][:].reshape(-1)
        np.testing.assert_allclose(flat[wet], truth[s.iteration][prefix][wet], rtol=1e-6)
        assert np.all(np.isnan(flat[~wet]))
    np.testing.assert_array_equal(g['face'][:], np.arange(13))
    np.testing.assert_array_equal(g['j'][:], np.arange(FS))

    # Idempotent: a second run skips everything.
    stats = llc_v2.extract_surface(out_dir, str(dest), fields=['SST', 'SSS'],
                                   mask_dir=mask_dir, FS=FS)
    assert stats['written'] == 0 and stats['skipped'] == 5

    # Asking for an extra variable makes the stores "incomplete" -> rewritten.
    stats = llc_v2.extract_surface(out_dir, str(dest), fields=['SST', 'SSS', 'Eta'],
                                   mask_dir=mask_dir, FS=FS, limit=1)
    assert stats['written'] == 1 and stats['skipped'] == 0
    g = zarr.open_group(str(dest / '20230101T01.zarr'), mode='r', use_consolidated=False)
    assert g.attrs['variables'] == ['Eta', 'Salt', 'Theta']

    # A store with complete=False is rewritten.
    g = zarr.open_group(str(dest / s.store_name), mode='a', use_consolidated=False)
    g.attrs['complete'] = False
    stats = llc_v2.extract_surface(out_dir, str(dest), fields=['SST', 'SSS'],
                                   mask_dir=mask_dir, FS=FS)
    assert stats['written'] == 1 and stats['skipped'] == 4

    # Without a mask dir, exact zeros become NaN -- identical result here.
    dest2 = tmp_path / 'dest2'
    stats = llc_v2.extract_surface(out_dir, str(dest2), fields=['SST'], FS=FS, limit=2,
                                   write_grid=False)
    assert stats['written'] == 2
    assert sorted(p.name for p in dest2.iterdir()) == ['20230101T01.zarr', '20230101T02.zarr']
    g2 = zarr.open_group(str(dest2 / '20230101T01.zarr'), mode='r', use_consolidated=False)
    g1 = zarr.open_group(str(dest / '20230101T01.zarr'), mode='r', use_consolidated=False)
    np.testing.assert_array_equal(g2['Theta'][:], g1['Theta'][:])

    # A requested field that is missing on disk: store written, left incomplete.
    with caplog.at_level('WARNING', logger='wrangler.ogcm.llc_v2'):
        stats = llc_v2.extract_surface(out_dir, str(tmp_path / 'dest3'), fields=['SST', 'SSU'],
                                       FS=FS, limit=1, write_grid=False)
    assert stats == {'discovered': 1, 'written': 0, 'skipped': 0, 'incomplete': 1,
                     'dt_seconds': []}
    g3 = zarr.open_group(str(tmp_path / 'dest3' / '20230101T01.zarr'), mode='r',
                         use_consolidated=False)
    assert g3.attrs['complete'] is False and g3.attrs['variables'] == ['Theta']
    assert 'missing' in caplog.text

    # extract_sst is the SST-only wrapper.
    stats = llc_v2.extract_sst(out_dir, str(tmp_path / 'dest4'), FS=FS, limit=1, write_grid=False)
    assert stats['written'] == 1


def test_write_grid_store_from_grid_dir(tmp_path):
    zarr = pytest.importorskip('zarr')
    rng = np.random.default_rng(21)
    n = 13 * FS * FS
    grid_dir = tmp_path / 'grid'
    grid_dir.mkdir()
    depth = np.where(rng.random(n) < 0.7, rng.uniform(10, 5000, n), 0.0).astype(np.float32)
    xc = rng.uniform(-180, 180, n).astype(np.float32)
    _write_data(grid_dir / 'Depth.data', depth)
    _write_data(grid_dir / 'XC.data', xc)
    _write_data(grid_dir / 'RAC.data', np.full(n, 1e6, np.float32))
    # hFacC with 3 levels: level 0 open where depth > 0.
    hfac = np.concatenate([(depth > 0).astype(np.float32),
                           np.zeros(n, np.float32), np.zeros(n, np.float32)])
    _write_data(grid_dir / 'hFacC.data', hfac)
    _write_data(grid_dir / 'RC.data', -np.arange(0.5, 3.5))          # 3 levels
    _write_data(grid_dir / 'RF.data', -np.arange(0.0, 4.0))          # 4 interfaces
    _write_data(grid_dir / 'YC.data', np.zeros(n // 2, np.float32))  # wrong size -> skipped

    url = llc_v2.write_grid_store(str(tmp_path / 'dest'), grid_dir=str(grid_dir), FS=FS)
    g = zarr.open_group(url, mode='r', use_consolidated=False)
    assert sorted(g.attrs['variables']) == ['Depth', 'RC', 'RF', 'XC', 'hFacC_k0', 'maskC', 'rA']
    assert any(s.startswith('YC.data (size') for s in g.attrs['skipped'])
    assert any(s == 'XG.data (missing)' for s in g.attrs['skipped'])
    assert g.attrs['complete'] is True
    np.testing.assert_array_equal(g['Depth'][:].reshape(-1), depth)
    np.testing.assert_array_equal(g['XC'][:].reshape(-1), xc)
    assert g['rA'].attrs['source_file'] == 'RAC.data'
    assert g['hFacC_k0'].attrs['levels_in_file'] == 3
    np.testing.assert_array_equal(g['maskC'][:].reshape(-1), depth > 0)
    assert g['maskC'].attrs['source'] == 'hFacC.data level 0 > 0'
    assert g['RC'].shape == (3,) and tuple(g['RC'].metadata.dimension_names) == ('k',)
    assert g['RF'].shape == (4,) and tuple(g['RF'].metadata.dimension_names) == ('k_p1',)

    # Mask dir takes precedence for maskC; second call skips a complete store.
    with pytest.raises(ValueError):
        llc_v2.write_grid_store(str(tmp_path / 'dest5'), FS=FS)
    assert llc_v2.write_grid_store(str(tmp_path / 'dest'), grid_dir=str(grid_dir), FS=FS) == url


def test_s3_filesystem_settings(monkeypatch):
    pytest.importorskip('s3fs')
    monkeypatch.setenv('ENDPOINT_URL', 'https://example.invalid')
    monkeypatch.setenv('AWS_PROFILE', 'nautilus-test')
    fs = llc_v2.s3_filesystem(asynchronous=False)
    assert fs.client_kwargs['endpoint_url'] == 'https://example.invalid'
    assert fs.config_kwargs['s3']['addressing_style'] == 'path'
    assert fs.kwargs['profile'] == 'nautilus-test'   # forwarded to AioSession(profile=)
    # Explicit args win over the environment.
    fs2 = llc_v2.s3_filesystem(endpoint='https://s3-west.nrp-nautilus.io/',
                               profile='other', asynchronous=False)
    assert fs2.client_kwargs['endpoint_url'] == 'https://s3-west.nrp-nautilus.io'
    assert fs2.kwargs['profile'] == 'other'
    assert llc_v2.is_s3('s3://llc4320-v2/SURFACE') and not llc_v2.is_s3('/tmp/x')


def test_cli_parser_and_dry_run(tmp_path, capsys):
    from wrangler.scripts import llc_v2_surface, llc_v2_sst
    out_dir, mask_dir, _ = _make_out_tree(tmp_path)
    args = llc_v2_surface.parser([out_dir, str(tmp_path / 'd'), '--FS', str(FS), '--dry-run',
                                  '--start', '2023-01-01T02', '--end', '2023-01-01',
                                  '--fields', 'SST, SSS,Eta', '--mask-dir', mask_dir])
    assert args.start == datetime(2023, 1, 1, 2, tzinfo=timezone.utc)
    assert args.end == datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert args.fields == ['SST', 'SSS', 'Eta'] and args.mask_dir == mask_dir
    args = llc_v2_surface.parser([out_dir, str(tmp_path / 'd'), '--FS', str(FS),
                                  '--dry-run', '--start', '2023-01-01T03'])
    assert args.fields == ['SST']
    stats = llc_v2_surface.main(args)
    assert stats['discovered'] == 3 and stats['written'] == 0
    assert 'discovered=3' in capsys.readouterr().out
    # The SST alias exposes the same parser/main.
    assert llc_v2_sst.parser is llc_v2_surface.parser and llc_v2_sst.main is llc_v2_surface.main


NAMELIST = """# Model parameters
# Continuous equation parameters
 &PARM01
 tRef= 51*20.,
 viscAr=5.6614e-04,
 &
# Time stepping parameters
 &PARM03
 nIter0=0,
 nTimeSteps=125280,
 deltaT=5.,
 pChkptFreq=626400.,
 dumpFreq=3600.,
 &
"""


def _write_eta(path, FS, wet, rng):
    eta = np.zeros(13 * FS * FS, dtype=np.float32)
    eta[wet] = rng.normal(0, 0.5, size=int(wet.sum())).astype(np.float32)
    eta[wet & (eta == 0)] = 0.1
    with open(path, 'wb') as f:
        f.write(eta.astype('>f4').tobytes())
    return eta.reshape(13, FS, FS)


def test_read_data_field(tmp_path):
    rng = np.random.default_rng(11)
    wet = rng.random(13 * FS * FS) < 0.6
    path = tmp_path / 'Eta.0000000720.data'
    eta = _write_eta(path, FS, wet, rng)
    back = llc_v2.read_data_field(str(path), FS)
    assert back.shape == (13, FS, FS) and back.dtype == np.float32
    np.testing.assert_array_equal(back, eta)
    with pytest.raises(ValueError):
        llc_v2.read_data_field(str(path), FS + 8)      # wrong FS
    with pytest.raises(ValueError):
        llc_v2.read_data_field(str(path), FS, level=1)  # only one level


def test_read_data_namelist(tmp_path):
    assert llc_v2.read_data_namelist(str(tmp_path)) == {}   # no `data` file
    (tmp_path / 'data').write_text(NAMELIST)
    nml = llc_v2.read_data_namelist(str(tmp_path))
    assert nml['deltaT'] == 5.0
    assert nml['nIter0'] == 0.0
    assert nml['nTimeSteps'] == 125280.0
    assert nml['dumpFreq'] == 3600.0
    assert 'startTime' not in nml
    # Several assignments per line, D exponents, case-insensitive keys.
    (tmp_path / 'data').write_text(" &PARM03\n NITER0=1440, deltat=2.5D1, startTime=0.,\n &\n")
    nml = llc_v2.read_data_namelist(str(tmp_path))
    assert nml == {'deltaT': 25.0, 'nIter0': 1440.0, 'startTime': 0.0}


def test_read_stdout_and_run_params(tmp_path):
    assert llc_v2.read_stdout_params(str(tmp_path)) == {}
    assert llc_v2.read_run_params(str(tmp_path)) == {'source': None}
    (tmp_path / 'STDOUT.0000').write_text(
        "(PID.TID 0000.0001) // =======================================================\n"
        "(PID.TID 0000.0001) > nIter0=125280,\n"
        "(PID.TID 0000.0001) > deltaT=20.,\n"
        "(PID.TID 0000.0001) deltaT =  2.000000000000000E+01 /* Time step ( s ) */\n"
        "(PID.TID 0000.0001) nIter0 =        125280 /* Run starting timestep number */\n")
    so = llc_v2.read_stdout_params(str(tmp_path))
    assert so['deltaT'] == 20.0 and so['nIter0'] == 125280.0
    # data wins where present; STDOUT fills the rest.
    (tmp_path / 'data').write_text(" &PARM03\n deltaT=20.,\n &\n")
    prm = llc_v2.read_run_params(str(tmp_path))
    assert prm['deltaT'] == 20.0 and prm['nIter0'] == 125280.0
    assert prm['source'] == 'data+STDOUT'
    (tmp_path / 'data').unlink()
    assert llc_v2.read_run_params(str(tmp_path))['source'] == 'STDOUT'


def test_discover_warns_when_namelist_disagrees(tmp_path, caplog):
    out_dir, _, _ = _make_out_tree(tmp_path)
    folder = os.path.join(out_dir, SEG1[0])
    # Files sit at iterations 720, 1440, 2160 = +1, +2, +3 h at deltaT=5 s.
    # Claim deltaT=25 s: first file would be at +5 h -> disagreement warning.
    with open(os.path.join(folder, 'data'), 'w') as f:
        f.write(" &PARM03\n nIter0=0,\n deltaT=25.,\n &\n")
    with caplog.at_level('INFO', logger='wrangler.ogcm.llc_v2'):
        steps = llc_v2.discover_timesteps(out_dir, 'SST')
    assert [s.date.hour for s in steps] == [1, 2, 3, 4, 5]     # dates unchanged
    assert 'numbering convention: UNMATCHED' in caplog.text
    assert 'disagrees' in caplog.text
    # Relative-to-nIter0 numbering is also accepted: folder 2's files are at
    # 2520, 2880 with nIter0=2160, deltaT=10 (absolute).  Rewrite them as
    # relative (360, 720) and point STDOUT at the same parameters.
    caplog.clear()
    f2 = os.path.join(out_dir, SEG2[0])
    for old_it, new_it in ((2520, 360), (2880, 720)):
        for fn in os.listdir(f2):
            if f'.{old_it:010d}.' in fn:
                os.rename(os.path.join(f2, fn), os.path.join(f2, fn.replace(f'{old_it:010d}', f'{new_it:010d}')))
    with caplog.at_level('INFO', logger='wrangler.ogcm.llc_v2'):
        steps = llc_v2.discover_timesteps(out_dir, 'SST')
    assert [s.date.hour for s in steps] == [1, 2, 3, 4, 5]
    assert 'numbering convention: relative-to-nIter0' in caplog.text


def test_inspect_cli(tmp_path, capsys):
    pytest.importorskip('zarr')
    from wrangler.scripts import llc_v2_inspect
    out_dir, mask_dir, truth = _make_out_tree(tmp_path)
    dest = tmp_path / 'dest'
    llc_v2.extract_surface(out_dir, str(dest), fields=['SST'], mask_dir=mask_dir, FS=FS,
                           limit=1, write_grid=False)
    store = str(dest / '20230101T01.zarr')
    wet = truth['wet']

    # The real Eta file from the same folder/iteration: same land pattern -> OK.
    eta_ok = os.path.join(out_dir, SEG1[0], 'Eta.0000000720.data')
    res = llc_v2_inspect.main(llc_v2_inspect.parser([store, '--eta', eta_ok, '--FS', str(FS)]))
    out = capsys.readouterr().out
    assert res['n_wet'] == int(wet.sum()) and res['n_bad'] == 0
    assert res['eta_wet_but_field_nan'] == 0.0 and res['field_wet_but_eta_zero'] == 0.0
    assert 'value check: OK' in out and 'mask check: OK' in out
    assert 'selected_iteration: 720' in out and 'selected_date_utc: 2023-01-01 01:00:00' in out

    # Eta with a shuffled wet pattern -> MISMATCH.
    rng = np.random.default_rng(5)
    eta_bad = tmp_path / 'Eta.bad.data'
    _write_eta(eta_bad, FS, rng.permutation(wet), rng)
    res = llc_v2_inspect.main(llc_v2_inspect.parser([store, '--eta', str(eta_bad), '--FS', str(FS)]))
    assert res['eta_wet_but_field_nan'] > 0.1
    assert 'MISMATCH' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Inventory (step 2 on Pleiades found no Theta.*.shrunk files)
# ---------------------------------------------------------------------------

def _touch(path):
    open(path, 'wb').close()


def test_inventory_and_summary(tmp_path, capsys):
    out_dir = tmp_path / 'OUT'
    out_dir.mkdir()
    # Folder A: namelists + 2D .data only (what step 2 saw).
    a = out_dir / '2023_01_01_000000_to_2023_01_08_060000'
    a.mkdir()
    for fn in ('data', 'data.cal', 'data.exch2_36x36x113847', 'data.kpp'):
        _touch(a / fn)
    for h in range(1, 4):
        _touch(a / f'Eta.{720 * h:010d}.data')
        _touch(a / f'Eta.{720 * h:010d}.meta')
        _touch(a / f'oceQnet.{720 * h:010d}.data')
    (a / 'pickups').mkdir()
    # Folder B: also has shrunk 3D fields.
    b = out_dir / '2023_03_27_000000_to_2023_03_30_000000'
    b.mkdir()
    for h in range(1, 3):
        _touch(b / f'Eta.{720 * h:010d}.data')
        for fld in ('U', 'V', 'Theta', 'Salt'):
            _touch(b / f'{fld}.{720 * h:010d}.shrunk')
    (out_dir / 'matlab_v0').mkdir()               # ignored: not an output folder
    _touch(out_dir / 'Theta.0000000720.shrunk')   # ignored: not inside a folder

    invs = llc_v2.inventory(str(out_dir))
    assert [os.path.basename(i.folder) for i in invs] == [a.name, b.name]
    ia, ib = invs
    assert ia.n_files == 4 + 9 + 1
    assert ia.fields[('Eta', 'data')] == (3, 720, 2160)
    assert ia.fields[('Eta', 'meta')] == (3, 720, 2160)
    assert ia.fields_with_ext('shrunk') == []
    assert ia.fields_with_ext('data') == ['Eta', 'oceQnet']
    assert 'pickups/' in ia.other and 'data.cal' in ia.other
    assert ib.fields_with_ext('shrunk') == ['Salt', 'Theta', 'U', 'V']
    assert ib.has('Theta', 'shrunk') and not ia.has('Theta', 'shrunk')

    summ = llc_v2.summarize_inventory(invs, 'Theta', 'shrunk')
    assert summ['n_folders'] == 2 and summ['with_field'] == 1
    assert summ['first_with'] == datetime(2023, 3, 27, 1, tzinfo=timezone.utc)  # first file = start + 1 h
    assert summ['last_with'] == datetime(2023, 3, 30, tzinfo=timezone.utc)
    assert summ['without_field'] == [a.name]
    assert summ['combos'][('Eta', 'data')] == 2
    assert summ['combos'][('Theta', 'shrunk')] == 1

    assert llc_v2.inventory(str(out_dir), max_folders=1)[0].folder == str(a)

    # Bounded find: depth 1 sees only the stray file, depth 2 also folder B's.
    hits, total = llc_v2.find_files(str(out_dir), '.shrunk', max_depth=1)
    assert total == 1 and hits[0].endswith('Theta.0000000720.shrunk')
    hits, total = llc_v2.find_files(str(out_dir), '.shrunk', max_depth=2, limit=3)
    assert total == 9 and len(hits) == 3

    # CLI
    from wrangler.scripts import llc_v2_inventory
    summ2 = llc_v2_inventory.main(llc_v2_inventory.parser(
        [str(out_dir), '--field', 'Theta', '--ext', 'shrunk', '--find', str(out_dir), '--depth', '2']))
    out = capsys.readouterr().out
    assert summ2 == summ
    assert 'shrunk:[-]  data:[Eta,oceQnet]' in out
    assert 'shrunk:[Salt,Theta,U,V]' in out
    assert 'with Theta.*.shrunk: 1   (2023-03-27 01:00 to 2023-03-30 00:00)' in out
    assert f'folders lacking it: {a.name}' in out
    assert '*.shrunk files under' in out and ': 9' in out
