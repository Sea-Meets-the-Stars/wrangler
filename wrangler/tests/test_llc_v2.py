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

STRIDE = 144  # 25 s model timestep -> 144 iterations per hour (v1 value)


def _make_out_tree(tmp_path, FS=FS, seed=0):
    """Synthetic /nobackupp27/.../OUT: two adjacent folders of hourly Theta files.

    Folder 1 covers 00h..03h (3 files, end-exclusive), folder 2 covers
    03h..05h (2 files).  Also drops in distractor files (U, a 2D .data
    field) and a non-output folder that discovery must ignore.

    Returns:
        (out_dir, mask_dir, {iteration: (mask_bits, values)})
    """
    rng = np.random.default_rng(seed)
    n = 13 * FS * FS
    mask_bits = rng.random(n) < 0.7
    out_dir = tmp_path / 'OUT'
    mask_dir = tmp_path / 'mask'
    out_dir.mkdir()
    mask_dir.mkdir()
    with open(mask_dir / 'hFacC.bits', 'wb') as f:
        f.write(np.packbits(mask_bits.astype(np.uint8), bitorder='little').tobytes())

    truth = {}
    folders = {
        '2023_01_01_000000_to_2023_01_01_030000': [0, 1, 2],
        '2023_01_01_030000_to_2023_01_01_050000': [3, 4],
    }
    for name, hours in folders.items():
        d = out_dir / name
        d.mkdir()
        for h in hours:
            it = 1000 + h * STRIDE
            vals = rng.uniform(-2, 32, size=int(mask_bits.sum())).astype(np.float32)
            with open(d / f'Theta.{it:010d}.shrunk', 'wb') as f:
                f.write(vals.astype('>f4').tobytes())
            (d / f'U.{it:010d}.shrunk').write_bytes(b'')       # other field
            (d / f'Eta.{it:010d}.data').write_bytes(b'')       # 2D field, .data
            truth[it] = (mask_bits, vals)
    (out_dir / 'not_an_output_folder').mkdir()
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


def test_discover_timesteps_dates_and_ordering(tmp_path):
    out_dir, _, truth = _make_out_tree(tmp_path)
    steps = llc_v2.discover_timesteps(out_dir, 'Theta')

    assert len(steps) == 5
    assert [s.iteration for s in steps] == sorted(truth)
    expected = [datetime(2023, 1, 1, h, tzinfo=timezone.utc) for h in range(5)]
    assert [s.date for s in steps] == expected
    assert [s.n_in_folder for s in steps] == [0, 1, 2, 0, 1]
    assert steps[3].folder.endswith('2023_01_01_030000_to_2023_01_01_050000')
    assert all(os.path.isfile(s.path) for s in steps)
    assert steps[0].store_name == '20230101T00.zarr'
    assert steps[4].store_name == '20230101T04.zarr'

    # Model timestep inferred from the within-folder iteration stride.
    assert llc_v2.infer_timestep_seconds(steps) == pytest.approx(3600 / STRIDE)

    # Other fields discover independently.
    assert len(llc_v2.discover_timesteps(out_dir, 'U')) == 5
    assert llc_v2.discover_timesteps(out_dir, 'Salt') == []


def test_discover_timesteps_date_window(tmp_path):
    out_dir, _, _ = _make_out_tree(tmp_path)
    start = datetime(2023, 1, 1, 1)             # naive -> treated as UTC
    end = datetime(2023, 1, 1, 4, tzinfo=timezone.utc)
    steps = llc_v2.discover_timesteps(out_dir, 'Theta', start=start, end=end)
    assert [s.date.hour for s in steps] == [1, 2, 3]


def test_discover_timesteps_warns_on_gap(tmp_path, caplog):
    out_dir, _, _ = _make_out_tree(tmp_path)
    # Add a folder claiming 4 hours (00..04) but holding hours 0, 2, 3 only:
    # non-constant iteration stride + span mismatch -> both warnings.
    gap = os.path.join(out_dir, '2023_01_02_000000_to_2023_01_02_040000')
    os.mkdir(gap)
    for h in (0, 2, 3):
        open(os.path.join(gap, f'Theta.{5000 + h * STRIDE:010d}.shrunk'), 'wb').close()
    with caplog.at_level('WARNING', logger='wrangler.ogcm.llc_v2'):
        steps = llc_v2.discover_timesteps(out_dir, 'Theta')
    assert len(steps) == 8
    assert 'stride is not constant' in caplog.text
    assert 'spans 4.00 h but holds 3 files' in caplog.text
    # The good folders alone still date correctly; the gap folder's dates
    # after the hole are (knowingly) position-based.
    gap_steps = [s for s in steps if s.folder == gap]
    assert [s.date.hour for s in gap_steps] == [0, 1, 2]
    assert llc_v2.infer_timestep_seconds(steps) is None


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

def test_extract_sst_end_to_end_local(tmp_path, caplog):
    zarr = pytest.importorskip('zarr')
    out_dir, mask_dir, truth = _make_out_tree(tmp_path)
    dest = tmp_path / 'dest'

    # Dry run writes nothing.
    stats = llc_v2.extract_sst(out_dir, mask_dir, str(dest), FS=FS, dry_run=True)
    assert stats['discovered'] == 5 and stats['written'] == 0
    assert not dest.exists()

    stats = llc_v2.extract_sst(out_dir, mask_dir, str(dest), FS=FS)
    assert stats == {'discovered': 5, 'written': 5, 'skipped': 0,
                     'dt_seconds': pytest.approx(25.0)}

    names = sorted(p.name for p in dest.iterdir())
    assert names == ['20230101T00.zarr', '20230101T01.zarr', '20230101T02.zarr',
                     '20230101T03.zarr', '20230101T04.zarr', 'grid.zarr']

    # grid.zarr: wet mask from hFacC.bits
    g = zarr.open_group(str(dest / 'grid.zarr'), mode='r', use_consolidated=False)
    mask_bits = next(iter(truth.values()))[0]
    np.testing.assert_array_equal(g['maskC'][:].reshape(-1), mask_bits)
    assert g.attrs['complete'] is True

    # One timestep store: dims/chunks/attrs/values, NaN over land.
    steps = llc_v2.discover_timesteps(out_dir, 'Theta')
    s = steps[3]
    g = zarr.open_group(str(dest / s.store_name), mode='r', use_consolidated=False)
    z = g['Theta']
    assert z.shape == (13, FS, FS)
    assert z.chunks == (1, FS, FS)            # (1, 720, 720) clipped to FS=8
    assert tuple(z.metadata.dimension_names) == ('face', 'j', 'i')
    assert g.attrs['selected_iteration'] == s.iteration
    assert g.attrs['selected_date_utc'] == '2023-01-01 03:00:00'
    assert g.attrs['source_folder'] == '2023_01_01_030000_to_2023_01_01_050000'
    assert g.attrs['complete'] is True
    assert z.attrs['units'] == 'degC'
    flat = z[:].reshape(-1)
    mask_bits, vals = truth[s.iteration]
    np.testing.assert_allclose(flat[mask_bits], vals, rtol=1e-6)
    assert np.all(np.isnan(flat[~mask_bits]))
    np.testing.assert_array_equal(g['face'][:], np.arange(13))
    np.testing.assert_array_equal(g['j'][:], np.arange(FS))

    # Idempotent: a second run skips everything.
    stats = llc_v2.extract_sst(out_dir, mask_dir, str(dest), FS=FS)
    assert stats['written'] == 0 and stats['skipped'] == 5

    # An incomplete store gets rewritten.
    g = zarr.open_group(str(dest / s.store_name), mode='a', use_consolidated=False)
    g.attrs['complete'] = False
    stats = llc_v2.extract_sst(out_dir, mask_dir, str(dest), FS=FS)
    assert stats['written'] == 1 and stats['skipped'] == 4

    # --limit
    stats = llc_v2.extract_sst(out_dir, mask_dir, str(tmp_path / 'dest2'), FS=FS,
                               limit=2, write_grid=False)
    assert stats['written'] == 2
    assert sorted(p.name for p in (tmp_path / 'dest2').iterdir()) == \
        ['20230101T00.zarr', '20230101T01.zarr']


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
    from wrangler.scripts import llc_v2_sst
    out_dir, mask_dir, _ = _make_out_tree(tmp_path)
    args = llc_v2_sst.parser([out_dir, mask_dir, str(tmp_path / 'd'), '--FS', str(FS),
                              '--dry-run', '--start', '2023-01-01T02', '--end', '2023-01-01'])
    assert args.start == datetime(2023, 1, 1, 2, tzinfo=timezone.utc)
    assert args.end == datetime(2023, 1, 1, tzinfo=timezone.utc)
    args = llc_v2_sst.parser([out_dir, mask_dir, str(tmp_path / 'd'), '--FS', str(FS),
                              '--dry-run', '--start', '2023-01-01T02'])
    stats = llc_v2_sst.main(args)
    assert stats['discovered'] == 3 and stats['written'] == 0
    assert 'discovered=3' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Plain .data fields, `data` namelist, and the inspection CLI
# ---------------------------------------------------------------------------

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


def test_discover_warns_when_namelist_disagrees(tmp_path, caplog):
    out_dir, _, _ = _make_out_tree(tmp_path)
    folder = os.path.join(out_dir, '2023_01_01_000000_to_2023_01_01_030000')
    # Files sit at iterations 1000, 1144, 1288. With deltaT=25 s and nIter0=0
    # the first is at +6.94 h, not +0 h -> disagreement warning.
    with open(os.path.join(folder, 'data'), 'w') as f:
        f.write(" &PARM03\n nIter0=0,\n deltaT=25.,\n &\n")
    with caplog.at_level('INFO', logger='wrangler.ogcm.llc_v2'):
        steps = llc_v2.discover_timesteps(out_dir, 'Theta')
    assert len(steps) == 5                       # dates unchanged
    assert 'namelist deltaT=25 s' in caplog.text
    assert 'disagrees' in caplog.text
    # Consistent namelist (nIter0 = first iteration, hourly stride) -> no warning.
    caplog.clear()
    with open(os.path.join(folder, 'data'), 'w') as f:
        f.write(f" &PARM03\n nIter0=1000,\n deltaT={3600 / STRIDE},\n &\n")
    with caplog.at_level('INFO', logger='wrangler.ogcm.llc_v2'):
        llc_v2.discover_timesteps(out_dir, 'Theta')
    assert 'disagrees' not in caplog.text
    assert 'first file at +0.000 h' in caplog.text


def test_inspect_cli(tmp_path, capsys):
    pytest.importorskip('zarr')
    from wrangler.scripts import llc_v2_inspect
    out_dir, mask_dir, truth = _make_out_tree(tmp_path)
    dest = tmp_path / 'dest'
    llc_v2.extract_sst(out_dir, mask_dir, str(dest), FS=FS, limit=1, write_grid=False)
    store = str(dest / '20230101T00.zarr')
    mask_bits, vals = truth[1000]

    # Matching Eta: same wet pattern -> mask check OK.
    rng = np.random.default_rng(5)
    eta_ok = tmp_path / 'Eta.0000001000.data'
    _write_eta(eta_ok, FS, mask_bits, rng)
    res = llc_v2_inspect.main(llc_v2_inspect.parser(
        [store, '--eta', str(eta_ok), '--FS', str(FS)]))
    out = capsys.readouterr().out
    assert res['n_wet'] == int(mask_bits.sum()) and res['n_bad'] == 0
    assert res['eta_wet_but_field_nan'] == 0.0 and res['field_wet_but_eta_zero'] == 0.0
    assert 'value check: OK' in out and 'mask check: OK' in out
    assert 'selected_iteration: 1000' in out

    # Eta with a shuffled wet pattern -> MISMATCH.
    eta_bad = tmp_path / 'Eta.bad.data'
    _write_eta(eta_bad, FS, rng.permutation(mask_bits), rng)
    res = llc_v2_inspect.main(llc_v2_inspect.parser(
        [store, '--eta', str(eta_bad), '--FS', str(FS)]))
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
    assert ib.has('Theta') and not ia.has('Theta')

    summ = llc_v2.summarize_inventory(invs, 'Theta')
    assert summ['n_folders'] == 2 and summ['with_field'] == 1
    assert summ['first_with'] == datetime(2023, 3, 27, tzinfo=timezone.utc)
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
        [str(out_dir), '--find', str(out_dir), '--depth', '2']))
    out = capsys.readouterr().out
    assert summ2 == summ
    assert 'shrunk:[-]  data:[Eta,oceQnet]' in out
    assert 'shrunk:[Salt,Theta,U,V]' in out
    assert 'with Theta.*.shrunk: 1   (2023-03-27 00:00 to 2023-03-30 00:00)' in out
    assert f'folders lacking it: {a.name}' in out
    assert '*.shrunk files under' in out and ': 9' in out
