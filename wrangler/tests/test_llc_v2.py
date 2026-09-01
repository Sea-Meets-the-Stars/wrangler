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
