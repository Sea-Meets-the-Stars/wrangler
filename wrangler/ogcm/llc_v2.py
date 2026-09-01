""" Python reader for compressed ("shrunk") LLC4320 v2 raw output.

This is a pure-Python/NumPy port of the bit-mask decompression scheme
implemented in the MATLAB/MEX toolbox Dimitris shared
(``$OS_OGCM/LLC_v2/extract/matlab_v0/llc_shrunk_mex.c``), so the LLC4320 v2
raw output on NASA Pleiades can be read without MATLAB or a compiled
MEX/C++ binary.  The decompression algorithm here is a line-for-line port
of that C source (see ``uncompress()`` and ``compute_offsets()`` there);
see ``matlab_v0/README.md`` for the authoritative format description.

File layout
-----------
Data:  ``<dataDir>/<fieldName>.<10-digit zero-padded timestep>.shrunk``
    A stream of big-endian float32 values, one per *wet* grid point, for
    every vertical level in turn.  Dry (masked-out) points are simply
    omitted -- that's the "shrunk" compression.  ``timestep`` is the
    MITgcm iteration number.

Mask:  ``<maskDir>/hFacC.bits`` / ``hFacS.bits`` / ``hFacW.bits``
    One bit per (level, facet, row, col) grid point, ``1`` = wet.  Bits
    are packed LSB-first within each byte.  NZ (vertical levels) is
    inferred from the mask file size.  Field -> mask mapping: ``U`` uses
    ``hFacW.bits``, ``V`` uses ``hFacS.bits``, everything else (including
    scalars like ``Theta``) uses ``hFacC.bits``.

Grid
----
13 native LLC facets of FS x FS each (FS=4320 for full LLC4320
resolution), flattened per level as 13*FS*FS points in facet-major,
row-major order.  This module returns that native ``(13, FS, FS)`` layout
directly -- no lat/lon "map" reprojection (that's what the MATLAB
toolbox's ``twodify``/'map' mode does) -- matching the ``face``/``j``/``i``
convention already used for LLC4320 elsewhere in this codebase.
"""

import os

import numpy as np

N_FACETS = 13

# Fields carried as single-level (2D) diagnostics -- no real depth axis.
# From llc_shrunk_read_field.m's "2D FIELDS" list.
FIELDS_2D = {
    'Eta', 'KPPhbl', 'PhiBot', 'SIarea', 'SIheff', 'SIhsalt',
    'SIhsnow', 'SIuice', 'SIvice', 'oceFWflx', 'oceQnet',
    'oceQsw', 'oceSflux', 'oceTAUX', 'oceTAUY',
}


def mask_file_for_field(field_name: str) -> str:
    """Return the mask filename (hFacC/S/W.bits) for *field_name*.

    Mirrors ``llc_shrunk_read_field.m``'s FIELD -> MASK MAPPING: ``U``
    lives on hFacW points, ``V`` on hFacS points, everything else
    (including cell-center scalars like Theta/SST) on hFacC points.

    Args:
        field_name (str): e.g. 'Theta', 'Salt', 'U', 'V', 'Eta'.

    Returns:
        str: mask file basename, e.g. 'hFacC.bits'.
    """
    if field_name == 'U':
        return 'hFacW.bits'
    elif field_name == 'V':
        return 'hFacS.bits'
    return 'hFacC.bits'


def mask_bytes_per_level(FS: int) -> int:
    """Number of mask bytes per vertical level for facet side *FS*.

    Args:
        FS (int): facet side length (must be divisible by 8).

    Returns:
        int: 13*FS*FS/8
    """
    if FS % 8 != 0:
        raise ValueError(f"FS={FS} must be divisible by 8")
    return N_FACETS * FS * FS // 8


def detect_nz(mask_file: str, FS: int) -> int:
    """Auto-detect the number of vertical levels from a mask file's size.

    Mirrors ``llc_shrunk_mex('nz_detect', ...)``.

    Args:
        mask_file (str): path to hFacC.bits / hFacS.bits / hFacW.bits.
        FS (int): facet side length.

    Returns:
        int: number of vertical levels (NZ).
    """
    bytes_per_level = mask_bytes_per_level(FS)
    nbytes = os.path.getsize(mask_file)
    if nbytes % bytes_per_level != 0:
        raise ValueError(
            f"{mask_file}: size {nbytes} is not a multiple of "
            f"{bytes_per_level} bytes/level (FS={FS}); "
            "wrong FS, or a corrupt/truncated mask file.")
    return nbytes // bytes_per_level


def read_mask_level(mask_file: str, FS: int, level: int = 0) -> np.ndarray:
    """Read and unpack one level's wet-point mask.

    Args:
        mask_file (str): path to hFacC.bits / hFacS.bits / hFacW.bits.
        FS (int): facet side length.
        level (int, optional): 0-based vertical level. Defaults to 0
            (surface).

    Returns:
        np.ndarray: boolean array, length 13*FS*FS -- True where the grid
            point is wet (i.e. has a value in the .shrunk data file) --
            in the same facet-major, row-major flat order the .shrunk
            data is packed in.  Bit order is LSB-first within each byte,
            matching the C reference (``mask[i>>3] & (1 << (i&7))``).
    """
    bytes_per_level = mask_bytes_per_level(FS)
    offset = level * bytes_per_level
    with open(mask_file, 'rb') as f:
        f.seek(offset)
        raw = f.read(bytes_per_level)
    if len(raw) != bytes_per_level:
        raise IOError(
            f"{mask_file}: short read for level {level} "
            f"(got {len(raw)}, expected {bytes_per_level} bytes)")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')
    return bits.astype(bool)


def level_byte_offset(mask_file: str, FS: int, level: int, nz: int = None) -> int:
    """Byte offset of *level* within the matching .shrunk data file.

    Each level's compressed length is ``popcount(level mask) * 4`` bytes
    (4 bytes/float32); the offset of level L is the cumulative sum of
    the compressed lengths of levels 0..L-1.  Level 0 is always at
    offset 0, so surface-only callers (e.g. SST) can skip this entirely.

    Note this only computes *level* offsets, not the finer-grained
    per-row offsets ``llc_shrunk_precompute.m`` also builds -- those are
    only needed for reading an x/y sub-region of a level, which this
    module doesn't yet support (we always read the full global field).

    Args:
        mask_file (str): path to hFacC.bits / hFacS.bits / hFacW.bits.
        FS (int): facet side length.
        level (int): 0-based vertical level whose offset is wanted.
        nz (int, optional): vertical level count, to skip auto-detection
            if already known.

    Returns:
        int: byte offset of *level*'s data within the .shrunk file.
    """
    if level == 0:
        return 0
    if nz is None:
        nz = detect_nz(mask_file, FS)
    if not (0 <= level < nz):
        raise ValueError(f"level={level} out of range for NZ={nz}")
    offset = 0
    for lev in range(level):
        offset += int(read_mask_level(mask_file, FS, lev).sum()) * 4
    return offset


def decompress_level(shrunk_file: str, mask: np.ndarray, byte_offset: int = 0) -> np.ndarray:
    """Decompress one level of a .shrunk file into the native (13, FS, FS) grid.

    Port of the C reference's ``uncompress()``: walk the mask bit by bit
    and pull the next big-endian float32 from the compressed stream
    wherever the bit is set, else leave 0.0.

    Args:
        shrunk_file (str): path to ``<field>.<timestep>.shrunk``.
        mask (np.ndarray): boolean wet-point mask for this level, from
            `read_mask_level` (length 13*FS*FS for some FS).
        byte_offset (int, optional): byte offset of this level's
            compressed data within *shrunk_file* (0 for the surface
            level; see `level_byte_offset` otherwise). Defaults to 0.

    Returns:
        np.ndarray: float32 array, shape (13, FS, FS). Dry points are 0.0.
    """
    FS = int(round((mask.size // N_FACETS) ** 0.5))
    if N_FACETS * FS * FS != mask.size:
        raise ValueError(f"mask length {mask.size} is not 13*FS*FS for integer FS")

    npts = int(mask.sum())
    nbytes = npts * 4
    with open(shrunk_file, 'rb') as f:
        f.seek(byte_offset)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise IOError(
            f"{shrunk_file}: short read at offset {byte_offset} "
            f"(got {len(raw)}, expected {nbytes} bytes -- "
            "wrong mask/level/FS, or a truncated file)")

    # Values are packed big-endian float32 (standard MITgcm/Fortran order).
    values = np.frombuffer(raw, dtype='>f4').astype(np.float32)

    flat = np.zeros(mask.size, dtype=np.float32)
    flat[mask] = values
    return flat.reshape(N_FACETS, FS, FS)


def read_shrunk_field(data_dir: str, mask_dir: str, field_name: str,
                      timestep, FS: int, level: int = 0,
                      nz: int = None) -> np.ndarray:
    """High-level reader: resolve paths, pick the mask, decompress one level.

    Mirrors ``llc_shrunk_read_field.m``.

    Args:
        data_dir (str): directory containing
            ``<field_name>.<timestep>.shrunk``.
        mask_dir (str): directory containing hFacC.bits / hFacS.bits /
            hFacW.bits.
        field_name (str): e.g. 'Theta', 'Salt', 'U', 'V', 'Eta'.
        timestep (int or str): MITgcm iteration number; zero-padded to
            10 digits in the filename.
        FS (int): facet side length (4320 for full-resolution LLC4320).
        level (int, optional): 0-based vertical level (0 = surface).
            Forced to 0 for `FIELDS_2D` members. Defaults to 0.
        nz (int, optional): vertical level count, to skip auto-detection
            if already known.

    Returns:
        np.ndarray: float32 array, shape (13, FS, FS).
    """
    if field_name in FIELDS_2D:
        level = 0

    ts_str = f"{int(timestep):010d}"
    shrunk_file = os.path.join(data_dir, f"{field_name}.{ts_str}.shrunk")
    mask_file = os.path.join(mask_dir, mask_file_for_field(field_name))

    if not os.path.isfile(shrunk_file):
        raise IOError(f"Not found: {shrunk_file}")
    if not os.path.isfile(mask_file):
        raise IOError(f"Not found: {mask_file}")

    mask = read_mask_level(mask_file, FS, level)
    byte_offset = level_byte_offset(mask_file, FS, level, nz=nz)
    return decompress_level(shrunk_file, mask, byte_offset)


def read_sst(data_dir: str, mask_dir: str, timestep, FS: int = 4320) -> np.ndarray:
    """Convenience wrapper: read global SST (Theta, k=0) for one timestep.

    Args:
        data_dir (str): directory containing the ``Theta.<timestep>.shrunk``
            file for this timestep.
        mask_dir (str): directory containing hFacC.bits.
        timestep (int or str): MITgcm iteration number.
        FS (int, optional): facet side length. Defaults to 4320
            (full-resolution LLC4320).

    Returns:
        np.ndarray: float32 array, shape (13, FS, FS), degrees C.
    """
    return read_shrunk_field(data_dir, mask_dir, 'Theta', timestep, FS, level=0)
