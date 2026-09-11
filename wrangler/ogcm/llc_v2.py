""" Readers and extraction pipeline for LLC4320 v2 raw output on NASA Pleiades.

Two kinds of raw files matter (facts from Dan Whitt, 2026-09-09; see
``claude_prompts/llc4320_v2.md``):

* **Uncompressed surface / 2D fields** -- ``SST/SSS/SSU/SSV.<iter>.data``
  (surface Theta/Salt/U/V) and the native 2D diagnostics (``Eta``, ...) are
  plain MITgcm "compact" binaries, 4320 x 56160 big-endian real*4, i.e. the
  native ``(13, 4320, 4320)`` facet layout.  `read_data_field` reads them and
  `extract_surface` turns them into dbof-style Zarr stores.  This is the
  path used for the surface-field product.
* **Compressed 3D fields** -- ``<field>.<iter>.shrunk`` + ``hFac*.bits``
  masks, decoded by the pure-Python port below (kept for the later
  full-depth phase).

Compressed ("shrunk") format
============================

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
import re
import glob
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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


def decompress_level(shrunk_file: str, mask: np.ndarray, byte_offset: int = 0,
                     dry_value: float = 0.0) -> np.ndarray:
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
        dry_value (float, optional): value assigned to dry (masked-out)
            points. Defaults to 0.0, matching the C reference; the Zarr
            pipeline passes ``np.nan`` to match the dbof convention.

    Returns:
        np.ndarray: float32 array, shape (13, FS, FS). Dry points are
            *dry_value*.
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

    flat = np.full(mask.size, dry_value, dtype=np.float32)
    flat[mask] = values
    return flat.reshape(N_FACETS, FS, FS)


def read_shrunk_field(data_dir: str, mask_dir: str, field_name: str,
                      timestep, FS: int, level: int = 0,
                      nz: int = None, dry_value: float = 0.0) -> np.ndarray:
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
        dry_value (float, optional): value for dry points. Defaults to 0.0.

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
    return decompress_level(shrunk_file, mask, byte_offset, dry_value=dry_value)


def read_sst(data_dir: str, mask_dir: str, timestep, FS: int = 4320,
             dry_value: float = 0.0) -> np.ndarray:
    """Convenience wrapper: read global SST (Theta, k=0) for one timestep.

    Args:
        data_dir (str): directory containing the ``Theta.<timestep>.shrunk``
            file for this timestep.
        mask_dir (str): directory containing hFacC.bits.
        timestep (int or str): MITgcm iteration number.
        FS (int, optional): facet side length. Defaults to 4320
            (full-resolution LLC4320).
        dry_value (float, optional): value for dry (land) points.
            Defaults to 0.0.

    Returns:
        np.ndarray: float32 array, shape (13, FS, FS), degrees C.
    """
    return read_shrunk_field(data_dir, mask_dir, 'Theta', timestep, FS,
                             level=0, dry_value=dry_value)


# ---------------------------------------------------------------------------
# Plain (uncompressed) MITgcm .data fields and the run's `data` namelist
# ---------------------------------------------------------------------------

def read_data_field(data_file: str, FS: int, level: int = 0,
                    dtype: str = '>f4') -> np.ndarray:
    """Read one level of a plain MITgcm binary ``.data`` file (2D fields).

    The native 2D diagnostics (``Eta``, ``KPPhbl``, ``oceQnet``, ... --
    see `FIELDS_2D`) are *not* shrunk: each ``<field>.<iteration>.data``
    is the full ``13*FS*FS`` grid as big-endian float32 (MITgcm's default
    ``real*4`` output, what ``read_llc_fkij`` reads in ``ExtractFields.m``),
    with 0.0 over land.  Multi-level files are just levels concatenated.

    Args:
        data_file (str): path to ``<field>.<iteration>.data``.
        FS (int): facet side length.
        level (int, optional): 0-based level to read. Defaults to 0.
        dtype (str, optional): on-disk dtype. Defaults to big-endian float32.

    Returns:
        np.ndarray: float32 array, shape (13, FS, FS).
    """
    npts = N_FACETS * FS * FS
    itemsize = np.dtype(dtype).itemsize
    nbytes_level = npts * itemsize
    fsize = os.path.getsize(data_file)
    if fsize % nbytes_level != 0:
        raise ValueError(
            f"{data_file}: size {fsize} is not a multiple of one level "
            f"({nbytes_level} bytes for FS={FS}); wrong FS or dtype?")
    nlev = fsize // nbytes_level
    if not (0 <= level < nlev):
        raise ValueError(f"level={level} out of range for {nlev} level(s) in {data_file}")
    with open(data_file, 'rb') as f:
        f.seek(level * nbytes_level)
        raw = f.read(nbytes_level)
    return np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(N_FACETS, FS, FS)


NAMELIST_KEYS = ('deltaT', 'nIter0', 'startTime', 'nTimeSteps', 'endTime',
                 'dumpFreq', 'taveFreq')


def _parse_params(text: str, keys) -> dict:
    """Regex-parse ``key=value`` numeric assignments out of *text* (see `read_data_namelist`)."""
    out = {}
    for key in keys:
        m = re.search(rf'(?i)(?<![A-Za-z0-9_]){key}\s*=\s*([-+0-9.eEdD]+)', text)
        if m is None:
            continue
        val = m.group(1).replace('D', 'E').replace('d', 'e').rstrip('.')
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def read_data_namelist(folder: str, keys=NAMELIST_KEYS) -> dict:
    """Pull a few numeric parameters out of a folder's MITgcm ``data`` namelist.

    Each raw-output folder on Pleiades holds the namelists of the run
    segment that produced it (``data``, ``data.cal``, ``data.diagnostics``,
    ...).  ``deltaT`` (model timestep, s) and ``nIter0`` (first iteration
    of the segment) let a file be dated from its iteration number
    independently of the position rule `discover_timesteps` uses, so the
    two can be cross-checked.  Tolerant, regex-based parsing (Fortran
    namelist: ``key=value,`` pairs, several per line allowed, ``#`` comment
    lines); anything unparseable is simply absent from the result.

    Args:
        folder (str): raw-output folder (must contain a file named ``data``).
        keys (tuple, optional): parameter names to look for (case-insensitive).

    Returns:
        dict: ``{key: float}`` for every key found; ``{}`` if there is no
            ``data`` file.
    """
    path = os.path.join(folder, 'data')
    try:
        if not os.path.isfile(path):
            # Some folders on Pleiades have a *directory* named data/ instead.
            return {}
        with open(path, errors='replace') as f:
            lines = [ln for ln in f if not ln.lstrip().startswith('#')]
    except OSError as e:
        logger.warning("%s: cannot read (%s); no namelist parameters", path, e.strerror)
        return {}
    return _parse_params('\n'.join(lines), keys)


def read_stdout_params(folder: str, keys=NAMELIST_KEYS, max_bytes: int = 4_000_000) -> dict:
    """Parse the same parameters from the folder's ``STDOUT.0000`` (or first ``STDOUT.*``).

    Dan Whitt: "deltaT ranges from 5 to 20 seconds depending on the
    folder/run ... you can find it in STDOUT.0* in each folder", and
    ``nIter0`` varies too.  MITgcm echoes its namelists near the top of
    STDOUT (``(PID.TID 0000.0001) > deltaT=5.,``) and again as
    ``deltaT = 5.000000000000000E+00 /* ... */``; both forms parse.  Only
    the first *max_bytes* are read -- the echo is at the top and STDOUT can
    be huge.

    Returns:
        dict: ``{key: float}``; ``{}`` if no STDOUT file exists.
    """
    try:
        cands = sorted(fn for fn in os.listdir(folder) if fn.startswith('STDOUT.'))
        if not cands:
            return {}
        path = os.path.join(folder, 'STDOUT.0000' if 'STDOUT.0000' in cands else cands[0])
        with open(path, errors='replace') as f:
            text = f.read(max_bytes)
    except OSError as e:
        logger.warning("%s: cannot read STDOUT (%s); no run parameters", folder, e.strerror)
        return {}
    return _parse_params(text, keys)


def read_run_params(folder: str, keys=NAMELIST_KEYS) -> dict:
    """``deltaT``/``nIter0``/... for a folder: from ``data``, filled in from ``STDOUT``.

    Returns:
        dict: merged parameters (``data`` wins where both have a key), plus
            ``'source'``: 'data', 'STDOUT', 'data+STDOUT' or None.
    """
    nml = read_data_namelist(folder, keys)
    missing = [k for k in keys if k not in nml]
    out = dict(nml)
    src = 'data' if nml else None
    if missing:
        so = read_stdout_params(folder, missing)
        if so:
            out.update(so)
            src = 'data+STDOUT' if nml else 'STDOUT'
    out['source'] = src
    return out


# ---------------------------------------------------------------------------
# Discovery: raw-output folders -> (iteration, date) records
# ---------------------------------------------------------------------------
#
# Raw output on Pleiades lives under one parent (e.g.
# /nobackupp27/dbwhitt/llc_4320/OUT) in folders named by the time range they
# cover, e.g. ``2023_01_01_000000_to_2023_01_08_060000``, each holding one
# ``<field>.<10-digit iteration>.<ext>`` file per output hour.  The
# filenames carry the MITgcm iteration number, not a date, and deltaT
# (5-20 s) and nIter0 differ from folder to folder, so iteration numbers
# alone cannot date a file.  The rule, per Dan Whitt (run owner, 2026-09-09):
#
#     hourly snapshots; the first file of a folder is ONE HOUR AFTER the
#     folder's start time; the last file is AT the folder's end time; no
#     hour appears in two folders.  =>  date(n) = start + (n + 1) hours,
#     n = 0-based position in the sorted listing, and nfiles == span hours.
#
# (This is one hour later than the ``start + n hours`` loop in
# ``ExtractFields.m``, which is what earlier versions of this module used.)
# ``discover_timesteps`` implements that rule, checks the file count against
# the folder span, and cross-checks against iteration*deltaT using the
# folder's ``data``/``STDOUT`` parameters, logging warnings on any
# inconsistency rather than silently mis-dating files.

FOLDER_RE = re.compile(
    r'^(?P<y0>\d{4})_(?P<m0>\d{2})_(?P<d0>\d{2})_(?P<t0>\d{6})'
    r'_to_'
    r'(?P<y1>\d{4})_(?P<m1>\d{2})_(?P<d1>\d{2})_(?P<t1>\d{6})$')
SHRUNK_RE = re.compile(r'^(?P<field>[A-Za-z0-9]+)\.(?P<iteration>\d{10})\.shrunk$')

OUTPUT_CADENCE = timedelta(hours=1)

logger = logging.getLogger(__name__)


def parse_folder_name(name: str):
    """Parse a raw-output folder name into its (start, end) datetimes.

    Args:
        name (str): folder basename (or full path), e.g.
            ``2023_01_01_000000_to_2023_01_08_060000``.

    Returns:
        tuple: (start, end) as timezone-aware UTC ``datetime`` objects.

    Raises:
        ValueError: if *name* does not follow the
            ``YYYY_MM_DD_HHMMSS_to_YYYY_MM_DD_HHMMSS`` convention.
    """
    base = os.path.basename(os.path.normpath(name))
    m = FOLDER_RE.match(base)
    if m is None:
        raise ValueError(f"Not a raw-output folder name: {base!r}")

    def _dt(y, mo, d, hms):
        return datetime(int(y), int(mo), int(d), int(hms[0:2]), int(hms[2:4]),
                        int(hms[4:6]), tzinfo=timezone.utc)

    start = _dt(m['y0'], m['m0'], m['d0'], m['t0'])
    end = _dt(m['y1'], m['m1'], m['d1'], m['t1'])
    return start, end


@dataclass(frozen=True)
class Timestep:
    """One available output step of one field.

    Attributes:
        field (str): field name, e.g. 'Theta'.
        folder (str): full path of the raw-output folder holding the file.
        path (str): full path of the ``.shrunk`` file.
        iteration (int): MITgcm iteration number from the filename.
        n_in_folder (int): 0-based position within the folder's sorted listing.
        date (datetime): assigned UTC date: folder start + (n_in_folder + 1) hours.
    """
    field: str
    folder: str
    path: str
    iteration: int
    n_in_folder: int
    date: datetime

    @property
    def store_name(self) -> str:
        """Zarr store name for this step, dbof-style: ``YYYYMMDDTHH.zarr``."""
        return store_name_for_date(self.date)


def _check_folder_consistency(folder: str, start, end, iterations):
    """Warn if a folder's files don't look like an unbroken hourly sequence.

    Two independent checks on the ``start + (n+1) hours`` dating rule:

    1. Iteration stride: with a fixed model timestep and hourly output,
       consecutive iteration numbers differ by a constant.  A varying
       stride means a missing/extra file, and the position-based dates
       after the gap would be wrong.
    2. Folder span: the last file is *at* the folder's end time and the
       first is one hour after its start, so nfiles must equal the span in
       hours exactly.

    Returns:
        int or None: the constant iteration stride, if there is one.
    """
    nfiles = len(iterations)
    stride = None
    if nfiles >= 2:
        diffs = np.diff(np.asarray(iterations, dtype=np.int64))
        if np.all(diffs == diffs[0]):
            stride = int(diffs[0])
        else:
            logger.warning(
                "%s: iteration stride is not constant (%s) -- a missing or "
                "extra file? Position-based dates may be wrong here.",
                folder, sorted(set(int(d) for d in diffs)))
    span_hours = (end - start) / OUTPUT_CADENCE
    if span_hours != nfiles:
        logger.warning(
            "%s: folder spans %.2f h but holds %d files -- expected exactly %d "
            "for hourly output whose last file is at the folder end time. "
            "Dates in this folder may be wrong.",
            folder, span_hours, nfiles, int(round(span_hours)))
    return stride


def _check_against_namelist(folder: str, iterations, stride):
    """Compare the position-based dating with iteration*deltaT from ``data``/``STDOUT``.

    With ``deltaT`` (and ``nIter0``) known, the first file should sit one
    hour after the segment start and files one hour apart.  Two numbering
    conventions are accepted: absolute (``(iter0 - nIter0) * deltaT``) or
    relative to the segment (``iter0 * deltaT``).  Logs which one matched
    and warns if neither does; never changes the dates.

    Returns:
        dict: the parsed run parameters (possibly empty).
    """
    prm = read_run_params(folder)
    if 'deltaT' not in prm or not iterations:
        return prm
    dt = prm['deltaT']
    n_iter0 = prm.get('nIter0', 0.0)
    hour = OUTPUT_CADENCE.total_seconds() / 3600.0
    first_abs_h = (iterations[0] - n_iter0) * dt / 3600.0
    first_rel_h = iterations[0] * dt / 3600.0
    stride_h = stride * dt / 3600.0 if stride else None
    if abs(first_abs_h - hour) < 1e-6:
        conv = 'absolute'
    elif abs(first_rel_h - hour) < 1e-6:
        conv = 'relative-to-nIter0'
    else:
        conv = None
    logger.info("%s: %s gives deltaT=%g s, nIter0=%g -> first file at +%.3f h "
                "(absolute numbering) / +%.3f h (relative), file spacing %s h; "
                "numbering convention: %s",
                folder, prm.get('source'), dt, n_iter0, first_abs_h, first_rel_h,
                f"{stride_h:.3f}" if stride_h is not None else "n/a", conv or 'UNMATCHED')
    if conv is None or (stride_h is not None and abs(stride_h - hour) > 1e-6):
        logger.warning(
            "%s: iteration*deltaT dating disagrees with the 'start + (n+1) hours' "
            "rule (first file +%.3f h abs / +%.3f h rel, spacing %s h). Dates "
            "assigned here follow the position rule; check this folder.",
            folder, first_abs_h, first_rel_h,
            f"{stride_h:.3f}" if stride_h is not None else "n/a")
    return prm


def discover_timesteps(out_dir: str, field: str = 'SST', ext: str = 'data',
                       start: datetime = None, end: datetime = None,
                       unreadable: list = None):
    """Find every ``<field>.<iteration>.<ext>`` file under *out_dir* and date it.

    Date rule (Dan Whitt, 2026-09-09): folder start time + (n + 1) hours for
    the n-th file of the folder in iteration order -- the first file is one
    hour after the folder start, the last is at the folder end.

    Args:
        out_dir (str): parent of the ``YYYY_MM_DD_HHMMSS_to_...`` folders,
            e.g. ``/nobackupp27/dbwhitt/llc_4320/OUT``.
        field (str, optional): filename prefix, e.g. 'SST', 'Eta', 'Theta'.
            Defaults to 'SST'.
        ext (str, optional): 'data' (uncompressed) or 'shrunk'. Defaults
            to 'data'.
        start (datetime, optional): keep only steps with date >= start.
        end (datetime, optional): keep only steps with date < end.
        unreadable (list, optional): if given, paths of output folders that
            could not be listed (permission denied, e.g. the segment still
            being written) are appended to it. They are always skipped
            with a warning.

    Returns:
        list[Timestep]: sorted by date.
    """
    if not os.path.isdir(out_dir):
        raise IOError(f"Not a directory: {out_dir}")
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    steps = []
    for entry in sorted(os.listdir(out_dir)):
        folder = os.path.join(out_dir, entry)
        if not os.path.isdir(folder):
            continue
        try:
            t0, t1 = parse_folder_name(entry)
        except ValueError:
            logger.debug("Skipping non-output folder %s", folder)
            continue

        try:
            names = os.listdir(folder)
        except OSError as e:
            logger.warning("%s: cannot list folder (%s); skipped", folder, e.strerror)
            if unreadable is not None:
                unreadable.append(folder)
            continue
        files = []
        for fn in names:
            m = OUTPUT_FILE_RE.match(fn)
            if m is not None and m['field'] == field and m['ext'] == ext:
                files.append((int(m['iteration']), fn))
        if not files:
            continue
        files.sort()  # by iteration number

        iterations = [it for it, _ in files]
        stride = _check_folder_consistency(folder, t0, t1, iterations)
        _check_against_namelist(folder, iterations, stride)

        for n, (iteration, fn) in enumerate(files):
            date = t0 + (n + 1) * OUTPUT_CADENCE
            if start is not None and date < start:
                continue
            if end is not None and date >= end:
                continue
            steps.append(Timestep(field=field, folder=folder,
                                  path=os.path.join(folder, fn),
                                  iteration=iteration, n_in_folder=n,
                                  date=date))

    steps.sort(key=lambda s: (s.date, s.iteration))

    # Global sanity check: dates must be unique (no hour in two folders).
    dates = [s.date for s in steps]
    if len(set(dates)) != len(dates):
        dupes = sorted({d for d in dates if dates.count(d) > 1})
        logger.warning("Duplicate dates across folders: %s",
                       [d.isoformat() for d in dupes[:5]])
    return steps


def infer_timestep_seconds(steps):
    """Model timesteps (seconds) implied by hourly-spaced iteration numbers.

    Uses consecutive steps *within* the same folder only.  deltaT varies
    between folders (5-20 s), so a list of the distinct values seen is
    returned rather than a single number.

    Args:
        steps (list[Timestep]): from `discover_timesteps`.

    Returns:
        list[float]: sorted distinct values of 3600 / stride; empty if no
            two consecutive same-folder steps exist.
    """
    strides = set()
    for a, b in zip(steps[:-1], steps[1:]):
        if a.folder == b.folder and b.n_in_folder == a.n_in_folder + 1:
            d = b.iteration - a.iteration
            if d > 0:
                strides.add(d)
    return sorted(OUTPUT_CADENCE.total_seconds() / d for d in strides)


# ---------------------------------------------------------------------------
# Inventory: what fields / file types does each raw-output folder hold?
# ---------------------------------------------------------------------------
#
# Written after step 2 of the Pleiades checklist found *no* Theta.*.shrunk
# files in 2023_01_01_000000_to_2023_01_08_060000.  Rather than guess at
# names, summarise every folder compactly: for each (field, extension)
# present, the file count and iteration range.

OUTPUT_FILE_RE = re.compile(
    r'^(?P<field>[A-Za-z0-9_]+)\.(?P<iteration>\d{6,12})\.(?P<ext>[A-Za-z0-9_.]+)$')


@dataclass
class FolderInventory:
    """Summary of one raw-output folder.

    Attributes:
        folder (str): full path.
        start, end (datetime): from the folder name.
        n_files (int): number of entries in the folder.
        fields (dict): ``{(field, ext): (count, min_iteration, max_iteration)}``
            for every ``<field>.<iteration>.<ext>`` file present.
        other (list): up to 20 filenames not matching that pattern
            (namelists, logs, subdirectories, ...).
    """
    folder: str
    start: datetime
    end: datetime
    n_files: int
    fields: dict
    other: list

    def fields_with_ext(self, ext: str):
        """Sorted field names present with extension *ext* (e.g. 'shrunk')."""
        return sorted(f for (f, e) in self.fields if e == ext)

    def has(self, field: str, ext: str = 'data') -> bool:
        return (field, ext) in self.fields


def inventory_folder(folder: str) -> FolderInventory:
    """Summarise the files in one raw-output folder (see `FolderInventory`)."""
    start, end = parse_folder_name(folder)
    names = sorted(os.listdir(folder))
    fields, other = {}, []
    for fn in names:
        m = OUTPUT_FILE_RE.match(fn)
        if m is None:
            if len(other) < 20:
                other.append(fn + ('/' if os.path.isdir(os.path.join(folder, fn)) else ''))
            continue
        key = (m['field'], m['ext'])
        it = int(m['iteration'])
        cnt, lo, hi = fields.get(key, (0, it, it))
        fields[key] = (cnt + 1, min(lo, it), max(hi, it))
    return FolderInventory(folder=folder, start=start, end=end, n_files=len(names),
                           fields=fields, other=other)


def inventory(out_dir: str, max_folders: int = None, unreadable: list = None):
    """Inventory every ``YYYY_MM_DD_HHMMSS_to_...`` folder under *out_dir*.

    Args:
        out_dir (str): raw-output parent.
        max_folders (int, optional): stop after this many folders.
        unreadable (list, optional): if given, folders that cannot be
            listed (permission denied) are appended; they are skipped with
            a warning either way.

    Returns:
        list[FolderInventory]: in name (= chronological) order.
    """
    if not os.path.isdir(out_dir):
        raise IOError(f"Not a directory: {out_dir}")
    out = []
    for entry in sorted(os.listdir(out_dir)):
        folder = os.path.join(out_dir, entry)
        if not os.path.isdir(folder) or FOLDER_RE.match(entry) is None:
            continue
        try:
            out.append(inventory_folder(folder))
        except OSError as e:
            logger.warning("%s: cannot list folder (%s); skipped", folder, e.strerror)
            if unreadable is not None:
                unreadable.append(folder)
            continue
        if max_folders is not None and len(out) >= max_folders:
            break
    return out


def summarize_inventory(invs, field: str = 'SST', ext: str = 'data') -> dict:
    """Roll an `inventory` up into the few facts we need.

    Returns:
        dict: ``n_folders``; ``combos`` -- ``{(field, ext): n_folders_present}``
            over all folders; ``with_field`` -- folders holding
            ``<field>.*.<ext>``; ``first_with``/``last_with`` -- their date
            range (or None); ``without_field`` -- folder basenames lacking it.
    """
    combos = {}
    with_field, without = [], []
    for inv in invs:
        for key in inv.fields:
            combos[key] = combos.get(key, 0) + 1
        (with_field if inv.has(field, ext) else without).append(inv)
    return {
        'n_folders': len(invs),
        'combos': dict(sorted(combos.items())),
        'with_field': len(with_field),
        'first_with': with_field[0].start + OUTPUT_CADENCE if with_field else None,
        'last_with': with_field[-1].end if with_field else None,
        'without_field': [os.path.basename(i.folder) for i in without],
    }


def find_files(root: str, suffix: str = '.shrunk', max_depth: int = 3,
               limit: int = 20):
    """Locate files ending in *suffix* under *root*, at most *max_depth* levels down.

    A bounded ``find root -maxdepth N -name "*suffix"`` for "where did the
    compressed 3D output go?": files directly in *root* are at depth 1,
    files in its immediate subdirectories at depth 2, and so on.
    Unreadable directories are skipped silently.

    Returns:
        tuple: (list of up to *limit* matching paths, total number of matches).
    """
    root = os.path.normpath(root)
    base_depth = root.count(os.sep)
    hits, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        file_depth = dirpath.count(os.sep) - base_depth + 1
        if file_depth > max_depth:
            dirnames[:] = []
            continue
        if file_depth >= max_depth:
            dirnames[:] = []
        for fn in filenames:
            if fn.endswith(suffix):
                total += 1
                if len(hits) < limit:
                    hits.append(os.path.join(dirpath, fn))
    return hits, total


# ---------------------------------------------------------------------------
# Zarr output on Nautilus S3 (or a local directory)
# ---------------------------------------------------------------------------
#
# Layout mirrors llc4320-native-grid-preprocessing's LLC4320_RAW convention
# (docs/Data_Organization.md there): one ``grid.zarr`` plus one
# ``{YYYYMMDDTHH}.zarr`` store per output hour, each variable a
# ``(face, j, i)`` array chunked ``(1, 720, 720)``, NaN over land.
#
# Credentials follow the PAB repo's Nautilus pattern (``nautilus/s3_push.py``,
# ``pab/report/publish.py::NautilusS3Backend``): endpoint from
# ``ENDPOINT_URL`` (default s3-west), keys from the standard boto3 chain --
# ``AWS_PROFILE`` (default "default") in ``~/.aws/credentials`` or
# ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` -- and path-style
# addressing, which Nautilus' Ceph RGW requires.
#
# ``zarr`` / ``fsspec`` / ``s3fs`` are imported lazily so the decoder above
# stays importable without them.

DEFAULT_ENDPOINT = 'https://s3-west.nrp-nautilus.io'
DEFAULT_BUCKET = 'llc4320-v2'
DEFAULT_PREFIX = 'SURFACE'
SURF_CHUNKS = (1, 720, 720)   # (face, j, i): dbof's llc_surf_timestep_chunks
GRID_STORE_NAME = 'grid.zarr'


def store_name_for_date(date: datetime) -> str:
    """dbof-style per-hour store name: 2023-01-01 06:00 -> '20230101T06.zarr'."""
    return date.strftime('%Y%m%dT%H') + '.zarr'


def s3_endpoint(endpoint: str = None) -> str:
    """Resolve the S3 endpoint: explicit arg, else ``$ENDPOINT_URL``, else Nautilus west."""
    return (endpoint or os.environ.get('ENDPOINT_URL') or DEFAULT_ENDPOINT).rstrip('/')


def s3_filesystem(endpoint: str = None, profile: str = None, asynchronous: bool = True):
    """Build an s3fs filesystem for Nautilus with the PAB/dbof settings.

    Args:
        endpoint (str, optional): S3 endpoint URL; see `s3_endpoint`.
        profile (str, optional): boto3 credentials profile. Defaults to
            ``$AWS_PROFILE`` if set, else the boto3 default chain.
        asynchronous (bool, optional): True for use inside a zarr
            ``FsspecStore`` (zarr v3 drives the store asynchronously),
            False for ordinary listing/exists calls. Defaults to True.

    Returns:
        fsspec.AbstractFileSystem
    """
    import fsspec
    kwargs = dict(
        asynchronous=asynchronous,
        client_kwargs={'endpoint_url': s3_endpoint(endpoint)},
        config_kwargs={
            'signature_version': 's3v4',
            'request_checksum_calculation': 'when_required',
            's3': {'addressing_style': 'path',
                   'payload_signing_enabled': False,
                   'use_accelerate_endpoint': False,
                   'use_dualstack_endpoint': False},
        },
    )
    profile = profile or os.environ.get('AWS_PROFILE')
    if profile:
        kwargs['profile'] = profile
    return fsspec.filesystem('s3', **kwargs)


def is_s3(dest: str) -> bool:
    """True if *dest* is an ``s3://`` URL."""
    return str(dest).startswith('s3://')


def _zarr_store(store_url: str, endpoint: str = None, profile: str = None):
    """Return a zarr store object for a local path or an ``s3://`` URL."""
    if is_s3(store_url):
        import zarr
        fs = s3_filesystem(endpoint, profile, asynchronous=True)
        return zarr.storage.FsspecStore(path=store_url, fs=fs)
    return store_url


def open_zarr_group(store_url: str, mode: str = 'a', endpoint: str = None,
                    profile: str = None):
    """Open (or create) a zarr group at a local path or ``s3://`` URL.

    Args:
        store_url (str): e.g. ``/scratch/out/20230101T00.zarr`` or
            ``s3://llc4320-v2/SURFACE/20230101T00.zarr``.
        mode (str, optional): 'r' (read), 'a' (read/write, create if
            missing) or 'w' (create, overwriting anything there). Defaults
            to 'a'.
        endpoint, profile: see `s3_filesystem`.

    Returns:
        zarr.Group
    """
    import zarr
    store = _zarr_store(store_url, endpoint, profile)
    if mode == 'w':
        return zarr.group(store=store, overwrite=True)
    return zarr.open_group(store=store, mode=mode, use_consolidated=False)


def store_is_complete(store_url: str, variables, endpoint: str = None,
                      profile: str = None) -> bool:
    """True if *store_url* exists, holds all *variables*, and was fully written.

    A store is marked ``complete=True`` in its attributes only after every
    variable has been written and verified, so an interrupted run leaves a
    store this returns False for, and a re-run rewrites it.

    Args:
        variables (str or iterable of str): variable name(s) required.
    """
    if isinstance(variables, str):
        variables = [variables]
    try:
        g = open_zarr_group(store_url, mode='r', endpoint=endpoint, profile=profile)
    except Exception:
        return False
    return bool(g.attrs.get('complete', False)) and all(v in g for v in variables)


def _ensure_coords(root, FS: int):
    """Write the face/j/i index coordinates if not already present (dbof style)."""
    for name, n, chunk in (('face', N_FACETS, N_FACETS),
                           ('j', FS, min(FS, SURF_CHUNKS[1])),
                           ('i', FS, min(FS, SURF_CHUNKS[2]))):
        if name in root:
            continue
        vals = np.arange(n, dtype=np.int64)
        zc = root.create_array(name, shape=vals.shape, chunks=(chunk,),
                               dtype=vals.dtype, overwrite=True,
                               dimension_names=(name,))
        zc[:] = vals


def write_surface_variable(root, name: str, field: np.ndarray, attrs: dict = None,
                           chunks=SURF_CHUNKS, verify: bool = True):
    """Write one global (13, FS, FS) surface field into an open zarr group.

    Args:
        root (zarr.Group): open for writing.
        name (str): variable name, e.g. 'Theta'.
        field (np.ndarray): shape (13, FS, FS).
        attrs (dict, optional): variable attributes.
        chunks (tuple, optional): (face, j, i) chunk shape. Defaults to
            `SURF_CHUNKS`. Clipped to the field shape for small test grids.
        verify (bool, optional): read every face back and compare (as
            dbof's ``_verify_tile`` does). Defaults to True.
    """
    if field.ndim != 3 or field.shape[0] != N_FACETS:
        raise ValueError(f"expected (13, FS, FS), got {field.shape}")
    FS = field.shape[1]
    chunks = tuple(min(c, s) for c, s in zip(chunks, field.shape))
    _ensure_coords(root, FS)

    is_float = np.issubdtype(field.dtype, np.floating)
    z = root.create_array(name, shape=field.shape, chunks=chunks,
                          dtype=field.dtype, overwrite=True,
                          fill_value=np.nan if is_float else 0,
                          dimension_names=('face', 'j', 'i'))
    for k, v in (attrs or {}).items():
        try:
            z.attrs[k] = v
        except Exception:  # unserialisable attr -- skip, as dbof does
            pass
    # One face at a time keeps the in-flight buffer to ~1/13 of the field.
    for f in range(N_FACETS):
        z[f] = field[f]
        if verify:
            back = z[f]
            if not np.array_equal(back, field[f], equal_nan=is_float):
                raise RuntimeError(f"{name} face {f}: read-back mismatch")
    return z


# ---------------------------------------------------------------------------
# Surface-field table, land masking, grid store, per-hour stores, pipeline
# ---------------------------------------------------------------------------

# Filename prefix on Pleiades -> (variable name in the Zarr store, attributes).
# Variable names follow dbof / xmitgcm conventions (Theta, Salt, U, V, Eta,
# ...) so llc4320-v2 stores read like the LLC4320_RAW/SURFACE ones.
# SSU/SSV are the model's native face-relative velocity components (not
# rotated to east/north); rotation needs CS/SN from grid.zarr.
SURFACE_FIELDS = {
    'SST': ('Theta', {'long_name': 'sea surface temperature (Theta, k=0)', 'units': 'degC'}),
    'SSS': ('Salt', {'long_name': 'sea surface salinity (Salt, k=0)', 'units': 'psu'}),
    'SSU': ('U', {'long_name': 'surface velocity, native i-component (U, k=0), '
                                'face-relative, not rotated', 'units': 'm s-1'}),
    'SSV': ('V', {'long_name': 'surface velocity, native j-component (V, k=0), '
                                'face-relative, not rotated', 'units': 'm s-1'}),
    'Eta': ('Eta', {'long_name': 'sea surface height anomaly', 'units': 'm'}),
    'KPPhbl': ('KPPhbl', {'long_name': 'KPP boundary layer depth', 'units': 'm'}),
    'PhiBot': ('PhiBot', {'long_name': 'bottom pressure potential anomaly', 'units': 'm2 s-2'}),
    'oceQnet': ('oceQnet', {'long_name': 'net surface heat flux into the ocean', 'units': 'W m-2'}),
    'oceQsw': ('oceQsw', {'long_name': 'net shortwave radiation into the ocean', 'units': 'W m-2'}),
    'oceFWflx': ('oceFWflx', {'long_name': 'net surface freshwater flux into the ocean',
                              'units': 'kg m-2 s-1'}),
    'oceSflux': ('oceSflux', {'long_name': 'net surface salt flux into the ocean', 'units': 'g m-2 s-1'}),
    'oceTAUX': ('oceTAUX', {'long_name': 'surface wind stress, native i-component', 'units': 'N m-2'}),
    'oceTAUY': ('oceTAUY', {'long_name': 'surface wind stress, native j-component', 'units': 'N m-2'}),
    'SIarea': ('SIarea', {'long_name': 'sea-ice fractional coverage', 'units': '1'}),
    'SIheff': ('SIheff', {'long_name': 'sea-ice effective thickness', 'units': 'm'}),
    'SIhsnow': ('SIhsnow', {'long_name': 'snow effective thickness', 'units': 'm'}),
    'SIuice': ('SIuice', {'long_name': 'sea-ice velocity, native i-component', 'units': 'm s-1'}),
    'SIvice': ('SIvice', {'long_name': 'sea-ice velocity, native j-component', 'units': 'm s-1'}),
}

# Fields whose value is legitimately 0 over most of the ocean, so exact
# zeros cannot be used as a land indicator without a mask.
ZERO_IS_NOT_LAND = {'SIarea', 'SIheff', 'SIhsnow', 'SIuice', 'SIvice'}


def field_variable(prefix: str):
    """(variable name, attrs) for a filename prefix; unknown prefixes map to themselves."""
    if prefix in SURFACE_FIELDS:
        var, attrs = SURFACE_FIELDS[prefix]
        return var, dict(attrs, source_file_prefix=prefix, level=0)
    return prefix, {'source_file_prefix': prefix, 'level': 0}


def surface_wet_mask(mask_dir: str, var_name: str, FS: int) -> np.ndarray:
    """k=0 wet mask, shape (13, FS, FS), from the right hFac*.bits for *var_name*.

    U lives on hFacW points, V on hFacS points, everything else on hFacC
    (`mask_file_for_field`).
    """
    mask_file = os.path.join(mask_dir, mask_file_for_field(var_name))
    return read_mask_level(mask_file, FS, level=0).reshape(N_FACETS, FS, FS)


def apply_land_mask(field: np.ndarray, wet: np.ndarray = None, prefix: str = None) -> np.ndarray:
    """Set land points to NaN.

    With *wet* (bool, same shape) given, land is ``~wet``.  Without it, land
    is taken to be the points that are exactly 0.0 -- what MITgcm writes
    over land in ``.data`` output -- except for `ZERO_IS_NOT_LAND` fields,
    which are returned unchanged with a warning.

    Returns:
        np.ndarray: float32, modified in place and returned.
    """
    if field.dtype != np.float32:
        field = field.astype(np.float32)
    if wet is not None:
        field[~wet] = np.nan
    elif prefix in ZERO_IS_NOT_LAND:
        logger.warning("%s: zero is a valid ocean value; no mask dir given, so land "
                       "is left as 0.0 -- pass --mask-dir for NaN land.", prefix)
    else:
        field[field == 0.0] = np.nan
    return field


# MITgcm grid file name -> (xmitgcm/dbof variable name, attrs).  All 2D,
# one (13, FS, FS) level in compact format.
GRID_2D_FILES = {
    'XC': ('XC', {'long_name': 'longitude of cell centre', 'units': 'degrees_east'}),
    'YC': ('YC', {'long_name': 'latitude of cell centre', 'units': 'degrees_north'}),
    'XG': ('XG', {'long_name': 'longitude of cell corner', 'units': 'degrees_east'}),
    'YG': ('YG', {'long_name': 'latitude of cell corner', 'units': 'degrees_north'}),
    'RAC': ('rA', {'long_name': 'cell area', 'units': 'm2'}),
    'RAS': ('rAs', {'long_name': 'cell area at south face', 'units': 'm2'}),
    'RAW': ('rAw', {'long_name': 'cell area at west face', 'units': 'm2'}),
    'RAZ': ('rAz', {'long_name': 'cell area at corner', 'units': 'm2'}),
    'DXC': ('dxC', {'long_name': 'cell x size at west face', 'units': 'm'}),
    'DYC': ('dyC', {'long_name': 'cell y size at south face', 'units': 'm'}),
    'DXG': ('dxG', {'long_name': 'cell x size at south face', 'units': 'm'}),
    'DYG': ('dyG', {'long_name': 'cell y size at west face', 'units': 'm'}),
    'Depth': ('Depth', {'long_name': 'ocean depth', 'units': 'm'}),
    'AngleCS': ('CS', {'long_name': 'cosine of grid orientation angle', 'units': '1'}),
    'AngleSN': ('SN', {'long_name': 'sine of grid orientation angle', 'units': '1'}),
}
# 3D grid files: only level 0 is stored (as <name>_k0), plus a bool mask.
GRID_3D_FILES = ('hFacC', 'hFacS', 'hFacW')
# 1D vertical grid files (nz or nz+1 big-endian float32 values).
GRID_1D_FILES = {'RC': 'k', 'RF': 'k_p1', 'DRC': 'k_p1', 'DRF': 'k'}


def write_grid_store(dest: str, grid_dir: str = None, mask_dir: str = None,
                     FS: int = 4320, endpoint: str = None, profile: str = None,
                     skip_existing: bool = True) -> str:
    """Write ``grid.zarr`` for v2 from its own grid files (and/or the bit masks).

    v2 shares v1's horizontal grid but has its own bathymetry, hFac, land
    mask and vertical grid (Dan Whitt), so everything is read from
    *grid_dir* (``/nobackupp27/dbwhitt/llc_4320/grid_90x90x19493`` on
    Pleiades).  Each `GRID_2D_FILES` entry whose ``.data`` file exists with
    the size of one compact ``13*FS*FS`` float32 level is written; level 0
    of each `GRID_3D_FILES` file is written as ``<name>_k0``; `GRID_1D_FILES`
    are written as 1D arrays; ``maskC`` (True = wet at k=0) comes from
    ``hFacC.bits`` in *mask_dir* if given, else from ``hFacC.data`` level 0,
    else from ``Depth > 0``.  Files that are missing or of unexpected size
    are skipped with a warning, so the store is as complete as the inputs
    allow.

    Args:
        dest (str): destination prefix: local directory or ``s3://bucket/prefix``.
        grid_dir (str, optional): directory of MITgcm grid ``.data`` files.
        mask_dir (str, optional): directory holding ``hFacC.bits``.
        FS (int, optional): facet side. Defaults to 4320.
        endpoint, profile: see `s3_filesystem`.
        skip_existing (bool, optional): leave a complete store alone.

    Returns:
        str: the store URL.
    """
    url = f"{str(dest).rstrip('/')}/{GRID_STORE_NAME}"
    if skip_existing and store_is_complete(url, 'maskC', endpoint, profile):
        logger.info("grid store complete, skipping: %s", url)
        return url
    if grid_dir is None and mask_dir is None:
        raise ValueError("write_grid_store needs grid_dir and/or mask_dir")

    level_bytes = N_FACETS * FS * FS * 4
    root = open_zarr_group(url, mode='w', endpoint=endpoint, profile=profile)
    root.attrs.update({'model': 'LLC4320_v2', 'FS': FS, 'complete': False,
                       'grid_dir': grid_dir or '', 'mask_dir': mask_dir or '',
                       'description': 'Static grid information for LLC4320 v2 '
                                      'surface output (see wrangler.ogcm.llc_v2)'})
    written, skipped = [], []
    hfacc_k0 = depth = None

    if grid_dir is not None:
        for fname, (var, attrs) in GRID_2D_FILES.items():
            path = os.path.join(grid_dir, f"{fname}.data")
            if not os.path.isfile(path):
                skipped.append(f"{fname}.data (missing)")
                continue
            if os.path.getsize(path) != level_bytes:
                skipped.append(f"{fname}.data (size {os.path.getsize(path)} != {level_bytes})")
                continue
            try:
                arr = read_data_field(path, FS)
            except OSError as e:
                skipped.append(f"{fname}.data ({e.strerror})")
                continue
            write_surface_variable(root, var, arr, attrs=dict(attrs, source_file=f"{fname}.data"))
            written.append(var)
            if fname == 'Depth':
                depth = arr
        for fname in GRID_3D_FILES:
            path = os.path.join(grid_dir, f"{fname}.data")
            if not os.path.isfile(path):
                skipped.append(f"{fname}.data (missing)")
                continue
            size = os.path.getsize(path)
            if size % level_bytes != 0:
                skipped.append(f"{fname}.data (size {size} not a multiple of one level)")
                continue
            try:
                arr = read_data_field(path, FS, level=0)
            except OSError as e:
                skipped.append(f"{fname}.data ({e.strerror})")
                continue
            write_surface_variable(root, f"{fname}_k0", arr,
                                   attrs={'long_name': f'{fname} at k=0 (open fraction)',
                                          'units': '1', 'source_file': f"{fname}.data",
                                          'levels_in_file': size // level_bytes})
            written.append(f"{fname}_k0")
            if fname == 'hFacC':
                hfacc_k0 = arr
        for fname, dim in GRID_1D_FILES.items():
            path = os.path.join(grid_dir, f"{fname}.data")
            if not os.path.isfile(path):
                skipped.append(f"{fname}.data (missing)")
                continue
            try:
                vals = np.fromfile(path, dtype='>f4').astype(np.float32)
            except OSError as e:
                skipped.append(f"{fname}.data ({e.strerror})")
                continue
            if vals.size == 0 or vals.size > 10_000:
                skipped.append(f"{fname}.data ({vals.size} values; not a 1D vertical file)")
                continue
            z = root.create_array(fname, shape=vals.shape, chunks=vals.shape,
                                  dtype=vals.dtype, overwrite=True, dimension_names=(dim,))
            z[:] = vals
            z.attrs['source_file'] = f"{fname}.data"
            z.attrs['units'] = 'm'
            written.append(fname)

    wet = None
    if mask_dir is not None:
        try:
            wet = read_mask_level(os.path.join(mask_dir, 'hFacC.bits'), FS, 0).reshape(N_FACETS, FS, FS)
            mask_src = 'hFacC.bits'
        except OSError as e:
            skipped.append(f"hFacC.bits ({e.strerror}); falling back for maskC")
    if wet is not None:
        pass
    elif hfacc_k0 is not None:
        wet = hfacc_k0 > 0
        mask_src = 'hFacC.data level 0 > 0'
    elif depth is not None:
        wet = depth > 0
        mask_src = 'Depth.data > 0'
    else:
        wet = None
        skipped.append('maskC (no hFacC.bits, hFacC.data or Depth.data)')
    if wet is not None:
        write_surface_variable(root, 'maskC', wet,
                               attrs={'long_name': 'wet-point mask at cell centres, k=0',
                                      'source': mask_src})
        written.append('maskC')

    for item in skipped:
        logger.warning("grid.zarr: skipped %s", item)
    logger.info("grid.zarr: wrote %s", written)
    root.attrs['variables'] = written
    root.attrs['skipped'] = skipped
    root.attrs['complete'] = 'maskC' in written
    return url


def write_timestep_store(dest: str, step: Timestep, fields: dict, complete: bool = True,
                         endpoint: str = None, profile: str = None) -> str:
    """Write one ``{YYYYMMDDTHH}.zarr`` store holding one or more surface variables.

    Args:
        dest (str): destination prefix (local dir or ``s3://bucket/prefix``).
        step (Timestep): the output hour (supplies date/iteration/source).
        fields (dict): ``{var_name: (array (13, FS, FS), attrs dict)}``.
        complete (bool, optional): mark the store complete at the end.
            Pass False when some requested field was unavailable so a re-run
            revisits the store. Defaults to True.
        endpoint, profile: see `s3_filesystem`.

    Returns:
        str: the store URL.
    """
    url = f"{str(dest).rstrip('/')}/{step.store_name}"
    root = open_zarr_group(url, mode='w', endpoint=endpoint, profile=profile)
    any_arr = next(iter(fields.values()))[0]
    root.attrs.update({
        'model': 'LLC4320_v2',
        'selected_iteration': int(step.iteration),
        'selected_date_utc': step.date.strftime('%Y-%m-%d %H:%M:%S'),
        'source_folder': os.path.basename(step.folder),
        'FS': int(any_arr.shape[1]),
        'variables': sorted(fields),
        'complete': False,
    })
    for var, (arr, attrs) in fields.items():
        write_surface_variable(root, var, arr, attrs=attrs)
    root.attrs['complete'] = bool(complete)
    return url


def extract_surface(out_dir: str, dest: str, fields=('SST',), mask_dir: str = None,
                    grid_dir: str = None, FS: int = 4320, start: datetime = None,
                    end: datetime = None, limit: int = None, skip_existing: bool = True,
                    dry_run: bool = False, write_grid: bool = True, endpoint: str = None,
                    profile: str = None) -> dict:
    """End-to-end: discover hourly surface ``.data`` output, write one Zarr store per hour.

    Idempotent (like PAB's ``s3_push.py``): stores already marked complete
    are skipped, so an interrupted run can simply be restarted, and a
    re-run after new output lands on disk only processes the new hours.

    Args:
        out_dir (str): raw-output parent, e.g. ``/nobackupp27/dbwhitt/llc_4320/OUT``.
        dest (str): ``s3://llc4320-v2/SURFACE`` or a local directory.
        fields (iterable of str, optional): filename prefixes to extract,
            e.g. ``['SST', 'SSS', 'Eta']`` (see `SURFACE_FIELDS`). Discovery
            is driven by the first one. Defaults to ``('SST',)``.
        mask_dir (str, optional): directory with ``hFac*.bits``; land -> NaN
            from the k=0 masks. Without it, exact zeros are treated as land.
        grid_dir (str, optional): v2 grid ``.data`` directory for ``grid.zarr``.
        FS (int, optional): facet side. Defaults to 4320.
        start, end (datetime, optional): date window (see `discover_timesteps`).
        limit (int, optional): process at most this many hours (testing).
        skip_existing (bool, optional): skip complete stores. Defaults to True.
        dry_run (bool, optional): only list what would be done.
        write_grid (bool, optional): also write ``grid.zarr`` (needs
            *grid_dir* and/or *mask_dir*; skipped with a warning otherwise).
        endpoint, profile: see `s3_filesystem`.

    Returns:
        dict: {'discovered', 'written', 'skipped', 'incomplete', 'dt_seconds'}.
    """
    fields = list(fields)
    if not fields:
        raise ValueError("fields must not be empty")
    var_of = {f: field_variable(f) for f in fields}
    variables = [var_of[f][0] for f in fields]

    unreadable = []
    steps = discover_timesteps(out_dir, fields[0], 'data', start=start, end=end,
                               unreadable=unreadable)
    if limit is not None:
        steps = steps[:limit]
    dts = infer_timestep_seconds(steps)
    logger.info("Discovered %d hourly %s steps under %s (model dt values: %s s); "
                "%d unreadable folder(s) skipped",
                len(steps), fields[0], out_dir, dts, len(unreadable))
    stats = {'discovered': len(steps), 'written': 0, 'skipped': 0, 'incomplete': 0,
             'dt_seconds': dts,
             'unreadable_folders': [os.path.basename(u) for u in unreadable]}

    if dry_run:
        for s in steps:
            logger.info("would write %s/%s  <- %s  [%s]", dest, s.store_name,
                        os.path.basename(s.path), ','.join(fields))
        return stats

    if write_grid:
        if grid_dir is None and mask_dir is None:
            logger.warning("no --grid-dir / --mask-dir given: grid.zarr not written")
        else:
            write_grid_store(dest, grid_dir, mask_dir, FS, endpoint, profile,
                             skip_existing=skip_existing)

    wet_cache = {}

    def wet_for(var):
        if mask_dir is None:
            return None
        key = mask_file_for_field(var)
        if key not in wet_cache:
            wet_cache[key] = surface_wet_mask(mask_dir, var, FS)
        return wet_cache[key]

    for s in steps:
        url = f"{str(dest).rstrip('/')}/{s.store_name}"
        if skip_existing and store_is_complete(url, variables, endpoint, profile):
            logger.info("complete, skipping: %s", url)
            stats['skipped'] += 1
            continue
        logger.info("%s (iteration %d): reading %s", s.date.isoformat(), s.iteration,
                    ','.join(fields))
        arrays, missing = {}, []
        for prefix in fields:
            var, attrs = var_of[prefix]
            path = os.path.join(s.folder, f"{prefix}.{s.iteration:010d}.data")
            if not os.path.isfile(path):
                logger.warning("missing %s", path)
                missing.append(prefix)
                continue
            try:
                arr = read_data_field(path, FS)
            except (OSError, ValueError) as e:
                logger.warning("cannot read %s (%s); treated as missing", path, e)
                missing.append(prefix)
                continue
            arr = apply_land_mask(arr, wet_for(var), prefix)
            arrays[var] = (arr, dict(attrs, source_file=os.path.basename(path)))
        if not arrays:
            logger.warning("no requested fields found for %s; store not written", s.store_name)
            stats['incomplete'] += 1
            continue
        write_timestep_store(dest, s, arrays, complete=not missing, endpoint=endpoint,
                             profile=profile)
        if missing:
            stats['incomplete'] += 1
            logger.warning("wrote %s without %s (left incomplete)", url, missing)
        else:
            stats['written'] += 1
            logger.info("wrote %s", url)
    return stats


def extract_sst(out_dir: str, dest: str, mask_dir: str = None, **kwargs) -> dict:
    """SST-only convenience wrapper around `extract_surface` (fields=['SST'])."""
    return extract_surface(out_dir, dest, fields=['SST'], mask_dir=mask_dir, **kwargs)
