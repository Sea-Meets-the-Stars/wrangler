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
# Discovery: raw-output folders -> (iteration, date) records
# ---------------------------------------------------------------------------
#
# Raw output on Pleiades lives under one parent (e.g.
# /nobackupp27/dbwhitt/llc_4320/OUT) in folders named by the time range they
# cover, e.g. ``2023_01_01_000000_to_2023_01_08_060000``, each holding one
# ``<field>.<10-digit iteration>.shrunk`` file per output step.  The .shrunk
# filenames carry the MITgcm iteration number, not a date.  The date rule is
# the one Dimitris uses in ``ExtractFields.m`` / ``dimitris_notes_v2.txt``:
#
#     date(file) = folder start time + n hours,
#
# where n is the file's 0-based position in the *sorted* listing of its
# folder (output is hourly).  ``discover_timesteps`` implements exactly
# that, and additionally cross-checks it against the iteration numbers
# (constant stride within a folder <=> constant cadence) and against the
# folder's end time, logging warnings on any inconsistency rather than
# silently mis-dating files.

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
        date (datetime): assigned UTC date (folder start + n_in_folder hours).
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

    Two independent checks on Dimitris' "start + n hours" dating rule:

    1. Iteration stride: with a fixed model timestep and hourly output,
       consecutive iteration numbers differ by a constant.  A varying
       stride means a missing/extra file, and the position-based dates
       after the gap would be wrong.
    2. Folder span: the ``_to_`` end time should equal start + nfiles
       hours (end-exclusive) or start + (nfiles-1) hours (end-inclusive).

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
    if span_hours not in (nfiles, nfiles - 1):
        logger.warning(
            "%s: folder spans %.2f h but holds %d files -- expected %d "
            "(end-exclusive) or %d (end-inclusive) for hourly output.",
            folder, span_hours, nfiles, nfiles, nfiles - 1)
    return stride


def discover_timesteps(out_dir: str, field: str = 'Theta',
                       start: datetime = None, end: datetime = None):
    """Find every available ``<field>.*.shrunk`` file under *out_dir* and date it.

    Mirrors the ``dir('*/U*.shrunk')`` scan in ``ExtractFields.m`` (but for
    *field*, so SST extraction doesn't depend on U being present) and its
    date assignment: folder start time + one hour per file in sorted order.

    Args:
        out_dir (str): parent of the ``YYYY_MM_DD_HHMMSS_to_...`` folders,
            e.g. ``/nobackupp27/dbwhitt/llc_4320/OUT``.
        field (str, optional): field whose files to discover. Defaults to
            'Theta'.
        start (datetime, optional): keep only steps with date >= start.
        end (datetime, optional): keep only steps with date < end.

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

        files = []
        for fn in os.listdir(folder):
            m = SHRUNK_RE.match(fn)
            if m is not None and m['field'] == field:
                files.append((int(m['iteration']), fn))
        if not files:
            continue
        files.sort()  # by iteration number

        iterations = [it for it, _ in files]
        _check_folder_consistency(folder, t0, t1, iterations)

        for n, (iteration, fn) in enumerate(files):
            date = t0 + n * OUTPUT_CADENCE
            if start is not None and date < start:
                continue
            if end is not None and date >= end:
                continue
            steps.append(Timestep(field=field, folder=folder,
                                  path=os.path.join(folder, fn),
                                  iteration=iteration, n_in_folder=n,
                                  date=date))

    steps.sort(key=lambda s: (s.date, s.iteration))

    # Global sanity check: dates must be unique (adjacent folders share a
    # boundary time in their names; if both contained that hour we'd get a
    # duplicate and one of the two files would be mis-dated).
    dates = [s.date for s in steps]
    if len(set(dates)) != len(dates):
        dupes = sorted({d for d in dates if dates.count(d) > 1})
        logger.warning("Duplicate dates across folders: %s",
                       [d.isoformat() for d in dupes[:5]])
    return steps


def infer_timestep_seconds(steps):
    """Infer the model timestep (seconds) from hourly-spaced iteration numbers.

    Only uses consecutive steps *within* the same folder, where the hourly
    cadence assumption is what the dating rule already relies on.

    Args:
        steps (list[Timestep]): from `discover_timesteps`.

    Returns:
        float or None: 3600 / (iteration stride), or None if no two
            consecutive same-folder steps exist or strides disagree.
    """
    strides = set()
    for a, b in zip(steps[:-1], steps[1:]):
        if a.folder == b.folder and b.n_in_folder == a.n_in_folder + 1:
            strides.add(b.iteration - a.iteration)
    if len(strides) != 1:
        return None
    stride = strides.pop()
    return OUTPUT_CADENCE.total_seconds() / stride if stride > 0 else None


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


def store_is_complete(store_url: str, variable: str, endpoint: str = None,
                      profile: str = None) -> bool:
    """True if *store_url* exists, holds *variable*, and was fully written.

    A store is marked ``complete=True`` in its attributes only after every
    variable has been written and verified, so an interrupted run leaves a
    store this returns False for, and a re-run rewrites it.
    """
    try:
        g = open_zarr_group(store_url, mode='r', endpoint=endpoint, profile=profile)
    except Exception:
        return False
    return bool(g.attrs.get('complete', False)) and variable in g


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


def write_grid_store(dest: str, mask_dir: str, FS: int = 4320, endpoint: str = None,
                     profile: str = None, skip_existing: bool = True) -> str:
    """Write ``grid.zarr`` with the surface wet-point mask (from hFacC.bits).

    Only the information we can derive from the v2 mask files is written for
    now: ``maskC`` (True = wet at k=0).  Horizontal grid variables (XC, YC,
    ...) are deferred until we know where the v2 grid files live / whether
    the grid is identical to v1's (see Q&A in claude_prompts/llc4320_v2.md).

    Args:
        dest (str): destination prefix: local directory or
            ``s3://bucket/prefix``.
        mask_dir (str): directory holding hFacC.bits.
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
    mask = read_mask_level(os.path.join(mask_dir, 'hFacC.bits'), FS, level=0)
    root = open_zarr_group(url, mode='w', endpoint=endpoint, profile=profile)
    root.attrs.update({'model': 'LLC4320_v2', 'FS': FS,
                       'description': 'Static grid information for LLC4320 v2 '
                                      'surface output (see wrangler.ogcm.llc_v2)'})
    write_surface_variable(root, 'maskC', mask.reshape(N_FACETS, FS, FS),
                           attrs={'long_name': 'wet-point mask at cell centres, k=0',
                                  'source': 'hFacC.bits'})
    root.attrs['complete'] = True
    return url


def write_timestep_store(dest: str, step: Timestep, field: np.ndarray, var_name: str,
                         var_attrs: dict = None, endpoint: str = None,
                         profile: str = None) -> str:
    """Write one ``{YYYYMMDDTHH}.zarr`` store holding one surface variable.

    Args:
        dest (str): destination prefix (local dir or ``s3://bucket/prefix``).
        step (Timestep): the output step (supplies date/iteration/source).
        field (np.ndarray): (13, FS, FS) surface field.
        var_name (str): variable name in the store, e.g. 'Theta'.
        var_attrs (dict, optional): variable attributes.
        endpoint, profile: see `s3_filesystem`.

    Returns:
        str: the store URL.
    """
    url = f"{str(dest).rstrip('/')}/{step.store_name}"
    root = open_zarr_group(url, mode='w', endpoint=endpoint, profile=profile)
    root.attrs.update({
        'model': 'LLC4320_v2',
        'selected_iteration': int(step.iteration),
        'selected_date_utc': step.date.strftime('%Y-%m-%d %H:%M:%S'),
        'source_folder': os.path.basename(step.folder),
        'source_file': os.path.basename(step.path),
        'FS': int(field.shape[1]),
        'complete': False,
    })
    write_surface_variable(root, var_name, field, attrs=var_attrs)
    root.attrs['complete'] = True
    return url


SST_ATTRS = {'long_name': 'sea surface temperature (Theta, k=0)',
             'units': 'degC', 'source_field': 'Theta', 'level': 0}


def extract_sst(out_dir: str, mask_dir: str, dest: str, FS: int = 4320,
                start: datetime = None, end: datetime = None, limit: int = None,
                skip_existing: bool = True, dry_run: bool = False,
                write_grid: bool = True, endpoint: str = None,
                profile: str = None) -> dict:
    """End-to-end: discover Theta output, decode SST, write one Zarr store per hour.

    Idempotent (like PAB's ``s3_push.py``): stores already marked complete
    are skipped, so an interrupted run can simply be restarted, and a
    re-run after new output lands on disk only processes the new hours.

    Args:
        out_dir (str): raw-output parent, e.g.
            ``/nobackupp27/dbwhitt/llc_4320/OUT``.
        mask_dir (str): directory holding hFacC.bits.
        dest (str): ``s3://llc4320-v2/SURFACE`` or a local directory.
        FS (int, optional): facet side. Defaults to 4320.
        start, end (datetime, optional): date window (see
            `discover_timesteps`).
        limit (int, optional): process at most this many steps (testing).
        skip_existing (bool, optional): skip complete stores. Defaults to True.
        dry_run (bool, optional): only list what would be done.
        write_grid (bool, optional): also write/refresh ``grid.zarr``.
        endpoint, profile: see `s3_filesystem`.

    Returns:
        dict: counts -- {'discovered', 'written', 'skipped'} -- plus
            'dt_seconds' (inferred model timestep, or None).
    """
    steps = discover_timesteps(out_dir, 'Theta', start=start, end=end)
    if limit is not None:
        steps = steps[:limit]
    dt_sec = infer_timestep_seconds(steps)
    logger.info("Discovered %d Theta steps under %s (inferred model dt = %s s)",
                len(steps), out_dir, dt_sec)
    stats = {'discovered': len(steps), 'written': 0, 'skipped': 0,
             'dt_seconds': dt_sec}

    if dry_run:
        for s in steps:
            logger.info("would write %s/%s  <- %s", dest, s.store_name, s.path)
        return stats

    if write_grid:
        write_grid_store(dest, mask_dir, FS, endpoint, profile,
                         skip_existing=skip_existing)

    for s in steps:
        url = f"{str(dest).rstrip('/')}/{s.store_name}"
        if skip_existing and store_is_complete(url, 'Theta', endpoint, profile):
            logger.info("complete, skipping: %s", url)
            stats['skipped'] += 1
            continue
        logger.info("reading %s (iteration %d, %s)", s.path, s.iteration,
                    s.date.isoformat())
        sst = read_shrunk_field(s.folder, mask_dir, 'Theta', s.iteration, FS,
                                level=0, dry_value=np.nan)
        write_timestep_store(dest, s, sst, 'Theta', SST_ATTRS, endpoint, profile)
        logger.info("wrote %s", url)
        stats['written'] += 1
    return stats
