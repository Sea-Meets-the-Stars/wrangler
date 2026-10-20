""" Storage / object-count / runtime estimate for the LLC4320 v2 surface extraction.

Updated 2026-09-24 with the numbers measured on Pleiades (see the "On Pleiades"
section of claude_prompts/llc4320_v2.md) instead of the earlier guesses:

* wet fraction 0.581 (was assumed 0.70) -- from the first real SST store's
  per-face statistics, global `wet fraction 0.581`.
* 9,692 hourly stores -- the 2026-10-08 restart's `Discovered 9692 hourly SST
  steps`, i.e. 2023-01-01 01:00 through 2024-02-08 20:00 with no gaps (that is
  exactly the number of hours in that closed interval).  It was 9,503 while the
  final folder was unreadable.

Run with::

    conda run -n ocean14 python wrangler/scripts/llc_v2_sst_volume.py
"""

FS = 4320
N_FACETS = 13
BYTES_PER_VALUE = 4            # float32
CHUNK = 720                    # (face, j, i) = (1, 720, 720)

# --- measured on Pleiades ---------------------------------------------------
WET_FRACTION = 0.581           # finite (ocean) points; land is NaN
# 2026-10-08: the last OUT folder (2024_01_31_230000_to_2024_02_08_200000, a
# symlink into the live run) became readable and added exactly its full span of
# 189 h, so the record now runs to 2024-02-08T20 and the earlier 9,503 stands
# only for the period before that folder opened up.
N_STORES = 9692                # hourly stores, 2023-01-01T01 .. 2024-02-08T20
# Compression, MEASURED 2026-09-25 from three consecutive stores in the live
# run (473.5 MB each, 419 objects each): 563.8 MB of wet float32 per store
# lands as 473.5 MB, i.e. 1.19x -- zstd does much less well on SST than the
# 1.5x first assumed. All-land chunks are not written at all, which is why a
# store holds 419 objects rather than the 468 chunks + 26 coord/metadata
# objects a fully populated array would need.
OCEAN_COMPRESSION = 1.19
MEASURED_STORE_MB = 473.5          # one field, one hour
MEASURED_OBJECTS_PER_STORE = 419   # one field
MEASURED_SEC_PER_STORE = 78.0      # observed 74.6 overall / 82.4 recent

# Throughputs to bracket the wall-clock estimate. The pipeline writes each
# chunk and reads it back to verify, so bytes on the wire are ~2x the stored
# size.
THROUGHPUTS_MB_S = (10, 25, 50, 100)


def expected_store_count(first='2023-01-01 01:00', last='2024-02-08 20:00') -> int:
    """Hourly stores in the closed interval [*first*, *last*].

    Used to check the dry run's ``discovered=`` against the span the folder
    names cover: if the two agree, the position-based dating produced no
    gaps and no duplicated hours anywhere in the tree.
    """
    from datetime import datetime
    fmt = '%Y-%m-%d %H:%M'
    t0 = datetime.strptime(first, fmt)
    t1 = datetime.strptime(last, fmt)
    return int((t1 - t0).total_seconds() // 3600) + 1


def estimate(n_fields: int = 1) -> dict:
    """Storage, object count and runtime for *n_fields* surface fields."""
    npts = N_FACETS * FS * FS
    raw_bytes = npts * BYTES_PER_VALUE                      # full grid, uncompressed
    wet_bytes = npts * WET_FRACTION * BYTES_PER_VALUE       # actual data
    store_bytes = wet_bytes / OCEAN_COMPRESSION             # what lands in the bucket
    chunks_per_var = N_FACETS * (FS // CHUNK) ** 2
    # Measured: 419 objects for one field, of which 25 are coordinate chunks
    # and metadata documents shared by every variable in the store.
    objects_per_store = n_fields * (MEASURED_OBJECTS_PER_STORE - 25) + 25

    total_bytes = store_bytes * n_fields * N_STORES
    wire_bytes = 2 * total_bytes                            # write + verify read
    gb = 1e9
    out = {
        'n_fields': n_fields,
        'grid_points': npts,
        'raw_full_grid_GB_per_field_hour': raw_bytes / gb,
        'stored_GB_per_field_hour': store_bytes / gb,
        'stored_GB_per_store': store_bytes * n_fields / gb,
        'stored_TB_total': total_bytes / gb / 1e3,
        'read_from_lustre_TB': raw_bytes * n_fields * N_STORES / gb / 1e3,
        'days_at_measured_rate': n_fields * MEASURED_SEC_PER_STORE * N_STORES / 86400,
        'chunks_per_variable': chunks_per_var,
        'objects_per_store': objects_per_store,
        'objects_total': objects_per_store * N_STORES,
    }
    for mb_s in THROUGHPUTS_MB_S:
        out[f'days_at_{mb_s}MB_s'] = wire_bytes / (mb_s * 1e6) / 86400
    return out


if __name__ == '__main__':
    n_expected = expected_store_count()
    print(f"hours in [2023-01-01 01:00, 2024-02-08 20:00] = {n_expected:,}   "
          f"dry run discovered = {N_STORES:,}   "
          f"{'MATCH (no gaps, no duplicate hours)' if n_expected == N_STORES else 'MISMATCH'}")
    for n in (1, 4, 12):
        label = {1: 'SST only', 4: 'SST+SSS+SSU+SSV', 12: 'all 12 surface fields'}[n]
        print(f"\n=== {label} ({n} field{'s' if n > 1 else ''}), {N_STORES} hourly stores ===")
        for k, v in estimate(n).items():
            print(f"{k:34s} {v:,.3f}" if isinstance(v, float) else f"{k:34s} {v:,}")
