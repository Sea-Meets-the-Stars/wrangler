""" Back-of-envelope storage estimate for global hourly LLC4320 v2 SST in Zarr.

Written to answer the "how big is the SST-only pass?" question in
claude_prompts/llc4320_v2.md (Q&A round of 2026-09-03).  Run with::

    conda run -n ocean14 python wrangler/scripts/llc_v2_sst_volume.py
"""

FS = 4320
N_FACETS = 13
BYTES_PER_VALUE = 4          # float32
HOURS_PER_YEAR = 365 * 24
# Fraction of LLC grid points that are wet at the surface.  The LLC4320
# native grid has ~29% land/masked points (the Arctic cap and most of the
# 13 facets are ocean), so use 0.7 wet.
WET_FRACTION = 0.70
# Zarr's default zstd codec on float32 SST with NaN land: land chunks
# compress to ~nothing; ocean chunks compress modestly.  1.3x-2.0x on the
# wet points is typical for float32 temperature fields; take 1.5x.
OCEAN_COMPRESSION = 1.5


def estimate():
    """Return a dict of sizes in GB for one field-hour, one day, one year."""
    npts = N_FACETS * FS * FS
    raw_bytes = npts * BYTES_PER_VALUE                   # uncompressed, full grid
    shrunk_bytes = npts * WET_FRACTION * BYTES_PER_VALUE  # what's on Pleiades disk
    zarr_bytes = shrunk_bytes / OCEAN_COMPRESSION         # what lands in the bucket
    gb = 1e9
    return {
        'grid_points': npts,
        'raw_full_grid_GB_per_hour': raw_bytes / gb,
        'shrunk_on_disk_GB_per_hour': shrunk_bytes / gb,
        'zarr_GB_per_hour': zarr_bytes / gb,
        'zarr_GB_per_day': zarr_bytes * 24 / gb,
        'zarr_TB_per_year': zarr_bytes * HOURS_PER_YEAR / gb / 1e3,
        'stores_per_year': HOURS_PER_YEAR,
        'chunks_per_store': N_FACETS * (FS // 720) ** 2,
        'objects_per_year': HOURS_PER_YEAR * (N_FACETS * (FS // 720) ** 2 + 4),
    }


if __name__ == '__main__':
    for k, v in estimate().items():
        print(f"{k:32s} {v:,.3f}" if isinstance(v, float) else f"{k:32s} {v:,}")
