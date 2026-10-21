# LLC4320 v2 surface fields -- design

**Date:** 2026-10-20 (revised the same day after the full inventory of the
raw tree, section 2.1). **Status:** agreed design, not yet implemented.
**Audience:** Xavier, and anyone (person or Claude session) who implements
or runs this later. It stands on its own; the discussion that produced it is
in `llc4320_v2_other_fields.md` (Q&A 1-13) and the history of the SST
product in `llc4320_v2.md` (its `## Logs`).

Everything in sections 1-3 describes code and data as they exist today
(`wrangler/ogcm/llc_v2.py`, `llc_v2_access.py`, `wrangler/scripts/llc_v2_*.py`,
`bin/wr_llc_v2_*`, `docs/howto/llc4320_v2.rst`). Sections 4-11 are the
design for what comes next. Where a number is measured it says so; where it
is inferred it says so.

---

## 1. Summary

The public product `s3://llc4320-v2/SURFACE` holds hourly global sea surface
temperature (`Theta`) from Dan Whitt's LLC4320 v2 run, 9,501 Zarr stores
over 2023-01-01 01:00 to 2024-01-31 23:00 UTC, plus `grid.zarr`. This design
adds every other 2D surface field the run wrote -- **twenty of them**:
`SSS`, `SSU`, `SSV`, `Eta`, `KPPhbl`, `PhiBot`, six surface fluxes, two
ice-shelf fluxes and six sea-ice fields (section 4.1) -- **into the same
hourly stores**, without rewriting the `Theta` already there, in **one pass
over the record run as several processes over disjoint date windows**, and
closes the 191 missing SST hours.

Two things the 2026-10-20 inventory changed: the raw tree now has **95
folders, not 83**, so the record is probably longer than the 9,692 hours
SST was planned for (section 2.1; needs one paste to settle), and twenty
fields rather than fifteen put **every storage estimate above the 55 TB
accepted so far** (section 8). Both are decisions for Xavier before launch.

Decisions already taken (Q&A 1-13):

| decision | answer |
|---|---|
| Preserve existing data when adding fields | yes: open stores `mode='a'`, create only absent arrays |
| Which fields | all 2D fields present in every raw folder -- 20 besides SST (section 4.1); one pass |
| Storage ceiling | 55 TB was accepted; **every estimate for 20 fields exceeds it** (section 8) -- quota or field list to be decided |
| Parallelism | yes; latency-bound, so more in-flight requests per process *and* several processes |
| `SSU`/`SSV` | native face-relative components only; rotation documented, not stored |
| SST gaps | the two outage hours now; the 189-hour block when Dan opens the folder |
| Extent of new fields | the full record, same as SST |
| Layout | one store per hour holding all variables |
| Land mask | the k=0 `hFac*.bits` masks, chosen by each variable's grid location |
| Read-back verification | `--verify {full,sample,none}`, default `full` |
| Launch | benchmark first (40 min), then launch with the worker count it indicates |

---

## 2. The raw source on Pleiades

**Location.** `/nobackupp27/dbwhitt/llc_4320/OUT`, folders named
`YYYY_MM_DD_HHMMSS_to_YYYY_MM_DD_HHMMSS` (the simulated interval each run
segment covers), e.g. `2023_01_01_000000_to_2023_01_08_060000`. There were
83 readable folders when SST was extracted (84 once the live-run symlink
became listable); **the 2026-10-20 inventory counts 95** (section 2.1). Each holds
the segment's MITgcm namelists (`data`, `data.cal`, `data.diagnostics`,
`data.seaice`, `data.shelfice`, ...), `STDOUT.*`, and one file per output
hour per field. The last folder, `2024_01_31_230000_to_2024_02_08_200000`,
is a symlink into the still-active run directory
`MITgcm/run_sal_90x90x19493/run_sep32026`; it lists but its data files are
not readable by us (189 hours).

**Files.** `<FIELD>.<10-digit iteration>.data`: `SST`, `SSS`, `SSU`, `SSV`
(the k=0 level of Theta, Salt, U, V -- Dan Whitt, 2026-09-09) and the 2D
diagnostics `Eta`, `KPPhbl`, `PhiBot`, `oceQnet`, `oceQsw`, `oceFWflx`,
`oceSflux`, `oceTAUX`, `oceTAUY`, `SHIfwFlx`, `SHIhtFlx`, `SIarea`,
`SIheff`, `SIhsalt`, `SIhsnow`, `SIuice`, `SIvice` -- **21 prefixes, each
present in all 95 folders** (inventory of 2026-10-20). Every one is
**uncompressed MITgcm "compact" format**: 4320 x 56160 big-endian `real*4`,
970,444,800 bytes, no header, no `.meta` file (confirmed 2026-10-20:
`ls $F1 | grep -c '\.meta$'` prints 0). The compressed `.shrunk` +
`hFac*.bits` format exists only for the 3D fields and is irrelevant here
(the pure-Python decoder in `llc_v2.py` is kept for a later depth phase).

### 2.1 What the 2026-10-20 inventory showed

The `(field, ext) -> number of folders` block of `wr_llc_v2_inventory $OUT`
(pasted in `llc4320_v2_other_fields.md`, prompt 3) counts **95 folders**
holding every one of the 21 prefixes above, against 83 (plus one unlistable)
on 2026-09-24. The paste did not include the tool's first lines -- the
folder count with the first and last dates, and any unreadable folders --
so **from here I cannot tell what the extra folders are**. The likely
reading (inference): Dan's run has continued past 2024-02-08 and new
segments have landed in `OUT`, roughly eleven of them. At the 4-7 simulated
days per folder seen so far that would be of order 50-60 more days, i.e.
**~1,300-1,500 more hours, a record of roughly 11,000 hours** rather than
9,692. Alternatives that would also give 95: a different `$OUT`, or
re-run/duplicate segments overlapping dates already covered (which
`discover_timesteps` would report as `Duplicate dates across folders`).

What it affects: the 9,692-hour span and its count-equals-span proof must
be re-established for the new tree (the dry run does it: the discovered
count must equal the hours in the closed interval first-to-last); the SST
record itself would need extending before SST is "finished"; every storage
and runtime figure in section 8 scales with the hour count. It does not
affect the design of the code. The one paste that settles it:

```bash
wr_llc_v2_inventory $OUT --quiet | head -4          # folders, first..last date, unreadable
wr_llc_v2_surface $OUT /tmp/x --fields SST --dry-run 2>&1 | grep -E "Discovered|Duplicate|spans .* but holds|unreadable"
```

**Other file families in the tree, all out of scope for the one-pass run**
(what they probably are is inference from the names; none is a global
hourly 2D surface field present throughout the record):

| files | folders | probable meaning |
|---|---|---|
| `R_Eta`, `R_SIarea`, `R_SIheff`, `R_SIhsalt`, `R_SIhsnow`, `R_SIuice`, `R_SIvice`, `R_Salt`, `R_Theta`, `R_U`, `R_V` (`R_W`, `R_SaltF`, `R_ThetaF` once) | 5 | a second output stream of the same fields with an `R_` prefix -- restart/pickup-time snapshots or a regional or re-run subset; only 5 segments have it |
| `EtaF`, `SIheffF`, `SaltF`, `ThetaF`, `SaltFF`, `ThetaFF` | 1-4 | variants written at a different frequency or filtered ("F") during a few segments |
| `Eta_tide` (57), `Eta_tide_presal` (29) | 57 / 29 | tidal sea-surface-height diagnostics; `presal` presumably from before the salinity-related change that names the run directory `run_sal_*`; together they span 86 of 95 folders, so not a complete record either |
| `Theta/Salt/U/V/W.shrunk` (39-40), `*.every5.shrunk` (11), `*.sections5deg` (11), `*.bloat` (13-16), `W.level1` (1) | partial | the compressed 3D output and derived products (every 5th level, 5-degree sections, decompressed "bloat" copies); the depth phase, not this one |

The run-time field resolution (section 4.3) ignores all of these because
they are not present in every folder; nothing needs to exclude them by name.

**Compact layout.** The 13 x 4320 x 4320 points are stored as five facets
back to back: facets 1-2 (faces 0-5) each `(3*4320, 4320)`, facet 3 (face 6,
Arctic) `(4320, 4320)`, facets 4-5 (faces 7-9, 10-12) each `(4320, 3*4320)`.
A plain reshape to `(13, 4320, 4320)` is right for faces 0-6 and scrambles
faces 7-12; `llc_v2.compact_to_faces` does it correctly (section 3.2).

**Dating rule** (Dan Whitt, 2026-09-09; implemented in
`discover_timesteps`): hourly snapshots; the n-th file of a folder (0-based,
sorted by iteration) is at `folder start + (n + 1) h`; the last file is at
the folder's end time; no hour appears in two folders. The code requires
`nfiles == span in hours` per folder and warns otherwise. Proof it is right:
the discovered count, 9,692, equals the number of hours in the closed
interval 2023-01-01 01:00 to 2024-02-08 20:00 exactly, with no duplicate
dates across 83 folders.

**`deltaT` and `nIter0`** vary per segment (`deltaT` observed as 5, 8, 12,
18, 20 s), so iteration numbers cannot date files across folders. They are
parsed from `data` or `STDOUT.0000` (`read_run_params`) only to cross-check
the position rule; the check is advisory and one folder
(`2023_02_28_150000_to_2023_03_03_170000`) logs `UNMATCHED`, most likely a
namelist overwritten by a restart. The `selected_iteration` stored in each
Zarr store is therefore not comparable across segments; use the timestamps.

**Masks.** `/nobackupp29/bcnelson/MITgcm/compressedOutput/work/bitmask.4320x173/hFac{C,S,W}.bits`:
one bit per (level, point), LSB-first within each byte, 30,326,400 bytes per
level, 173 levels (5,246,467,200 bytes each). Level 0 gives the surface
wet masks at cell centres (C), west faces (W, U points) and south faces (S,
V points). The bits are in compact order and pass through
`compact_to_faces` too.

**Grid.** `/nobackupp27/dbwhitt/llc_4320/grid_90x90x19493`: fifteen 2D
compact files (`XC YC XG YG RAC RAS RAW RAZ DXC DYC DXG DYG Depth AngleCS
AngleSN`, each 970,444,800 bytes), 3D `hFacC/S/W.data` (173 levels), and 1D
`RC RF DRC DRF`. The horizontal grid is identical to v1's; bathymetry,
`hFac`, land mask and vertical grid are v2's own.

---

## 3. How SST was extracted (the pipeline as it stands)

`wr_llc_v2_surface $OUT s3://llc4320-v2/SURFACE --fields SST --mask-dir
$MASKDIR --grid-dir $GRIDDIR` (`llc_v2.extract_surface`). Run from a
Pleiades front end as a `nohup`/`setsid` process (a PBS script exists at
`wrangler/ogcm/pbs/llc_v2_surface.pbs` but was not needed once the crashes
were understood).

### 3.1 Reading

`read_data_field(path, FS=4320)`: size-check the file against one compact
level, `np.frombuffer(..., '>f4')`, cast to native float32,
`compact_to_faces`. Result `(13, 4320, 4320)` float32, 970 MB, 0.0 over
land (MITgcm's convention).

### 3.2 Face order -- the bug and its repair

Until 2026-10-02 the reader reshaped the compact stream directly to
`(13, 4320, 4320)`, which left faces 7-12 of every store, and of
`grid.zarr`, scrambled (each stored "face" interleaved rows of a rotated
facet). The land masks were reshaped the same way, so NaN patterns, per-face
statistics and binned global maps all looked right; a native-resolution
Gulf Stream zoom exposed it. The values were only permuted, so nothing was
lost.

* **Reader**: `compact_to_faces` applied at every reshape site; verified
  against `xmitgcm.utils._reshape_llc_data` and against v1's `grid.zarr`
  (XC/YC of faces 7-12 identical after the fix).
* **Marker**: stores written since carry the group attribute
  `face_layout = "llc_faces"`; `llc_v2.stored_faces(group, var)` reads old
  and new stores correctly.
* **Repair**: `wr_llc_v2_fix_faces` (`repair_store_faces`) rewrote faces
  7-12 of the 5,492 old stores in place on 2026-10-09 (6 workers, 7.48 h,
  ~5 s/store from the workstation). It infers each store's layout from its
  own NaN pattern against the true-layout mask (`classify_face_layout`)
  before touching it, backs up the faces locally, writes, reads back,
  marks. Verified afterwards: every store on the bucket has the attribute,
  NaN patterns match `grid.zarr/maskC` at 100.000%, and median row-to-row
  SST steps on faces 7-12 fell from 27.5 degC to 0.003-0.009 degC.

### 3.3 Masking and NaN

`apply_land_mask(field, wet, prefix)`: with `--mask-dir`, land is `~wet`
where `wet` is the k=0 bit mask (`surface_wet_mask`) -> NaN. Without a mask
dir, exact zeros are treated as land -- wrong for the sea-ice fields, which
is why `ZERO_IS_NOT_LAND` exists and `--mask-dir` is required from now on.
SST used the mask. Known artefact: **3,477 wet points per snapshot
(0.0025%) are exactly 0.0** -- cells the bit mask calls wet that the run
wrote as 0 (probably ice-shelf cavities, `data.shelfice` is active, or a
small bathymetry difference). Documented in the HOWTO; not worth a re-run.

Current mask selection (`mask_file_for_field`) picks `hFacW` only for a
variable literally named `U` and `hFacS` only for `V`; everything else gets
`hFacC`. Section 5.2 changes this.

### 3.4 Store layout, chunking, compression

One Zarr v3 group per hour, `{YYYYMMDD}T{HH}.zarr` (`store_name_for_date`),
under `s3://llc4320-v2/SURFACE/`. Written through `zarr.storage.FsspecStore`
over an asynchronous `s3fs` filesystem with path-style addressing,
`signature_version s3v4`, no payload signing (`s3_filesystem`); credentials
from `AWS_PROFILE`/`~/.aws/credentials`, endpoint from `ENDPOINT_URL`
(default `https://s3-west.nrp-nautilus.io`).

| item | value |
|---|---|
| variable | `Theta`, `(face, j, i)`, float32, fill `NaN` |
| chunks | `(1, 720, 720)` -> 13 x 36 = 468 chunks per variable |
| codecs | zarr 3 defaults (bytes + zstd); measured ratio 1.19x on ocean points |
| objects | all-NaN (land) chunks are never written: 419 objects per store before the face fix, ~384 after (more chunks are all land in true order) |
| size | 473.5 MB per store measured pre-fix, ~445 MB after; 4.4 TB for 9,501 stores |
| coordinates | `face` (13), `j`, `i` (4320) as int64 with COMODO `axis` attrs; `j_g`/`i_g` only when a staggered variable needs them |
| group attrs | `model`, `selected_iteration`, `selected_date_utc`, `source_folder`, `FS`, `variables`, `face_layout`, `complete` |
| variable attrs | `long_name`, `units`, `source_file_prefix` (`SST`), `source_file`, `level` (0) |

Store names, chunking and attribute names follow dbof's
`LLC4320_RAW/SURFACE` convention so v2 reads like v1.

### 3.5 `grid.zarr` and C-grid staggering

`write_grid_store` reads the 23 grid variables (15 2D files under their
xmitgcm names -- `RAC`->`rA`, `AngleCS`->`CS`, ... -- level 0 of
`hFacC/S/W` as `hFacC_k0` etc., `RC/RF/DRC/DRF` on `k`/`k_p1`, and `maskC`
from `hFacC.bits`). 4.8 GB.

LLC4320 is an Arakawa C-grid: tracers at centres `(j, i)`, U on west faces
`(j, i_g)`, V on south faces `(j_g, i)`, vorticity at corners `(j_g, i_g)`.
Until 2026-10-20 the writer hard-coded `(face, j, i)` for everything, so 11
of 19 2D grid variables were mislabelled and `xgcm.Grid` could not be built.
Fixed: `VAR_HDIMS` maps variable -> horizontal dims (matching what v1
declares), `write_surface_variable` uses it, `_ensure_coords` writes
`c_grid_axis_shift = -0.5` on `j_g`/`i_g` (MITgcm's west/south convention),
and `wr_llc_v2_fix_grid_dims` (`repair_store_dims`, metadata only) repaired
the published grid. `xgcm.Grid` now builds from the public store. `VAR_HDIMS`
already lists `U`, `oceTAUX`, `SIuice` as west-face and `V`, `oceTAUY`,
`SIvice` as south-face, so the new fields will be labelled correctly.

### 3.6 Consolidated metadata

`consolidate_store` writes zarr consolidated metadata after every store so a
reader opens it in one request. It must use a *synchronous* fsspec mapper:
the asynchronous store's listing comes back empty and would record zero
members (a bug found and fixed 2026-10-09; a guard refuses to write a
zero-member document). All 9,498 complete stores were backfilled on
2026-10-14.

### 3.7 Completeness, verification, retries

* `write_surface_variable` writes each face and **reads it back**, comparing
  with `np.array_equal(..., equal_nan=True)`; a mismatch raises.
* The group is created with `complete=False` and set `True` only after every
  requested variable is written and verified; `store_is_complete(url, vars)`
  requires `complete` and the presence of each variable. Runs are idempotent:
  complete stores are skipped, incomplete ones rewritten.
* A store's write is retried up to 4 times with 10 s doubling backoff; a
  store that still fails is logged, counted (`failed=`) and the run moves
  on (added 2026-10-09 after a single transient Ceph 403 killed the run
  three times).
* A requested field missing on disk leaves the store `complete=False`
  (`incomplete=`), so a later run revisits it.

### 3.8 Public access and the reader

The bucket policy (`docs/nautilus/s3_llc4320v2_policy.json`) grants
anonymous `GetObject` and `ListBucket`, nothing else; the original
dbof-derived policy had granted the world `PutObject` and `DeleteObject`.
`wr_llc_v2_check_access` asserts read/list pass and put/delete fail.
`wrangler.ogcm.llc_v2_access` provides `open_grid`, `open_store`,
`open_range(start, end, variables=...)` (a lazy dataset with a `time` axis,
store names generated from the cadence so no listing is needed, missing
hours skipped with a warning), `nearest_index`, `extract_point`,
`region_slices`. The HOWTO is `docs/howto/llc4320_v2.rst`, live at
`https://wrangler.readthedocs.io/en/llc4320_v2/howto/llc4320_v2.html`.

### 3.9 State of the record and measured performance

9,501 stores complete, each `variables=['Theta']`, `face_layout` set,
consolidated. Missing: `20231019T03` (absent) and `20231228T23` (group
exists, no array) -- both lost in the Nautilus outage of 2026-10-12 -- and
`20240201T00`..`20240208T20` (189 hours, unreadable raw files).

Measured throughput of the SST run (object timestamps): 74.6 s/store over
the first 1,028 stores, 82.4 s/store over a later 24, range 46-116 s; a
burst of five stores at ~30 s each (2026-09-30); degradation to 133 s/store
while another client ran six workers against the endpoint. 473.5 MB
PUT + 473.5 MB GET per store = 11 MB/s on the wire. **Interpretation**
(section 5.3): the loop is latency-bound on Nautilus round trips -- at most
10 of ~419 PUTs in flight, GETs never overlapping PUTs -- with CPU and
Lustre inferred at ~10 s per store per field.

---

## 4. The new fields

### 4.1 Field table

Raw prefix -> store variable (xmitgcm/dbof names, so v2 reads like v1),
units, grid location (and hence mask and dims), compressibility class
(section 8) and time sampling. All 21 prefixes are present in all 95
folders (inventory 2026-10-20). The store variables `SHIfwFlx`, `SHIhtFlx`
and `SIhsalt` are not yet in `SURFACE_FIELDS` and are added by this work.

| raw prefix | variable | units | location | dims | mask | class | sampling |
|---|---|---|---|---|---|---|---|
| `SST` | `Theta` | degC | centre | `(face, j, i)` | hFacC | dense | snapshot (done) |
| `SSS` | `Salt` | psu | centre | `(face, j, i)` | hFacC | dense | snapshot |
| `SSU` | `U` | m s-1 | west face | `(face, j, i_g)` | hFacW | dense | snapshot |
| `SSV` | `V` | m s-1 | south face | `(face, j_g, i)` | hFacS | dense | snapshot |
| `Eta` | `Eta` | m | centre | `(face, j, i)` | hFacC | dense | snapshot |
| `KPPhbl` | `KPPhbl` | m | centre | `(face, j, i)` | hFacC | dense | see 4.2 |
| `PhiBot` | `PhiBot` | m2 s-2 | centre | `(face, j, i)` | hFacC | dense | see 4.2 |
| `oceQnet` | `oceQnet` | W m-2 | centre | `(face, j, i)` | hFacC | dense | see 4.2 |
| `oceQsw` | `oceQsw` | W m-2 | centre | `(face, j, i)` | hFacC | dense | see 4.2 |
| `oceFWflx` | `oceFWflx` | kg m-2 s-1 | centre | `(face, j, i)` | hFacC | dense | see 4.2 |
| `oceSflux` | `oceSflux` | g m-2 s-1 | centre | `(face, j, i)` | hFacC | dense | see 4.2 |
| `oceTAUX` | `oceTAUX` | N m-2 | west face | `(face, j, i_g)` | hFacW | dense | see 4.2 |
| `oceTAUY` | `oceTAUY` | N m-2 | south face | `(face, j_g, i)` | hFacS | dense | see 4.2 |
| `SHIfwFlx` | `SHIfwFlx` | kg m-2 s-1 | centre | `(face, j, i)` | hFacC | sparse | see 4.2 |
| `SHIhtFlx` | `SHIhtFlx` | W m-2 | centre | `(face, j, i)` | hFacC | sparse | see 4.2 |
| `SIarea` | `SIarea` | 1 | centre | `(face, j, i)` | hFacC | sparse | see 4.2 |
| `SIheff` | `SIheff` | m | centre | `(face, j, i)` | hFacC | sparse | see 4.2 |
| `SIhsnow` | `SIhsnow` | m | centre | `(face, j, i)` | hFacC | sparse | see 4.2 |
| `SIhsalt` | `SIhsalt` | g m-2 | centre | `(face, j, i)` | hFacC | sparse | see 4.2 |
| `SIuice` | `SIuice` | m s-1 | west face | `(face, j, i_g)` | hFacW | sparse | see 4.2 |
| `SIvice` | `SIvice` | m s-1 | south face | `(face, j_g, i)` | hFacS | sparse | see 4.2 |

"Dense" means non-zero over the whole ocean and expected to compress like
SST (1.19x, measured); "sparse" means zero except under sea ice
(`SI*`) or ice shelves (`SHI*`, `pkg/shelfice` fluxes) and expected to
compress far better -- inference, measured by benchmark run D.
`ZERO_IS_NOT_LAND` grows to cover all eight sparse fields.

`SSU`/`SSV` are the model's face-relative components on the C-grid faces.
Eastward/northward velocity is obtained by interpolating both to cell
centres (`xgcm` `interp`) and rotating with `CS`/`SN` from `grid.zarr`:
`u_east = CS*u_c - SN*v_c`, `v_north = SN*u_c + CS*v_c`. This goes in the
HOWTO as a worked example; it is not stored. `SIuice`/`SIvice` are placed on
the U/V points, which is MITgcm's C-grid sea-ice dynamics default; Dan is
asked to confirm (section 11).

### 4.2 Snapshot or hourly mean -- unresolved, handled

Dan Whitt stated "hourly snapshots of every variable". That is certain for
the state variables. `KPPhbl`, `PhiBot`, the fluxes, the ice-shelf fluxes
and the sea-ice fields are `pkg/diagnostics` output, and MITgcm averages a
diagnostic over its output interval unless its `frequency` in
`data.diagnostics` is negative. The checks so far established only that
there are **no `.meta` files** (which would have carried `timeInterval`);
the inventory of 2026-10-20 says which fields exist, not how they were
sampled. So the question is **still open**. It changes variable attributes
and documentation, not the store layout, so it does not block the code
work. The design handles it as follows:

* **Pre-launch check** (one paste from Pleiades, before the one-pass run):
  ```bash
  ls -l $F1/data.diagnostics
  grep -iE "frequency|fileName|fields\(|levels\(" $F1/data.diagnostics | head -40
  grep -iE "frequency|fileName" $F1/STDOUT.0000 | head -40     # MITgcm echoes the namelist here
  ```
  Negative `frequency` for the diagnostic feeding a file = snapshot;
  positive = mean over the interval. If `data.diagnostics` is absent or
  silent, Dan is the remaining source.
* **CLI**: `--time-mean PREFIX,...` names the raw prefixes that are hourly
  means. Those variables get `cell_methods = "time: mean (interval: 1 hour)"`
  and `time_sampling = "mean over the hour ending at selected_date_utc"`;
  all others get `time_sampling = "snapshot"`. Until the check is done the
  diagnostics would be written with `time_sampling = "snapshot per run
  owner; averaging over the output hour not independently verified"`, so the
  uncertainty is in the data, not only in a document.
* **Fix-up path**: if the answer arrives after the run, a metadata-only tool
  (`repair_store_attrs`, same shape as `repair_store_dims`) rewrites the
  attributes and re-consolidates -- seconds per store, no chunks touched.

### 4.3 The field list: settled, and still resolved at run time

The 2026-10-20 inventory settles the list: the 21 prefixes of section 4.1
are present in all 95 folders, and nothing else is (section 2.1). `KPPhbl`
and `PhiBot`, which the 2026-09-24 notes had not mentioned, are present
throughout; `SHIfwFlx`, `SHIhtFlx` and `SIhsalt` were not in
`SURFACE_FIELDS` and are added.

`--fields all` still resolves the list at run time: `llc_v2.inventory` over
the tree, keep every prefix whose `.data` files are present in **every
readable folder** and which is in `SURFACE_FIELDS`. This guards against the
tree changing again (new segments with a different diagnostic set would drop
a field with a warning rather than silently leaving stores incomplete), and
it ignores the partial-coverage families of section 2.1 without naming
them. Prefixes present everywhere but unknown to the table are logged and
skipped unless named explicitly. The resolved list is logged and stored in
each run's stats, and should equal the 21 above.

---

## 5. Design changes

### 5.1 Append mode and per-variable completeness

`write_timestep_store` opens each store with `mode='a'`
(`open_zarr_group(..., 'a')`, which creates a missing group), sets the group
attributes only if the group is new, and writes **only the requested
variables that are not already complete** (`create_array(overwrite=True)`
replaces an incomplete one). Existing arrays are never read or rewritten.

Completeness becomes per variable:

* after verification an array gets `complete = True` (plus `written_utc`);
* `attrs['variables']` becomes the **union** of what is present;
* the group's `complete` means "every variable in `attrs['variables']` is
  complete";
* `store_is_complete(url, vars)` is true when each requested variable is
  present and either carries `complete = True` or (legacy rule) the group is
  `complete` and lists it. The 9,501 existing stores satisfy the legacy rule
  untouched.

Consequence: a store missing one field on disk no longer forces the other
fields to be re-read and rewritten on each re-run, which also makes the 189
unreadable hours cost nothing until they open. `--no-skip-existing` means
"rewrite the requested arrays", not "rewrite the store".

**One-writer rule.** Two processes appending different arrays to one group
would race on the group document and the consolidated metadata. Therefore
no two processes may touch the same store; section 5.4 enforces it by
construction, and operationally: one extraction command at a time over any
date range, and no repair/consolidate tool against the bucket while an
extraction runs.

### 5.2 Mask by staggering

`wet_for(var)` chooses the mask from `var_hdims(var)`: `i_g` in the dims ->
`hFacW`, `j_g` -> `hFacS`, else `hFacC`. This corrects `oceTAUX`,
`oceTAUY`, `SIuice`, `SIvice`, which the name-based rule would have masked
with `hFacC` (leaving land-adjacent face points as 0 and giving a NaN
pattern matching no mask). `--mask-dir` becomes mandatory when any
`ZERO_IS_NOT_LAND` field is requested, and that set grows to all eight
sparse fields (`SI*`, `SHI*`), which are legitimately 0 over most of the
ocean. Masks are read once per process.

### 5.3 More requests in flight: `--concurrency` and whole-variable writes

Where the time goes (section 3.9) is waiting on S3 with at most 10
requests in flight. Two changes, both inside one process and with no effect
on store safety:

* `--concurrency N` (default 32) sets `zarr.config.set({'async.concurrency': N})`
  at process start (zarr 3.4.1; the environment variable
  `ZARR_ASYNC__CONCURRENCY` does the same and is what the benchmark uses).
* `write_surface_variable` writes the whole `(13, 4320, 4320)` array in one
  `z[:] = field` and verifies with one `z[:]` read, instead of face by face,
  so all 468 chunks are eligible to be in flight. Memory per variable in
  flight: ~1 GB raw + compressed buffers; fine for a node.

Expected effect: 85 s -> 30-45 s per store per field (inference; the
benchmark measures it). Threads of our own are not added: zarr's asyncio
already provides the I/O concurrency and compression runs in its thread
pool; the GIL would block anything else.

### 5.4 `--workers N`: disjoint windows by construction

`extract_surface` discovers the hours once, splits the list into N
contiguous blocks (`np.array_split`), and runs one process per block
(`ProcessPoolExecutor`, `spawn` context, as `wr_llc_v2_fix_faces` does).
Each worker builds its own `s3fs` filesystem and runs the sequential loop
over its block, so the per-process connection pool is independent; the
parent aggregates the stats. `grid.zarr` is handled by the parent before
the workers start (or skipped with `--no-grid`). Because the blocks
partition the discovered hours, no store can be visited by two workers.
`--start/--end` still work and are applied before the split.

Hand-launching several `wr_llc_v2_surface` commands with typed windows
remains possible but is discouraged; `--workers` exists so that disjointness
does not depend on anyone's arithmetic.

### 5.5 `--verify {full,sample,none}`

`full` (default): read the whole variable back and compare. `sample`: read
back one face, chosen as `(hour index) mod 13` so every face is exercised
across the record, 1/13 of the GET traffic. `none`: trust the PUT (a failed
PUT raises; S3 does not write partial objects). The mode is recorded in the
array attributes (`verified = "full" | "sample" | "none"`). The first
window's stores of the pass run at `full`; the benchmark says whether the
bulk should drop to `sample`.

### 5.6 Streaming per field, retries per variable

Within a store the loop is read -> mask -> write -> verify -> mark, one
field at a time, so a sixteen-field pass needs ~3 GB per process rather
than ~16 GB. The retry (4 attempts, 10 s doubling) wraps a single variable;
a variable that still fails stays incomplete, the store stays incomplete,
the run continues, and the next invocation picks it up.

### 5.7 Re-consolidation and coordinates

After the last variable of a store, the group's `variables`/`complete` are
updated and `consolidate_store` rewrites the consolidated document, as now.
`_ensure_coords` adds `j_g`/`i_g` (with COMODO attributes) to an hourly
store the first time a staggered variable lands in it; `face`, `j`, `i`
already exist.

### 5.8 `open_range` with mixed variable sets

For weeks the bucket will hold stores with different variable sets.
`xr.concat` as called today fails when a variable is present in some
datasets and not others. Change: when `variables` is not given, intersect
the variable sets of the opened stores, warn naming what was dropped, and
concatenate the intersection; when `variables` is given, behave as now
(a store lacking one is an error, or skipped under `skip_missing`).
`extract_point`'s default `variables=('Theta',)` is unchanged.

### 5.9 CLI summary (`wr_llc_v2_surface`)

| option | meaning | default |
|---|---|---|
| `--fields` | prefixes, or `all` (section 4.3) | `SST` |
| `--mask-dir` | `hFac*.bits` directory; required for `all` or any sea-ice field | -- |
| `--workers N` | processes over disjoint windows | 1 |
| `--concurrency N` | zarr `async.concurrency` per process | 32 |
| `--verify` | `full`, `sample`, `none` | `full` |
| `--time-mean` | prefixes that are hourly means (section 4.2) | none |
| `--no-grid`, `--start`, `--end`, `--limit`, `--dry-run`, `--no-skip-existing`, `--max-retries`, `--retry-wait`, `--endpoint`, `--profile` | as now | |

The summary line gains `variables=...` (the resolved list) and per-worker
counts.

---

## 6. Benchmark plan

About 40 minutes on Pleiades, no code change, writes to a throwaway prefix.
Each run covers six **different** hours (Lustre caches re-reads, which
would flatter later runs). `--fields SST --no-grid --mask-dir $MASKDIR`
throughout; dest `s3://llc4320-v2/BENCH`.

| run | setup | reading |
|---|---|---|
| A | 1 process, defaults, hours 1-6 of some day | baseline s/store; expect ~85 |
| B | 1 process, `ZARR_ASYNC__CONCURRENCY=40`, hours 7-12 | 35-45 s confirms latency-bound; equal to A means CPU (`top`: one core pinned) or per-connection bandwidth -- processes help either way |
| C | 4 processes (`setsid`/`&`), four disjoint 6-hour windows, same variable | aggregate = wall / 24 stores; near B/4 -> use 6-8 workers; near B -> egress or Nautilus is the ceiling, use 2-3 |
| D | 1 hour, `--fields all` (21 variables), to local scratch `$SCR/BENCH` | `du -s` per variable directory: compressed size per field; the eight sparse fields decide whether the total is ~65 TB or ~100 TB (section 8) |

Also, before launch: `qsub -v LIMIT=1 wrangler/ogcm/pbs/llc_v2_surface.pbs`
tells whether compute nodes reach Nautilus; if they do, the one-pass run
goes on a PBS node (`select=1:ncpus=8:model=bro`, 120 h walltime,
resubmittable since the run is idempotent), otherwise on a PFE with at most
4 workers. Afterwards `aws --endpoint-url $ENDPOINT s3 rm --recursive
s3://llc4320-v2/BENCH`. Record A-D in `llc4320_v2_other_fields.md` `## Logs`.

---

## 7. Run order

1. **Now, current code**: the two outage hours.
   ```bash
   wr_llc_v2_surface $OUT s3://llc4320-v2/SURFACE --fields SST --no-grid --mask-dir $MASKDIR --start 2023-10-19T03 --end 2023-10-19T04
   wr_llc_v2_surface $OUT s3://llc4320-v2/SURFACE --fields SST --no-grid --mask-dir $MASKDIR --start 2023-12-28T23 --end 2023-12-29T00
   ```
   `mode='w'` is harmless here (one store absent, one with no array).
   Afterwards the SST record is contiguous to 2024-01-31 23:00.
2. **Code changes and tests** (sections 5 and 10); commit, push, `git pull`
   + `pip install -e .` on Pleiades.
3. **Pre-launch pastes**: the folder count, date range and dry-run
   discovered count for the 95-folder tree (2.1); `data.diagnostics` /
   `STDOUT` frequencies (4.2); PBS egress test (6). The field list itself is
   settled (4.3).
4. **Decide storage** (section 8): confirm a quota of ~100 TB with Nautilus,
   or trim the field list, before anything is written.
5. **Benchmark** A-D; choose `--workers`, `--verify`, node type; clean up.
6. **One pass over the full record**:
   ```bash
   wr_llc_v2_surface $OUT s3://llc4320-v2/SURFACE --fields all --mask-dir $MASKDIR --no-grid \
       --workers N --concurrency 32 --verify full [--time-mean ...]
   ```
   If the tree now extends past 2024-02-08 (2.1), this same pass writes
   those new hours with **all 21 variables including SST**, so the SST
   record is extended as a by-product. Any hours whose files are still
   unreadable (the 189, if `run_sep32026` is still closed) are discovered
   but every field is "missing", so no store is written for them. Monitor
   with the log (`grep -c " wrote "`, `WARNING|Error`) and the per-hour S3
   probe used on 2026-10-14. Re-running the same command finishes anything
   that failed; it is idempotent at the variable level.
7. **Audit** from the workstation: every expected hour's group attributes
   (`variables`, `complete`, per-array `complete`), a NaN-pattern check of
   `U`/`V` against `hFacW_k0`/`hFacS_k0`, and value ranges per variable.
8. **HOWTO** update and Read the Docs; remove "Current gaps" as they close.
9. **When Dan opens `run_sep32026`**: the same command with
   `--start 2024-02-01 --end 2024-02-09` writes the 189 stores with all
   variables, SST included, in one go.

---

## 8. Storage and runtime estimates

Measured constants (SST): 0.47 GB stored per field-hour (4.59 TB per dense
field over 9,692 hours), ~400 objects per variable per store, 85 s per
store per field in one process as the code stands. Hour count: **9,692** as
known today; if the 95-folder tree extends the record to ~11,000 hours
(section 2.1, inference) multiply everything below by about 1.15.

**Storage, 20 new fields** (12 dense + 8 sparse; class per section 4.1):

| | stored | objects | basis |
|---|---|---|---|
| `Theta` (done) | 4.4 TB | 3.9 M | measured |
| 12 dense fields (`Salt`, `U`, `V`, `Eta`, `KPPhbl`, `PhiBot`, 6 `oce*`) | **~55 TB** | ~47 M | inference: same as SST per field |
| 8 sparse fields (`SI*` x 6, `SHI*` x 2) | **~1-4 TB** if they compress 10-50x (mostly zeros) ... **37 TB** if they compress like SST | ~31 M | unmeasured; run D measures it |
| total, 21 variables | **~60-65 TB likely; ~97 TB upper bound** | ~82 M | **every case exceeds the 55 TB accepted in Q3** |

So the 55 TB figure no longer fits even if the sparse fields cost nothing:
the twelve dense fields alone fill it. The choice before launch is either a
quota of ~70 TB (or ~80 TB if the record is ~11,000 hours) confirmed with
Nautilus and run D's measurement of the sparse fields, or a shorter list.
If trimming, the candidates with the least surface-ocean interest are
`PhiBot` (bottom pressure) and `KPPhbl` (mixed-layer depth), 4.6 TB each;
the sparse fields are cheap in bytes and not worth dropping for storage.

The object count matters as much as the bytes: in a latency-bound loop a
sparse field costs the same round trips as a dense one even if its bytes
are negligible. Zarr omits only all-*fill* (NaN) chunks; all-zero ocean
chunks are written (small).

**Runtime, 20 fields over 9,692 hours** (inference until the benchmark):

| configuration | s per store per field | wall clock |
|---|---|---|
| as the code stands, 1 process | 85 | ~190 days |
| `--concurrency 32`, whole-variable writes, 1 process | 40 (expected) | ~90 days |
| same, 4 workers | 40 | ~22 days |
| same, 6 workers | 40 | ~15 days |
| same, 8 workers | 40 | ~11 days |
| same, 6 workers, `--verify sample` | ~30 | ~11 days |

Lustre reads: 0.97 GB per field-hour, ~190 TB in all for 20 fields;
single-stream reads are inferred at 1-3 s each and are not the bottleneck.

---

## 9. HOWTO and documentation changes

`docs/howto/llc4320_v2.rst`: title and summary no longer "temperature
only"; at-a-glance table lists all variables with units; the layout section
gets the per-variable table of section 4.1 (location, mask); the staggering
section notes which hourly variables sit on `i_g`/`j_g`; a new "Velocity
components" section with the rotation example; "Known caveats" gains the
time-sampling statement from section 4.2 and that the exact-zero artefact
applies to every variable masked with `hFacC`; "What is here, and what may
follow" shrinks to the depth phase; "Current gaps" tracks per-field coverage
while the pass runs and is removed when done. `README.rst`'s pointer is
unchanged. `docs/nautilus/s3_llc4320v2.md` is unchanged.

---

## 10. Tests (`wrangler/tests/test_llc_v2.py`, `test_llc_v2_access.py`)

* Appending a variable to an existing store leaves the existing arrays'
  chunk objects byte-identical (compare chunk bytes before and after) and
  leaves the group attributes as a union.
* Per-variable completeness: `store_is_complete` with a partial set; the
  legacy rule on a store that has only the group-level `complete`; an
  incomplete array is rewritten, complete ones are not.
* Mask by staggering: `oceTAUX` masked with `hFacW`, `oceTAUY` with `hFacS`,
  a tracer with `hFacC`, on the synthetic tree.
* `--workers`: the window split partitions the discovered hours (every hour
  exactly once, blocks contiguous), and a 2-worker local run produces the
  same stores as 1 worker.
* `--verify` modes record the attribute; `sample` reads exactly one face
  (mock the read).
* `--fields all` resolves from a synthetic inventory with one field missing
  from one folder (dropped with a warning) and one unknown prefix (skipped).
* `--time-mean` sets `cell_methods`/`time_sampling`.
* Retry per variable: a failing variable leaves the others complete and the
  store incomplete; a second run completes it.
* `open_range` over a mixed-variable range without `variables` returns the
  intersection with a warning; with `variables` behaves as before.
* Existing tests (38 + 8) keep passing; run in both `dbof` and `ocean14`.

---

## 11. Risks and open items

| item | status / mitigation |
|---|---|
| **Storage exceeds 55 TB in every estimate** (~60-65 TB likely, ~97 TB upper bound, more if the record is longer) | decision before launch: confirm ~70-80 TB with Nautilus, or trim (`PhiBot`, `KPPhbl` first); run D measures the sparse fields |
| **95 folders, not 83** -- the record's extent is unknown from here | one paste (section 2.1) gives the span and the dry-run count; the design is unchanged, the numbers scale |
| **Snapshot vs. mean for the diagnostics** | still open (section 4.2); pre-launch check; attributes carry the uncertainty; metadata-only fix-up afterwards if needed |
| **`SIuice`/`SIvice` grid location** (C-grid U/V points assumed) | ask Dan; a wrong assumption is a metadata-only fix (`repair_store_dims`) plus re-masking |
| **Endpoint throttling** (transient 403s seen 2026-10-08 under 6 external workers) | more in-flight requests raise the odds; per-variable retries with backoff are in place; if `failed=` climbs, lower `--workers`/`--concurrency` |
| **Two writers on one store** | `--workers` partitions; rule: one command at a time, no repair/consolidate tools during extraction |
| **Compute-node egress to Nautilus unknown** | `qsub -v LIMIT=1` test; fall back to the PFE with <= 4 workers |
| **Mixed variable sets during the run** | `open_range` intersection (5.8); HOWTO states per-field coverage |
| **The 189 hours** | depend on Dan relaxing permissions on `run_sep32026`; the email draft is in `llc4320_v2_other_fields.md` |
| **Exact-zero artefact** | will appear identically in every `hFacC`-masked field; documented, not fixed |
| **Field list** | settled by the 2026-10-20 inventory: 21 prefixes in all 95 folders; still resolved at run time as a guard |
| **Out-of-scope file families** (`R_*`, `*F`, `Eta_tide*`, 3D products) | partial coverage, ignored by construction; what they are is inference (section 2.1) -- worth one line in the Dan email |

Items that need Xavier before launch: the two outage-hour commands (7.1);
the pastes of 7.3 (folder span and dry-run count, diagnostic frequencies,
PBS egress); the storage decision (7.4); the benchmark (6); sending the Dan
email (adding the `SIuice`/`SIvice` grid question, the `R_*`/`Eta_tide`
meaning, and whether the run now extends past 2024-02-08).
