# LLC4320 v2 — other surface fields

Successor to `llc4320_v2.md`, which is closed. That doc covers the SST product
end to end: the raw-format reader, the extraction pipeline, the face-order bug
and its repair, the public bucket, the HOWTO and the C-grid staggering fix.
Read its `## Logs` for how anything here came to be.

## Goals

1. **Add the remaining surface fields** — `SSS`, `SSU`, `SSV` first, and the
   native 2D diagnostics after if we want them — alongside the `Theta`
   already published.
2. **Finish SST**: 191 hours are still missing of the 9,692.

## Where things stand (2026-10-20)

The published product is `s3://llc4320-v2/SURFACE`, public and read-only:

* **9,501 hourly stores**, 2023-01-01 01:00 to 2024-01-31 23:00 UTC, each one
  `Theta` on `(face, j, i)`, float32, NaN over land, chunked `(1, 720, 720)`,
  about 470 MB per store and 4.4 TB in all.
* **`grid.zarr`** with 23 variables on the correct Arakawa C-grid staggering,
  COMODO attributes present, `xgcm.Grid` builds from it.
* **Consolidated metadata** on every store, so one request opens one store.
* **191 hours still missing**:
  * `20231019T03` (absent) and `20231228T23` (present but incomplete), both
    lost to a Nautilus outage on 2026-10-12. Each needs a one-hour re-run.
  * `20240201T00` to `20240208T20` (189 hours). The raw output is in
    `2024_01_31_230000_to_2024_02_08_200000`, a symlink into a run directory
    that was still active; its `SST.*.data` files are not readable. See
    **Email Dan** below.

Code lives in `wrangler/ogcm/llc_v2.py` (extraction), `llc_v2_access.py`
(public reader), `wrangler/scripts/llc_v2_*.py` with `bin/wr_llc_v2_*`
wrappers. Docs: `docs/howto/llc4320_v2.rst`.

## Prompts

### Design

1. Read the pre-run Q&A below see my answers to the questions.  
If you have more, please ask them.  After we are done discussing,
we will generate a design document before proceeding.
We will call it `claude_prompts/llc4320_v2/llc4320_v2_design.md`.
It should include a description of how all the fields were extracted,
including SST. 
Use Fable if you can.
Log your work.

## Pre-run Q&A

**Claude (2026-10-20):** Questions before any of this is designed. The first
is much the most important.

1. **Adding a field must not rewrite what is already there.** Today
   `write_timestep_store` opens each store with `mode='w'`, which replaces the
   whole store. So running `--fields SST,SSS` over the existing record would
   destroy and rewrite 4.4 TB of perfectly good `Theta`, re-reading 9.4 TB from
   Lustre to do it — for nothing. The fix is to open `mode='a'` and create only
   the arrays that are absent, leaving `Theta` untouched; `store_is_complete`
   would then ask only whether the *requested* variables are present. Shall I
   implement that? It is the difference between adding a field in one pass over
   new data and re-doing the entire dataset each time.
>A. Yes, we wish to preserve the existing data.

2. **Which fields, and in what order?** `SSS`, `SSU`, `SSV` are the obvious
   three. The inventory also shows `Eta`, `oceQnet`, `oceQsw`, `oceFWflx`,
   `oceSflux`, `oceTAUX`, `oceTAUY` and the `SI*` sea-ice fields present in all
   83 folders. Do you want all of them eventually, and if so is there an order
   that matters to you scientifically?
>A. We want to add all of them eventually.  Any order that makes it the most efficient is best

3. **Volume and quota.** Each field costs about the same as `Theta`: 4.6 TB
   over the full 9,692 hours. Four fields is **18.4 TB**, twelve would be
   55 TB. The bucket has never had a stated quota. Can you find out what we are
   allowed before we commit to a number of fields?
>A. It is fine to use 55Tb

4. **Runtime, and whether to parallelise.** At the measured 85 s per store per
   field, `SSS`+`SSU`+`SSV` over the full record is about **26 days** in one
   process. Splitting by date range across several processes is
   straightforward — `--start`/`--end` already exist, the stores are disjoint —
   but with the `mode='a'` change above, two processes must never touch the
   same store at once. Four or six windows, run from Pleiades? Or accept the
   single long run now that it survives outages?
>A. Is the time to run I/O limited or CPU limited?  That is, will it help to 
run in multiple processes?  If so, let us do it in parallel.

   **Claude:** It is I/O-bound, and more precisely *latency*-bound on
   Nautilus round trips rather than bandwidth- or CPU-bound. Multiple
   processes will help, and so will more concurrency inside one process.
   The evidence, then what I infer, then the layout I recommend.

   *Measured* (object timestamps on the bucket; `llc4320_v2.md` logs of
   2026-09-25, 10-02 and 10-09):

   * 74.6 s/store over the first 1,028 stores, 82.4 s over a later 24,
     range 46-116 s. A store is 473.5 MB in 419 objects (~1.1 MB each).
     Every byte crosses the wire twice -- once as a PUT, once as a GET for
     the read-back verification -- so 947 MB per 85 s is **11 MB/s**.
   * On 2026-09-30 23:01-23:04 the *same process on the same node* wrote
     five full stores at **~30 s each**. Nothing changed but the network.
     So the whole cycle (Lustre read, masking, compression, PUT, read-back)
     fits in 30 s, and the usual 55 s on top is waiting on S3.
   * While the face repair ran six workers against the endpoint from the
     workstation (2026-10-08), the extraction slowed from 94 to 133
     s/store; the repair itself ran at 5 s/store with six processes against
     a ~50 s single-process estimate, i.e. ~120 MB/s aggregate. The endpoint
     sustains an order of magnitude more than one Pleiades process gets, so
     the ceiling is per-connection latency, not the bucket. It also shows
     that contention at the endpoint is real.

   *Inferred* (not timed on Pleiades): one 970 MB field from Lustre at a
   single-stream 0.3-1 GB/s is 1-3 s; `compact_to_faces`, the mask and the
   float32 cast are 2-3 s of numpy; zstd at the observed 1.19x on ~1 GB is
   2-4 s, and zarr 3 already runs the codec in a thread pool. So CPU is
   about 10 s per store per field, consistent with the 30 s floor.

   *What the code does* (`write_surface_variable`): writes one face (36
   chunks) at a time, reads that face back, then the next face. zarr's
   default `async.concurrency` is 10, so at most 10 of the 419 PUTs are ever
   in flight, the GETs never overlap the PUTs, and nothing overlaps the
   Lustre read of hour n+1 with the upload of hour n. There is no threading
   or dask of ours anywhere. 838 operations in 85 s with 10 in flight is
   about one second per round trip -- the signature of a latency-bound loop.

   *Recommended layout:*

   1. **In-process first** (cheap, no safety implications): expose zarr's
      `async.concurrency` as `--concurrency` (default 32) and write and
      verify a whole variable in one call instead of face by face, so all
      chunks can be in flight. I expect 85 s to become 30-45 s.
   2. **Processes over disjoint windows**: a `--workers N` option in
      `wr_llc_v2_surface` that splits the discovered hours into N contiguous
      windows and runs one process per window (the `ProcessPoolExecutor`
      pattern already in `wr_llc_v2_fix_faces`). Disjointness is then
      guaranteed by construction, which is how the `mode='a'` rule -- no two
      processes ever touch one store -- should be enforced, not by typed
      dates. Start at 4; go to 6-8 if the benchmark scales. Threads inside
      one process are not an alternative: zarr's asyncio already provides
      the I/O concurrency, and the GIL blocks the rest.
   3. **Where**: a PBS node (`select=1:ncpus=8:model=bro`, as in
      `wrangler/ogcm/pbs/llc_v2_surface.pbs`) if compute nodes reach
      Nautilus -- `qsub -v LIMIT=1` answers that in five minutes --
      otherwise the PFE with at most 4 processes. Budget ~3 GB per process
      per field in flight.
   4. **Rules**: one extraction command at a time over any date range
      (two passes with different field lists over overlapping dates would
      collide on the same stores); no repair or consolidate tool against
      the bucket while extraction runs; retries per variable, not per
      store.

   *Benchmark* (about 40 minutes, no code change, writes to a throwaway
   prefix, and settles it rather than guessing). Three runs of
   `wr_llc_v2_surface $OUT s3://llc4320-v2/BENCH --fields SST --no-grid
   --mask-dir $MASKDIR`, each over six *different* hours (Lustre caches
   re-reads, which would flatter the later runs):

   * **A.** one process, defaults: the baseline s/store (expect ~85).
   * **B.** one process with `ZARR_ASYNC__CONCURRENCY=40` in the environment
     (I verified zarr 3.4.1 honours that variable). B at 35-45 s confirms
     latency-bound. B equal to A means either CPU (`top` shows one core
     pinned) or per-connection bandwidth -- processes help in both cases.
   * **C.** four processes over four disjoint six-hour windows, same
     variable. Aggregate s/store is wall clock / 24. C near B/4 means scale
     to 6-8 workers. C near B means Pleiades egress or Nautilus is the
     ceiling: use 2-3 and accept the time.

   Then `aws s3 rm --recursive s3://llc4320-v2/BENCH`. For scale: the
   fifteen remaining fields over 9,692 hours at 85 s is ~140 days in one
   process as the code stands; at 40 s with six workers it is ~11 days.
   (The sea-ice fields are mostly zeros and compress almost to nothing, so
   their bytes are negligible, but their object count -- and so their
   latency cost -- is the same as any other field.)


5. **`SSU`/`SSV`: native, rotated, or both?** They come on the native
   face-relative axes, `U` on the west cell face and `V` on the south. Rotating
   to east/north needs `CS`/`SN` and an interpolation to cell centres, which
   loses the C-grid staggering and doubles the storage if we keep both. My
   inclination is to publish the native components only, document the rotation
   in the HOWTO with a worked example, and let users do it — that is lossless
   and matches what v1 does. Do you agree, or do your users want
   `u_east`/`v_north` served ready-made?
>A. I agree

6. **Do the 191 SST hours first?** Two of them are one-hour re-runs that could
   be done in minutes. The 189 depend on Dan. Doing the two now means the
   record is contiguous to 2024-01-31 before any new field starts, which keeps
   the HOWTO simple. Any reason not to?
>A. Yes, start with SST

7. **Should new fields cover the full record, or only where SST already is?**
   If the 189 hours never become available, do we publish `SSS` for the same
   9,503 hours as SST, or stop at whatever is readable and let the fields
   differ in extent? I would keep them aligned and document one coverage.
>A. Full record

8. **One store per hour, or one per field?** Keeping every field in the same
   `{YYYYMMDD}T{HH}.zarr` means a single open gives a user everything for that
   hour, and `open_range` needs no change. The alternative, a prefix per field,
   makes partial re-runs independent but doubles the request count for anyone
   wanting two fields. I favour the first; confirm.
>A. I confirm

9. **Sea-ice fields need a real land mask.** `SIarea`, `SIheff` and friends are
   legitimately 0 over open ocean, so the "exact zeros are land" fallback is
   wrong for them and `--mask-dir` becomes mandatory. Worth noting if those
   fields are wanted.
>A. We have a land mask already, so do not worry about this issue.

**Claude (2026-10-20, round 2):** Your answers hold together; nothing
conflicts. Checking each against the code and the logs turned up a few
consequences, most of which I can settle myself (listed as decisions at the
end). Three need you, plus one short paste from Pleiades.

10. **Are the 2D diagnostics snapshots or hourly means?** Dan said "hourly
    snapshots of every variable", and that is certainly true of `SST`, `SSS`,
    `SSU`, `SSV` and `Eta`. But `oceQnet`, `oceQsw`, `oceFWflx`, `oceSflux`,
    `oceTAUX`, `oceTAUY` and the `SI*` fields are `pkg/diagnostics` output,
    and MITgcm averages a diagnostic over its output interval unless its
    `frequency` in `data.diagnostics` is negative. If they are means, the
    hour stamp means "the hour ending at", not "the instant", and that has
    to go into the variable attributes (`cell_methods: time: mean`), the
    HOWTO's "snapshots, not averages" caveat, and the design doc. The data
    would still go in the same hourly store either way. The check is a
    paste from Pleiades:
    ```bash
    grep -E "frequency|fileName|fields|levels" $F1/data.diagnostics
    ls $F1 | grep -c '\.meta$'; ls $F1/oceQnet.*.meta 2>/dev/null | head -1 | xargs cat
    ```
    (a `.meta` file with two `timeInterval` values is an average; one value
    is a snapshot). While you are there, the one line that settles the field
    list -- `SURFACE_FIELDS` also knows `KPPhbl` and `PhiBot`, which your
    inventory did not mention, and `SIhsalt` exists in MITgcm but is not in
    the table:
    ```bash
    wr_llc_v2_inventory $OUT | grep -A3 "^combos"
    ```
    My recommendation: extract whatever is present in all 83 folders, and
    label averages honestly if that is what they are.
>A.

11. **All remaining fields in one pass, or `SSS`/`SSU`/`SSV` first?** One
    pass is the most efficient order you asked for in Q2: every store is
    opened, re-consolidated and completeness-checked once instead of once
    per pass, and the per-store S3 overhead is paid once. The cost is that
    nothing is published until the first window finishes, and if Q10 turns
    out to matter the attributes on the diagnostics would need a metadata
    fix-up afterwards. Two passes (the three state variables, then the
    diagnostics and sea ice) publish the fields people want first within
    days and let the Q10 answer land before the diagnostics are written.
    My recommendation: two passes, unless Q10 is answered before launch, in
    which case one.
>A.

12. **Keep the full read-back verification?** Every chunk written is read
    back and compared, which is why each store crosses the wire twice. In a
    latency-bound loop that is roughly 40 percent of the time per store. It
    has never caught a mismatch (the one incomplete store on the bucket was
    a crash mid-write, which `complete` already handles), and a failed PUT
    raises rather than writing a wrong object. The options are `full`
    (today), `sample` (read back one face per variable, 1/13 of the GETs),
    or `none`. My recommendation: add `--verify {full,sample,none}`,
    default `full`, run the first window of each pass at `full`, and let
    the benchmark in Q4 tell us whether dropping to `sample` for the bulk
    is worth it.
>A.

13. **Run the Q4 benchmark first, or launch at four windows?** The benchmark
    is 40 minutes and three commands; launching blind at four processes
    would most likely work but could leave a 2-4x speed-up unused for a
    multi-week run. My recommendation: benchmark, then launch with whatever
    N it indicates.
>A.

**Decisions I am settling myself** (say so if you disagree with any):

* **Completeness becomes per variable.** Each array gets `complete: true`
  after its verification; the group's `complete` means "every variable in
  `attrs['variables']` is complete"; `attrs['variables']` becomes the union
  on append; `store_is_complete(vars)` asks only about the requested ones.
  The 9,501 existing stores (`complete=True`, `variables=['Theta']`, no
  per-array attribute) stay valid under that rule without being touched.
  A store missing one field no longer forces the other fields to be
  rewritten on the next run, which also means the 189 hours in the
  unreadable folder cost nothing until they open up.
* **The 189 hours wait for Dan and are then done in one pass with all
  fields**, not SST first and the rest later; the two outage hours are done
  now with the code as it stands (Q6).
* **The land mask follows the staggering, not the name.** `mask_file_for_field`
  picks `hFacW` only for a variable literally named `U` and `hFacS` only for
  `V`, so `oceTAUX`/`SIuice` would be masked with `hFacC` and `oceTAUY`/`SIvice`
  likewise, leaving land-adjacent face points as 0 instead of NaN and giving
  them a NaN pattern that matches no mask. The choice will be driven by
  `var_hdims` (`i_g` -> `hFacW`, `j_g` -> `hFacS`, else `hFacC`), so your
  mask handles the sea-ice fields with no further option (Q9).
* **Names stay as the table says**: `Salt`, `U`, `V`, `Eta`, `oceQnet`, ...,
  `SIarea`, ... -- the xmitgcm/dbof names, so v2 stores read like v1's.
  `SIhsalt` is added to the table if the inventory shows it.
* **Writes stream field by field** within a store (read, mask, write, next)
  rather than reading all requested fields into memory first, so a
  fifteen-field pass needs ~3 GB per process, not ~15 GB.
* **Consolidated metadata is rewritten after every append**, as
  `write_timestep_store` already does; `_ensure_coords` adds `i_g`/`j_g` to
  an hourly store the first time `U`/`V` land in it, which is what the HOWTO
  already says will happen.
* **`open_range` tolerates mixed variable sets.** As written, `xr.concat`
  fails if some stores in the range hold `Salt` and others do not, which
  will be the state of the bucket for weeks. Without `variables=` it will
  warn and keep the variables common to all stores opened; with
  `variables=` it behaves as now.
* **HOWTO** gains the new variables in the at-a-glance table and the layout
  section, the `SSU`/`SSV` rotation worked example (Q5), per-variable
  staggering in the existing table, and a note on coverage per field while
  the passes run; "Current gaps" shrinks as they close.
* **Tests**: appending to a store leaves the existing `Theta` chunks
  byte-identical; per-variable completeness and `store_is_complete` with a
  partial set; the mask-by-staggering choice; `--workers` windows are
  disjoint and cover every hour exactly once; streaming writes; `open_range`
  on a mixed-variable range; `--verify` modes.

## Email Dan

Draft for Xavier to send; adjust as you like.

> Subject: LLC4320 v2 — last week of output, and adding more surface fields
>
> Hi Dan,
>
> A quick update and one request.
>
> The global hourly SST extraction is done and public. It covers
> 2023-01-01 01:00 to 2024-01-31 23:00 UTC, 9,501 hourly snapshots on the
> native grid, about 4.4 TB, served as Zarr from
> `s3://llc4320-v2/SURFACE` on the NSF Nautilus endpoint
> (`https://s3-west.nrp-nautilus.io`). It needs no credentials. There is a
> HOWTO with worked examples here:
> https://wrangler.readthedocs.io/en/llc4320_v2/howto/llc4320_v2.html
> You are credited as the source of the simulation; please tell me if you
> would like that worded differently, or a citation added.
>
> The request: the last week, 2024-02-01 to 2024-02-08, is the only gap. Its
> raw output is in
> `/nobackupp27/dbwhitt/llc_4320/OUT/2024_01_31_230000_to_2024_02_08_200000`,
> which is a symlink into `run_sep32026`. The directory lists fine, but the
> `SST.*.data` files inside are not readable by me, so the extraction skipped
> them. If that segment is finished and you are happy to relax the permissions,
> I can pick up those 189 hours in a single short run.
>
> Separately, we are about to extract `SSS`, `SSU` and `SSV` the same way.
> Two things would help before we start:
>
> 1. Are `SSU`/`SSV` the native face-relative components, on the west and
>    south cell faces respectively, as in the standard MITgcm layout? We plan
>    to publish them unrotated and document the rotation with `AngleCS`/`AngleSN`.
> 2. Is the run still producing output beyond 2024-02-08, and should we expect
>    to extend the record later?
>
> Thanks again — the dataset has been a pleasure to work with, and the only
> problems we hit were of our own making.
>
> Best,
> Xavier

## Logging

Append dated entries to `## Logs` below, in the format used by
`llc4320_v2.md`:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-10-20 (Design prompt 1, discussion round: answered Q4 from the run evidence; raised questions 10-13)

Read this file, the `## Logs`, step 6 and the `## Q&A` of `llc4320_v2.md`
(the predecessor), `wrangler/ogcm/llc_v2.py` in full, `llc_v2_access.py`,
`wrangler/scripts/llc_v2_surface.py`, `llc_v2_fix_faces.py`,
`llc_v2_sst_volume.py`, the PBS script and `docs/howto/llc4320_v2.rst`. Also
checked the `dbof` environment: zarr 3.4.1, `async.concurrency` 10 by
default, `threading.max_workers` None, and that `ZARR_ASYNC__CONCURRENCY=40`
in the environment is honoured -- so the Q4 benchmark needs no code change.
No code changed; nothing on the bucket touched; the design doc is not yet
written, per the prompt.

**Q4 conclusion: latency-bound on Nautilus, not CPU- or Lustre-bound.**
Measured: 74.6-82.4 s/store typical (range 46-116 s), 473.5 MB in 419
objects per store, each byte PUT then GET back for verification, so 11 MB/s
on the wire; five stores at ~30 s each on 2026-09-30 from the same process
and node, which caps the whole cycle at 30 s and attributes the usual
remaining 55 s to S3 waits; the extraction slowed to 133 s/store while the
workstation repair ran six workers, and that repair itself reached ~120 MB/s
aggregate, so the endpoint is not the ceiling. Inferred: Lustre read 1-3 s,
numpy 2-3 s, zstd 2-4 s, i.e. ~10 s CPU per store per field. Code: one face
at a time with at most 10 PUTs in flight and no overlap of GETs, PUTs or
the next Lustre read -- 838 operations per 85 s is one second per round
trip. Recommended: `--concurrency` (default 32) with whole-variable writes;
`--workers N` splitting the discovered hours into disjoint windows so the
`mode='a'` one-writer-per-store rule is enforced by construction; 4 workers
to start, 6-8 if the benchmark scales; a PBS node if compute nodes reach
Nautilus, else the PFE with at most 4; one extraction command at a time and
no repair/consolidate tools alongside. Proposed a 40-minute, three-run
benchmark into `s3://llc4320-v2/BENCH` (defaults; concurrency 40; four
windows) with the reading of each outcome.

**Consequences checked.** Reading several fields per hour in one pass is
cheaper than one field per pass (store open, consolidation and completeness
check paid once), but a store missing one field would, as the code stands,
force every field to be re-read and rewritten on each re-run -- hence
per-variable completeness. `mask_file_for_field` keys on the literal names
`U`/`V`, so `oceTAUX`/`oceTAUY`/`SIuice`/`SIvice` would be masked with
`hFacC`; it will key on `var_hdims` instead. `open_range`'s `xr.concat`
fails on stores with differing variable sets, which the bucket will have
for weeks; it will fall back to the common variables with a warning. The
`oce*` and `SI*` fields are `pkg/diagnostics` output and may be hourly
means rather than snapshots, which changes attributes and the HOWTO caveat
but not the store layout. `SURFACE_FIELDS` lists `KPPhbl`/`PhiBot`, which
the inventory did not mention, and lacks `SIhsalt`. Fifteen fields remain,
not twelve; the sea-ice ones compress to almost nothing but cost the same
number of round trips.

**Questions raised (10-13):** snapshot vs. mean for the diagnostics, with
the `data.diagnostics`/`.meta` commands to settle it and the inventory
`combos` line for the field list; one pass vs. two (recommend two unless
Q10 is settled first); keep full read-back verification or add
`--verify {full,sample,none}` (recommend adding it, default `full`);
benchmark before launch (recommend yes). Settled as decisions: per-variable
completeness with the existing stores valid unchanged; the 189 hours done
with all fields once Dan opens the folder; mask by staggering; xmitgcm names
kept; streaming per-field writes; re-consolidation on append; tolerant
`open_range`; HOWTO and test scope.

Model note: this round ran as Fable 5.1, as the prompt asked.
