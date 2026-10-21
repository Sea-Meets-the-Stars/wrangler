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

2. See my answers to your second round of questions.  Then write the
design document.  Use Fable if you can. Log your work.

3. I have reviewed the design document.  Here is the output from the `wr_llc_v2_inventory` command from the design doc:
```
(field, ext) -> number of folders containing it:
  Eta.data       95
  EtaF.data       1
  Eta_tide.data       57
  Eta_tide_presal.data       29
  KPPhbl.data       95
  PhiBot.data       95
  R_Eta.data       5
  R_SIarea.data       5
  R_SIheff.data       5
  R_SIhsalt.data       5
  R_SIhsnow.data       5
  R_SIuice.data       5
  R_SIvice.data       5
  R_Salt.data       5
  R_SaltF.data       1
  R_Theta.data       5
  R_ThetaF.data       1
  R_U.data       5
  R_V.data       5
  R_W.data       1
  SHIfwFlx.data       95
  SHIhtFlx.data       95
  SIarea.data       95
  SIheff.data       95
  SIheffF.data       4
  SIhsalt.data       95
  SIhsnow.data       95
  SIuice.data       95
  SIvice.data       95
  SSS.data       95
  SST.data       95
  SSU.data       95
  SSV.data       95
  Salt.bloat      15
  Salt.every5.shrunk 11
  Salt.sections5deg 11
  Salt.shrunk     39
  SaltF.data       3
  SaltFF.data       1
  Theta.bloat      13
  Theta.bloat.tmp  1
  Theta.every5.shrunk 11
  Theta.sections5deg 11
  Theta.shrunk     40
  ThetaF.data       2
  ThetaFF.data       1
  U.bloat      16
  U.every5.shrunk 11
  U.sections5deg 11
  U.shrunk     39
  V.bloat      16
  V.bloat.bad  1
  V.every5.shrunk 11
  V.sections5deg 11
  V.shrunk     39
  W.every5.shrunk 11
  W.level1     1
  W.sections5deg 11
  W.shrunk     39
  oceFWflx.data       95
  oceQnet.data       95
  oceQsw.data       95
  oceSflux.data       95
  oceTAUX.data       95
  oceTAUY.data       95
```
Please revise the design document to reflect this information.
Then geneate a slide deck that summarizes the design.
Put it in `claude_prompts/llc4320_v2/llc4320_v2_design.pptx`.
Be sure to include a one line summary on each slide
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
>A.  The first command (grep) returns "0".  The second command gives nothing.
  Use your recommendation

    **Claude:** How I read this. The only command above that prints a bare
    `0` is `ls $F1 | grep -c '\.meta$'`, so: **there are no `.meta` files**
    (the `timeInterval` check is unavailable). "The second command gives
    nothing" fits the inventory line, and that one was my mistake: the tool
    prints the heading `(field, ext) -> number of folders containing it:`,
    not the word `combos`, so `grep "^combos"` can never match. The
    `data.diagnostics` grep result is therefore unknown to me, and the
    snapshot-vs-mean question for the `oce*`/`SI*` fields **remains open**.
    It changes attributes and the HOWTO caveat, not the store layout, so the
    design proceeds: `--fields all` resolves the field list at run time from
    what is present in every folder, a `--time-mean` option labels averages
    with `cell_methods`, the uncertainty is recorded in the variable
    attributes until settled, and a metadata-only fix-up exists if the
    answer arrives after the run. The corrected pre-launch commands are in
    the design doc, sections 4.2 and 4.3:
    ```bash
    grep -iE "frequency|fileName|fields\(|levels\(" $F1/data.diagnostics | head -40
    grep -iE "frequency|fileName" $F1/STDOUT.0000 | head -40
    wr_llc_v2_inventory $OUT --quiet | sed -n '/number of folders containing it/,$p'
    ```

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
>A.  I have answered Q10.  Do one pass

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
>A.  Use your recommendation

13. **Run the Q4 benchmark first, or launch at four windows?** The benchmark
    is 40 minutes and three commands; launching blind at four processes
    would most likely work but could leave a 2-4x speed-up unused for a
    multi-week run. My recommendation: benchmark, then launch with whatever
    N it indicates.
>A.  Use your recommendation

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

### 2026-10-20 (Design prompt 2: read answers 10-13; wrote llc4320_v2_design.md)

Reread this file. Answers: Q10 "the first command (grep) returns 0, the
second gives nothing, use your recommendation"; Q11 one pass of all
remaining fields; Q12 and Q13 my recommendations (`--verify {full,sample,none}`
default `full`; benchmark first, then launch with the N it indicates).

**Q10 reading.** The only command in that block that prints a bare `0` is
`ls $F1 | grep -c '\.meta$'`, so there are no `.meta` files and the
`timeInterval` test is unavailable. "The second command gives nothing" is
the inventory line, and that was my error: `wr_llc_v2_inventory` prints the
heading `(field, ext) -> number of folders containing it:`, not `combos`
(checked `wrangler/scripts/llc_v2_inventory.py`). So whether the `oce*`/`SI*`
diagnostics are snapshots or hourly means is still unresolved, and the field
list is still unconfirmed. Did not invent a result: the design resolves the
field list at run time from what is present in every folder, adds
`--time-mean` to label averages with `cell_methods`, records the uncertainty
in the variable attributes, provides a metadata-only fix-up, and lists the
corrected pre-launch commands (`data.diagnostics`/`STDOUT.0000` frequency
greps, inventory summary via `sed`). Added an indented note under the Q10
answer saying so.

**Design doc** written to `claude_prompts/llc4320_v2/llc4320_v2_design.md`,
self-contained. Sections: summary and decision table; the raw source
(folders, compact layout, dating rule, deltaT/nIter0, masks, grid); how SST
was extracted (reading, the face-order bug and its in-place repair, masking
and the exact-zero artefact, store layout/chunking/compression, `grid.zarr`
and the C-grid staggering fix, consolidated metadata, completeness and
verification and retries, public access and the reader, state and measured
performance); the new fields (table of prefix -> variable, units, location,
dims, mask, sampling; the open snapshot/mean question and how it is handled;
run-time field list); design changes (append mode and per-variable
completeness with the legacy rule for existing stores, mask by staggering,
`--concurrency` and whole-variable writes, `--workers` with disjoint windows
by construction, `--verify`, streaming per field and per-variable retries,
re-consolidation, `open_range` on mixed variable sets, CLI table); the
benchmark plan (A defaults, B concurrency 40, C four windows, D one hour of
all fields locally for per-field sizes); run order (two outage hours now ->
code and tests -> pre-launch pastes -> benchmark -> one pass -> audit ->
HOWTO -> the 189 hours); storage and runtime estimates; HOWTO changes;
tests; risks and open items.

**New finding while estimating.** Re-ran `llc_v2_sst_volume.estimate`: 15
remaining fields at SST's measured 0.47 GB per field-hour is 69 TB stored
(73 TB with Theta), above the 55 TB accepted in Q3. The ten full-ocean
fields alone are ~46 TB; the five sea-ice fields are mostly zeros and should
compress 10-50x (inference, unmeasured), which would bring the total to
~52-55 TB. Benchmark run D (one hour of all fields to local scratch, `du`
per variable) measures this before launch; flagged as the first risk.

No code changed; nothing on the bucket touched. Open items for Xavier: run
the two outage-hour commands; paste the three pre-launch checks; run the
benchmark; send the Dan email (add the `SIuice`/`SIvice` grid-location
question); confirm the bucket quota against the upper-bound estimate. This
round ran as Fable 5.1.

### 2026-10-20 (Design prompt 3: full inventory read; design doc revised; slide deck built)

Read the `(field, ext) -> number of folders` block Xavier pasted from
`wr_llc_v2_inventory`. Three things in it changed the design.

**95 folders, not 83.** Every one of 21 prefixes is present in 95 folders;
SST was extracted from 83 (84 once the live-run symlink became listable).
The paste omitted the tool's first lines (folder count with first/last
dates, unreadable folders), so I cannot tell from here what the extra
folders are. Most likely (inference) Dan's run continued past 2024-02-08 and
~11 new segments landed, which at 4-7 simulated days per folder would add
~1,300-1,500 hours for a record of ~11,000 hours instead of 9,692. The
9,692-hour count-equals-span proof must be redone on the new tree. Asked
for one paste (`wr_llc_v2_inventory $OUT --quiet | head -4` and the dry-run
`Discovered`/`Duplicate`/`spans` lines). The code design is unchanged; the
numbers scale.

**20 new fields, not 15.** Present throughout: `SSS SSU SSV Eta KPPhbl
PhiBot oceQnet oceQsw oceFWflx oceSflux oceTAUX oceTAUY SHIfwFlx SHIhtFlx
SIarea SIheff SIhsalt SIhsnow SIuice SIvice`. `SHIfwFlx`/`SHIhtFlx`
(ice-shelf fluxes) and `SIhsalt` are not in `SURFACE_FIELDS` and are added
by the design. Classified as 12 dense (compress like SST, 1.19x measured)
and 8 sparse (`SI*`, `SHI*`: zero away from ice; compression inferred, to be
measured by benchmark run D). Storage redone: the 12 dense fields alone are
~55 TB at SST's measured 0.47 GB per field-hour; with the sparse fields
~60-65 TB likely and ~97 TB upper bound including Theta, times ~1.15 if the
record is ~11,000 hours. **Every case exceeds the 55 TB accepted in Q3**, so
the quota or the field list is a decision before launch (`PhiBot`, `KPPhbl`
named as the first candidates to trim). Runtime redone for 20 fields:
~190 days single-process as the code stands; ~15 days with concurrency 32
and 6 workers; ~11 days with 8 workers or `--verify sample` (inference).

**Partial-coverage families, out of scope.** `R_*` (5 folders), `*F`/`*FF`
(1-4), `Eta_tide` (57), `Eta_tide_presal` (29), and the 3D `.shrunk`,
`every5.shrunk`, `sections5deg`, `bloat`, `W.level1` products. What they are
is inference from the names (a second output stream, other-cadence
variants, tidal SSH, the depth phase); the run-time field resolution ignores
them by construction since they are not in every folder. Snapshot vs. mean
for the diagnostics is **not** settled by the inventory and stays open.

**Design doc** (`llc4320_v2_design.md`): new section 2.1 (what the inventory
showed, the 95-folder question, the out-of-scope table); section 4.1
rebuilt as a 21-row table with a compressibility class; 4.2 marked still
open; 4.3 "settled, and still resolved at run time"; 5.2 `ZERO_IS_NOT_LAND`
covers all eight sparse fields; benchmark run D now all 21 fields; run order
gains "decide storage" and notes the pass extends SST if the tree is
longer; section 8 and the risk table rewritten with the new totals.

**Slide deck** `llc4320_v2_design.pptx`, 17 slides, built by
`make_llc4320_v2_design_slides.py` (python-pptx; node/pptxgenjs is not
installed here), same palette and 20 pt floor as `docs/slides/llc4320_v2.pptx`.
Every slide has a one-line takeaway under its title. Arc: title; goal and
status; raw source; pipeline; face-order bug; store layout and grid; access;
the 20 fields in three groups; what the inventory changed; snapshot/mean;
append mode and masks; concurrency, workers, verify; the I/O-bound evidence
(native bar chart of the measured s/store); benchmark plan; run order;
storage and runtime; risks and decisions. Validated with the pptx skill's
`validate.py` (passed), rendered through LibreOffice to images and every
slide inspected; fixed six defects on the first pass (wrapping names on the
title card, a mid-word path break, an undersized facet schematic, wrapping
monospace labels, a duplicated chart title, a two-line summary), re-rendered
and re-checked those slides.

No code changed; nothing on the bucket touched. Open for Xavier: the
folder-span paste; the storage decision; the diagnostic-frequency paste;
the outage-hour runs; the benchmark; the Dan email (now also: does the run
continue past 2024-02-08, and what are the `R_*`/`Eta_tide*` files). This
round ran as Fable 5.1.
