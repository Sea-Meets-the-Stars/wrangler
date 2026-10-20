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

1. Please answer the pre-run Q&A below, then propose a plan. Do not write code
   until the plan is agreed.

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

2. **Which fields, and in what order?** `SSS`, `SSU`, `SSV` are the obvious
   three. The inventory also shows `Eta`, `oceQnet`, `oceQsw`, `oceFWflx`,
   `oceSflux`, `oceTAUX`, `oceTAUY` and the `SI*` sea-ice fields present in all
   83 folders. Do you want all of them eventually, and if so is there an order
   that matters to you scientifically?

3. **Volume and quota.** Each field costs about the same as `Theta`: 4.6 TB
   over the full 9,692 hours. Four fields is **18.4 TB**, twelve would be
   55 TB. The bucket has never had a stated quota. Can you find out what we are
   allowed before we commit to a number of fields?

4. **Runtime, and whether to parallelise.** At the measured 85 s per store per
   field, `SSS`+`SSU`+`SSV` over the full record is about **26 days** in one
   process. Splitting by date range across several processes is
   straightforward — `--start`/`--end` already exist, the stores are disjoint —
   but with the `mode='a'` change above, two processes must never touch the
   same store at once. Four or six windows, run from Pleiades? Or accept the
   single long run now that it survives outages?

5. **`SSU`/`SSV`: native, rotated, or both?** They come on the native
   face-relative axes, `U` on the west cell face and `V` on the south. Rotating
   to east/north needs `CS`/`SN` and an interpolation to cell centres, which
   loses the C-grid staggering and doubles the storage if we keep both. My
   inclination is to publish the native components only, document the rotation
   in the HOWTO with a worked example, and let users do it — that is lossless
   and matches what v1 does. Do you agree, or do your users want
   `u_east`/`v_north` served ready-made?

6. **Do the 191 SST hours first?** Two of them are one-hour re-runs that could
   be done in minutes. The 189 depend on Dan. Doing the two now means the
   record is contiguous to 2024-01-31 before any new field starts, which keeps
   the HOWTO simple. Any reason not to?

7. **Should new fields cover the full record, or only where SST already is?**
   If the 189 hours never become available, do we publish `SSS` for the same
   9,503 hours as SST, or stop at whatever is readable and let the fields
   differ in extent? I would keep them aligned and document one coverage.

8. **One store per hour, or one per field?** Keeping every field in the same
   `{YYYYMMDD}T{HH}.zarr` means a single open gives a user everything for that
   hour, and `open_range` needs no change. The alternative, a prefix per field,
   makes partial re-runs independent but doubles the request count for anyone
   wanting two fields. I favour the first; confirm.

9. **Sea-ice fields need a real land mask.** `SIarea`, `SIheff` and friends are
   legitimately 0 over open ocean, so the "exact zeros are land" fallback is
   wrong for them and `--mask-dir` becomes mandatory. Worth noting if those
   fields are wanted.

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
