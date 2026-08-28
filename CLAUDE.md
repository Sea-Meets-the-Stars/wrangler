# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**Wrangler** is a Python package for efficiently downloading, processing, and
analyzing ocean data. The current focus is VIIRS Sea Surface Temperature (SST)
from PODAAC, and the package is being extended to handle native-grid output from
the LLC4320 ocean model (`wrangler/ogcm/`).

Repository layout:

- `wrangler/` — the Python package source
  - `grab/` — data download (e.g. PODAAC)
  - `datasets/`, `extract/`, `preproc/` — dataset definitions, field extraction, preprocessing
  - `ogcm/` — LLC4320 / ocean model support
  - `tables/`, `plotting/`, `scripts/`, `tests/`
- `bin/` — command-line entry points
- `docs/` — Sphinx documentation (Read the Docs)
- `projects/` — project-specific analyses
- `claude_prompts/` — prompts and task definitions that drive this work

## Working Rules

- **Git**: The user (J. Xavier Prochaska) performs all git commands. Do not run
  `git add`, `git commit`, `git push`, `git reset`, or any other command that
  changes repository state. Read-only git (`git status`, `git diff`, `git log`,
  `git show`) is fine when helpful. Your job is to edit files.
- **Calculations**: If you do any calculation, generate it as a Python script and
  write it to disk so that it can be added to the repository. Do not perform
  one-off calculations only in memory or in the chat.
- **Python environment**: If you need to run Python, use the `ocean14` conda
  environment (e.g. `conda run -n ocean14 python script.py`). Never use the
  system Python.

## Conventions

- Write clear, well-documented Python with docstrings.

## Related Repositories

- **llc4320-native-grid-preprocessing** — on this computer at
  `/Users/xavier/Oceanography/python/llc4320-native-grid-preprocessing`. It
  generates training datasets from the LLC4320 ocean model (Zarr image patches
  plus Parquet metadata). The LLC4320 native-grid work in `wrangler/ogcm/`
  is coupled to that pipeline; consult it for conventions on native-grid
  sampling, dataset structure, and downstream use.

## Logging

The work log lives in [claude_prompts/start_up.md](claude_prompts/start_up.md)
under the "Logs" section. When asked to log work, append a dated entry using the
format documented there:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```
