# Getting started

## Goals

This repository will download, process, and analyze ocean data, starting with
VIIRS Sea Surface Temperature (SST) from PODAAC. It is being extended to handle
native-grid output from the LLC4320 ocean model (`wrangler/ogcm/`), in step with
the sibling `llc4320-native-grid-preprocessing` pipeline.

## Prompts

1. Read this file.  Execute the 1st task under "Claude/CLAUDE.md file"
2. Read this file.  Execute the 1st task under "Claude/Skills"
3. Read this file.  Execute the 1st task under "Claude/Settings"
4. Read this file.  Execute the 1st task under "Basic start up"

## Claude

### CLAUDE.md file

Generate or update the `CLAUDE.md` file for this Repository.  It must state:

- I will perform git commands.  Read-only git (`status`, `diff`, `log`, `show`)
  is fine.
- If you do any calculation, generate it as a python script and write it to disk
  so that I can add it to the Repository.
- If you need to run Python, use the `ocean14` conda environment.
- The related repository `llc4320-native-grid-preprocessing` is on this computer
  at `/home/xavier/Oceanography/python/llc4320-native-grid-preprocessing`.

(Done 2026-08-27 — see Logs.)

### Skills

Copy over the skills/ files from the `IOPtics` repository.

(Done 2026-08-27 — `critical-partner` and `grill-me`; see Logs.)

### Settings

Copy over the settings.json file from the `IOPtics` repository.  Copy the
permission policy (allow / deny / ask), not the accumulated path-specific
entries that reference IOPtics' own files.

## Basic start up

Generate the basic files that one needs for a Python GitHub repository, e.g. a
file for dependencies.  Examine the other Repositories in `Oceanography/python`
to see how I tend to organize things.  Also, make a suggestion of which of their
settings.json files to copy into this one.

Note: this repo already has `setup.py` (with `install_requires`), `LICENSE`,
`.gitignore`, `README.rst`, and Sphinx `docs/`.  It does not yet have a
`requirements.txt` or `pytest.ini`.

### Report

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-27 (Generate CLAUDE.md and skills)

Followed the house pattern documented in `~/bin/claude_prompts/claude_start_up.md`
(a synthesis of the `start_up.md` convention across Xavier's other repos) to
bootstrap Claude support for this repo, which had neither a `CLAUDE.md` nor a
`.claude/` directory before this.

Wrote `CLAUDE.md` at the repo root with: a short project overview (Wrangler
downloads/processes/analyzes ocean data, currently VIIRS SST from PODAAC, and
is being extended on branch `llc4320_v2` to handle LLC4320 native-grid output
in `wrangler/ogcm/`); the three canonical working rules (Xavier runs all git
commands — Claude may use read-only git but not `add`/`commit`/`push`/`reset`;
calculations go into Python scripts written to disk, not one-off inline code;
Python runs via the `ocean14` conda environment, never system Python); and a
"Related Repositories" pointer to the sibling repo
`llc4320-native-grid-preprocessing` (generates the LLC4320 training datasets
that `wrangler/ogcm/` is coupled to), following the same sibling-pointer
pattern used in `cugn-climatology`'s `CLAUDE.md` for `cugn`.

Copied `.claude/skills/critical-partner/SKILL.md` and
`.claude/skills/grill-me/SKILL.md` verbatim from IOPtics, which the pattern
doc identifies as the upstream source of truth for skills — confirmed
byte-identical with `diff`.

Learned along the way: the prompt's reference to `claude_start_up.md` in
`bin/claude_prompts` did not resolve inside this repo or any sibling repo —
it lives at `/Users/xavier/bin/claude_prompts/claude_start_up.md` (Xavier's
personal `~/bin`, not a repo-local `bin/`), confirmed with the user before
proceeding.

Scoped this run to just `CLAUDE.md` + skills, per prompt #1's literal ask;
did not generate `.claude/settings.json` or the "Basic start up" repo files
(dependencies, `.gitignore`, etc.) — those are separate prompts in the house
pattern and weren't requested here.

No git commands were run beyond read-only `status`/`branch`/`diff` checks.
`CLAUDE.md` and `.claude/` are new, untracked files; Xavier will stage and
commit per the git-is-Xavier's rule this same log documents.

### 2026-10-23 (Restructure start_up.md to the house skeleton)

Rewrote `claude_prompts/start_up.md` to follow the canonical skeleton in
`~/bin/claude_prompts/repo_start_up.md`: Goals, numbered Prompts, a Claude
block (CLAUDE.md file / Skills / Settings), Basic start up (with an empty
Report header), Logging, and Logs.  The previous single prompt (pointing at
`claude_start_up.md`, now renamed `repo_start_up.md`) was replaced by the
standard four numbered prompts.  The existing 2026-08-27 log entry was kept
verbatim.

State of the repo when this was written: `CLAUDE.md` and
`.claude/skills/{critical-partner,grill-me}` exist, so prompts 1–2 are marked
done.  There is no `.claude/settings.json` yet (prompt 3 is open).  For Basic
start up, `setup.py`, `LICENSE`, `.gitignore`, `README.rst` and `docs/` already
exist; `requirements.txt` and `pytest.ini` do not.  Note this repo uses
`README.rst` (Read the Docs), unlike the `README.md` in the house layout.

There is no `prompts/` directory; the house convention (and this repo's
`CLAUDE.md`) uses `claude_prompts/`, so the file stays there.  No git commands
were run.
