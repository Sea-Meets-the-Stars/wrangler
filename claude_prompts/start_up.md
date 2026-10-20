# Start up

## Prompts

1. Please see the file `claude_start_up.md` in `bin/claude_prompts`.  Following that, please generate the `CLAUDE.md` file and SKILLS files for this Repository.  If you have any questions, please ask in the Q&A section below.  Log your work.  Please use Opus

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