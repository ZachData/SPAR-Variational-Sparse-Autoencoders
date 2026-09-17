# FINISHING.md — turning the notebook into a finished project

Started 2026-09-17. The science is done; this file tracks the work of making the
repository read as finished. Check boxes as they land; every step is a PR onto
`master`. When everything is checked, this file moves to `docs/notebook/`.

## STATUS — read this first (updated 2026-09-17, end of session 1)

**Where we are:** Step 1 (the paper) is done except for four editorial calls
listed under it. Steps 2–5 are untouched. Branch `finishing-plan` holds
everything so far as open **PR #8** onto `master`; merge it (or keep working
on the branch) before starting step 2.

**Next action:** step 2, "Verifiability". Concretely, in this order:

1. `git add -f` every `experiments/**/{config.json,experiment_config.json,evaluation_results.json,comprehensive_summary_*.json}`
   (≈1,949 files, 2.4 MB; `experiments/` is git-ignored so `-f` or a
   `.gitignore` negation is needed — prefer the negation so future runs are
   picked up). Leave `ae.pt`, `.npz`, `.png`, `logs/` ignored.
2. Write `reproduce.py` at the root: regenerates every figure in
   `workshop/figs/` and every numeric table in the paper from the committed
   JSONs, CPU only. The existing readers are the starting point —
   `falsification/read_a2_dose_response.py`, `read_a3_dose_response.py`,
   `read_a4_dose_response.py`, `read_a4_followup.py`, `compare_arms.py`,
   `frontier.py`, `workshop/make_fig_beta.py`. Check whether each one reads
   checkpoints (GPU) or summaries (CPU); the Jaccard numbers in particular
   were read from checkpoints and are cached in
   `falsification/a{2,3,4}_dose_response_results.json`, so read the cache.
3. While regenerating figures, fix what the paper read-through flagged:
   drop the internal codes ("A2:", "A3:", "A4:") and `log_var_init=`
   labels from titles/annotations, use paper-facing titles and larger fonts
   (they are 0.72\textwidth two-panel figures).
4. Add a row per figure/table → JSON source in `docs/REPRODUCE.md` (create
   the file; step 3 will fold RUNBOOK into it).

**Environment for the next session:** use `/usr/bin/python3` (torch 2.10
cu128 + nnsight in `~/.local`), *not* bare `python` (miniforge base, CPU-only,
no nnsight). `./workshop/build_paper.sh` builds the paper (fetches `tectonic`
to `~/.local/bin` on first use). `pytest falsification/tests/ -q` → 115 must
stay green.

**Do not** re-audit the repo, re-read the paper, or re-derive the plan — all of
that is below and in PR #8. Start at step 2 item 1.

## Why

The repo is written as a lab notebook for future-us: ~7,600 lines of dated
narrative across 11 docs, session logs, "next steps", landmines addressed to an
AI. A finished research repo has three readers, none of them us:

1. someone who read the paper and wants to check a number,
2. someone who wants to reuse `falsification/`,
3. a reviewer deciding whether to trust any of it.

Everything below serves those three.

## What the audit found (2026-09-17)

- **No `config.json` from any of the 437 runs is committed.** The 13-seed
  battery exists only on one disk. All run metadata is 2.4 MB (1,949 JSONs);
  the weights are 2.3 GB total.
- **`pip install -e .` never worked** — no `pyproject.toml`/`setup.py`; no
  `LICENSE` despite the MIT claim.
- **CI is dead.** `.github/workflows/build.yml` is upstream's Poetry/PyPI
  workflow targeting a `main` branch that no longer exists; "checks passing"
  on PRs was CodeRabbit skipping itself. Root `tests/` fails at collection.
- **Dead code:** `sae_vis/`, 4 of 5 `analysis_scripts/`, root-level
  `train_jumprelu.py` / `train_vsae_jump_relu.py`, `workshop/paper.tex` +
  `00/01/02_*.md`, 12+ tracked `.pyc` files — none referenced by
  `falsification/` or the run scripts.
- **The paper has never been compiled.**

## Keep — the deliverables

| | Why |
|---|---|
| `falsification/` core (`evalues.py`, `permutation.py`, `simulate.py`, `run_arm.py`, readers/scorers, `tests/`) | Deliverable #2, the reusable package. 115 tests. |
| `dictionary_learning/trainers/vsae_*.py`, `training_scripts/` | The fixed trainers; the bugs found in them *are* findings. |
| Every `*_results.json` in `falsification/`, `workshop/figs/` | The data behind every number in the paper. |
| `workshop/mechanism_paper.tex`, `references.bib` | Deliverable #4. |
| `SAEBench-main/` (vendored v0.4.2) | Needed for E4. Document what was modified. |
| The *content* of `CLAUDE.md`'s landmines and the RESULTS addenda | The audit trail is part of the contribution. It needs a reader-facing home, not deletion. |

## Plan, in order

### 1. The paper — compile, fix, ship
- [x] Install a LaTeX toolchain (`tectonic`, standalone) and compile `workshop/mechanism_paper.tex` — `workshop/build_paper.sh`
- [x] Fix every compiler error/warning that matters — 1 hard error (9-column table in an 8-column spec), 2 wrong roundings in the seed-ceiling table (1.65→1.64, 2.66→2.65 per the committed survey JSON), 15 overfull boxes (3 tables running into the margin, unbreakable identifiers), a Unicode quote the font couldn't render, hyperref link boxes → coloured text, and two internal-doc leaks ("CLAUDE.md's two-threshold rule", "addendum 20") in the prose
- [x] Read the compiled PDF once end to end. Found and fixed: **Table 6 (the ReLU factor) carried the superseded 6-seed numbers (d = +20.8 / +19.8) while claiming 13 seeds** — replaced with the 13-seed values `compare_arms.py e3_masked_kl e3_masked_kl_relu` reproduces (0.1315±0.0074 vs 0.0287±0.0033, d = +19.3; 0.4655±0.0113 vs 0.3081±0.0092, d = +15.3), and the dependent "~4.4×", "d ≈ 20" and Table 7 "19.8–20.8" updated to match; §3.2's item count summed to 17 against the 15-item Table 12 (rewritten to match the table); Table 7's caption pointed at §6.2 instead of the seed survey §5.2.
- [ ] **Editorial decisions for the author** (not made unilaterally):
  - the abstract is ~370 words — most venues cap at 150–250;
  - the author block is a placeholder (`Zach`, a gmail address) — needs name, affiliation, and a decision on whether the preprint's co-authors are on this one;
  - the naive-comparison gap is described as "d ≈ 16 on reconstruction, d ≈ 13 on liveness" in the abstract/§3.4, but Table 1's first row is d = −5.7 on FVE; the 16.5 is the *unitinit* rung. Either say "up to d ≈ 16 along the ladder" or quote the first-row numbers;
  - figures: 0.72\textwidth two-panel PNG→PDF with matplotlib-default fonts, and internal experiment codes ("A2:", "A3:", "A4:") and `log_var_init=` labels in the suptitles/annotations — regenerate under step 2 with paper-facing titles and larger fonts.
- [x] Commit `workshop/mechanism_paper.pdf` (25 pages, 264 KB; rebuilt by `build_paper.sh`)

### 2. Verifiability — every number recomputable from the repo, no GPU
- [ ] Commit every run's `config.json`, `experiment_config.json`, `evaluation_results.json`, `comprehensive_summary_*.json` (≈2.4 MB; leave `ae.pt`, `.npz`, `.png` ignored)
- [ ] `reproduce.py` (or `make figures`): regenerates every figure in `workshop/figs/` and every table in the paper from committed JSONs; runs in CI
- [ ] Record which figure/table each result JSON feeds, in `docs/REPRODUCE.md`

### 3. Docs — 11 files → 5, frozen tense
- [ ] `README.md` — front door: the claim under test, the six findings (one paragraph + figure each), reproduce, cite. Today's status table becomes a *final state* block; the session log goes to the notebook.
- [ ] `docs/RESULTS.md` — findings in final form, past tense, no addendum numbering (from `RESULTS_2026-09-03.md` + PROJECT's "What is established")
- [ ] `docs/METHODS.md` — pre-registration, statistical design, power analysis (PROJECT.md's frozen lower half)
- [ ] `docs/ERRATA.md` — CLAUDE.md's landmines + corrections to the preprint, rewritten for a human reader
- [ ] `docs/REPRODUCE.md` — RUNBOOK cleaned: environment (`/usr/bin/python3`, pinned versions), one command per figure/table
- [ ] `docs/notebook/` — `PROJECT.md`, `HANDOFF.md`, `RESULTS_2026-09-03.md`, `FINDINGS_2026-09-02.md`, `REMEDIATION.md`, `OVERVIEW.md`, `workshop/00–02_*.md`, `workshop/paper.tex` moved **verbatim**, each with a one-line "frozen <date>; historical record" header
- [ ] `CLAUDE.md` shrinks to a pointer at `docs/ERRATA.md` + the working conventions

### 4. Hygiene — one PR
- [ ] `pyproject.toml` so `pip install -e .` is true
- [ ] `requirements-lock.txt` frozen from the working env (torch 2.10.0+cu128, nnsight 0.7.0, …)
- [ ] `LICENSE` (MIT, matching upstream `dictionary_learning`)
- [ ] `CITATION.cff`
- [ ] Replace `.github/workflows/build.yml` with a workflow that runs `pytest falsification/tests/` (and `reproduce.py`) on push/PR to `master`; badge in README
- [ ] `git rm` tracked `__pycache__/*.pyc`
- [ ] Remove root `train_jumprelu.py`, `train_vsae_jump_relu.py`, `sweep.out`
- [ ] Remove `sae_vis/` and the four unreferenced `analysis_scripts/` (history keeps them)
- [ ] Root `tests/`: fix or remove (not both)
- [ ] `wandb` sweep plumbing in `training_scripts/`: check whether any finished result depends on it before cutting
- [ ] `SAEBench-main/`: note the vendored version and the modified files in `docs/REPRODUCE.md`

### 5. Archive and release
- [ ] Upload the 437 checkpoints (2.3 GB) to a HuggingFace repo; link from README
- [ ] Tag `v1.0`; GitHub release with the PDF attached
- [ ] Zenodo DOI (optional)
- [ ] Move this file to `docs/notebook/`

## Log

| Date | Done |
|---|---|
| 2026-09-17 | Audit; this plan written. Branches consolidated to `master` (PRs #6, #7). Paper compiled for the first time (`build_paper.sh`); compiler + read-through fixes, incl. Table 6's stale 6-seed numbers; PDF committed. All on PR #8. |
