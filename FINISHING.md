# FINISHING.md — turning the notebook into a finished project

Started 2026-09-17. The science is done; this file tracks the work of making the
repository read as finished. Check boxes as they land; every step is a PR onto
`master`. When everything is checked, this file moves to `docs/notebook/`.

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
- [ ] Install a LaTeX toolchain (`tectonic`, standalone) and compile `workshop/mechanism_paper.tex`
- [ ] Fix every compiler error/warning that matters (undefined refs, missing figures, bib)
- [ ] Read the compiled PDF once end to end for things hand-checking missed
- [ ] Commit `workshop/mechanism_paper.pdf` (or attach to the `v1.0` release — decide)

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
| 2026-09-17 | Audit; this plan written. Branches consolidated to `master`. |
