# FINISHING.md — turning the notebook into a finished project

Started 2026-09-17. The science is done; this file tracks the work of making the
repository read as finished. Check boxes as they land; every step is a PR onto
`master`. When everything is checked, this file moves to `docs/notebook/`.

## STATUS — read this first (updated 2026-09-17, end of session 3)

**Where we are:** Steps 1–3 done. Three stacked PRs: **#8** `finishing-plan`
(step 1), **#9** `finishing-verifiability` (step 2, base #8), **#10**
`finishing-docs` (step 3, base #9). Merge in that order, retargeting each to
`master` as its base lands.

**Tabled by the author (2026-09-17):** the paper rewrite. Step 1's four
editorial calls (abstract length, author block, the "d ≈ 16" phrasing;
the figure item is done) and any fuller revision of
`workshop/mechanism_paper.tex` wait a few days. Do not start them in a
session that was not asked to. The README's citation block carries the
same author placeholder and changes with it.

**What step 3 delivered:** `docs/RESULTS.md` (findings in final form, past
tense, sectioned by finding with a pointer to the notebook addenda each
rests on), `docs/METHODS.md` (thesis, framework, floors, simulation
results, measurement conventions, the pre-registration verbatim with
outcomes beside it), `docs/ERRATA.md` (preprint corrections, every bug with
what it touched and its status, replaced metrics, retracted results,
SAEBench modifications), `docs/REPRODUCE.md` with RUNBOOK folded in,
`docs/notebook/` holding PROJECT / HANDOFF / RESULTS / FINDINGS /
REMEDIATION / RUNBOOK / OVERVIEW / the workshop notes and draft verbatim
under a frozen header plus a `README.md` index (and the old README's
session log), `README.md` rewritten as the front door, `CLAUDE.md` cut to a
pointer at `docs/ERRATA.md` plus the conventions. Code docstrings that cite
"PROJECT.md" / "RESULTS addendum N" / "REMEDIATION F6" were left alone;
`docs/notebook/README.md` resolves them.

**Next action:** step 4, "Hygiene — one PR", starting with `pyproject.toml`
and the CI workflow (`pytest falsification/tests/` + `python reproduce.py
--check` on push/PR to `master`). Check the `wandb` sweep plumbing before
cutting it; `git log -S` is enough.

**Environment for the next session:** `/usr/bin/python3` for anything with
torch; `reproduce.py` needs only numpy/scipy/matplotlib. `pytest
falsification/tests/ -q` → 115 must stay green; `python reproduce.py
--check` must exit 0.

**Do not** re-audit the repo, re-read the paper, or re-derive the plan.
Start at step 4.

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
  - ~~figures: 0.72\textwidth two-panel PNG→PDF with matplotlib-default fonts, and internal experiment codes ("A2:", "A3:", "A4:") and `log_var_init=` labels in the suptitles/annotations~~ — done in step 2 (full text width, 9 pt fonts, σ and achieved-L0 labels, no suptitle).
- [x] Commit `workshop/mechanism_paper.pdf` (25 pages, 264 KB; rebuilt by `build_paper.sh`)

### 2. Verifiability — every number recomputable from the repo, no GPU
- [x] Commit every run's `config.json`, `experiment_config.json`, `evaluation_results.json`, `comprehensive_summary_*.json` (1,940 files, 2.4 MB) — via a `.gitignore` negation so future runs are picked up; `ae.pt`, `.png`, logs stay ignored. **Also** the 170 `all_histograms_*.npz` (16 MB): `report_summaries.liveness()` reads the per-feature selection counts from the `.npz`, not the JSON, so without them Tables 1, 6 and 8 cannot be recomputed. The three `experiments/VSAEJumpReLU_*` smoke-test dirs stay ignored.
- [x] `reproduce.py`: every figure in the paper + `beta_sweep.pdf`, every numeric table, `--check` asserts 125 printed numbers against the paper (exit 1 on drift). CPU, ~45 s. Runs-in-CI is step 4's workflow item. The four dose-response figures now share one definition, `falsification/dose_response_figure.py`, used by the readers too. `frontier.pdf` is left as committed (exploratory; `frontier.py` would redraw it over 11 arms, not addendum 6's 8).
- [x] `docs/REPRODUCE.md` — figure/table → arms → data → function → equivalent script, plus the list of numbers that still need a GPU and a checkpoint.

### 3. Docs — 11 files → 5, frozen tense
- [x] `README.md` — front door: the claim under test, the six findings (one paragraph each, the A2 figure inline), reproduce, layout, framework quick start, cite. No status table; the session log went to `docs/notebook/README.md`.
- [x] `docs/RESULTS.md` — findings in final form, past tense; each section ends with a one-line pointer to the notebook addenda it rests on (a reviewer needs the trail), and a "predicted and wrong" section keeps the falsified predictions
- [x] `docs/METHODS.md` — thesis, framework, combinatorial floors, the three simulation results, κ/α, measurement conventions, training config, the pre-registration verbatim with outcomes beside it
- [x] `docs/ERRATA.md` — preprint corrections, every code bug (what / what it touched / status), replaced metrics, retracted results, SAEBench modifications (verified against `git log -- SAEBench-main`)
- [x] `docs/REPRODUCE.md` — RUNBOOK folded in (GPU commands, framework snippet); pinned versions are step 4's `requirements-lock.txt`
- [x] `docs/notebook/` — all eleven files moved with `git mv`, verbatim under a frozen header; `README.md` index explains the old names code comments still cite
- [x] `CLAUDE.md` → pointer at `docs/` + conventions (added: the paper's numbers are checked by `reproduce.py`, not typed)

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
| 2026-09-17 (session 3) | Step 3. Paper rewrite tabled by the author. `docs/{RESULTS,METHODS,ERRATA}.md` written; RUNBOOK folded into `docs/REPRODUCE.md`; eleven notebook files moved verbatim to `docs/notebook/`; README and CLAUDE.md rewritten. Branch `finishing-docs`, PR #10 (stacked on #9). |
| 2026-09-17 (session 2) | Step 2. Run metadata + `.npz` committed (2,110 files, 18 MB). `reproduce.py --check`: 125/125 paper numbers reproduce from committed data, CPU only. Figures regenerated at text width with paper-facing labels; paper's Table 3 (−2.0 row) and E2 gap (0.4410→0.4420) corrected. `docs/REPRODUCE.md`. Branch `finishing-verifiability`, PR #9 (stacked on #8). |
