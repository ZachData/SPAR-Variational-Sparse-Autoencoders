# FINISHING.md — turning the notebook into a finished project

Started 2026-09-17. The science is done; this file tracks the work of making the
repository read as finished. Check boxes as they land; every step is a PR onto
`master`. When everything is checked, this file moves to `docs/notebook/`.

## STATUS — read this first (updated 2026-09-17, end of session 4; handoff)

### Where the work is

Steps 1–4 are done. **`master` only has step 1.** The four step PRs were
merged into each other rather than into `master`: #8 (`finishing-plan`) →
`master`, then #9 → `finishing-plan`, #10 → `finishing-verifiability`, #11
→ `finishing-docs`. So the complete stack (steps 2, 3, 4 and this handoff)
lives on **`finishing-docs`**, and **PR #12 `finishing-docs` → `master`** is
the one merge that remains. After it lands:

```bash
git checkout master && git pull
git branch -d finishing-plan finishing-verifiability finishing-docs finishing-hygiene
git push origin --delete finishing-plan finishing-verifiability finishing-docs finishing-hygiene
```

CI (`.github/workflows/ci.yml`) runs for the first time on that merge. It
has been verified locally in a fresh venv (install, 115 tests with CPU
torch, `reproduce.py --check`) but never on a GitHub runner; if it fails,
the likely causes are the CPU-torch index URL line or the pip cache key,
not the tests.

### What each step produced (one line each; the checklists below have the detail)

| Step | Deliverable | Verify with |
|---|---|---|
| 1 | `workshop/mechanism_paper.pdf` compiles from `.tex` via `workshop/build_paper.sh`; compiler and read-through fixes | `./workshop/build_paper.sh` |
| 2 | `experiments/**` metadata + `.npz` committed (18 MB); `reproduce.py` rebuilds every paper figure and table, `--check` asserts 125 numbers; `docs/REPRODUCE.md` | `python reproduce.py --check` → exit 0 |
| 3 | `docs/{RESULTS,METHODS,ERRATA,REPRODUCE}.md`; `docs/notebook/` (frozen record + index); `README.md` front door; `CLAUDE.md` pointer | read `README.md` |
| 4 | `pyproject.toml`, `requirements-lock.txt`, `LICENSE`, `CITATION.cff`, `ci.yml`, dead code removed | `pip install -e ".[dev]" && pytest` → 115 |

### Tabled by the author (2026-09-17) — do not start unasked

The paper rewrite: step 1's editorial calls (abstract ~370 words vs a
150–250 cap; the author block placeholder; "d ≈ 16 on reconstruction" in
the abstract/§3.4 where Table 1's first row is −5.7 and 16.5 is the
*unitinit* rung) and any fuller revision of `workshop/mechanism_paper.tex`.
When it resumes: `reproduce.py`'s `CHECKS` (the `check(...)` calls) are the
paper's printed numbers — change both together and keep `--check` green.
The author placeholder is in three places that change together:
`mechanism_paper.tex`'s author block, `README.md`'s BibTeX, `CITATION.cff`.

### Next action: step 5, archive and release — needs the author

1. **Checkpoint archive to HuggingFace.** What to upload: the 434 final
   `experiments/<arm>/seed<n>/<run>/trainer_0/ae.pt` (2.47 GB; exclude the
   three top-level `experiments/VSAEJumpReLU_*` smoke-test dirs). Of the
   3,347 intermediate `ae_<step>.pt` (21 GB, the dense-schedule arms), the
   only ones a reported number depends on are `e2_sampling_only_early` and
   `e2_sigma_low_init_early` (2 x 70 files, 0.88 GB):
   `falsification/read_selection_jaccard.py` reads them for the step-wise
   Jaccard trajectory in RESULTS §3 (0.069 → 0.811), which is tabulated in
   notebook addendum 8 and cached nowhere else. Upload those two arms'
   intermediates too; skip the rest (the A2/A3/A4 arms' dense schedules
   were never read). Preserve the `experiments/...` path structure so
   `docs/REPRODUCE.md`'s GPU readers work on top of a download:
   ```bash
   pip install -U huggingface_hub && huggingface-cli login
   huggingface-cli upload <user>/<repo> experiments experiments --repo-type model \
       --include "*/seed*/*/trainer_0/ae.pt" "e2_*_early/seed*/*/trainer_0/ae_*.pt" \
       --exclude "VSAEJumpReLU_*/**"
   ```
   (~3.4 GB total. Check the `--include` globs against `huggingface-cli
   upload --help` for the installed version before trusting them.)
   Decisions that are the author's: the HF account/repo name; whether to
   also upload the 3 Pythia checkpoints under `experiments/e4_pythia_*`
   (they are in the glob above; keep them — E4 depends on them).
2. Link it: `README.md` → Reproduce ("the 2.3 GB of weights are not"),
   `docs/REPRODUCE.md` → "What is committed, and what is not".
3. `git tag v1.0 && git push origin v1.0`; GitHub release with
   `workshop/mechanism_paper.pdf` attached — **after** the paper rewrite
   lands, so the release PDF is the final one. `CITATION.cff` has
   `version: 1.0.0` and `date-released: 2026-09-17`; bump the date to the
   release day.
4. Zenodo DOI (optional; enable the GitHub integration before tagging if
   wanted, so the tag mints the DOI).
5. `git mv FINISHING.md docs/notebook/FINISHING.md` with the frozen header
   the other notebook files carry, and add its row to
   `docs/notebook/README.md`; drop the `FINISHING.md` bullet from
   `CLAUDE.md`.

### Known loose ends (not blocking; recorded so they are not rediscovered)

- `workshop/figs/frontier.pdf` is the 8-arm addendum-6 figure; running
  `falsification/frontier.py` now draws 11 arms with overlapping labels.
  Left as committed on purpose (`docs/REPRODUCE.md` says so). Fix the label
  placement if the figure is ever needed again.
- `reproduce.py` regenerates figures but CI does not byte-diff them
  (matplotlib stamps its version into the files). Locally, regenerated
  PDFs were byte-identical to the committed ones under matplotlib 3.10.8.
- Docstrings in `falsification/` and the trainers still cite "PROJECT.md",
  "RESULTS addendum N", "REMEDIATION F6" by their old names;
  `docs/notebook/README.md` resolves them. Deliberate.
- `experiments/e2_learned_var` has 6 seeds (the 13-seed battery superseded
  it); `a2_sigma_init_m1` has analysis `.npz` for 3 seeds only — those were
  a frontier side-check, not part of any table.
- The `VSAEBatchTopK` `frac_recovered` bug (ERRATA §2) is documented, not
  fixed; nothing in the paper uses that number.
- E0 (the pipeline negative control) was never run; RESULTS §9 says so.
- The wandb sweep plumbing in the training scripts is dormant and kept
  (step 4 checklist explains why).

### Environment

`/usr/bin/python3` (Python 3.14, torch 2.10 cu128, nnsight 0.7) for
anything with torch; bare `python`/`python3` on this machine is a CPU-only
conda base — fine for `reproduce.py` and the paper, not for training.
`pip install -e ".[dev]"` covers `reproduce.py` and the tests anywhere.
Must stay green: `pytest` → 115 (109 + 6 torch-only), `python reproduce.py
--check` → exit 0. `./workshop/build_paper.sh` builds the PDF (`tectonic`
in `~/.local/bin`).

**Do not** re-audit the repo, re-read the paper, or re-derive the plan.
Start by merging #12, then step 5 item 1 with the author.

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
- [x] `pyproject.toml` so `pip install -e .` is true — setuptools, packages `dictionary_learning`, `dictionary_learning.trainers`, `falsification`; base deps numpy/scipy/matplotlib; extras `train`, `e4`, `interp`, `dev`. Verified in a fresh venv: install, 109 tests + 1 skip without torch, 115 with CPU torch + einops, `reproduce.py --check` green.
- [x] `requirements-lock.txt` frozen from the working env (`pip freeze` of `/usr/bin/python3`, 294 lines, header says what it is); `requirements.txt` is now `-e .[train,e4,dev]`
- [x] `LICENSE` (MIT; upstream `dictionary_learning`'s copyright line kept, SAEBench's MIT noted — the vendored copy predates upstream's LICENSE file)
- [x] `CITATION.cff` (author placeholder, same as the paper — changes with the rewrite)
- [x] `.github/workflows/ci.yml` replaces upstream's Poetry/PyPI workflow: `pip install -e .[dev]` + CPU torch, `pytest falsification/tests/`, `python reproduce.py --check`, on push/PR to `master`. Figures are regenerated but not diffed (matplotlib stamps its version into the files). Badge in README.
- [x] `git rm` 43 tracked `__pycache__/*.pyc`
- [x] Removed root `train_jumprelu.py`, `train_vsae_jump_relu.py`, `sweep.out`
- [x] Removed `sae_vis/` and the four unreferenced `analysis_scripts/` (history keeps them); `analysis_scripts/` un-ignored so `online_histogram_analyzer.py` is a normal tracked file
- [x] Root `tests/`: removed. They were upstream `dictionary_learning`'s and import classes this fork does not have (`AutoEncoderNew`); the end-to-end one needs a GPU. `falsification/tests/` is the suite (`pytest` now defaults to it via `pyproject.toml`).
- [x] `wandb` sweep plumbing: **kept, deliberately.** `--sweep` mode / `BaseSweepRunner` is dormant and no result depends on it, but `train_vsae_topk.py`, `train_vsae_batchtopk.py` and `train_vsae_topk_masked_kl.py` import `dictionary_learning.base_sweep` at module scope and every result was produced through those scripts; cutting it means editing the trainers that produced the data for no gain. `training.py` also imports `wandb` at top level, so `wandb` stays in the `train` extra.
- [x] `SAEBench-main/`: version and modified files in `docs/ERRATA.md` §5, pointer in `docs/REPRODUCE.md`

### 5. Archive and release
- [ ] Upload the 434 final checkpoints (2.47 GB) + the two `*_early` arms' intermediates (0.88 GB) to a HuggingFace repo; link from README (STATUS has the command)
- [ ] Tag `v1.0`; GitHub release with the PDF attached
- [ ] Zenodo DOI (optional)
- [ ] Move this file to `docs/notebook/`

## Log

| Date | Done |
|---|---|
| 2026-09-17 | Audit; this plan written. Branches consolidated to `master` (PRs #6, #7). Paper compiled for the first time (`build_paper.sh`); compiler + read-through fixes, incl. Table 6's stale 6-seed numbers; PDF committed. All on PR #8. |
| 2026-09-17 (session 4, handoff) | PRs #9–#11 were merged into each other, not `master`; full stack is on `finishing-docs`; PR #12 `finishing-docs` → `master` opened. STATUS rewritten as a cold-start handoff: PR state, per-step deliverables, the tabled rewrite, step 5 with commands, loose ends. |
| 2026-09-17 (session 4) | Step 4. `pyproject.toml` (+ extras), `requirements-lock.txt`, `LICENSE`, `CITATION.cff`, `ci.yml` (tests + `reproduce.py --check`), 43 `.pyc` untracked, `sae_vis/`, four analysis scripts, root train scripts and root `tests/` removed; wandb sweep plumbing kept with reason. Branch `finishing-hygiene`, PR #11 (stacked on #10). |
| 2026-09-17 (session 3) | Step 3. Paper rewrite tabled by the author. `docs/{RESULTS,METHODS,ERRATA}.md` written; RUNBOOK folded into `docs/REPRODUCE.md`; eleven notebook files moved verbatim to `docs/notebook/`; README and CLAUDE.md rewritten. Branch `finishing-docs`, PR #10 (stacked on #9). |
| 2026-09-17 (session 2) | Step 2. Run metadata + `.npz` committed (2,110 files, 18 MB). `reproduce.py --check`: 125/125 paper numbers reproduce from committed data, CPU only. Figures regenerated at text width with paper-facing labels; paper's Table 3 (−2.0 row) and E2 gap (0.4410→0.4420) corrected. `docs/REPRODUCE.md`. Branch `finishing-verifiability`, PR #9 (stacked on #8). |
