# HANDOFF — cold-start orientation

Short by design. Read this first, act from `PROJECT.md`. When this file and
`PROJECT.md` disagree, `PROJECT.md` wins — it is the living document and this one
is a table of contents with a heartbeat.

Last touched: 2026-09-09.

## What this repo is

A fork of `dictionary_learning` extended with variational SAEs (vSAEs), plus a
vendored SAEBench copy and a sequential-falsification framework (e-values +
permutation tests) that decides what the vSAE experiments actually license.
Two lines of work:

1. **vSAE architectures + experiments** — `dictionary_learning/`,
   `training_scripts/`, `analysis_scripts/`. Behind the arXiv preprint.
2. **Falsification battery** — `falsification/`. The confirmatory work. This is
   where active effort goes.

## Which doc for what

| File | Use it for |
|---|---|
| `CLAUDE.md` | **Landmines in the vSAE code.** Read before touching `dictionary_learning/` or describing any checkpoint. Several are bugs that already produced false claims. |
| `PROJECT.md` | The living document: Status → Where things stand → What is established → Next steps, then the pre-registration. Cold-start reading order is stated at its top. |
| `falsification/RESULTS_2026-09-03.md` | Numbered addenda (currently through **#12**). Each experiment result lands here in full before `PROJECT.md`'s summary is updated. |
| `RUNBOOK.md` | Copy-pasteable commands, ordered so failures surface cheaply. |
| `OVERVIEW.md` | Plain-language "what is this project" for a non-specialist. |

## Current state (2026-09-10)

- Confirmatory battery **complete at 13 seeds/arm**, 5σ on every comparison.
  11 arms, 153 checkpoints, 0 failures. Framework: 115 tests green.
- **E1, E2, E3 have landed.** E0 pre-registered, never reported.
- **E4 has a first reading** (RESULTS addenda 10–14): SCR and TPP give opposite
  verdicts on the same two Pythia checkpoints; the disagreement is
  threshold-uniform (addendum 12) and survives both the conditional
  (addendum 13) and the full `--resample-train` (addendum 14) bootstrap — though
  SCR's side of it is ~15× wider and rests on the N=2 ablation threshold alone.
  Single seed each side, one dataset/class-pair — descriptive by design, not a
  permutation test.
- **`PROJECT.md` Next steps #0 boxes (1) and (2) are done** (addenda 12–14).
- **A second paper is now planned** — see `PROJECT.md` **Next steps A**, which
  takes priority over #0's remaining boxes. Two threads, nothing run yet.
- Branch `claude/falsification-framework`, pushed to origin, merged to `master`.

### Next task — `PROJECT.md` Next steps A, the two mechanism threads

**The paper this serves:** *"What does adding a KL term to a TopK SAE actually
do?"* Three parts — (1) at fixed variance, nothing: it is an L2 penalty and null
once implementation is matched (**established**, E1); (2) with sampling on it
hurts, and the mechanism is TopK **selection churn** (**Claim #3 confirmed**, but
without a dose-response curve — that is A2); (3) the literature's effect sizes
are the size of **implementation variance** (**established** — E1's 5 factors,
E3's ReLU d≈15–19, the decoder-gradient projection d=−14.3, the initial weight
draw d=−4.7/−7.3, none of which appear in any equations). E4 becomes a section,
not the thesis.

- **A1 — which features does each metric select? (zero GPU, do first)**
  Read what `get_effects_per_class_precomputed_acts` picks as top-effect for SCR
  and for each of TPP's five classes; cross-reference each against its rank in
  `all_histograms_*.npz`'s `feature_selection_counts`.
  **Prediction: SCR's top-effect features rank LOW in usage, TPP's rank HIGH.**
  That would explain addenda 13/14's puzzle — the two metrics agree on the vSAE
  (≈0.10 both) and disagree only on where the *masked baseline* lands — and would
  indict usage-frequency pruning, dead-feature counting and "more live features =
  better" as selecting against the features doing the interesting work. It would
  also explain addendum 6's frontier sign reversal.
  **Falsifier: if both metrics' top-effect features sit at comparable usage
  ranks, the explanation is dead.** Record that either way.

- **A2 — the σ_init dose-response (~80 min training)**
  13 seeds × ~6 `log_var_init` values on gelu-1l, `var_flag=1`, else matched to
  `e2_sampling_only`. Per checkpoint measure FVE damage and selection Jaccard
  (`falsification/read_selection_jaccard.py`).
  **Prediction: a threshold, not smooth decay** — noise flips the TopK argmax
  when σ exceeds the k/(k+1) pre-activation gap, so FVE and Jaccard should knee
  *together*. **Measure the gap distribution on a baseline checkpoint first**
  (free) so the knee is predicted, not fitted.
  **Watch:** `log_var_init=−8.0` is already below the clamp floor — straddle the
  clamp or points collapse onto each other. Liveness analysis is ~6.5
  min/checkpoint (~8.5 h for 78), so subset it. Use `run_arm.py` with a per-seed
  `--output-dir`.

Deferred behind A: #0 boxes (4) second SAEBench dataset / class pairs, and (5)
train a size-matched `dict_size=1474` Pythia baseline. Box (3) is promoted into
A1.

### Done 2026-09-09/10 — box (2), the bootstrap error bars (addenda 13–14)

**Conditional bootstrap (addendum 13, both metrics).** TPP ran to completion on
the 5-point grid `500 1000 1474 2000 3000` (200 draws, 5.06 h) after a ~15-line
fix: `e4_bootstrap.py` now writes `*_results.json.partial` every 10 draws and
resumes from it exactly (draw RNG seeded by draw index). Both
`vsae − top_usage@1474` margins clear zero — SCR **+0.082 [+0.024, +0.150]**,
TPP **−0.086 [−0.090, −0.081]** — so the disagreement is not a test-set-noise
artifact. SCR's CI is ~15× wider, per threshold clears zero only at N=2; TPP
clears zero at N=2/5/10.

**Full `--resample-train` bootstrap (addendum 14, SCR).** Node effects
re-derived from a train-set resample every draw — 200 draws, full 9-point grid,
7.2 h — after adding 10GB-card memory handling to `e4_bootstrap.py`'s
`--resample-train` path. SCR's mean margin widens to **+0.088 [+0.008, +0.171]**
but does NOT cross zero: addendum 13's caveat is resolved. SCR's N=2/N=5 feature
selection is bit-identical between the two bootstraps; all extra CI width is
N≥10. `falsification/e4_bootstrap_scr_resample_train_results.json`.

## What is established (one line each — detail in RESULTS + PROJECT.md)

- **E1:** a fixed-variance vSAE *is* a TopK SAE with an L2 penalty (identity
  verified to 6 decimals). The measured d≈16 / d≈13 gap between the two
  implementations decomposes entirely into 5 optimiser/init details; with all 5
  matched the arms are null everywhere. Code diff enumerated at 15 items, all
  settled.
- **E2:** it is the sampling, not the KL. Removing the KL entirely recovers only
  6.4% of the FVE gap to baseline; the other 93.6% is the reparameterisation.
  Robust to the `scale_biases` correction (addenda 8–9).
- **E3:** the `F.relu(mu)` that only one trainer applies is a d≈15–19 effect,
  run as its own arm rather than patched away.
- **E4:** SCR says the vSAE's advantage is *not* explained by dictionary size;
  TPP says it *is*. Same checkpoints, same grid. The disagreement is the finding
  (thesis Failure 1, reproduced fresh). It is threshold-uniform (addendum 12)
  and survives both the conditional (addendum 13) and the full `--resample-train`
  (addendum 14) bootstrap in both directions — but SCR's margin CI is ~15×
  wider than TPP's and rests on the N=2 threshold alone.

## Next action

`PROJECT.md` **Next steps A** — **A1 first** (zero GPU, has a stated falsifier),
then **A2** (~80 min training). Both are scoped in full there; the summary is
under "Next task" above. Read Claims-worth-opening **#3** (the selection-churn
mechanism, confirmed for TopK — A2 is its noise axis) and **#6** before starting,
plus RESULTS addenda 10–14 for E4's state.

Neither thread is scoped *in code* yet — A1 needs a reader for
`get_effects_per_class_precomputed_acts`'s output; A2 needs a `log_var_init`
sweep added to `run_arm.py`'s `ARMS`.

## Environment

`python falsification/preflight.py` tells you which environment you are in.
Local RTX 3080 (10 GB) can train; remote/web sessions have no GPU or torch and
are limited to reading, analysis of committed JSONs, figures, and writing.
Training OOMs easily at 10 GB — use `falsification/run_arm.py`, never hand-edit
`create_full_config()`.
