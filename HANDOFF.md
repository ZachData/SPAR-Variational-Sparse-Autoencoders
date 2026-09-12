# HANDOFF — cold-start orientation

Short by design. Read this first, act from `PROJECT.md`. When this file and
`PROJECT.md` disagree, `PROJECT.md` wins — it is the living document and this one
is a table of contents with a heartbeat.

Last touched: 2026-09-12.

## Done 2026-09-12 — E4 box (5), the size-matched baseline (addendum 21): no confound, E4's checklist is fully closed

Trained a real TopK SAE from scratch at `dict_size=1474` (the vSAE's exact
live-feature count), config otherwise identical to the recovered baseline
(`falsification/train_e4_size_matched_baseline.py`, ~79s). Scored it directly
(no masking) with SCR and TPP on the original professor/nurse comparison point
(`falsification/score_e4_size_matched_baseline.py`). **Result: SCR's
trained-from-scratch score (0.020127) matches the masked-curve reference
(0.020127) to 6×10⁻⁸ — indistinguishable; TPP's (0.2025) is slightly *higher*
than the masked reference (0.1905, +0.0121).** Both headline verdicts survive
unchanged (SCR) or marginally reinforced (TPP) — "masked vs. trained-small"
was the last open confound in this design and it is not live. Side finding:
even matched to the vSAE's exact live-feature count, this baseline only kept
65.5% of its 1474 entries alive. **PROJECT.md Next steps #0 — all 5 boxes in
the SCR/TPP disagreement checklist — is now fully closed.**

## Done 2026-09-12 — A4, JumpReLU's discreteness test (addendum 20): confounded, not a clean replication

91/91 runs, 0 failures. The naive Pearson r(FVE, Jaccard) = +0.9334 looks like
a looser version of TopK's +0.9993 / BatchTopK's +0.9979, but it's confounded:
unlike those two hard-k architectures, JumpReLU's soft L0-target lets achieved
sparsity itself swing 36.8→290.8 across the sigma grid, and `r(FVE, L0)` is
just as tight (+0.9494). Restricting to the 4 grid points where L0 sits near
the 256 target, the coupling weakens to **r=+0.6297 (n=4)**, non-monotonically.
**Verdict: neither a clean replication nor refutation — the real finding is
that a soft, gradient-learned threshold isn't noise-robust in sparsity level,
not just selection stability, a failure mode hard top-k cannot exhibit by
construction.** Full account: RESULTS addendum 20, PROJECT.md's top entry and
"Next steps A" companion paragraph.

Getting here required writing `training_scripts/train_vsae_jumprelu.py` from
scratch (none existed) and fixing four real bugs in
`dictionary_learning/trainers/vsae_jump_relu.py`, all now in CLAUDE.md: a dead
threshold gradient (no STE), a missing L0-target sparsity loss, the gate
running before sampling instead of after (would have made selection churn
structurally impossible — a genuine design fork, resolved by asking the user,
who chose to restructure to match `vsae_topk.py`'s noise-before-selection
order), and `normalize_decoder()` not rescaling `threshold` (silently
corrupting evaluated output on any non-unit-norm decoder). All 115
falsification tests stayed green throughout.

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

## Current state (2026-09-12)

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
- **A1 is done and falsified (RESULTS addendum 15):** the usage-rank
  hypothesis for *why* SCR and TPP disagree does not hold. Both metrics'
  top-effect features sit in the same 0.55–0.73 usage-percentile band on the
  baseline SAE at every N in {2,5,10,20}; the sign of the small gap flips
  between N≤5 and N≥10; within-metric variance across classes dwarfs any
  between-metric difference. The SCR/TPP disagreement itself is still
  unexplained.
- **`PROJECT.md` Next steps #0 boxes (1)–(3) are done** (addenda 12–15).
- **A2 is DONE (RESULTS addendum 17).** Pre-flight (addendum 16) measured the
  k/(k+1) pre-activation gap on a baseline checkpoint and predicted no reachable
  knee; the sweep (4 new arms × 13 seeds, 52/52 runs, 0 failures) confirmed the
  curve is smooth, **but** found FVE tracks selection Jaccard almost exactly
  (Pearson r = +0.9993) across the whole 6-point grid — a sharper confirmation
  of Claim #3's mechanism than a threshold would have been, not the predicted
  decoupling. **Both threads of `Next steps A` are now closed.**
- **A3 (BatchTopK half of Claim #3's discreteness companion) is DONE (RESULTS
  addendum 18).** Found and fixed a live `scale_biases` bug in
  `vsae_batch_topk.py` (the same `var_encoder.bias` corruption addendum 8 fixed
  in `vsae_topk.py`) and found-but-didn't-fix a second bug that makes this
  trainer's `frac_recovered` unreliable (both now in CLAUDE.md). Seven new arms
  (deterministic baseline + A2's exact sigma grid) × 13 seeds, 91/91 runs, 0
  failures. **Result: the coupling generalises** — BatchTopK's r(FVE, Jaccard)
  = +0.9979, essentially as tight as TopK's +0.9993 — and the "elastic global
  budget should be more robust" intuition mildly *reverses* (BatchTopK closes
  80.5% of its own gap at the clamp floor vs. TopK's 84.1%).
- **A4 (JumpReLU, the sharpest discreteness test) is DONE (RESULTS addendum
  20) — see the top of this file for the full account.** Not a clean
  replication: the naive r=+0.9334 is confounded by an 8x swing in achieved
  sparsity A2/A3 never had (their hard-k always yields exactly k); controlling
  for it weakens the coupling to r=+0.6297 (n=4). The real finding is that a
  soft, gradient-learned threshold isn't noise-robust in sparsity level, a
  failure mode hard top-k cannot exhibit by construction. Wrote
  `training_scripts/train_vsae_jumprelu.py` from scratch and fixed four real
  bugs in `vsae_jump_relu.py` first (dead threshold gradient, missing
  L0-target loss, gate-before-noise ordering, `normalize_decoder` not
  rescaling threshold).
- **E4 box (4) is DONE (RESULTS addendum 19).** Widened SCR/TPP coverage to 3
  more `bias_in_bios` pairs and the second SAEBench dataset
  (`canrager/amazon_reviews_mcauley_1and5`). **Both verdicts replicate as the
  dominant pattern**: SCR "not explained by size" 4/4 on new `bias_in_bios`
  pairs, 3/4 on amazon (one weak exception); 7/8 combined. TPP "explained by
  size" replicates on amazon via `top_usage` but its `random` bracket goes
  from decisive to a coin flip. Found and fixed a real cache-contamination
  trap en route (`--smoke`-testing a pair before its real run silently
  poisons the "full" run's cache — now a CLAUDE.md landmine).
- **E4 box (5) is DONE (RESULTS addendum 21) — see the top of this file for
  the full account. `PROJECT.md` Next steps #0 is now FULLY CLOSED, all 5
  boxes.** "Masked vs. trained-small" is not a live confound: a TopK SAE
  trained from scratch at the vSAE's exact live count (1474) scores
  indistinguishably from the masked curve on SCR (6×10⁻⁸ apart) and slightly
  *better* than the masked curve on TPP — both headline verdicts survive.
- Branch `claude/falsification-framework`, pushed to origin, merged to `master`.

### Next task — nothing pre-selected; pick from the options below

Both mechanism-paper threads (`PROJECT.md` **Next steps A**), Claim #3's
discreteness companion for all three architectures (A2 TopK, A3 BatchTopK, A4
JumpReLU), and E4's ENTIRE SCR/TPP disagreement checklist (all 5 boxes) are
closed as of this session and the prior one: A1 (RESULTS addendum 15,
falsified), A2 (RESULTS addendum 17, confirmed sharper than predicted),
A3/BatchTopK (RESULTS addendum 18, generalises), A4/JumpReLU (RESULTS
addendum 20, confounded — neither a clean replication nor refutation, surfaces
a different finding about soft-threshold sparsity collapse under noise), E4
box (4) (RESULTS addendum 19, replicates), E4 box (5) (RESULTS addendum 21,
no confound). **The paper's three parts are now all established** — (1)
fixed-variance KL is a null L2 penalty (E1); (2) sampling-on damage tracks
selection churn almost linearly for hard top-k mechanisms (TopK, BatchTopK),
and a soft learned-threshold mechanism (JumpReLU) fails differently — its
sparsity level itself isn't noise-robust; (3) the literature's effect sizes
are implementation-variance-sized (E1's 5 factors, E3's ReLU, the
decoder-gradient projection, the initial weight draw). Nothing is queued
next — read `PROJECT.md`'s "Next steps" section fresh and pick from what
remains:

- **Write up the mechanism paper itself** — the material is now complete per
  the framing in `PROJECT.md` Next steps A's intro, including A4's more
  nuanced JumpReLU finding (confounded comparison, different failure mode)
  and E4's now-fully-closed checklist (the SCR/TPP disagreement itself is
  still unexplained, but every planned diagnostic against it, including the
  masking confound, has been run).
- **A follow-up A4 design that controls the L0 confound** — e.g. many more
  grid points concentrated in the L0-matched sigma region, or a hard L0 cap
  enforced architecturally rather than as a soft loss term (though the latter
  arguably turns JumpReLU into another top-k variant). Not scoped in code.
- **Extend E4 box (5)'s confound check to the other 7 (dataset, pair) points**
  addendum 19 added — box (5) as scoped only re-ran the original professor/
  nurse comparison; the 7 other points still rest on the masked-not-trained
  reference. Not required to close the box, but would extend the check's
  coverage the way box (4) extended boxes (1)-(3)'s.
- **Claims-worth-opening #4/#5** — desk work, no GPU, still open.

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

## Done 2026-09-11 — A1, falsified (addendum 15)

`falsification/read_e4_node_effects.py` reused the already-cached SCR/TPP
train-set activations (no new LLM or probe work, ~30s of SAE inference) to read
each metric's top-effect features and their usage-rank percentile on the
recovered baseline SAE. The stated prediction (SCR's top-effect features rank
low in usage, TPP's rank high) does not hold: both sit in the 0.55–0.73 band at
every N in {2,5,10,20}, the sign of the gap flips between N≤5 and N≥10, and
within-metric variance across classes (SCR's own `professor / nurse` vs.
`male / female`) dwarfs any between-metric difference. Per the stated falsifier,
the explanation is dead — the SCR/TPP disagreement stays unexplained. Closes
`PROJECT.md` Next steps #0 box (3).

## Done 2026-09-11 — A2, the sigma_init dose-response (addenda 16–17)

**Pre-flight (addendum 16).** `falsification/read_preact_gap.py` measured the
k/(k+1) pre-activation gap on `experiments/baseline/seed1`: median 0.0001, p99
0.0006 in training space. Even the `reparameterize()` clamp floor's sigma
(0.0498) sits ~80–550× above this, so no `log_var_init` keeps sigma below the
boundary gap — refined the "FVE and Jaccard knee together" prediction to "no
knee reachable; expect FVE less dose-sensitive than Jaccard."

**Sweep + result (addendum 17).** Four new arms
(`a2_sigma_init_{m1,m3,m4,m5}`, `log_var_init` ∈ {−1,−3,−4,−5},
`falsification/run_arm.py`) × 13 seeds ran via `falsification/run_a2_sweep.sh`
(52/52 runs, 0 failures, ~52 min). `falsification/read_a2_dose_response.py`
combined them with the two pre-existing endpoints (`e2_sampling_only` at −2.0,
`e2_sigma_low_init` at −8.0/clamped −6.0) into a 6-point curve. **The curve is
smooth (no knee, confirming half the pre-flight's refinement) but FVE tracks
Jaccard almost exactly — Pearson r = +0.9993 — across the whole grid (the
*other* half of the refinement, "decoupling," is wrong.)** FVE: 0.377 → 0.484
→ 0.614 → 0.743 → 0.817 → 0.834 as sigma falls 0.607 → 0.050; Jaccard: 0.766 →
0.812 → 0.870 → 0.918 → 0.945 → 0.952 in lockstep. Re-deriving
`e2_sigma_low_init`'s "84.1% of the gap closed" from this independent pipeline
reproduces addendum 7's number exactly. Even at the floor a real residual
remains (FVE gap 0.066, Jaccard 0.952) that the clamp prevents probing further.
Figure: `workshop/figs/a2_dose_response.pdf`.

## Done 2026-09-11 — A3, the BatchTopK discreteness companion (addendum 18)

Two pre-existing bugs in `vsae_batch_topk.py`, found while scoping this (both
now in CLAUDE.md): `scale_biases` carried the exact `var_encoder.bias`
corruption bug addendum 8 fixed in `vsae_topk.py` — fixed identically here.
`_apply_topk_sparsity`'s global `k*batch_size` budget silently breaks by a
factor of `~ctx_len` inside `loss_recovered()`'s 3D-activation hook (not
fixed — out of scope; use `frac_variance_explained`, not `frac_recovered`, for
this trainer). A third issue was purely cosmetic (a stale `d8192` in the
checkpoint directory name; the model itself is correctly `d2048`, verified
directly against `encoder.weight`'s shape).

Seven new arms (`a3_batchtopk_baseline` + A2's exact 6-point `log_var_init`
grid) × 13 seeds ran via `falsification/run_a3_sweep.sh` (91/91 runs, 0
failures, ~78 min). `falsification/read_a3_dose_response.py` mirrors A2's
reader but reads Jaccard from `VSAEBatchTopK.encode()`'s `selection_mask`
boolean tensor directly (BatchTopK doesn't guarantee exactly k active features
per token, so the TopK-specific `2k-intersection` union shortcut doesn't
apply). **Result: the coupling generalises.** r(FVE, Jaccard) = +0.9979
(BatchTopK) vs. +0.9993 (TopK) — both near-perfect. BatchTopK's own
deterministic baseline is notably better than TopK's (0.9509 vs. 0.900159),
but *relative* to each architecture's own baseline, BatchTopK is very slightly
*more* exposed to sampling noise at every matched sigma, not less (closes
80.5% of its own gap at the clamp floor vs. TopK's 84.1%) — the "elastic
global budget = more robust" intuition doesn't just fail to hold, it mildly
reverses. Figure: `workshop/figs/a3_batchtopk_dose_response.pdf`.

## Done 2026-09-11 — E4 box (4), coverage widened (addendum 19)

`run_e4_scr.py`/`run_e4_tpp.py` generalised with `--dataset`/`--column1-vals`
CLI args (default preserves the original invocation exactly). Found and fixed
a real trap first: `run_eval_single_sae`'s cache loads by filename existence
only, not by config, so `--smoke`-testing a new pair before its real run
silently poisons the "full" run with a ~20x-undersized cache (caught by
comparing cache file sizes against the known-good default; contaminated cache
and results deleted and redone) — now a CLAUDE.md landmine ("never `--smoke`
a pair/dataset you're about to run for real"). Five of the ten SAEBench eval
runs below were also killed mid-scoring by the harness's low-memory guard
(after the activation cache was already built) and needed one retry each —
an operational hazard, not a methodology bug; no results were affected.

**Result: both verdicts replicate as the dominant pattern.** SCR "not
explained by size": 4/4 on the 3 new `bias_in_bios` pairs (architect/
journalist +0.169, attorney/teacher +0.129, surgeon/psychologist +0.084;
mean +0.116, tight spread) and 3/4 on all 4 amazon pairs (mean +0.029, one
exception — Books/CDs_and_Vinyl at −0.100, though its `random` bracket is
near-zero, a weak rather than clean reversal). Combined: 7 of 8 (pair,
dataset) points say "not explained." TPP "explained by size" replicates on
amazon via `top_usage` (−0.061, vSAE score 0.1031 vs. `bias_in_bios`'s 0.1055
— barely moved) but its `random` bracket flips from decisive (−0.051) to a
coin flip (+0.0004) — what shifted is the baseline's own curve, not the vSAE.
Closes `PROJECT.md` Next steps #0 box (4); box (5) is the one item left open.

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
- **A2/Claim #3:** sampling-induced FVE damage tracks TopK selection-Jaccard
  instability almost exactly (r=+0.9993) across a 6-point, 13-seed-per-point
  sigma_init sweep — a smooth dose-response, not a threshold (addenda 16–17).
- **A3/Claim #3 (BatchTopK):** the same coupling holds for BatchTopK's global
  selection (r=+0.9979) — not TopK-specific — and BatchTopK is very slightly
  *more*, not less, exposed to noise at matched sigma (addendum 18).
- **A4/Claim #3 (JumpReLU):** NOT a clean third data point. The naive
  r=+0.9334 is confounded by an 8x swing in achieved sparsity (36.8→290.8)
  that TopK/BatchTopK's hard-k never allowed; L0-matched points weaken it to
  r=+0.6297 (n=4). The real finding: a soft, gradient-learned threshold's
  sparsity level isn't noise-robust, unlike hard top-k (addendum 20).
- **E4:** SCR says the vSAE's advantage is *not* explained by dictionary size;
  TPP says it *is*. Same checkpoints, same grid. The disagreement is the finding
  (thesis Failure 1, reproduced fresh). It is threshold-uniform (addendum 12)
  and survives both the conditional (addendum 13) and the full `--resample-train`
  (addendum 14) bootstrap in both directions — but SCR's margin CI is ~15×
  wider than TPP's and rests on the N=2 threshold alone. Both verdicts
  replicate as the dominant pattern across 3 more `bias_in_bios` pairs and a
  second dataset (7/8 SCR points "not explained," TPP's `top_usage` verdict
  holding on both datasets) — addendum 19. The masked-curve reference itself
  is confound-checked: a trained-from-scratch baseline at the vSAE's exact
  live count reproduces the masked curve's SCR score to 6×10⁻⁸ and scores
  slightly *higher* than it on TPP — "masked vs. trained-small" is not a live
  confound (addendum 21). **All 5 boxes of E4's SCR/TPP checklist are closed;
  the disagreement itself remains unexplained.**

## Next action

Both threads of `PROJECT.md` **Next steps A** are closed (A1 addendum 15, A2
addenda 16–17), Claim #3's discreteness companion is closed for all three
architectures (A3 BatchTopK addendum 18, A4 JumpReLU addendum 20 — confounded,
see above), and E4's entire SCR/TPP checklist is closed (box (4) addendum 19,
box (5) addendum 21) — nothing is pre-selected for the next session. See
"Next task" above for the menu (writing up the mechanism paper, a follow-up
A4 design that controls the L0 confound, or extending box (5)'s confound
check to the other 7 dataset/pair points). Read `PROJECT.md`'s Next steps
section fresh and pick.

## Environment

`python falsification/preflight.py` tells you which environment you are in.
Local RTX 3080 (10 GB) can train; remote/web sessions have no GPU or torch and
are limited to reading, analysis of committed JSONs, figures, and writing.
Training OOMs easily at 10 GB — use `falsification/run_arm.py`, never hand-edit
`create_full_config()`.
