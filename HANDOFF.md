# HANDOFF — cold-start orientation

Short by design. Read this first, act from `PROJECT.md`. When this file and
`PROJECT.md` disagree, `PROJECT.md` wins — it is the living document and this one
is a table of contents with a heartbeat.

Last touched: 2026-09-11.

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

## Current state (2026-09-11)

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
- Branch `claude/falsification-framework`, pushed to origin, merged to `master`.

### Next task — nothing pre-selected; pick from the options below

Both mechanism-paper threads (`PROJECT.md` **Next steps A**) are closed as of
this session: A1 (RESULTS addendum 15, falsified) and A2 (RESULTS addendum 17,
confirmed with a sharper result than predicted). **The paper's three parts are
now all established** — (1) fixed-variance KL is a null L2 penalty (E1); (2)
sampling-on damage tracks TopK selection churn almost linearly, no threshold
(A2); (3) the literature's effect sizes are implementation-variance-sized (E1's
5 factors, E3's ReLU, the decoder-gradient projection, the initial weight
draw). Nothing is queued next — read `PROJECT.md`'s "Next steps" section fresh
and pick from what remains:

- **E4 boxes (4)–(5)** — the second SAEBench dataset / other `bias_in_bios`
  class pairs (cheap, widens coverage), or training a size-matched baseline
  from scratch (expensive, removes the last confound in the reference curve).
- **Claim #3's discreteness companion** — JumpReLU vs. BatchTopK vs. TopK,
  the orthogonal axis to A2's noise sweep, now the natural next lever since
  addendum 17 found a real residual at A2's clamp floor that can't be probed
  further along the sigma axis. Touches `vsae_jump_relu.py`'s untested
  `scale_biases` path — verify that path is correct before trusting any
  `log_var` read off a JumpReLU checkpoint.
- **Write up the mechanism paper itself** — the material is now complete per
  the framing in `PROJECT.md` Next steps A's intro.
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
- **E4:** SCR says the vSAE's advantage is *not* explained by dictionary size;
  TPP says it *is*. Same checkpoints, same grid. The disagreement is the finding
  (thesis Failure 1, reproduced fresh). It is threshold-uniform (addendum 12)
  and survives both the conditional (addendum 13) and the full `--resample-train`
  (addendum 14) bootstrap in both directions — but SCR's margin CI is ~15×
  wider than TPP's and rests on the N=2 threshold alone.

## Next action

Both threads of `PROJECT.md` **Next steps A** are closed (A1 addendum 15, A2
addenda 16–17) — nothing is pre-selected for the next session. See "Next
task" above for the menu (E4 boxes 4–5, Claim #3's discreteness companion, or
writing up the mechanism paper). Read `PROJECT.md`'s Next steps section fresh
and pick.

## Environment

`python falsification/preflight.py` tells you which environment you are in.
Local RTX 3080 (10 GB) can train; remote/web sessions have no GPU or torch and
are limited to reading, analysis of committed JSONs, figures, and writing.
Training OOMs easily at 10 GB — use `falsification/run_arm.py`, never hand-edit
`create_full_config()`.
