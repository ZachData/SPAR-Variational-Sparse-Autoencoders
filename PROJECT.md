# PROJECT.md — Falsification-based validation of sparse autoencoder claims

**This is the living document.** It carries current state, what is established,
what to do next, and the pre-registration the battery runs under. It absorbed the
old `HANDOFF.md` on 2026-09-03; a new, deliberately short `HANDOFF.md` was
re-added on 2026-09-08 as a cold-start table of contents that points here — this
file stays authoritative when the two drift. `CLAUDE.md` has the repo's standing
landmines about the vSAE code — read that too, and read it before touching
anything in `dictionary_learning/`.

Reading order for a cold start: **Status** → **Where things stand** → **What is
established** → **Next steps**. Everything after that is the design and the
pre-registration, which change rarely; the sections before it change every session.

Last updated: 2026-09-11. **This session:** ran **A1**, the first of the two
mechanism-paper threads planned last session —
`falsification/read_e4_node_effects.py`, zero new GPU work beyond SAE inference
on already-cached activations (~30s). **Result: the usage-rank prediction is
FALSIFIED — RESULTS addendum 15.** The stated prediction (SCR's top-effect
features sit low in the baseline's usage ranking, TPP's sit high, which would
explain why the baseline's `top_usage` masking curve is flat for SCR but rising
for TPP) does not hold: both metrics' top-N features sit in the same 0.55–0.73
usage-percentile band on the baseline SAE at every N in {2,5,10,20}, the small
gap between them flips sign between N≤5 and N≥10, and within a single metric
individual classes scatter across almost the full usage-rank range (SCR's
`professor / nurse` and `male / female` disagree with each other as much as
either disagrees with any TPP class). Per the falsifier stated in advance, the
explanation is dead. PROJECT.md Next steps #0 box (3) closes with this
(promoted into A1, per the prior session's plan); the SCR/TPP disagreement
itself (addenda 10–14) stands unexplained by this hypothesis.

**Then ran A2, the σ_init dose-response, end to end.** Pre-flight
(`falsification/read_preact_gap.py`, **RESULTS addendum 16**) measured the
k/(k+1) pre-activation gap on a baseline checkpoint before choosing the sweep
grid and found even the `reparameterize()` clamp floor's sigma sits ~80–550×
above it — refining the original "FVE and Jaccard knee together at a
threshold" prediction to "no knee is reachable; expect FVE to be less
dose-sensitive than Jaccard." The sweep (4 new arms × 13 seeds,
`falsification/run_a2_sweep.sh`, ~52 min, 0 failures) then ran to completion.
**Result (RESULTS addendum 17): the first half of the refined prediction is
right — the curve is smooth, not kneed — but the second half is wrong.** FVE
tracks convergence-checkpoint selection Jaccard almost exactly across the
entire 6-point, >12×-sigma-range grid: **Pearson r = +0.9993** (13 seeds per
point). That is a sharper, more specific confirmation of Claim #3's mechanism
(selection churn drives the FVE damage) than a threshold would have been — a
single near-linear relationship holding continuously across six well-separated
operating points, not just two. Re-deriving `e2_sigma_low_init`'s "84.1% of the
gap closed" from this run's independent pipeline reproduces addendum 7's number
exactly, validating the measurement. Even at the achievable floor (clamp-limited
sigma=0.0498), a real residual FVE gap (0.066) and non-unit Jaccard (0.952)
remain — the clamp mechanically prevents testing lower sigma, so Claim #3's
other open half (varying the *discreteness* of the sparsity mechanism —
JumpReLU vs. BatchTopK vs. TopK) is the next lever, not a finer sigma sweep
against the same clamp. **Both threads of Next steps A are now closed.**

**Then ran A3, Claim #3's discreteness companion: does A2's FVE-vs-Jaccard
coupling depend on TopK's per-token selection, or hold for BatchTopK's global,
more elastic budget too?** Two pre-existing bugs in `vsae_batch_topk.py` had
to be handled first (both now in CLAUDE.md): `scale_biases` carried the exact
`var_encoder.bias` corruption bug addendum 8 fixed in `vsae_topk.py` (fixed
identically here) and `_apply_topk_sparsity`'s global budget silently breaks
by a factor of `~ctx_len` inside `loss_recovered()`'s 3D hook (not fixed — out
of scope; `frac_variance_explained` is unaffected and is what this uses).
Seven new arms (deterministic baseline + A2's exact 6-point sigma grid) × 13
seeds, 91/91 runs, 0 failures, ~78 min. **Result (RESULTS addendum 18): the
coupling generalises.** BatchTopK's own r(FVE, Jaccard) = **+0.9979**, just as
tight as TopK's +0.9993 — the "more elastic budget should be more robust"
intuition is not just falsified but mildly reverses: at every matched sigma
BatchTopK recovers a slightly *smaller* fraction of its own (higher) baseline
than TopK does of its own (80.5% vs. 84.1% of the gap closed at the clamp
floor). Claim #3's mechanism — selection churn costs reconstruction in direct
proportion — looks like a property of hard top-k-style selection under noise
in general, not a TopK-specific quirk. JumpReLU (no training script exists;
CLAUDE.md flags the trainer as never exercised end to end) remains the
sharper test of *discreteness itself* versus *hard top-k specifically*, and is
not attempted this session — deliberately scoped out given the size of that
lift versus BatchTopK's.

**Then closed E4 box (4): widened SCR/TPP coverage to 3 more `bias_in_bios`
pairs and the second SAEBench dataset (`canrager/amazon_reviews_mcauley_
1and5`).** `run_e4_scr.py`/`run_e4_tpp.py` generalised with `--dataset`/
`--column1-vals` CLI args. Found and fixed a real methodological trap before
trusting any of it: `run_eval_single_sae`'s cache loads by filename existence
only, so `--smoke`-testing a new pair before its real run silently poisons the
"full" run with a 20x-undersized cache (caught by comparing cache file sizes;
contaminated cache and results deleted and redone) — now a standing landmine
in CLAUDE.md ("never `--smoke` a pair/dataset you're about to run for real").
**Result (RESULTS addendum 19): both verdicts replicate as the dominant
pattern, with real added variance from the second dataset, not a reversal.**
SCR "not explained by size" replicates 4/4 on the 3 new `bias_in_bios` pairs
(mean margin +0.116) and 3/4 on amazon's own 4 pairs (mean +0.029, one
reversal — Books/CDs_and_Vinyl — with a near-zero `random` bracket, so a weak
exception rather than a clean flip); combined, 7 of 8 (pair, dataset) points
say "not explained." TPP's "explained by size" verdict replicates on amazon
via `top_usage` (−0.061) but its `random` bracket goes from decisive
(−0.051 on `bias_in_bios`) to a coin flip (+0.0004) — the vSAE's own TPP score
barely moves (0.1055 vs. 0.1031); what shifts is where the baseline's curve
sits. Box (4) is done; box (5) (train a size-matched baseline from scratch)
is the one item left in E4's checklist.

Prior session (2026-09-10): ran the full (non-conditional)
`--resample-train` SCR bootstrap — **RESULTS addendum 14** — the one thing
addendum 13 flagged as unrun. 200 draws, full 9-point grid, 7.2 h, after adding
10GB-card memory handling to `e4_bootstrap.py`'s `--resample-train` path.
**Result: SCR's mean margin `vsae − top_usage@1474` widens from +0.082
[+0.024, +0.150] to +0.088 [+0.008, +0.171] but does NOT cross zero** — the
addendum-13 caveat is resolved, the verdict survives. New sub-finding: SCR's
N=2 and N=5 feature selection is *bit-identical* between the conditional and
full bootstrap (resampling the train set doesn't change which top-2/top-5
features get picked); all the extra CI width comes from N≥10. Box (2) fully
done; boxes (3)–(5) untouched at that point.

Also that session: **planned the mechanism paper and its two remaining threads —
`Next steps A`.** The framing is *"What does adding a KL term to a TopK SAE
actually do?"*: fixed-variance KL is a null L2 penalty (established), sampling-on
damage is TopK selection churn (Claim #3, confirmed but without a dose-response
curve), and the literature's effect sizes are the size of implementation variance
(established). **A1** = which features SCR/TPP actually select, with a usage-rank
prediction and a stated falsifier, zero GPU. **A2** = the σ_init dose-response,
~80 min of training, predicting a *threshold* at the k/(k+1) pre-activation gap.

Prior session (2026-09-09): finished box (2)'s conditional bootstrap — **RESULTS
addendum 13**. TPP ran to completion on the 5-point grid `500 1000 1474 2000
3000` in 5.06 h after a ~15-line `.partial`/resume fix to `e4_bootstrap.py`.
Both metrics' paired margins are resolved away from zero under test-set
resampling in opposite directions — SCR **+0.082 [+0.024, +0.150]**, TPP
**−0.086 [−0.090, −0.081]** — so the SCR/TPP disagreement is not a test-set-noise
artifact. SCR's side is much thinner and rests on the N=2 ablation threshold
alone (N=5/10/20 straddle zero); TPP's clears zero at N=2/5/10. The SCR half's
conditional run had completed 2026-09-08 (`e4_bootstrap_scr_results.json`, full
9-point grid, ~5.7 h); the TPP half was killed twice by session/machine limits
before the fix.

Prior session (2026-09-07): Next steps #0 box (1) — the E4 per-threshold logging
gap (Claims-worth-opening #6a) — closed. `run_e4_scr.py`/`run_e4_tpp.py` now
store the full per-`n_value` breakdown at every baseline grid point; both rerun
against the warm caches and reproduced addenda 10–11's mean verdicts exactly, and
`falsification/e4_per_threshold_analysis.py` shows the SCR/TPP disagreement is
**not an averaging artifact** — SCR is "not explained by size" at all four
ablation thresholds, TPP is "explained by size" at all four (decisively only at
N≤10; a tie at N=20). RESULTS addendum 12.

The four things that landed across the prior two sessions, in order:

1. **Next steps #0 from the prior session is closed.**
   `falsification/reeval_var_flag1.py` re-ran the official `evaluate()` pipeline
   on all 37 on-disk `var_flag=1` checkpoints with the `scale_biases` bug
   corrected. **The result is reassuring, not alarming**: officially-reported FVE
   moves by ≈0.003-0.006 once corrected, not the ≈0.12 a pre-existing uncommitted
   proxy script had estimated — that proxy figure does not survive contact with
   the real pipeline and is superseded (RESULTS addendum 9). The 94.7%/5.3%
   reparameterisation/KL split that is E2's headline finding is essentially
   unchanged (93.6%/6.4% corrected). `evaluate()`'s own small-sample `frac_alive`
   did move a lot (0.85→1.00 for `e2_confirm`) but the pre-registered liveness
   thresholds are untouched by this bug entirely, because the histogram analyzer
   that computes them never samples (`training=False` unconditionally). Corrected
   results are written to `evaluation_results_corrected.json` alongside every
   affected checkpoint's original file (left untouched for the record);
   `compare_arms.py` and `frontier.py` now prefer the corrected file
   automatically.
2. **E4's "missing weights" blocker is resolved — the checkpoints were found, not
   retrained.** A search of other drives mounted on this machine turned up a
   prior/parallel copy of this project on `HDD_1TB` holding the actual `ae.pt`
   files for both the preprint's Pythia baseline
   (`TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto`) and its actual vSAE
   (`VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var`, `var_flag=0`).
   Both verified to load and run under the current codebase, checksummed and
   copied into `experiments/e4_pythia_baseline/seed42/` and
   `experiments/e4_pythia_vsae/seed42/`. `config.json` recovers the original
   hyperparameters directly (single seed 42; `auxk_alpha` 0.03125 vs. 0, CLAUDE.md
   landmine 3's confound confirmed in the checkpoint itself). The per-feature
   usage array E4's size-response curve needs was already sitting in the
   committed `all_histograms_*.npz`. **No retraining decision was needed after
   all** — the only remaining gap was writing the SAEBench-side scorer, closed
   next (below).
3. **E4's SAEBench SCR scorer is written and run against the recovered
   checkpoints — the vSAE's SCR advantage is not explained by dictionary size.**
   `falsification/e4_local_sae.py` (local, non-Hub checkpoint loaders) and
   `falsification/run_e4_scr.py` (the masking grid, scorer, and verdict) close
   Next steps #0. Masking the baseline TopK SAE down to a grid of N features by
   usage and scoring SCR at each point produces a curve that stays flat and near
   zero from N=100 to N=7379 (max 0.033); the vSAE's own score at its natural,
   unmasked 1474-feature live count is 0.102 — above every single point on that
   curve and above every one of 90 random-subset draws bracketing it from below.
   Getting the vendored SAEBench copy to import at all needed three environment
   fixes (a `sae_lens` API path that moved between the pinned version and the
   installed one, a `beartype` upgrade for Python 3.14 compatibility, and one
   missing package) — none touch evaluation logic; the falsification test suite
   and `preflight.py` are still green. Single seed, single dataset, single class
   pair — descriptive, not a permutation test, exactly as E4's design
   anticipated; see RESULTS addendum 10 for the full table and caveats,
   including what this does *not* rule out (CLAUDE.md landmine 3's AuxK
   confound).
4. **E4's TPP scorer reverses the SCR verdict — SCR and TPP disagree on
   whether size explains the vSAE's advantage.** `falsification/run_e4_tpp.py`
   mirrors the SCR runner for TPP; the vSAE's own TPP score (0.106) sits
   *below* the baseline's same-size `top_usage` curve (0.190 at n=1474) and
   below the `random` reference too (RESULTS addendum 11) — the opposite of
   addendum 10. This is CLAUDE.md's thesis Failure 1 (no principled way to
   combine metrics that disagree) reproduced fresh, on the same two
   checkpoints, same day. **Time ran out before any follow-up could be
   executed**, so the session instead worked out *why* the two metrics might
   disagree — four hypotheses (a logging gap that collapsed a per-threshold
   shape into one scalar; SCR's ratio-based score vs. TPP's difference-based
   one; specialisation-vs-coverage; a masked-vs-trained-small confound in the
   reference curve) — and recorded them as Claims-worth-opening #6 and a
   5-item TODO checklist at Next steps #0, in priority order, cheapest first.
   **Nothing in that checklist has been started** — it is queued for the next
   session with GPU time.

Branch `claude/falsification-framework`, GPU idle, pushed to origin and merged to
`master`. 11 arms in the confirmatory battery (153 checkpoints) plus the 2
single-seed Pythia checkpoints for E4.

Prior session (2026-09-04, second session), for context: (1) the sigma-annealing
arm (`e2_sigma_low_init`, addendum 7) closed 84% of E2's FVE gap to baseline; (2)
E4 was believed **blocked** on missing Pythia checkpoint weights (later found —
see above); (3) building the Jaccard-overlap instrumentation to explain (1)'s
residual gap surfaced **a real bug** in `scale_biases` that corrupted every saved
`var_flag=1` checkpoint's learned sigma (addendum 8) — corrected, RESULTS
addendum 3's "the posterior collapses completely" and "sampling noise is harmless
at eval time" both reverse, and the corrected Jaccard read then confirms
Claims-worth-opening #3 cleanly. Bug fixed in the trainer code; addendum 7's own
numbers were unaffected.

## Status

| | |
|---|---|
| Stage | Battery complete at 13 seeds; **E1, E2 and E3 have all landed**; E2's mechanism was corrected twice (addenda 7, 8) and then officially re-measured (addendum 9) |
| Framework | `falsification/` implemented, **115 tests green**, Type-I control verified |
| Newest figure | `workshop/figs/a3_batchtopk_dose_response.pdf` — BatchTopK's own FVE/Jaccard curve and r=+0.9979 scatter (addendum 18), companion to `a2_dose_response.pdf`'s TopK version (r=+0.9993, addendum 17). `workshop/figs/frontier.pdf` (liveness/reconstruction frontier, 8 arms) is older still. |
| Data | 11 arms, 153 checkpoints, 13 seeds/arm (2 new arms at 5 seeds each), 0 failures |
| Newest result | **E4 box (4) done (addendum 19): the SCR/TPP disagreement generalises across pairs and datasets, mostly.** SCR "not explained by size" replicates 4/4 on 3 new `bias_in_bios` pairs and 3/4 on all 4 amazon pairs (one weak exception); combined 7/8. TPP "explained by size" replicates on amazon via `top_usage` but its `random` bracket goes from decisive to a coin flip. Found and fixed a real cache-contamination trap (`--smoke` before a real run silently poisons it) en route — CLAUDE.md landmine. |
| In progress | Nothing running. **A1/A2/A3 (addenda 15, 17, 18) and E4 box (4) (addendum 19) are all done this session.** Next up is undecided: see Next steps below (E4 box (5) — a size-matched baseline from scratch, JumpReLU's training script, or the mechanism-paper writeup). |
| Blocking | Nothing blocked on compute or data. Boxes (1)–(4) done (addenda 12–15, 19); box (5) open. |
| Prior artifact | arXiv preprint; workshop draft on `claude/vae-workshop-paper-condensing-zumu6b` |

## Where things stand

The confirmatory battery is **complete at 13 seeds per arm** and reaches **5 sigma**
on every comparison. E1, E2 and E3 have all landed; E0 has never been reported.
E4 has a first reading (addenda 10–14, SCR and TPP, one dataset/class-pair,
single seed each side, descriptive by design; the two metrics disagree, that
disagreement is threshold-uniform, and it survives both the conditional and the
full `--resample-train` bootstrap — though SCR's side of it is much the thinner)
— see "What is established" below.

**How E1 landed.** The code diff between the two arms was enumerated by reading
`top_k.py` and `vsae_topk.py` against each other, plus the two training scripts,
and frozen at **15 differences and no more** (RESULTS addendum 4): 2 matched by
config, 2 already run as factors, 2 measured to be no-ops, 7 static no-ops shown
by reading, 1 the seed-permutation design already treats as noise, and exactly one
never run. That last one — the initial weight draw — was then run as
`e1_vsae_ref_fullmatch` (RESULTS addendum 5), and **with every enumerated
difference matched the two implementations are indistinguishable on all four
metrics** (largest effect d = 0.8, nothing significant) at the power where every
previous generation of the arm was detected at 5 sigma. The factor set was fixed
by the code diff *before* the arm ran, so this is confirmatory rather than a
garden of forking paths, and the recorded prediction for the closing arm (that it
would be null) was **wrong** — d = −4.7 on FVE on its own.

The liveness/reconstruction frontier ("Claims worth opening" #2) has been read too
— RESULTS addendum 6.

**E2's story changed twice this session, in two directions.** The sigma-annealing
arm (addendum 7) first closed 84% of E2's FVE gap to baseline by pinning sigma at
the reparameterise clamp floor from step 0, leaving a real 5-sigma residual.
Building the instrumentation to explain that residual then found a genuine bug in
`scale_biases` (addendum 8) that had been corrupting every saved `var_flag=1`
checkpoint's learned sigma since `var_flag=1` was added — corrected, RESULTS
addendum 3's "posterior collapses completely" and "eval-time noise is harmless"
both reverse, and the corrected Jaccard-overlap read then confirms
Claims-worth-opening #3 (selection instability under sampling noise) cleanly.
Addendum 7's own numbers are unaffected. Full account under **What is
established** below, in the E2 section.

## What is established

### E1 — confirmed, once every enumerated implementation difference is matched

`CLAUDE.md` landmine 1: at `sigma = 1` a vSAE's KL reduces to `0.5*||mu||^2`, so a
fixed-variance vSAE *is* a TopK SAE with an L2 activation penalty. Verified as
algebra (both trainers return `511.895264` on the same batch, six decimals) and now
as a claim about trained checkpoints.

The pilot's apparent difference (d = 37–59) was **three stacked implementation
mismatches**, removed in sequence: `kl_warmup_steps`, `use_april_update_mode`, and
decoder init scale. With all three matched, on the pre-registered liveness metric at
13 seeds:

| | < 0.1x k/d | < 0.5x k/d | thresholds |
|---|---|---|---|
| TopK+L2 vs vSAE (init 0.1, historical) | d = −2.8, 5.03σ | d = +13.0, 5.03σ | **disagree in direction** |
| **TopK+L2 vs vSAE (init 1.0, matched)** | **d = +0.8, p = 0.114** | **d = +0.2, p = 0.614** | **agree → indistinguishable** |

Neither threshold significant, both agreeing, at the highest power run.

That table was one generation of the arm. Two more factors followed, and the
verdict is the last row of this ladder — every vSAE arm against `e1_penalty`, in
the order the factors were matched (RESULTS addendum 5):

| vSAE arm | factors matched | FVE | `frac_recovered` | `< 0.1x` | `< 0.5x` |
|---|---|---|---|---|---|
| `e1_vsae_ref` | KL warmup, bias form | d = −5.7 | −8.5 | −2.8 | **+13.0** |
| `e1_vsae_ref_unitinit` | + init scale | **+16.5** | +17.7 | +0.8 ns | +0.2 ns |
| `e1_vsae_ref_gradproj` | + gradient projection | +3.8 | +4.9 | −2.4 | −3.7 |
| **`e1_vsae_ref_fullmatch`** | **+ init draw** | **−0.3 ns** | **−0.8 ns** | **−0.1 ns** | **−0.6 ns** |

Every intermediate generation is significant on at least one axis and each one
*traded* — matching the init scale closed liveness and blew open reconstruction,
matching the projection closed most of reconstruction and re-opened liveness.
**Only the last row is null everywhere, and it is the row where the code diff is
empty.** The largest effect anywhere in it is d = 0.8 (p = 0.071), at the power
where every earlier generation was detected at 5 sigma.

The claim this licenses is the decomposition, not a bare equivalence verdict:
*two implementations of an algebraically identical objective differ at d ≈ 16 on
reconstruction and d ≈ 13 on liveness; the entire difference decomposes into five
optimiser- and initialisation-side details, none of which appears in either
paper's equations; with all five matched they become indistinguishable.*

### E1's newest factor — the decoder-gradient projection

`vsae_topk.py` imported `remove_gradient_parallel_to_decoder_directions` and never
called it while still renormalising the decoder to unit norm, so the radial gradient
component was applied and then undone. Run as a factor (`project_decoder_grad`,
default off), 13 seeds:

* **It was the right explanation for reconstruction.** 78.9% of the FVE gap closes,
  74.9% of `frac_recovered`.
* **It breaks the liveness result.** Both pre-registered thresholds now separate the
  arms and both agree in direction, so this is a robust effect by F8b, not the
  shape change the two-threshold rule caught twice before.
* **It moved the vSAE away from `e1_penalty` on liveness, not toward it** — 0.1816
  (off) → 0.2197 (on) against `e1_penalty`'s 0.1836, and `e1_penalty` has had the
  projection all along. The same update-rule change has opposite-signed effects in
  the two implementations. That is an interaction, not a missing match.

The degeneracy is an identity between *objectives* (verified to six decimals on a
shared batch). It says nothing about two optimisers descending that objective
landing in the same place, and these arms measure that they do not.

### E1's factor set is closed — the code diff is 15 items and all 15 are settled

Enumerated 2026-09-04 by reading `top_k.py` and `vsae_topk.py` against each other
(plus `top_k_with_feature_penalty.py`, which is `top_k.py` and the penalty term
and nothing else, and the two training scripts). Full table in RESULTS addendum 4.

The config surface is fully matched — every field both arms expose is equal in the
saved `config.json`s — so every remaining difference is in code, and there are 15:

| status | count | items |
|---|---|---|
| matched by config | 2 | `kl_warmup_steps`, bias form |
| run as a measured factor | 2 | `decoder_init_scale`, `project_decoder_grad` |
| **no-op, measured** | 2 | dead-feature rule; the **±10 penalty clamp** |
| no-op, static (unreachable or algebraically inert) | 7 | `abs()` selection, the `threshold` buffer, the geometric-median guard, two epsilon/clamp guards, the dead-counter update site, the 2.7% init-norm offset |
| not a factor by design | 1 | RNG consumption order — the seed is the unit of randomisation |
| **run as a measured factor (the closing arm)** | **1** | the initial weight draw: normalised uniform vs. normalised Gaussian — `e1_vsae_ref_fullmatch`, and it is **not** null (d = −4.7 on FVE) |

**The clamp is the one that mattered.** `vsae_topk.py:867` penalises
`clamp(z, -10, 10)` and `top_k_with_feature_penalty.py:570` penalises `f`
unclamped — the same tensor at `var_flag=0`, but not the same function, and above
the clamp the vSAE's penalty gradient is **exactly zero**. It was the only
enumerated difference in the *objective* rather than the optimiser, so it was the
only one that could have qualified the six-decimal loss identity E1 rests on.
Measured over 20,000 activations, 13 seeds, all three E1 arms: the largest pre-TopK
activation anywhere is **0.194 against a clamp at 10**, and no entry in any arm
exceeds it. The identity is unqualified, and **the entire measured gap between the
arms is optimiser-side.**

Read that number in the right space or it misleads: training runs on activations
normalised to unit mean squared norm and the clamp acts there, but checkpoints are
saved with biases scaled back up by `norm_factor`. In the saved space the maximum
reads 4.96 and the headroom looks like a factor of 2 rather than 51.

### E2 — it is the sampling, not the KL (93.6% / 6.4%, corrected)

`kl_coeff` is not a shared scale across `var_flag`. A pre-registered two-stage design
(pilot at seed 101, confirmatory at seeds disjoint from it) found that **no beta in
{1e-4 … 1} puts a `var_flag=1` model within 0.02 FVE of baseline** — FVE degrades
monotonically (0.4581 → 0.0001) and never approaches baseline's 0.9003.

The decisive run was a control with **sampling on and the KL entirely off**. Numbers
below are from the official `evaluate()` re-run with the `scale_biases` bug
corrected (RESULTS addendum 9); the originally-reported (buggy) values were
0.460658 and 0.484106 and gave a 94.7%/5.3% split — the corrected split below is
essentially the same:

| config | FVE (13 seeds, corrected) |
|---|---|
| baseline (deterministic) | 0.900159 ± 0.0006 |
| `var_flag=1`, beta=1e-4 (`e2_confirm`) | 0.458146 ± 0.0040 |
| `var_flag=1`, **beta=0** (`e2_sampling_only`) | 0.486276 ± 0.0068 |

Removing the KL entirely recovers **6.4%** of the gap. The other **93.6% is the
reparameterisation**. This is a claim about the architecture, not a hyperparameter,
and it is much stronger than the beta-tuning story E2 was built on. **The split is
robust to the `scale_biases` correction** — it moved by ~1 percentage point, well
inside this design's noise floor, which is itself worth stating plainly: the bug
that reversed the collapse story (addendum 8) left this headline number alone.

Caveats that must travel with any E2 statement:
* `e2_confirm` characterises a model at FVE ≈ 0.46, not a healthy one. Liveness on it
  belongs in `confounders_uncontrolled` — recorded *before* the runs were read.
* `e2_sampling_only` is **not pre-registered**; proposed after stage 1 was seen. It is
  a control, not a selection step, so it does not contaminate `e2_confirm`, but it is
  exploratory.

### E3 — the ReLU is a large effect, not a nuisance

`vsae_topk.py` applies `F.relu(mu)`; the masked-KL trainer did not; the preprint shows
none. Rather than patch one to match the other, `relu_mu` became a flag and both run
as arms. no-ReLU vs ReLU: **d = +19.3 (0.1x) and +15.3 (0.5x)**, both thresholds
agreeing, 5.03σ. Had either trainer simply been patched, every E3 number would have
silently inherited a d ≈ 20 effect attributed to the KL mask.

### E4 — SCR and TPP disagree, and that disagreement IS the finding

RESULTS addenda 10 (SCR) and 11 (TPP), 2026-09-05/06. Masking the recovered
Pythia baseline (`experiments/e4_pythia_baseline/seed42/`) down to a grid of N
features by usage and scoring each metric at every point, then scoring the
recovered vSAE (`experiments/e4_pythia_vsae/seed42/`) once, unmasked, at its own
natural 1474-feature live count:

| metric | baseline curve shape | vSAE score | vs. `top_usage` reference at n=1474 | verdict |
|---|---|---|---|---|
| SCR (professor/nurse) | flat, noisy, 0.004–0.033 | 0.1017 | 0.0201 (margin **+0.082**) | **not** explained by size |
| TPP (5 classes) | clean, near-monotonic, 0.087–0.212 | 0.1055 | 0.1905 (margin **−0.085**) | explained by size |

Both hold under the `random`-subset bracket too (SCR margin +0.133, TPP margin
−0.051) — this is not an artifact of which reference was chosen.

**And it is not an artifact of averaging across ablation thresholds either
(addendum 12).** Rerun with the per-`n_value` breakdown stored at every grid
point, the verdict is re-derived once per threshold: SCR is "not explained by
size" at all of N=2/5/10/20 (`top_usage` margins +0.047 to +0.153), TPP is
"explained by size" at all four — though TPP is decisive only at N≤10 (margins
−0.07 to −0.13) and a near-tie at N=20 (−0.006 vs `top_usage`, +0.006 vs
`random`). Each metric's verdict is its own at every point on SAEBench's
within-eval ablation sweep, so the disagreement is a property of the two
metrics, not of collapsing them to a mean.

**And it is not a test-set-noise artifact — but the two sides are not equally
sturdy (addendum 13).** `falsification/e4_bootstrap.py` resamples the cached
test set with replacement (200 draws, node effects held fixed from the full
train set — a conditional bootstrap) and re-scores through SAEBench's own
internals. Both mean-over-threshold margins `vsae − top_usage@1474` clear zero:
SCR **+0.082 [+0.024, +0.150]** (frac > 0 = 0.995), TPP **−0.086
[−0.090, −0.081]** (frac > 0 = 0.000). So the disagreement survives resampling.
But SCR's CI is ~15× wider than TPP's, and per threshold SCR's margin clears
zero only at N=2 (+0.155 [+0.109, +0.214]); at N=5/10/20 it straddles zero. TPP's
margin clears zero at N=2/5/10 and sits on the boundary at N=20 (−0.009
[−0.017, −0.000]), where the vSAE lands *between* a random and a usage-ranked
subset of the baseline. The bootstrap makes TPP's side the sturdier of the two
without making SCR's vanish.

**And it is not an artifact of the feature-selection decision either (addendum
14).** The full `--resample-train` SCR bootstrap — node effects re-derived from
a bootstrap resample of the *train* set every draw, 200 draws, full grid, 7.2 h
— widens SCR's mean margin CI from +0.082 [+0.024, +0.150] to **+0.088
[+0.008, +0.171]** (~2.4× wider, lower bound +0.024 → +0.008) but **does not
cross zero**. Addendum 13's open caveat is resolved: SCR's "not explained by
size" verdict survives the strongest bootstrap available on this seed. All the
extra width is at N ≥ 10 — SCR's N=2 and N=5 feature selection is *bit-identical*
between the conditional and full bootstrap (resampling the train set doesn't
change which top-2/top-5 features get picked), so the low-N end the verdict rests
on is the part that is stable to resampling. The baseline `top_usage` curve's
full-bootstrap CI includes zero at every grid point but N=5000 — addendum 12's
"flat and noisy" with error bars. What still is *not* controlled: single seed
each side, one dataset/pair, AuxK.

**This is not a contradiction to resolve by picking a favorite metric — it is
CLAUDE.md's thesis Failure 1, reproduced fresh.** The preprint's own Global and
Conclusion sections reached opposite verdicts on the same hypothesis "because
there was no rule for combining heterogeneous evidence." SCR and TPP disagreeing
here, on the same two checkpoints, same dataset, same masking grid, same day, is
that exact situation. Reporting only SCR ("the vSAE beats the size-matched
baseline") or only TPP ("the vSAE is explained by size") would each be a true
statement about one metric and a misleading one about the vSAE's features in
general.

**This is a first reading, not the full E4 design.** Single seed each side
(matches the preprint's own limitation — descriptive, not a permutation test,
exactly as PROJECT.md's E4 design anticipated when it specified this budget).
One dataset for both metrics; the second SAEBench dataset
(`canrager/amazon_reviews_mcauley_1and5`) and the other three bias_in_bios class
pairs are not yet run. And neither result says anything about CLAUDE.md
landmine 3: the baseline still has `auxk_alpha=0.03125` against the vSAE's 0,
and ruling out dictionary size as the explanation (or not) does not rule AuxK in
or out — the two confounds are independent and this design controls only the
first.

### The method earned its keep twice

Both times, the **two-threshold** liveness pre-registration (F8b: robust only if both
agree; a disagreement is itself the finding) caught a *shape* change in the usage
distribution that a single threshold would have reported as a clean one-directional
effect — once for E1's init factor, once for E1a. Keep reporting both.


### E2's learned sigma — CORRECTED 2026-09-04: it does not collapse, and the noise is not harmless

~~Read 2026-09-03 with `falsification/read_learned_sigma.py`... every one of 41
million measured `log_var` values is at or below the clamp floor... the noise at
convergence is harmless (0.000012 FVE)...~~ **Both claims were a measurement
artifact, not a finding — see RESULTS addendum 8 for the full account.**
`scale_biases` was multiplying `var_encoder.bias` by `norm_factor` (~25.75) at
every checkpoint save, which is not the correct transformation for a log-variance
bias (unlike `encoder.bias`/`decoder.bias`, for which it is correct) and drives
`log_var` to appear fully clamp-saturated regardless of what the model actually
learned. This is now fixed in the trainer code (`var_encoder.weight` is rescaled
instead, which preserves `log_var`'s true value on raw activations); every
checkpoint already on disk was saved before the fix and needs the same correction
applied by hand when read (CLAUDE.md).

**Corrected, on all 13 seeds each of `e2_confirm` and `e2_sampling_only`: mean
`log_var` is ≈ −2.58 and ≈ −2.74 respectively, and *zero* values sit at or below
the floor.** The encoder settles at a moderate `sigma ≈ 0.27`, not the clamp's
`0.0498` — **the posterior does not collapse, and the deterministic SAE is not the
variational SAE's optimum.** `e2_sigma_low_init` is the one arm unaffected by the
bug: `log_var_init=-8.0` already sits below the floor before the bug's
multiplication is even applied, so its own numbers (addendum 7, below) needed no
revision.

**And the "harmless at eval time" claim inverts too.** Recomputed with corrected
sigma (a reconstruction-quality proxy, 5 seeds each, addendum 8 — not yet the
official `evaluate()` pipeline): turning sampling off recovers ≈**0.12 FVE** for
both `e2_confirm` and `e2_sampling_only`, not 0.000012. **This specific 0.12
figure did not survive the official re-run (addendum 9, Next steps #0, now
closed)** — the real `evaluate()` pipeline, run on the same corrected checkpoints,
moves reported FVE by only ≈0.005 relative to the buggy version, an order of
magnitude below the proxy's estimate, and its own absolute numbers do not match
either side of the proxy's comparison. What does survive is the *qualitative*
claim the proxy was built to support — the posterior does not collapse and eval
still samples with real noise — just not this specific magnitude.

E2's "94.7% is the reparameterisation" was built on the (now-corrected) prediction
that sigma starts at exp(−1) = 0.368 (`log_var_init = −2.0`) when mu is still
small, so the noise-to-signal ratio is worst exactly when TopK selection is being
established — and the prediction that initialising `log_var` below the clamp floor
should recover most of the gap if a bad initial noise scale is the whole story.

**Tested 2026-09-04, and the prediction was right — addendum 7.** `e2_sigma_low_init`
(`log_var_init=-8.0`, otherwise identical to `e2_sampling_only`), 13 seeds: FVE
0.484 → **0.834** against baseline's 0.900 — **84.1%** of the gap closes — and
`frac_alive` reaches **1.0000**, exceeding baseline's 0.9934. A real 5-sigma
residual against baseline remains (d = +103 on FVE). Because `e2_sigma_low_init`'s
own measurement is unaffected by the bug, **this finding stands as reported**.
The speculation that `e2_sampling_only`'s reported 0.484 baseline was itself
suppressed by the bug, and so understated the true residual gap, **did not pan
out**: the official re-run (addendum 9) puts `e2_sampling_only`'s corrected FVE at
0.486, essentially identical to the buggy 0.484 it is compared against — so
addendum 7's 84.1% and its residual gap stand exactly as reported, not larger.

**The residual gap is now explained too — addendum 8.** Building the Jaccard-
overlap instrumentation this correction was found inside of (Next steps #1, now
done): with sigma correctly read, `e2_sampling_only`'s TopK selection starts at
essentially chance-level stability (Jaccard 0.069 against a random-subset
reference of 0.067) and never fully stabilises (0.811 at step 10000 — still
~19% churn between two forward passes on the same token at convergence).
`e2_sigma_low_init` is far more stable throughout (0.431 → 0.952) and the gap
never closes. Selection instability tracks reconstruction quality in exactly the
predicted direction, confirming Claims-worth-opening #3's mechanism (a first pass
at this same measurement, using the not-yet-discovered-as-buggy checkpoints, had
found the opposite and is superseded).

**The official re-evaluation is done — addendum 9, 2026-09-05.**
`falsification/reeval_var_flag1.py` re-ran the real `evaluate()` pipeline (not a
proxy) on all 37 on-disk `var_flag=1` checkpoints with the bug corrected. The
result: officially-reported FVE moves by ≈0.003-0.006, not the ≈0.12 the
uncommitted proxy script had estimated — that number is retracted as a measure of
what the *official* pipeline reports, though the qualitative posterior-does-not-
collapse finding it was built on stands. `evaluate()`'s own small-sample
`frac_alive` did rise substantially (0.85→1.00 for `e2_confirm`), but the
pre-registered liveness thresholds are untouched, because `online_histogram_
analyzer.py` calls `encode(..., training=False)` unconditionally and never
samples. The 94.7%/5.3% split is essentially unchanged at 93.6%/6.4% corrected.
See "E2 — it is the sampling, not the KL" above and RESULTS addendum 9 for the
full table.

### Pre-registered but never reported

One item in the battery's design still appears nowhere in
`RESULTS_2026-09-03.md`. It is not blocked; it was simply overtaken.

* **E0, the pipeline negative control.** 13 `baseline` seeds exist and are
  analysed. The test itself — split them into two arbitrary groups, run the real
  metric pipeline, ask the framework to validate "group A is better organised" —
  has not been run. It is the only check that the real pipeline is exchangeable
  across seeds, and every downstream p-value assumes it is.

~~The learned sigma.~~ **Read 2026-09-03, corrected 2026-09-04** — see "E2's
learned sigma" above and RESULTS addendum 8.

---

## Next steps, in priority order

The official re-evaluation is done (RESULTS addendum 9). E4's checkpoints were
recovered and both its SCR and TPP scorers are written and run, and they
disagree (RESULTS addenda 10-11 — see "Closed" below for all three). Nothing on
this list is blocked on compute or data.

**Section A below is lettered, not numbered, on purpose: items #0–#2 keep their
existing numbers so the cross-references in `RESULTS_2026-09-03.md` ("Next steps
#0 box (2)", etc.) stay valid. A takes priority over all of them.**

### A. The mechanism paper — both threads now closed (planned 2026-09-10, done 2026-09-11)

**The framing.** The material now supports a second paper alongside the methods
one in Deliverables: *"What does adding a KL term to a TopK SAE actually do?"*
Three parts, all now established:

1. **At fixed variance, nothing.** It is an L2 penalty (E1, identity verified to
   6 decimals), and once the 5 optimiser/init details are matched the arms are
   null everywhere. **Established.**
2. **With sampling on it hurts, and the mechanism is TopK *selection churn*, not
   representation degradation.** Claim #3 is CONFIRMED for TopK; A2 (below) turned
   it into a 6-point dose-response curve with r=+0.9993 between FVE and selection
   Jaccard. **Established, RESULTS addendum 17.**
3. **The apparent effects in the literature are the size of implementation
   variance**, so the standard comparison protocol cannot distinguish (1) from
   artifact — E1's 5 factors (d≈13–16), E3's lone `F.relu(mu)` (d≈15–19, 5.03σ),
   the decoder-gradient projection (d=−14.3) and the initial weight draw
   (d=−4.7/−7.3), none of which appear in any equations. **Established; Claim #4
   supplies the one control arm that generalises it beyond this repo.**

E4 becomes a section of this paper, not its thesis. The backing — 153
checkpoints, 13 seeds/arm, 5σ — is unusually strong for a paper of this shape.

The two threads, both now done:

- [x] **A1. Which features does each metric actually select? (zero GPU)** —
  **done 2026-09-11, FALSIFIED, RESULTS addendum 15.** This was Next steps #0
  box (3) / Claims-worth-opening #6c, sharpened into a falsifiable prediction.
  `falsification/read_e4_node_effects.py` read the features
  `get_effects_per_class_precomputed_acts` picks as top-effect for SCR and for
  each of TPP's five classes, and cross-referenced each against its rank in the
  committed `all_histograms_*.npz`'s `feature_selection_counts`.

  **Prediction (falsified): SCR's top-effect features sit LOW in the usage
  ranking; TPP's sit HIGH.** Reasoning had been: `top_usage` masking keeps
  high-frequency general-purpose features and discards rare narrow-firing ones;
  disentangling a *correlated pair* (professor-not-gender) plausibly needs a
  rare feature, separating five classes needs broad ones.

  **Result: both sit in the same 0.55–0.73 band on the baseline SAE at every N
  in {2,5,10,20}, the small gap between them flips sign between N≤5 (SCR higher)
  and N≥10 (TPP higher), and within-metric variance across classes swamps any
  between-metric difference** — SCR's own `professor / nurse` and `male /
  female` classes disagree with each other as much as either disagrees with any
  TPP class. Per the falsifier stated in advance, **the explanation is dead**:
  usage-frequency pruning is not selectively discarding SCR-critical features
  while sparing TPP-critical ones. The SCR/TPP disagreement (addenda 10–14)
  stands unexplained by this hypothesis; a live but untested possibility is a
  *distributional* property of the surviving feature set (how well it spans a
  contrast) rather than *which* individual features survive. See addendum 15
  for the full per-class table and the vSAE-side reading (uninformative — 82%
  of that dictionary is dead, so any live feature is trivially near the top of
  its own usage ranking).

- [x] **A2. The σ_init dose-response — the noise axis of Claim #3 (~80 min
  training)** — **done 2026-09-11, RESULTS addendum 17.** Claim #3 was
  confirmed for TopK but rested on two arms plus an annealing run; now a
  6-point curve, 13 seeds each.

  **Design.** 13 seeds × ~6 values of `log_var_init` on gelu-1l, `var_flag=1`,
  everything else matched to `e2_sampling_only`. Measure two things per
  checkpoint: FVE damage vs. baseline, and selection Jaccard instability
  (`falsification/read_selection_jaccard.py`, which already applies the addendum-8
  bias correction).

  **Prediction, and how to make it a real one.** TopK selection is an argmax over
  pre-activations; noise of scale σ flips the selection whenever the gap between
  the k-th and (k+1)-th pre-activation is below σ. So the damage should be a
  **threshold phenomenon**, not smooth degradation, with the knee where σ ≈ the
  typical k/k+1 gap — and the FVE and Jaccard curves should knee *together*.
  **Measure the pre-activation gap distribution on a baseline checkpoint first**
  (one checkpoint read, free): that predicts the knee location a priori instead
  of fitting it post hoc.

  **Pre-flight, RESULTS addendum 16 — the prediction needed refining before the
  sweep landed.** `falsification/read_preact_gap.py` measured the k/(k+1) gap
  on `experiments/baseline/seed1`: median 0.0001, p99 0.0006 (training space).
  Even the clamp floor's sigma (0.0498) sits ~80–550× above this, so no
  `log_var_init` on the planned grid keeps sigma below the boundary gap — a
  shared knee is not reachable within this architecture's clamp range. Half
  right: **RESULTS addendum 17** confirms the curve is smooth, not kneed. But
  the *other* half of the refined prediction — that FVE would decouple from
  Jaccard because boundary churn concentrates in low-importance features — is
  **wrong**: FVE tracks Jaccard almost exactly (Pearson r = +0.9993) across all
  6 points, 13 seeds each, spanning sigma 0.6065 down to 0.0498:

  | `log_var_init` | sigma | FVE | Jaccard |
  |---|---|---|---|
  | −1.0 | 0.6065 | 0.3766 ± 0.0028 | 0.7657 ± 0.0022 |
  | −2.0 (`e2_sampling_only`) | 0.3679 | 0.4841 ± 0.0022 | 0.8117 ± 0.0013 |
  | −3.0 | 0.2231 | 0.6140 ± 0.0020 | 0.8700 ± 0.0008 |
  | −4.0 | 0.1353 | 0.7427 ± 0.0015 | 0.9180 ± 0.0006 |
  | −5.0 | 0.0821 | 0.8169 ± 0.0010 | 0.9453 ± 0.0002 |
  | −8.0 (`e2_sigma_low_init`, clamped) | 0.0498 | 0.8338 ± 0.0007 | 0.9523 ± 0.0002 |

  (baseline: FVE 0.900159.) This is a sharper confirmation of Claim #3's
  mechanism than a threshold would have been: a single near-linear
  FVE-vs-selection-stability relationship holds continuously across six
  well-separated operating points, not just two arms plus an annealing run.
  Re-deriving `e2_sigma_low_init`'s "84.1% of the gap closed" from this run's
  independent pipeline reproduces addendum 7's number exactly. Even at the
  achievable floor a real residual remains (FVE gap 0.066, Jaccard 0.952, not
  1.0) — the clamp mechanically prevents testing lower sigma, so this residual
  cannot be probed further along this axis. Figure at
  `workshop/figs/a2_dose_response.pdf`. See addendum 17 for the full account
  and caveats (the correlation is over 6 arm-level points, not a permutation
  test — CLAUDE.md's unit-of-analysis rule, same status as `frontier.py`'s
  cross-arm `rho`).

  **Why it matters beyond this codebase.** It is the same tension Gumbel-Softmax
  and Concrete exist to resolve: continuous-relaxation stochastic latents do not
  compose with hard combinatorial selection. That yields a constructive
  recommendation rather than a negative result — *if you want stochastic sparse
  codes, put the noise in the selection, not in the magnitudes.*

  **Companion — A3, BatchTopK half done (RESULTS addendum 18), JumpReLU half
  not attempted.** Claim #3's other open half varies the *discreteness* of the
  sparsity mechanism (JumpReLU's learned threshold vs. BatchTopK vs. TopK).
  The BatchTopK side is done: the FVE-vs-Jaccard coupling generalises
  (r=+0.9979 vs. TopK's +0.9993), and if anything BatchTopK's more elastic
  global selection budget is *slightly more* exposed to noise at matched
  sigma, not less (80.5% vs. 84.1% of the gap closed at the clamp floor).
  **JumpReLU is the sharper remaining test of discreteness itself** (BatchTopK
  is still hard top-k, just batch-scoped) but needs a training script written
  from scratch — none exists in `training_scripts/` — and the trainer itself
  is flagged in CLAUDE.md as never exercised end to end, so expect to find and
  fix bugs in the trainer before the sigma question is even reachable, the way
  A3 needed two BatchTopK fixes first. `vsae_jump_relu.py`'s `scale_biases` is
  already correct (verified directly, unlike BatchTopK's, which was not) but
  that says nothing about the rest of the forward/backward path.

  **Costs — actuals vs. the original estimate.** Estimated ~80 min training +
  up to 8.5h of liveness analysis; actual was ~52 min for the 52 new runs (13
  seeds × 4 arms) and **zero** liveness-analysis time — addendum 17 only needed
  each run's own `RUN_COMPLETE.json` FVE and a convergence-checkpoint Jaccard
  read (`read_a2_dose_response.py`), not the full histogram analyzer, so the
  8.5h concern never materialised. Landmines that did matter, for the record:
  * `log_var_init = −8.0` (`e2_sigma_low_init`) sits below the clamp floor and
    saturates to the same effective sigma as −6.0 would — confirmed directly
    (addendum 17's table reports the clamped, not raw, sigma for that row).
  * The four new arms were trained today, post-`scale_biases`-fix, and read
    with NO bias correction; the two reused pre-existing arms
    (`e2_sampling_only`, `e2_sigma_low_init`) predate the fix and need it —
    `read_a2_dose_response.py` applies it per-arm, not globally.
  * `falsification/run_arm.py`'s per-arm, per-seed `save_dir` (not
    `--output-dir`) is what prevents seed collisions; used via
    `falsification/run_a2_sweep.sh`, which mirrors `run_overnight.sh`'s
    skip-if-`RUN_COMPLETE.json`-exists pattern.

### 0. E4 — understand the SCR/TPP disagreement before widening coverage

**STATUS: items (1)–(4) done (RESULTS addenda 12–15, 19); (5) open.** Boxes
(1)–(4) are checked below; pick up at box (5). Nothing here needs re-deriving
— the reasoning is written out in Claims-worth-opening #6, this is just the
checklist.

Addenda 10-11 (2026-09-05/06) found SCR and TPP give opposite verdicts on the
same two checkpoints, same dataset (`LabHC/bias_in_bios_class_set1`), same
masking grid. Claims-worth-opening #6 works through *why* they might disagree
and lays out four diagnostics, cheapest first. Run in this order — each
checkbox is one of that entry's lettered options — and check items off as they
land, recording the result in a new RESULTS addendum the way every prior step
has been:

- [x] **(1) Fix the logging gap, then rerun** (Claims-worth-opening #6a) —
  **done 2026-09-07, RESULTS addendum 12.** Both runners now store the full
  per-threshold dict at every baseline grid point
  (`baseline_curve[i]["per_threshold"]`); both rerun against the warm caches and
  reproduced addenda 10–11's mean verdicts exactly.
  `falsification/e4_per_threshold_analysis.py` redoes the verdict once per
  threshold. **Result: the disagreement is not an averaging artifact.** SCR is
  "NOT explained by size" at all of N=2/5/10/20 (`top_usage` margins +0.047 to
  +0.153); TPP is "explained by size" at all four, though only decisively at
  N≤10 (margins −0.07 to −0.13) — at N=20 it is a tie (−0.006 vs `top_usage`,
  +0.006 vs `random`). Closes 6a as an *explanation* for the disagreement and
  sharpens the case for (2) and (3).
- [x] **(2) Bootstrap error bars from the cached activations**
  (Claims-worth-opening #6b) — **done 2026-09-09/10, RESULTS addenda 13–14.**
  `falsification/e4_bootstrap.py` resamples the cached test set with replacement
  (200 draws) and re-scores through SAEBench's own internals
  (`get_all_node_effects_for_one_sae`, `perform_feature_ablations`,
  `get_scr_plotting_dict` / `create_tpp_plotting_dict`); conditional bootstrap
  (node effects fixed from the full train set) unless `--resample-train`. SCR ran
  on the full 9-point grid (5.7 h); TPP on the 5-point grid
  `500 1000 1474 2000 3000` (5.06 h) after a ~15-line fix that dumps a `.partial`
  checkpoint every 10 draws and resumes from it (both 2026-09-08 TPP attempts
  had been lost to interruption).
  **Result — the SCR/TPP disagreement is not a test-set-noise artifact, but the
  two sides are not equally sturdy.** Both mean-over-threshold margins
  `vsae − top_usage@1474` clear zero: SCR **+0.082 [+0.024, +0.150]** (frac > 0
  = 0.995), TPP **−0.086 [−0.090, −0.081]** (frac > 0 = 0.000). But SCR's CI is
  ~15× wider and per threshold clears zero only at N=2 (+0.155 [+0.109, +0.214];
  N=5/10/20 straddle it); TPP clears zero at N=2/5/10 and sits on the boundary
  at N=20 (−0.009 [−0.017, −0.000]). The bootstrap makes TPP's side the sturdier
  of the two without making SCR's vanish.
  **Full `--resample-train` bootstrap (2026-09-10, RESULTS addendum 14):** node
  effects re-derived from a train-set resample every draw, 200 draws, full grid,
  7.2 h, after adding 10GB-card memory handling to the `--resample-train` path.
  SCR's mean margin widens to **+0.088 [+0.008, +0.171]** but does not cross
  zero — addendum 13's caveat is resolved. SCR's N=2/N=5 selection is
  bit-identical between the two bootstraps; all extra width is N≥10. Box (2)
  fully done.
- [x] **(3) Read which features SCR's and TPP's own effect computation
  selects** (Claims-worth-opening #6c) — **done 2026-09-11 as Next steps A1,
  FALSIFIED, RESULTS addendum 15.** The usage-rank prediction (SCR's top-effect
  features rank low, TPP's rank high) does not hold — both sit in the same
  0.55–0.73 band at every N, the sign flips between N≤5 and N≥10, and
  within-metric across-class variance dwarfs the between-metric difference. See
  Next steps A1 above for the full account.
- [x] **(4) The second SAEBench dataset and the other three `bias_in_bios`
  class pairs** — **done 2026-09-11, RESULTS addendum 19.** `run_e4_scr.py`/
  `run_e4_tpp.py` generalised with `--dataset`/`--column1-vals`. **Result:
  both verdicts replicate as the dominant pattern, with real added variance,
  not a reversal.** SCR "not explained by size": 4/4 on the 3 new
  `bias_in_bios` pairs (mean margin +0.116), 3/4 on amazon's own 4 pairs (mean
  +0.029, one weak exception — Books/CDs_and_Vinyl, near-zero `random`
  bracket); 7/8 combined. TPP "explained by size" replicates on amazon via
  `top_usage` (−0.061) but its `random` bracket goes from decisive (−0.051) to
  a coin flip (+0.0004) — the vSAE's own TPP score barely moves, the
  baseline's curve is what shifts. A real methodological trap was found and
  fixed en route: `--smoke`-testing a new pair before its real run silently
  poisons the "full" run's cache (SAEBench loads by filename existence only,
  not by config match) — now a CLAUDE.md landmine.
- [ ] **(5) Train a size-matched baseline from scratch**
  (Claims-worth-opening #6d, most expensive) — removes "masked vs.
  trained-small" as a live confound in the reference curve itself, at the cost
  of a real training run.

Boxes (1)–(4) are done (addenda 12–15, 19); box (5) is not scoped in code yet
— it is the plan, recorded before picking it up, per this project's own working
style. When starting a fresh session on this: read this checklist,
Claims-worth-opening #6 in full, and RESULTS addenda 10–19, then pick up at box
(5). Box (2) confirmed the prediction from (1): TPP's N=20 margin is a
knife-edge (−0.009 [−0.017, −0.000]), so the "explained by size" reading rests
on N≤10; SCR's per-threshold margins straddle zero at N=5/10/20 (both bootstraps)
and its mean-level verdict is carried by N=2 alone — where, addendum 14 shows,
the feature selection is bit-identical under train resampling. Box (3) then
falsified the leading hypothesis for *why* SCR and TPP disagree (usage-rank
sorting) — addendum 15 — so the disagreement itself is still open.

### 1. Desk work — no GPU, no new code

Both items under **Claims worth opening** (#4, the field-level projection claim;
#5, the seed-count survey) are pure analysis/writing and can be done from a
remote/no-GPU session.

### 2. Optional — extend the E2 beta grid downward (1e-5, 1e-6)

The beta=0 control (`e2_sampling_only`) makes this much less interesting: if
94.7% of the damage is the sampling, no smaller beta recovers a healthy model.
Doing it anyway would close the question formally, but it is a **new
pre-registration**, not a continuation of this one.

---

## Closed — E4's checkpoint recovery and SCR/TPP scorers (2026-09-05/06)

Full detail in RESULTS addenda 10 (SCR) and 11 (TPP), and in **What is
established** above (E4's section). Kept here as the historical record since a
fresh reader may otherwise look for this under "next steps."

The original preprint's Pythia checkpoints had no `ae.pt` on this machine;
`comprehensive_histogram_analysis/` held only their derived analysis outputs.
A search of other drives found a prior/parallel copy of this project on
`HDD_1TB` holding both checkpoints intact; checksummed-copied into
`experiments/e4_pythia_baseline/seed42/` and `experiments/e4_pythia_vsae/seed42/`.
`config.json` confirmed CLAUDE.md landmine 3's AuxK confound directly in the
recovered checkpoints (`auxk_alpha` 0.03125 vs. 0) rather than by inference from
directory names. With the weights in hand, `falsification/e4_local_sae.py`,
`falsification/run_e4_scr.py` and `falsification/run_e4_tpp.py` closed the
remaining gap — real SCR and TPP scorers wired to `falsification/size_control.
py`'s tested-but-scorer-less framework — and produced two readings that
disagree: SCR says the vSAE's advantage is not explained by dictionary size,
TPP says it is. See "What is established" (E4) and RESULTS addenda 10-11 for
the numbers and every caveat (single seed, bias_in_bios only so far, AuxK not
controlled).

## Closed — the official re-evaluation of E2's FVE (2026-09-05)

Full detail in RESULTS addendum 9, and in **What is established** above (E2's
section and the learned-sigma section). Kept here as the historical record since
a fresh reader may otherwise look for this under "next steps."

The prior session's addendum 8 fixed a real bug (`scale_biases` corrupting
`var_encoder.bias`) but measured its consequences with an uncommitted proxy
script, flagged for official re-measurement. `falsification/reeval_var_flag1.py`
did that: it re-runs the actual `evaluate()` pipeline on all 37 on-disk
`var_flag=1` checkpoints with the bug corrected (bias-correction applied
in-memory after loading; a corrected copy of `evaluation_results.json` is written
alongside the original rather than overwriting it). The proxy's ≈0.12 FVE
estimate does not replicate — the official numbers move by ≈0.005 — but E2's
central finding (94.7%/5.3%, now 93.6%/6.4%) is robust to the correction, which is
the answer that actually matters for the paper. `compare_arms.py` and
`frontier.py` now prefer the corrected file automatically, so no other script
needed updating.

## Closed — E1's decomposition (2026-09-04)

Full detail in RESULTS addenda 4 and 5, and in **What is established** above.
Kept here only as the historical record of how the stopping rule was resolved,
since a fresh reader may otherwise look for it under "next steps."

**The decision this closed:** *"Which implementation is E1's claim about — the
objectives, or the released implementations?"* was blocking because each new
matched asymmetry arrived unannounced and flipped the verdict, so no reading
could be committed to without fearing the next factor. The fix was to stop
matching asymmetries as they were noticed and instead enumerate the full code
diff between `top_k.py` and `vsae_topk.py` up front, freeze it, and run only what
was on the list:

1. **Enumerate every difference by reading the two files.** Done — 15
   differences, frozen in RESULTS addendum 4. 2 matched by config, 2 already run
   as factors, 2 measured to be no-ops (the dead-feature rule; the ±10 penalty
   clamp — the only one of the 15 in the *objective* rather than the optimiser,
   and it never binds, 51x headroom), 7 static no-ops shown by reading, 1 the
   seed-permutation design already treats as noise, and exactly one — the initial
   weight draw — never run.
2. **Run that one factor as an arm, then stop.** Done —
   `e1_vsae_ref_fullmatch`, 13 seeds, RESULTS addendum 5. `decoder_init_dist` was
   added to `VSAETopKConfig` as a 2-level factor (default `"gaussian"`,
   bit-identical to every existing checkpoint). The recorded prediction that it
   would be null was **wrong** (d = −4.7 on FVE on its own) — and with it
   matched, `e1_vsae_ref_fullmatch` is indistinguishable from `e1_penalty` on
   every metric, closing the decomposition.

**What this leaves for a human:** decision (a) under **Open decisions** is
answered in practice (the entire measured gap was optimiser-side, never
objective-side); decision (b), the equivalence margin, still needs a number, but
it now sets how strongly to state a null rather than which way the verdict goes.

## Closed — the Jaccard-overlap instrumentation (2026-09-04)

Full detail in RESULTS addendum 8, and in **What is established** above (E2's
learned-sigma section). Kept here as the historical record since a fresh reader
may otherwise look for this under "next steps."

Instrumenting TopK selection instability during early training
(`read_selection_jaccard.py`, two new arms with dense checkpointing) was built to
test whether it explains `e2_sigma_low_init`'s residual gap against baseline. A
first pass at the measurement, on checkpoints not yet known to carry the
`scale_biases` bug, found the opposite of the prediction. Chasing that anomaly
down found the bug itself; corrected, the result reverses cleanly and confirms
Claims-worth-opening #3: `e2_sampling_only`'s true selection instability starts
at chance level and never fully stabilises, `e2_sigma_low_init` is far more
stable throughout, and the gap never closes.

---

## Claims worth opening

Ranked by value per unit of cost. The first two need no GPU and no new code beyond
a script; the third is the most scientifically interesting thing available.

### 1. ~~"The deterministic SAE is the variational SAE's optimum"~~ — ANSWERED, then REVERSED

**No.** Read 2026-09-03, answered "yes, emphatically" from mean raw `log_var` of
−66.9 against a clamp floor of −6 (100% of 41M values at or below the floor) —
**this reading was a measurement artifact of the `scale_biases` bug and does not
survive correction.** See "What is established" above (E2's learned-sigma
section) and RESULTS addendum 8: corrected, mean `log_var` is ≈ −2.6 to −2.9,
nowhere near the floor. What the original (wrong) answer opened — the
sigma-annealing arm — still landed a real result on its own terms (addendum 7),
and chasing *that* arm's residual gap is what surfaced the bug (addendum 8).

### 2. ~~Is there one liveness–reconstruction frontier?~~ — ANSWERED 2026-09-04

**Both answers, in different regimes.** `falsification/frontier.py`, figure at
`workshop/figs/frontier.pdf`, full detail in RESULTS addendum 6. Exploratory, not
pre-registered.

* **Pooled over all 11 replicated arms the two metrics look unrelated**
  (rho = −0.14, p = 0.69) — which is the condition under which someone would
  report them as two independent pieces of evidence.
* **That zero is two opposite-signed relationships cancelling.** Scanning every
  possible cut rather than choosing one: below the cut rho < 0 at *every* cut,
  above it rho > 0 at *every* cut, at both thresholds.
* **Among the 8 working models the frontier is real and slopes UP**: rho = +0.86
  (p = 0.007) and +0.69 (p = 0.058). Better reconstruction buys **more** near-dead
  features. 25 of 28 pairs trade.

The consequence is sharper than the double-counting worry that motivated it. Since
the frontier slopes up, *"architecture A reconstructs better AND has fewer dead
features"* is not a doubly-supported claim — it asserts A is **off** the frontier,
which is rarer and stronger than either half, and it is the shape of claim the
preprint made.

Two things the run corrected. The speculation recorded here — that `gradproj`
"bought reconstruction and paid in liveness, which is what movement along a
frontier looks like" — is **wrong**: `fullmatch` bought both, and `gradproj` is
the only working arm that is dominated outright (by three others). And within the
E1 family, which shares an objective exactly, the arms scatter instead of tracing
a curve (rho = +0.40 / −0.30, thresholds disagreeing in sign). **Implementation
details knock an arm off the frontier rather than sliding it along.**

### 3. ~~"The reparameterisation trick is incompatible with discrete top-k selection,
not with sparse autoencoders"~~ — CONFIRMED 2026-09-04, second session, continued

**Confirmed.** Selection instability under sampling noise tracks reconstruction
quality in exactly the predicted direction — see "What is established" above
(E2's learned-sigma section) and RESULTS addendum 8 for the full account
(`read_selection_jaccard.py`, the two `_early` arms, the chance-level-to-0.811
trajectory, and the `scale_biases` bug this measurement surfaced along the way).
One sub-hypothesis falsified in passing: TopK is *not* forced to pad its
selection with `mu == 0` features (0% of selections are exactly zero), so the
instability works through boundary flips among genuinely-positive features, not
zero-padding.

**Not yet tested — the second half of the original plan, still open:** vary the
discreteness of the sparsity mechanism itself. `vsae_jump_relu.py` has `var_flag`
and a *learned threshold* rather than top-k selection; `vsae_batch_topk.py`
selects discretely but across the batch. Running sampling-on vs. sampling-off for
each, the way `e2_sampling_only` did for TopK, would test whether the FVE damage
is large for TopK, intermediate for BatchTopK, and small for JumpReLU — the
prediction that would generalise this beyond a single architecture and beyond
this preprint. This is now the natural next step for turning this into a second
paper's headline, but it is new training against a not-yet-fixed part of the
codebase's bug exposure (`vsae_jump_relu.py`'s `scale_biases` was fixed
pre-emptively this session but never tested against a real `var_flag=1` run).

**Two axes, not one (added 2026-09-10).** The above varies *selection hardness*.
The orthogonal and much cheaper axis is *noise scale*: a σ_init dose-response on
TopK alone, predicting a **threshold** in σ at the typical k/(k+1) pre-activation
gap, with the FVE and Jaccard curves kneeing together. That is **Next steps A2**
(~80 min of gelu-1l training), and it should run first — it needs no new trainer
and does not touch `vsae_jump_relu.py`. Together the two axes are the mechanism
section of the second paper: noise scale × selection hardness.

### 4. "A one-line optimiser detail moves reconstruction more than the architecture
under study" — ~30 min, converts a local finding into a general one

The decoder-gradient projection is worth d = −14.3 on FVE in the vSAE. That is
larger than most published architectural interventions, and it appears in no
equations. Right now it reads as a bug in one file. One arm makes it general:
**turn the projection off in `top_k.py` and train the baseline** — a plain TopK SAE
with no penalty and no KL.

* Same magnitude there → the projection is a **generic and unreported factor in
  TopK SAE training**, and any comparison between codebases that differ on it is
  confounded. That is a claim about the field, not about this repo.
* **The closing arm strengthened this claim's premise (2026-09-04).** The
  projection is no longer the only such detail: the *initial weight draw* —
  normalised Gaussian versus normalised uniform, identical in every summary
  statistic anyone reports — is worth d = −4.7 on FVE and d = −7.3 on
  `frac_recovered` on its own (RESULTS addendum 5). Two one-line details, neither
  in any equations, each larger than many published architectural interventions.
* Different magnitude → it **interacts with the penalty**, which is already the
  leading interpretation given that the projection moved the vSAE *away* from
  `e1_penalty` on liveness (0.1816 → 0.2197 against 0.1836). An interaction between
  an optimiser detail and a loss term is a subtler and more interesting result.

Either way the E1 decomposition gets a control arm it currently lacks.

### 5. "Most published SAE comparisons cannot reach the significance they imply" —
desk work

The combinatorial floor is not a subtlety, it is arithmetic: 6 seeds per group
cannot beat 3.07 sigma however large the effect, and 5 cannot beat 2.6. Survey how
many training seeds recent SAE papers actually use — the modal answer appears to be
one — and state the ceiling that implies for each.

This is the empirical hook the methods paper currently lacks: it turns "you should
pre-register and use permutation tests" from advice into a measured gap. Handle it
carefully — the point is that the field's *design conventions* cap what its results
can say, not that particular authors erred.

### 6. Why do SCR and TPP disagree on E4? — four diagnostics, cheapest first

RESULTS addenda 10-11 found SCR and TPP give opposite verdicts on the same two
checkpoints, same dataset (`LabHC/bias_in_bios_class_set1`), same masking grid:
SCR says the vSAE's advantage is not explained by dictionary size, TPP says it
is. That disagreement is the addenda's finding on its own terms — it is
CLAUDE.md's thesis Failure 1, reproduced fresh — but it also raises a question
worth its own investigation: is it a real property of the two architectures'
features, or an artifact of how the two metrics happen to be measured here?
Four hypotheses, roughly cheapest-to-test first, none mutually exclusive:

**(a) — TESTED 2026-09-07, RESULTS addendum 12: it is not hiding one.** Both
runners now store the per-`n_value` breakdown; rerun against the warm caches,
each metric's verdict is threshold-uniform (SCR "not explained by size" at all
four thresholds, TPP "explained by size" at all four, decisive only at N≤10).
The disagreement survives the per-threshold cut. The original text is kept below
as the record of why the check was needed.

**(a) The mean-of-thresholds aggregate may be hiding a shape change, the way
liveness's single-threshold summary did twice before (F8b).** Neither
`run_e4_scr.py` nor `run_e4_tpp.py` currently saves the baseline curve's
per-`n_value` breakdown — only the mean across `n_values=[2,5,10,20]` is stored
in `e4_{scr,tpp}_results.json` (`baseline_curve[i]["scores"]` is one aggregate
float per draw). The vSAE's own per-threshold scores *are* saved, and by
inspection SCR's "not explained by size" verdict holds at every individual
threshold for the vSAE against the n=1474 baseline point (0.157/0.089/0.093/
0.068 vs. 0.004/0.034/0.047/−0.004) — but the baseline curve itself was never
saved at that granularity, so whether TPP's "explained by size" verdict is
threshold-uniform or driven by one or two large thresholds (its own
per-threshold vSAE scores range from 0.008 at N=2 to 0.252 at N=20, a much
wider spread than SCR's) is currently unknown. **Cheapest fix**: extend both
scorers to return/store the full per-threshold dict at every grid point, not
just its mean, and rerun against the warm caches — a bookkeeping change, no new
LLM/SAE forward passes needed logic-wise.

**(b) — TESTED 2026-09-09/10, RESULTS addenda 13–14: partly, but the
disagreement survives it.** A 200-draw conditional bootstrap of the cached test
set (addendum 13), then a full `--resample-train` bootstrap that also resamples
the train set and re-derives node effects every draw (addendum 14), each put a
CI on every grid point and both vSAE scores. SCR's curve *is* the noisier of the
two — its margin CI is ~15× wider than TPP's, and per threshold SCR's
`vsae − top_usage@1474` margin straddles zero at N=5/10/20 (clears it only at
N=2). But at the mean-over-thresholds level SCR's margin still excludes zero
under both bootstraps (+0.082 [+0.024, +0.150] conditional; +0.088
[+0.008, +0.171] full), so SCR's "not explained by size" is not merely noise
around zero — it is a thin, low-ablation-budget effect against TPP's wide one.
The original text is kept below as the record of why the check was needed.

**(b) SCR's score is a ratio, TPP's is a difference — that alone could explain
why one curve is noisy/flat and the other clean/monotonic.**
`get_scr_plotting_dict` divides by `(clean_acc − original_acc)`, a denominator
that is small and noisy whenever the spurious correlation itself is weak on a
given draw, and can swing the score sharply on measurement noise alone;
`create_tpp_plotting_dict`'s `total_metric` is a plain accuracy-drop
difference, no division. If this is the whole story, the SCR/TPP disagreement
says less about the vSAE's features than about which of the two published
metrics is measured more reliably at this model scale — worth knowing
regardless of what it implies about the vSAE. **Testable cheaply**:
bootstrap-resample the cached test-set activations
(`falsification/e4_scr_artifacts/`, `e4_tpp_artifacts/` are already on disk) to
put an error bar on every grid point and the vSAE's own score, for both
metrics, without any new LLM or SAE forward passes — just resampled indices
into what is already cached. If SCR's "flat" curve turns out to be
statistically indistinguishable from noise around zero, that changes how much
weight addendum 10's verdict should carry relative to addendum 11's.

**(c) Specialisation vs. coverage.** SCR here asks for one clean axis
(professor/nurse, net of gender); TPP asks for five simultaneously-separable
classes from the same feature budget. A dictionary with far fewer live
features (the vSAE, 1474) might still find one dedicated feature for a single
salient axis while being forced to overload features across five classes it
was never specifically pushed toward — SCR would look great, TPP would look
worse, on the same underlying representation, for a real mechanistic reason
rather than a metric artifact. **Testable**: read which specific features SCR's
and TPP's own effect computation (`get_effects_per_class_precomputed_acts`)
selects as top-effect for each class, at the vSAE's natural size, and check
overlap — do the same handful of vSAE features get selected as top-effect for
*multiple* TPP classes (evidence for overloading) while SCR's top-effect set is
disjoint from all of them (evidence for a dedicated axis)? A read of existing
per-run artifacts, not a new training run.

**(d) The baseline's masked-from-8192 curve may not be a fair reference for a
dictionary that was never trained at that size.** Every point on the
baseline's `size_control.py` curve is the *same* 8192-wide dictionary with
entries zeroed out post hoc; the vSAE's 1474 features were learned together,
with the rest of its capacity never used for anything else. A dictionary
trained from scratch at `dict_size=1474` might organise its limited capacity
differently — more efficiently, or less — than a masked subset of a bigger
one, and `size_control.py`'s own
`test_random_subset_null_would_falsely_confirm_the_hypothesis` already proves
this kind of reference-choice sensitivity matters a great deal for this
design. **Most expensive of the four**: train a plain TopK baseline at
`dict_size=1474` on Pythia-70m layer 3 (same config otherwise) and score it
directly, no masking. A real training run, not just a rerun of the existing
scorers — single-seed the same way E4's other checkpoints are, so still
descriptive rather than confirmatory, but it removes "masked vs. trained-small"
as a live confound in the comparison itself, which (a)-(c) do not.

None of these would overturn RESULTS addenda 10-11's central finding — SCR and
TPP disagree on this dataset today — they would narrow down *why*, which is
exactly the kind of gap Failure 1 exists to force into the open rather than
paper over. See Next steps #0 for the priority order this implies.

---

## Open decisions

- **Agentic or fixed battery?** POPPER's novelty is LLM agents *designing* the
  falsification tests. We could (a) hand-specify a battery — more rigorous, much
  cheaper, less novel; or (b) have an LLM propose tests against a metric schema —
  closer to POPPER, more moving parts, and the relevance checker becomes load-
  bearing. **Recommendation: (a) first.** The statistical contribution stands
  alone, E0 is meaningful either way, and (b) can be layered on once the fixed
  battery has established the error rates it should be compared against.
- ~~**α and κ.**~~ **RESOLVED: α = 0.1, κ = 0.3**, pre-registered. The κ sweep
  (`kappa_sweep()` in `falsification/simulate.py`) measured worst-case Type-I
  (5 fully redundant tests) against power at 5 seeds/group:

  | κ | Type-I (worst) | power d=1.0 | d=1.5 | d=2.0 |
  |---|---|---|---|---|
  | 0.2 | 0.066 | 0.27 | 0.63 | 0.90 |
  | **0.3** | **0.086** | **0.29** | **0.66** | **0.92** |
  | 0.4 | 0.095 | 0.27 | 0.63 | 0.91 |
  | 0.5 | 0.095 | 0.19 | 0.53 | 0.85 |
  | 0.7 | 0.061 | 0.01 | 0.08 | 0.27 |

  κ=0.3 maximises power while staying under α. Against the previous κ=0.5 this is
  free: power at d=1.5 rises 0.53 → 0.66 at 5 seeds, 0.73 → 0.82 at 6. The power
  table in "What the simulation settled" is κ=0.5 and is superseded by the κ=0.3
  table in `RUNBOOK.md`.
- **Equivalence margin for E1.** Needs to be pre-specified. What difference in live
  fraction would we consider a real departure from degeneracy? This is now the
  binding constraint on the whole experiment, not a formality: at 13 seeds the
  across-seed SDs are ~1e-3, so essentially any non-zero difference clears 5 sigma,
  and E1's verdict has already flipped once on a difference of 0.036 in live
  fraction. More power cannot supply it, and the numbers are now known, so it can
  no longer be chosen innocently.
- ~~**A stopping rule for matching asymmetries (new, 2026-09-03).**~~
  **RESOLVED 2026-09-04 by enumeration, not by exhaustion.** The rule is: *the
  factor set is the code diff; run what is on it and stop.* The diff is
  enumerated and frozen in RESULTS addendum 4 at 14 items, one of which is
  unrun. Nothing may be added to it without a code difference that justifies it,
  and the two candidates that arrived after the pilot and looked most alarming
  (the dead-feature rule, the ±10 clamp) were both closed by measurement rather
  than by a training run.

  What made this urgent stands as the record of why the rule was needed: matching
  `project_decoder_grad` **closed 79% of the reconstruction gap and opened a
  liveness gap that had been closed** (RESULTS addendum 2). "Keep matching until
  the arms agree" is a garden of forking paths with a pre-registered metric
  attached. The remaining question is not *when to stop* but which implementation
  E1's claim is *about*, and it still has to be answered explicitly:

  * the degeneracy is an identity between **objectives** (`0.5*||mu||^2`, verified
    to six decimals), and says nothing about optimisers, in which case the
    projection is a nuisance factor and the unprojected arm is a legitimate vSAE;
  * or the claim is about the **released implementations**, in which case every
    asymmetry between them is in scope and none of them should be matched at all.

  Those two readings license different arms and currently give different verdicts.
  Picking one is a human decision, and it should be recorded before the next factor
  is run.
## The thesis

Interpretability makes claims of the form "architecture A produces better features
than architecture B". These are validated informally: run a benchmark suite, read
the numbers, write a conclusion. That process has two failure modes that our own
preprint exhibits, which makes it an unusually well-documented case study.

**Failure 1 — no principled aggregation.** The preprint ran core metrics, SCR, TPP,
t-SNE, and feature visualisation, then combined them by narrative. Its Global
section concluded the dispersion hypothesis was *confirmed*; its Conclusion
concluded it was *rejected*. Nothing in the method could have adjudicated that,
because there was no rule for combining heterogeneous evidence.

**Failure 2 — sub-hypotheses that the main null does not imply.** The vSAE
outscored the baseline on SCR and TPP, and this was read as evidence of a more
disentangled feature space. But SCR and TPP reward *selective* ablation, and
selectivity is easier when fewer features are live — and the vSAE had 18% of its
dictionary alive against the baseline's 90%. The main null ("no better
organisation") being true does **not** imply the sub-null ("no higher SCR"). So the
result carries no evidence about organisation at all.

POPPER (Huang et al., 2025) supplies exactly the missing machinery: an implication
check (their Assumption 1, enforced by a relevance checker) and sequential
aggregation of e-values with Type-I error control under optional stopping. Our
contribution is to instantiate it where the hard part is different. POPPER's
difficulty is *proposing* good falsification tests over a static database. Ours is
*producing a valid p-value at all* — because the randomness in an SAE comparison is
the training seed, and seeds are expensive.

**Claim:** sequential falsification with an explicit implication check is a
practical validation protocol for interpretability claims, and applying it to a
real published SAE result reverses that result's conclusion.

## Why the implication check is the interesting part here

In POPPER's biology setting, relevance checking guards against a tangential test.
In interpretability it does something sharper: it is **confounder control for
benchmark metrics**. Nearly every SAE metric co-varies with a nuisance variable —
live-feature count, L0, reconstruction quality — and the field routinely compares
models that differ on those nuisances. Formalising "does the main null imply this
sub-null?" forces the nuisance into the open.

The worked example (`python falsification/worked_example.py`) shows the framework
excluding both of the preprint's headline wins before any evidence accrues, and
declining to validate. That is the correct answer, and the informal process got it
wrong in print.

## The power problem, and what it costs

The unit of analysis for an architecture claim is the training run. That makes an
exact permutation test the natural instrument, and its floor is brutal:

**Corrected 2026-09-02.** The figures below previously used a floor function whose
one-sided and two-sided branches were swapped, making every one-sided floor 2x too
pessimistic (`falsification/FINDINGS_2026-09-02.md`, item 1). Corrected values, for
the one-sided tests this design actually uses:

| seeds/group | assignments C(2n,n) | min attainable p | e (κ=0.5) | e (κ=0.3) |
|---|---|---|---|---|
| 3 | 20 | 5.0e-02 | 2.24 | 2.44 |
| 4 | 70 | 1.4e-02 | 4.18 | 5.87 |
| 5 | 252 | 4.0e-03 | 7.94 | **14.39** |
| 6 | 924 | 1.1e-03 | 15.20 | 35.73 |
| 8 | 12,870 | 7.8e-05 | 56.72 | 225.85 |

Validation at α=0.1 needs aggregate E ≥ 10.

**The old claim that "5 seeds per group cannot validate on a single test no matter
how large the effect" no longer holds at the pre-registered κ.** It was computed at
κ=0.5 from the inverted floor (e = 5.61). Corrected, 5 seeds gives e = 7.94 at
κ=0.5 — still short — but **e = 14.39 at κ=0.3, which validates**. Since κ=0.3 is
the pre-registered value, `seeds_required(alpha=0.1, kappa=0.3, n_tests=1)` now
returns **5**, not 6.

This does not change the direction of the project's advice, only its arithmetic:
replication across seeds is still what buys evidence, and a single-run-per-condition
sweep still cannot validate. The power tables elsewhere in this file and in
RUNBOOK.md are simulation-derived and use real permutation p-values, not the floor,
so they are unaffected by the correction.

**The measured cost of a run has since changed the planning picture more than any
of this.** At ~1 min/run on the 3080 (30 runs in 27 min, 2026-09-02), seeds are no
longer the binding constraint at all — 6 seeds/arm already delivers 2-3.6x the
required evidence. The binding cost is the 1M-sample feature-usage analysis at
~6.5 min/checkpoint, i.e. 6.5x the cost of the training run it measures.

The same arithmetic applies to the preprint's cleanest result. Its β dose-response
is monotone over six orders of magnitude, yet with one run per condition the exact
trend test floors at 1/4! = 0.042, giving e = 2.45 against a threshold of 10.
**Even a perfect dose-response does not validate without replication.** This is the
single most important planning fact in the project: buy seeds, not sweep points.

## What the simulation settled (no GPU required)

`python falsification/simulate.py` answers three questions that decide the design.
All three are locked into the test suite.

**1. Reusing the same runs across tests does not break validity — but it spends
the margin.** Several falsification tests reading metrics off the same 10 runs are
dependent, which is exactly what POPPER's Assumption 2 guards against (its tests
each query a different database; ours do not). Empirical Type-I at alpha = 0.1,
5 seeds/group:

| tests | correlation | Type-I |
|---|---|---|
| 2 | 0.0 | 0.002 |
| 3 | 1.0 | 0.050 |
| 5 | 0.95 | 0.092 |
| 5 | 1.0 | 0.095 |

Validity holds throughout, but redundant tests take the rate from 0.002 to the
edge of alpha. **Design rule: prefer few, genuinely different falsification tests
over many correlated ones.** Note also how conservative the independent case is —
permutation p-values with 5 seeds/group are discrete multiples of 1/252 and
therefore super-uniform, which costs power as well as error rate.

**2. Peeking at p-values to choose test order breaks Type-I control.** With 10
candidate metrics, running the most significant first gives Type-I = **0.123 >
alpha = 0.1**, against 0.020 when the order is fixed in advance. The
optional-stopping guarantee covers *when you stop*, never *which test you reach
for next*. **Design rule: the battery and its order are pre-registered in this
file before any seeded run is looked at.**

**3. Power is the binding constraint, and it is worse than expected.** Probability
of validating a true effect, two pre-specified tests, alpha = 0.1, effect size d in
across-seed standard deviations:

| seeds/group | d=0.5 | d=1.0 | d=1.5 | d=2.0 |
|---|---|---|---|---|
| 4 | 0.01 | 0.07 | 0.24 | 0.50 |
| 5 | 0.03 | 0.20 | 0.54 | 0.86 |
| 6 | 0.06 | 0.32 | 0.73 | 0.96 |
| 8 | 0.10 | 0.48 | 0.91 | 0.99 |
| 10 | 0.14 | 0.61 | 0.96 | 1.00 |

**A one-standard-deviation effect is not reliably detectable even with 10 seeds per
group (power 0.61).** This reframes the budget. The effects we expect to be huge —
the degeneracy (E1) and beta-driven feature death — are fine at 5-6 seeds. But any
subtle claim about "feature organisation" is out of reach at this scale, and we
should say so rather than run an underpowered arm and report a null.

Before committing GPU time to an arm, estimate its d from the existing single-seed
data and read the required seed count off this table.


---

## Experiments — the pre-registration

This is the battery as pre-registered, kept in its original form because that is
what makes it a pre-registration. The seed counts say 6 and 10; the confirmatory
battery ran at **13 seeds per arm** after the 6-seed generation hit the 3.07-sigma
combinatorial floor. Outcomes are **not** recorded here — they are in
`falsification/RESULTS_2026-09-03.md` and summarised under "What is established"
above. Where an experiment's design was changed after seeing results, the change is
noted in the section itself and the arm is marked exploratory.

Every arm is gelu-1l, `blocks.0.hook_resid_post`, d=2048, k=256, auxk=1/32,
lr=8e-4, 10k steps — the existing sweep configuration — varying only the stated
intervention and the seed.

### E0 — Pipeline negative control. 10 runs.
> **Outcome: never run.** 13 `baseline` seeds exist and are analysed; the split
> test itself has not been done. See "Pre-registered but never reported".

Train 10 TopK SAEs with **identical config, different seeds**, run the *real*
metric pipeline over them, split into two arbitrary groups of 5 and ask the
framework to validate "group A is better organised than group B". Ground truth:
null by construction.

**What this can and cannot establish.** An earlier version of this plan proposed
re-splitting the same 10 runs many times and reading the rejection rate as an
empirical Type-I error. That is close to circular: permutation p-values over
re-splits of a fixed set are uniform *by construction*, so the measurement is
guaranteed to pass and tests only that a permutation test is a permutation test.

What E0 genuinely tests is the **plumbing**, and that is worth testing: whether
the real metrics, computed by the real pipeline, are actually exchangeable across
seeds. Bugs that would show up here and nowhere else include a metric that depends
on checkpoint filename ordering, a shared data-loading order, a cached artifact
leaking between runs, or an evaluation that is not seed-independent. Any of these
would silently break exchangeability and invalidate every downstream test.

Error control of the *procedure* is established separately and for free by
`falsification/simulate.py`, which needs no GPU. See "What the simulation settled".

### E1 — The degeneracy claim. 6 runs.
> **Outcome: unresolved, and the reason is a decision.** Confirmed on the
> pre-registered liveness metric against `e1_vsae_ref_unitinit`, not against
> `e1_vsae_ref_gradproj`. See "What is established" and Open decisions (a), (b).

Train TopK SAE + explicit `(β/2)·||f||²` activation penalty, 6 seeds. Under the
degeneracy (`CLAUDE.md`, landmine 1) this should be **indistinguishable** from the
fixed-variance vSAE at matched β.

Note this is an equivalence claim, so a failure to reject is not evidence of
equivalence. Report it as a TOST-style equivalence test with a pre-specified
margin, or as a power statement ("we could have detected a difference of size X").
Do not report a null result as confirmation.

### E2 — Is it variational at all? 6 runs.
> **Outcome: landed, stronger than designed** — 94.7% of the damage is the
> reparameterisation, not the KL. The learned-sigma diagnostic below has been
> read and corrected twice since (it does **not** collapse; a `scale_biases` bug
> made it look like it had). See "What is established".

Train with `var_flag=1`, 6 seeds — the experiment the preprint claims to have run
and did not. Diagnostic of interest is the learned σ: if it collapses toward 0, the
degeneracy is the *optimum* rather than an implementation accident, which is a
substantially stronger result.

### E3 — The masked-KL intervention. 6 runs.
> **Outcome: landed.** The ReLU confound flagged below was resolved by running
> `relu_mu` as a two-level factor rather than patching either trainer; the ReLU
> alone is d = +19.3 / +15.3.

`vsae_topk_masked_kl`, 6 seeds. Tests the preprint's stated death mechanism
directly. **Beware the confound in landmine 2**: that trainer omits the `F.relu(mu)`
that `vsae_topk.py` applies, so either patch one to match the other or report the
comparison as confounded.

### E4 — Size-matched SCR/TPP control. No training. Implemented in `falsification/size_control.py`.
> **Outcome: both metrics landed, and they disagree.** `falsification/
> run_e4_scr.py` and `run_e4_tpp.py` wire `size_control.py` to real SAEBench
> scorers against the recovered checkpoints (`experiments/e4_pythia_baseline/`,
> `experiments/e4_pythia_vsae/`, seed 42), on `LabHC/bias_in_bios_class_set1`:
> SCR says the vSAE's advantage is not explained by dictionary size, TPP says
> it is (RESULTS addenda 10-11). The second SAEBench dataset and the other
> three class pairs are the remaining extension — see Next steps #0.

Measure SCR/TPP as a **function of dictionary size** for the baseline SAE, then
ask where the vSAE's score falls on that curve.

**The originally-planned version of this experiment was wrong, in the dangerous
direction.** It proposed restricting the baseline to a *random* subset matching
the vSAE's live count. But a random 18% subset of a dictionary trained to work as
a whole reconstructs badly, while the vSAE's 1474 features were learned together.
The vSAE beats that null trivially — so the test was rigged in its favour and
would have manufactured a positive result for the very hypothesis it exists to
check. `test_random_subset_null_would_falsely_confirm_the_hypothesis` demonstrates
this concretely: a vSAE with zero genuine advantage still clears the random null.

The correct reference is the baseline's **top-N most-used features** — its best
foot forward, and the strongest thing size alone can buy. Clearing that bar is the
demanding test; beating the random curve means nothing. Report both to bracket the
answer.

Sweep N over a grid, score each restricted dictionary, and place the vSAE's score
on the resulting curve. If it sits on or below the top-usage curve at N = 1474,
the advantage is explained by size, and the preprint's SCR/TPP reading collapses.
This also produces the figure that makes the argument visually, which a single
p-value would not.

Still to write: the SAEBench-side scorer that turns a kept-index set into an SCR
score. Everything around it is implemented and tested against synthetic scorers
with known ground truth.

**Budget:** no training runs — masking and re-scoring an existing dictionary at
each grid point is the whole design, and both checkpoints now exist on disk
(2026-09-05). The remaining cost is entirely the SAEBench scorer integration and
however long its SCR/TPP eval takes to run per grid point.

**Pre-registration.** Per simulation finding 2, the battery above and its order are
fixed before any seeded run is inspected. Adding an arm after seeing results is
permitted only if reported as exploratory and excluded from the evidence product.

## Deliverables

1. **Paper (methods).** Sequential falsification for interpretability claims, with
   the vSAE study as the case study whose conclusion it reverses. Target: a venue
   caring about measurement validity and falsifiability (the InterpScience CFP
   framing still fits).
2. **`falsification/`** as a reusable package, with the Type-I calibration from E0
   as its empirical warrant.
3. **Corrections** to the preprint's record: the degeneracy, the AuxK confound, the
   config mismatches, the self-contradiction between Global and Conclusion.
4. **Paper (mechanism), planned 2026-09-10.** *"What does adding a KL term to a
   TopK SAE actually do?"* — fixed-variance KL is a null L2 penalty (E1);
   sampling-on damage is TopK selection churn with a σ threshold (E2 + Claim #3 +
   Next steps A2); and the effect sizes reported in the literature are the size
   of implementation variance (E1's 5 factors, E3's ReLU, Claim #4's control arm).
   Constructive payload: *put the noise in the selection, not the magnitudes.*
   Remaining experiments are scoped at **Next steps A**.

## Landmines specific to continuing this work

These cost real time this session. They are not in `CLAUDE.md` because they are about
*operating* the battery, not about the science.

**Never poll for GPU-busy with `pgrep -f '<pattern>'`.** The Bash-tool wrapper
process's command line contains the entire script text, so a wait loop that greps for
`run_arm.py` matches *its own parent* and never exits. This deadlocked a job for six
minutes. If the GPU is already idle, omit the wait entirely.

**Do not run two GPU jobs concurrently.** Two analysers OOM on the 10GB card
(each needs ~3.5–3.9 GB with `n_ctxs=3000`). Analysis is serial by necessity. The
analyser *raises* on OOM so no bad summary is written — but a **training** OOM in
`loss_recovered()` is swallowed by an `except ... continue` and written as
`frac_recovered: 0.0` with NaN CE metrics. That failure is silent.

**Calling `analysis_scripts/online_histogram_analyzer.py` directly (bypassing
`run_analysis.sh`) needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set
by hand.** `run_analysis.sh` exports it (line 16); a bare invocation (e.g. to
re-analyse one straggler seed after a batch run gets killed) does not inherit it
and OOMs on allocator fragmentation even with ~9GB nominally free — it needs the
contiguous block `torch.addmm` wants for the unembed. Confirmed 2026-09-04:
`read_learned_sigma.py` and other one-off checkpoint-reading scripts likely have
the same exposure; export it before running any of them standalone.

**A long-running unattended shell command (`run_in_background: true`) can be
killed by *unrelated* memory pressure on the host, not by anything this repo's
jobs did.** Twice this session a multi-seed loop was killed mid-run with "system
is running low on memory" while a completely different project's process
(`relay_null`, ~8.5GB RSS, 99% CPU, from another Claude Code session on this
machine) was the actual cause. `ps aux --sort=-%mem` before assuming a killed job
means *this* repo's config is too heavy — the GPU card itself was never the
constraint (idle at ~1.3GB used throughout). After a kill, check which seeds
actually finished (`RUN_COMPLETE.json` / `comprehensive_summary_*.json` on disk)
before re-running — both `run_arm.py` and `run_analysis.sh`'s per-seed loop are
naturally resumable from wherever they stopped.

**Shrinking `n_ctxs` to enable parallelism would break comparability** — it changes
which tokens each checkpoint sees, and the existing runs were all measured at 3000.

**`run_analysis.sh` skips on mtime, not existence.** Retraining an arm leaves the old
generation's summary next to the new `ae.pt`; the mtime check catches that. If you add
a step that rewrites checkpoints, make sure `ae.pt` gets a newer mtime than its
summary or the stale numbers will silently persist.

**Flags that leave no trace in the weights need `config.json`.** `relu_mu`,
`decoder_init_scale`, `project_decoder_grad` and `decoder_init_dist` all change
behaviour but not parameter shapes, so none can be recovered from a state dict.
Each is written into the trainer's `config` property; `relu_mu` is also read back
in `utils.load_dictionary` because it changes `encode`, the others are provenance
only. Any new factor of this kind needs the same, and see the next item — writing
it into `config` is necessary but not sufficient.

**A trainer's `config` is a `@property`: writing to the dict `trainSAE` hands you
does nothing.** `trainSAE` does `trainer.config["norm_factor"] = norm_factor`
(`training.py:212`) expecting it to persist, but both `TopKTrainer.config` and
`VSAETopKTrainer.config` rebuild a fresh dict on every access, so the write lands
on a temporary and `norm_factor` is in no checkpoint's `config.json` (CLAUDE.md).
Every *factor* field survives because it is read from `self.model_config` /
`self.training_config` inside the property, not assigned into the dict from
outside — that is the pattern to copy, not the assignment `trainSAE` uses.

**`falsification/tests/` was never in git until 2026-09-03.** The stock `tests/`
rule in `.gitignore` — where it means a coverage artefact directory — matched it,
so the suite CLAUDE.md calls "must stay green" existed on one machine only. It is
re-included now (`!falsification/tests/`, with `__pycache__` put back after it,
since the last matching pattern wins). If you add a directory under `falsification/`
that the stock ignore list happens to name, check `git status` actually sees it.

**Timings, measured.** Training ≈ 1 min/run. Analysis ≈ 1.1 min/checkpoint at 1M
samples (was 6 min before `update_histograms` was vectorised — 59 of 60 output arrays
verified bit-identical after that change). A full 13-seed arm is ≈ 13 min train +
≈ 15 min analyse.

**~~`run_e4_scr.py` and `run_e4_tpp.py` discard the per-threshold breakdown at
every baseline grid point.~~ FIXED 2026-09-07 (RESULTS addendum 12).** Both
runners now store `baseline_curve[i]["per_threshold"]` — a list of the full
`{scr,tpp}_metric_threshold_N` dict, one entry per draw — reconstructed from
`size_response_curve`'s call order after the fact (the `Scorer` signature it
takes is a bare `keep_indices -> float`, so the per-threshold dict is captured
in a closure-local list and re-walked against `points`, not threaded through).
`falsification/e4_per_threshold_analysis.py` consumes it. Left here as the
record: the mean-across-`n_values` collapse was a real shape-behind-a-scalar
risk (F8b), it just turned out not to be masking anything — both verdicts are
threshold-uniform.

**SCR's per-threshold score is a ratio; TPP's is a difference.**
`get_scr_plotting_dict` divides by `(clean_acc − original_acc)`, which can be
small and noisy on any single draw; `create_tpp_plotting_dict`'s `total_metric`
is a plain subtraction. Observed consequence: the baseline `top_usage` SCR
curve is flat and noisy (0.004–0.033 across N=100–7379) while the equivalent
TPP curve is clean and near-monotonic (0.087–0.212) — plausibly a property of
how the two metrics are constructed rather than of the two dictionaries being
compared. **Verified partly (RESULTS addenda 13–14):** both the conditional and
the full `--resample-train` bootstrap confirm SCR is the noisier metric here
(margin CI ~15× wider than TPP's), but SCR's mean-level `vsae − top_usage@1474`
margin still excludes zero under both (+0.082 → +0.088), so the disagreement is
not purely a metric-construction artifact.

---

## Where the detail lives

| file | what it holds |
|---|---|
| **this file** | current state, what is established, next steps, the pre-registration, open decisions |
| `CLAUDE.md` | standing landmines in the vSAE code; read before touching `dictionary_learning/` |
| `falsification/RESULTS_2026-09-03.md` | all measured results. Addendum 1: 13-seed/5σ rerun. 2: gradient projection. 3: the learned sigma collapses **— CORRECTED by 8, do not trust in isolation**. 4: the E1 code diff, enumerated and frozen (15 items). 5: the closing arm — E1 lands. 6: the liveness/reconstruction frontier. 7: the sigma-annealing arm — 84% of E2's gap is the init, not the reparameterisation. 8: the `scale_biases` bug — corrects 3, confirms Claims-worth-opening #3. 9: the official `evaluate()` re-run — the proxy's ≈0.12 FVE does not replicate, the 94.7%/5.3% split does. 10: E4's SCR scorer — the vSAE's SCR score is not explained by dictionary size. 11: E4's TPP scorer — reverses addendum 10's verdict, a live case of the thesis's Failure 1. 12: the SCR/TPP disagreement is threshold-uniform, not an artifact of averaging across `n_values`. 13: E4 bootstrap error bars — the disagreement is not a test-set-noise artifact either (both margins clear zero, opposite signs), but SCR's side is ~15× wider and rests on the N=2 threshold alone. 14: E4 full `--resample-train` bootstrap — SCR's mean margin widens to +0.088 [+0.008, +0.171] but does not cross zero; N=2/N=5 feature selection is bit-identical under train resampling |
| `falsification/FINDINGS_2026-09-02.md` | the five instrumentation bugs the pilot exposed |
| `falsification/REMEDIATION.md` | fix tracking + the four author decisions and their rationale |
| `RUNBOOK.md` | commands, arm table, E4 design |
| `falsification/frontier.py`, `read_penalty_clamp.py`, `read_learned_sigma.py` | the checkpoint-reading analyses behind addenda 3, 4 and 6 — `read_learned_sigma.py`'s own numbers need the addendum-8 bias correction applied by hand; it does not do this itself |
| `falsification/read_selection_jaccard.py` | Jaccard-overlap-during-training analysis behind addendum 8; applies the bias correction itself — the pattern to copy for re-reading any other `var_flag=1` checkpoint |
| `falsification/reeval_var_flag1.py` | official `evaluate()` re-run behind addendum 9; writes `evaluation_results_corrected.json` per checkpoint, which `compare_arms.py` (and `frontier.py`) now prefer automatically |
| `falsification/e4_local_sae.py`, `falsification/run_e4_scr.py`, `falsification/run_e4_tpp.py` | E4's local (non-Hub) checkpoint loaders and the SCR/TPP masking-grid/scorer/verdict behind addenda 10-12; either runner's `--smoke` flag gives a fast pipeline check before a full run. Both now store `baseline_curve[i]["per_threshold"]` |
| `falsification/e4_per_threshold_analysis.py` | re-derives the E4 size verdict once per ablation threshold from the stored per-`n_value` breakdown (addendum 12) |
| `falsification/e4_bootstrap.py`, `falsification/e4_bootstrap_{scr,tpp}_results.json`, `falsification/e4_bootstrap_scr_resample_train_results.json` | E4's test-set bootstrap error bars: conditional (addendum 13, both metrics) and full `--resample-train` (addendum 14, SCR only — node effects re-derived per draw, 10GB-card memory handling, 7.2 h for 200 draws on the full grid). `--smoke` for a fast check; `.partial` checkpoint every 10 draws with exact resume |
| `workshop/figs/frontier.pdf` | the frontier figure (addendum 6) |
## Verify the environment is sane

```bash
python falsification/preflight.py                 # says which environment you are in
python -m pytest falsification/tests/ -q          # 112 passing; must stay green
python falsification/run_arm.py --check           # validates every arm without torch
python falsification/report_summaries.py --table  # cross-arm liveness
python falsification/compare_arms.py e1_penalty e1_vsae_ref_gradproj   # any two arms
```

`preflight.py` now imports each training module through
`run_arm.load_training_module()` as well as reading its source as text, which closes
FINDINGS item 5 — it used to pass on a machine that could not train, because every
wiring check read source as text and missing packages slipped through. The text
checks stay: they are what makes preflight useful in an environment without torch.
