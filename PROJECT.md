# PROJECT.md — Falsification-based validation of sparse autoencoder claims

**This is the living document.** It carries current state, what is established,
what to do next, and the pre-registration the battery runs under. It absorbed
`HANDOFF.md` on 2026-09-03; there is no separate handoff file. `CLAUDE.md` has the
repo's standing landmines about the vSAE code — read that too, and read it before
touching anything in `dictionary_learning/`.

Reading order for a cold start: **Status** → **Where things stand** → **What is
established** → **Next steps**. Everything after that is the design and the
pre-registration, which change rarely; the sections before it change every session.

Last updated: 2026-09-04 (second session). The sigma-annealing arm from Next steps
#1 has landed: `log_var_init` is now a factor, `e2_sigma_low_init` ran 13 seeds,
and it closes **84% of E2's FVE gap** to baseline and reaches **100% liveness**
(RESULTS addendum 7). E2's headline needs restating — see below and "What is
established". Earlier this session (previous entry): the E1 code diff was
enumerated, frozen, and closed by its last arm, and the liveness/reconstruction
frontier was read. What's left is E4 (no GPU needed) or desk work — see
**Next steps**.
Branch `claude/falsification-framework`, GPU idle. 11 arms, 143 checkpoints on disk.

## Status

| | |
|---|---|
| Stage | Battery complete at 13 seeds; **E1, E2 and E3 have all landed**; E2's mechanism now has a cause |
| Framework | `falsification/` implemented, **115 tests green**, Type-I control verified |
| Newest figure | `workshop/figs/frontier.pdf` — the liveness/reconstruction frontier over 8 working arms |
| Data | 11 arms, 143 checkpoints, 13 seeds/arm, 0 failures, all analysed at 1e6 samples |
| Newest result | **84% of E2's reconstruction damage is the initial noise scale, not the reparameterisation itself** — `e2_sigma_low_init`, RESULTS addendum 7 |
| Blocking | Nothing is blocked on compute. Decision (b) sets how strongly to state E1's null |
| Prior artifact | arXiv preprint; workshop draft on `claude/vae-workshop-paper-condensing-zumu6b` |

## Where things stand

The confirmatory battery is **complete at 13 seeds per arm** and reaches **5 sigma**
on every comparison. E2 and E3 landed earlier and are stronger than they were
designed to be. **E1 has now landed too.** E0 and E4 have never been reported at
all. **The learned sigma has been read** and it collapses completely — see below.

**How E1 landed.** The code diff between the two arms was enumerated by reading
`top_k.py` and `vsae_topk.py` against each other, plus the two training scripts,
and frozen at **15 differences and no more** (RESULTS addendum 4): 2 matched by
config, 2 already run as factors, 2 measured to be no-ops, 7 static no-ops shown
by reading, 1 the seed-permutation design already treats as noise, and exactly one
never run. That last one — the initial weight draw — was then run as
`e1_vsae_ref_fullmatch` (RESULTS addendum 5), and **with every enumerated
difference matched the two implementations are indistinguishable on all four
metrics** (largest effect d = 0.8, nothing significant) at the power where every
previous generation of the arm was detected at 5 sigma.

The order matters as much as the result. The factor set was fixed by a code diff
*before* the arm ran, so the arm is confirmatory rather than another step in a
garden of forking paths, and the prediction recorded for it (that the init draw
would be null) was **wrong** — it is worth d = −4.7 on FVE on its own. E1 stops
here because the list is empty, not because the arms finally agreed.

The liveness/reconstruction frontier ("Claims worth opening" 2) has been read too
— see there and RESULTS addendum 6.

**The sigma-annealing arm has landed (RESULTS addendum 7).** `log_var_init` is now
a factor on `VSAETopKConfig`/`ExperimentConfig`; `e2_sigma_low_init` sets it to
−8.0 (below the `[-6, 2]` reparameterise clamp, so sigma is pinned at
`exp(-3) = 0.0498` from step 0 rather than declining there over training) with
`kl_coeff=0.0`, `var_flag=1` — otherwise identical to `e2_sampling_only`. 13 seeds:
FVE rises 0.484 → **0.834** (baseline 0.900, so **84.1%** of the gap closes) and
`frac_alive` reaches **1.0000** — better than baseline's own 0.9934. A real 5-sigma
residual against baseline remains (FVE d = +103), so the reparameterisation is not
entirely free, but the dominant story — "94.7% of the damage is the
reparameterisation" — turns out to be substantially an **initialisation artifact**,
not an architectural incompatibility between sampling and discrete TopK selection.
`read_learned_sigma.py` confirms the mechanism: 100% of measured values sit at the
clamp floor from the first checkpoint onward, i.e. this arm is
deterministic-equivalent throughout, and that alone is what recovers most of the
gap. What's left needs either E4 (no GPU, next steps #1) or desk work (#2).

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

### E2 — it is the sampling, not the KL (94.7% / 5.3%)

`kl_coeff` is not a shared scale across `var_flag`. A pre-registered two-stage design
(pilot at seed 101, confirmatory at seeds disjoint from it) found that **no beta in
{1e-4 … 1} puts a `var_flag=1` model within 0.02 FVE of baseline** — FVE degrades
monotonically (0.4581 → 0.0001) and never approaches baseline's 0.9003.

The decisive run was a control with **sampling on and the KL entirely off**:

| config | FVE (13 seeds) |
|---|---|
| baseline (deterministic) | 0.900159 ± 0.0006 |
| `var_flag=1`, beta=1e-4 | 0.460894 ± 0.0025 |
| `var_flag=1`, **beta=0** | 0.484106 ± 0.0022 |

Removing the KL entirely recovers **5.3%** of the gap. The other **94.7% is the
reparameterisation**. This is a claim about the architecture, not a hyperparameter,
and it is much stronger than the beta-tuning story E2 was built on.

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

### The method earned its keep twice

Both times, the **two-threshold** liveness pre-registration (F8b: robust only if both
agree; a disagreement is itself the finding) caught a *shape* change in the usage
distribution that a single threshold would have reported as a clean one-directional
effect — once for E1's init factor, once for E1a. Keep reporting both.


### E2's learned sigma — the posterior collapses, and the damage is not the noise

Read 2026-09-03 with `falsification/read_learned_sigma.py`, 20,000 activations,
same batch for every checkpoint, 13 seeds per arm. Full detail in RESULTS
addendum 3.

**Every one of 41 million measured `log_var` values is at or below the [−6, 2]
clamp floor**, and the largest is 24 log-units beneath it. Mean raw `log_var` is
−66.92 (`e2_confirm`) and −71.32 (`e2_sampling_only`): the encoder asks for
sigma ≈ 3e−15 and the clamp gives it exp(−3) = 0.0498. **The variational SAE's
optimum is the deterministic SAE** — the strongest form the degeneracy claim could
take, and it needed no new training. The KL is minimised in sigma at sigma = 1 and
so pulls the other way; it moves `log_var` by 4.4 units and is overwhelmed, which
is the right direction and confirms the reading.

**The noise at convergence is harmless.** `evaluation.py:51` calls
`dictionary(x, output_features=True)` without passing `training=`, and
`VSAETopK.forward` defaults it to `True`, so every recorded FVE for a `var_flag=1`
arm was measured *with* sampling on. Turning it off on identical data changes FVE
by 0.000012. So E2's damage is **an optimisation-path effect**: the model has
already learned to switch the noise off as hard as it can and is still at FVE 0.46
against baseline's 0.90. Training under sampling lands the optimiser somewhere
worse and it does not recover once the posterior collapses.

E2's "94.7% is the reparameterisation" stood with a mechanism attached: sigma
starts at exp(−1) = 0.368 (`log_var_init = −2.0`) when mu is still small, so the
noise-to-signal ratio is worst exactly when the TopK selection is being
established, and the prediction was that initialising `log_var` below the clamp
floor should recover most of the gap if a bad initial noise scale is the whole
story.

**Tested 2026-09-04, and the prediction was right.** `e2_sigma_low_init`
(`log_var_init=-8.0`, otherwise identical to `e2_sampling_only`), 13 seeds
(RESULTS addendum 7): FVE 0.484 → **0.834** against baseline's 0.900 — **84.1%** of
the gap closes — and `frac_alive` reaches **1.0000**, exceeding baseline's 0.9934.
A real 5-sigma residual against baseline remains (d = +103 on FVE), so the
reparameterisation is not entirely free of cost, but **the dominant share of E2's
headline number was an initialisation artifact, not an architectural
incompatibility**. `read_learned_sigma.py` on the new arm shows 100% of values at
the clamp floor from the first checkpoint — this arm is deterministic-equivalent
throughout training, unlike `e2_sampling_only` which arrives there late — and that
alone is what recovers 84–89% of the gap. Restate E2 as: *the reparameterisation
costs little once the initial noise scale is matched to where the clamp will take
it anyway; a naive default (sigma starting at 0.368 while mu is small) makes the
KL-off control look far worse than the architecture actually is.*

### Pre-registered but never reported

One item in the battery's design still appears nowhere in
`RESULTS_2026-09-03.md`. It is not blocked; it was simply overtaken.

* **E0, the pipeline negative control.** 13 `baseline` seeds exist and are
  analysed. The test itself — split them into two arbitrary groups, run the real
  metric pipeline, ask the framework to validate "group A is better organised" —
  has not been run. It is the only check that the real pipeline is exchangeable
  across seeds, and every downstream p-value assumes it is.

~~The learned sigma.~~ **Done 2026-09-03** — see the section immediately above.

---

## Next steps, in priority order

The sigma-annealing arm (previously #1) landed this session — RESULTS addendum 7,
and see "What is established" above. What's left is E4 (no GPU) or desk work;
nothing is blocked.

### 1. E4 — the size-matched SCR/TPP control, no training required

Design under **Experiments** below and in `RUNBOOK.md` section 1. The missing
piece is the `scorer(keep_indices)` function that masks the dictionary and calls
SAEBench; everything else (usage counts from the baseline summaries) is on disk,
and the machinery around it is implemented and tested against synthetic scorers
with known ground truth.

### 2. The residual E2 gap — instrument Jaccard overlap during early training

`e2_sigma_low_init` closed 84% of the FVE gap and reached 100% liveness, but left
a real 5-sigma residual against baseline (d = +103 on FVE). That residual is now
the interesting quantity: Claims-worth-opening #3's hypothesis (selection
instability under noisy scores, not variational inference per se) predicts it
should show up as index churn early in training even with the low-sigma init,
just less of it than under the default init. Instrumenting Jaccard overlap of
selected indices **during** a training run (not on converged checkpoints, which
addendum 3 already showed reads as falsely stable) is the direct test and the
best new science left on the list — see Claims-worth-opening #3 for the two ways
to run it.

### 3. Desk work — no GPU, no new code

Both items under **Claims worth opening** (#4, the field-level projection claim;
#5, the seed-count survey) are pure analysis/writing and can be done from a
remote/no-GPU session.

### 4. Optional — extend the E2 beta grid downward (1e-5, 1e-6)

The beta=0 control (`e2_sampling_only`) makes this much less interesting: if
94.7% of the damage is the sampling, no smaller beta recovers a healthy model.
Doing it anyway would close the question formally, but it is a **new
pre-registration**, not a continuation of this one.

---

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

---

---

## Claims worth opening

Ranked by value per unit of cost. The first two need no GPU and no new code beyond
a script; the third is the most scientifically interesting thing available.

### 1. ~~"The deterministic SAE is the variational SAE's optimum"~~ — ANSWERED

**Yes, emphatically.** Read 2026-09-03; see "What is established" above and RESULTS
addendum 3. Mean raw `log_var` is −66.9 against a clamp floor of −6, with 100% of
41M values at or below the floor. What it opened instead is the annealing arm under
Next steps #1, and a correction to the method proposed for claim 3 below.

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

### 3. "The reparameterisation trick is incompatible with discrete top-k selection,
not with sparse autoencoders" — the best new science here

E2 established **that** 94.7% of the damage is the sampling rather than the KL,
and RESULTS addendum 7 has since shown 84% of *that* is an initialisation
artifact rather than the sampling itself. What remains — a real, 5-sigma residual
between `e2_sigma_low_init` and `baseline` — is smaller than the original number
but still needs a mechanism, and the obvious one is testable and sharp.

**Hypothesis.** TopK selection is a discrete argmax over noisy scores. Sampling
`z = mu + sigma*eps` before selection means the selected index set changes between
forward passes for the same input, so the decoder is asked to reconstruct from an
inconsistent feature assignment and every feature's decoder column is trained on a
moving target. The damage is then a property of *discrete selection under noise*,
not of variational inference.

**Two things are already known about it.** The damage is an optimisation-path
effect — it is in the trained weights, not in the eval-time noise (RESULTS
addendum 3) — which is what this hypothesis predicts. And one sub-hypothesis is
already falsified: TopK is *not* forced to pad its selection with `mu == 0`
features. There are 975–1064 strictly positive features per token against k = 256,
and the measured fraction of selected features with `mu == 0` is exactly 0. If
selection instability is the mechanism it works through boundary flips, not through
zero-padding.

**Two ways to test it, both cheap, and the repo already has the trainers:**

* **Measure the instability during EARLY TRAINING, not on converged checkpoints.**
  The original form of this test — Jaccard overlap of selected indices on the
  finished `e2_confirm` checkpoints — is now known to be useless: at the collapsed
  sigma, sampling on versus off moves FVE by 0.000012, so the converged forward
  pass is stable and the test would read as "no instability" and be misinterpreted
  as falsifying the mechanism. Instrument the Jaccard overlap **during** a training
  run instead, where sigma starts at 0.368 and mu is still small. Now that
  `e2_sigma_low_init` exists, run the instrumentation on **both** it and
  `e2_sampling_only`: the prediction is less early-training index churn under the
  low-sigma init (which is most of why its FVE recovers), with some churn
  remaining (which is the source of the residual 5-sigma gap against baseline).
* **Vary the discreteness of the sparsity** (training). `vsae_jump_relu.py` has
  `var_flag` and its sparsity is a *learned threshold*, not a top-k selection;
  `vsae_batch_topk.py` selects discretely but across the batch. Run
  sampling-on vs sampling-off for each, exactly as `e2_sampling_only` did for
  TopK. **Prediction: the FVE damage is large for TopK, intermediate for BatchTopK,
  and small for JumpReLU.** If that ordering holds, the claim is established and it
  generalises well beyond this preprint.

This is the one item on the list that produces a positive, mechanistic, transferable
result rather than a correction. It is also the natural headline for a second paper.

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
> reparameterisation, not the KL. The learned-sigma diagnostic below has now
> been read: it collapses past the clamp floor. See "What is established".

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
> **Outcome: not started.** Blocked only on the SAEBench-side scorer. Next steps #2.

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

**Budget:** ~28 training runs. At 20–40 min each on the 3080, roughly 10–19 hours,
parallelisable across days. E4 needs no training and should start immediately.

**Pre-registration.** Per simulation finding 2, the battery above and its order are
fixed before any seeded run is inspected. Adding an arm after seeing results is
permitted only if reported as exploratory and excluded from the evidence product.

## Deliverables

1. **Paper.** Sequential falsification for interpretability claims, with the vSAE
   study as the case study whose conclusion it reverses. Target: a venue caring
   about measurement validity and falsifiability (the InterpScience CFP framing
   still fits).
2. **`falsification/`** as a reusable package, with the Type-I calibration from E0
   as its empirical warrant.
3. **Corrections** to the preprint's record: the degeneracy, the AuxK confound, the
   config mismatches, the self-contradiction between Global and Conclusion.

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


---

## Where the detail lives

| file | what it holds |
|---|---|
| **this file** | current state, what is established, next steps, the pre-registration, open decisions |
| `CLAUDE.md` | standing landmines in the vSAE code; read before touching `dictionary_learning/` |
| `falsification/RESULTS_2026-09-03.md` | all measured results. Addendum 1: 13-seed/5σ rerun. 2: gradient projection. 3: the learned sigma collapses. 4: the E1 code diff, enumerated and frozen (15 items). 5: the closing arm — E1 lands. 6: the liveness/reconstruction frontier. 7: the sigma-annealing arm — 84% of E2's gap is the init, not the reparameterisation |
| `falsification/FINDINGS_2026-09-02.md` | the five instrumentation bugs the pilot exposed |
| `falsification/REMEDIATION.md` | fix tracking + the four author decisions and their rationale |
| `RUNBOOK.md` | commands, arm table, E4 design |
| `falsification/frontier.py`, `read_penalty_clamp.py`, `read_learned_sigma.py` | the three checkpoint-reading analyses behind addenda 3, 4 and 6 — no training, needs the GPU only to draw activations |
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
