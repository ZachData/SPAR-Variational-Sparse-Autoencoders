# Results

What the experiments in this repository established, in final form. Every
number below is recomputed from the committed data by `python reproduce.py
--check` (see `REPRODUCE.md`) unless marked *GPU*, in which case it was read
from a checkpoint by the named script and is cached in
`falsification/*_results.json`. The design and its pre-registration are in
`METHODS.md`; the bugs and retractions along the way are in `ERRATA.md`; the
dated record, including the order in which things were found, is in
`notebook/`.

Unless stated otherwise: gelu-1l, `blocks.0.hook_resid_post`, dictionary
d = 2048, k = 256, AuxK α = 1/32, lr 8e-4, 10,000 steps, **13 seeds per arm**.
Effect sizes are Cohen's *d* in across-seed standard deviations (pooled SD =
mean of the two groups' SDs); *p*-values are two-sided seed-permutation tests
with 4 M Monte Carlo draws, whose floor is 5.16σ. Liveness is reported at both
pre-registered thresholds — the fraction of features selected in fewer than
0.1× and 0.5× of k/d samples over 10⁶ tokens — and a liveness result counts
only if both agree in direction.

## The claim under test

The arXiv preprint this repository extends introduced variational sparse
autoencoders (vSAEs): a TopK SAE whose encoder emits a posterior mean and
log-variance, with a KL term to a standard normal prior, and concluded from a
Pythia-70M comparison and a β sweep that the KL term disperses features and
improves downstream interpretability (SCR, TPP). Six findings, summarised
here and detailed below:

1. **The evaluated vSAE was never variational.** With `var_flag = 0` there is
   no sampling and the KL reduces to ½‖μ‖²; every checkpoint the preprint
   evaluated is a TopK SAE with an L2 activation penalty. The two
   implementations of that identical objective nonetheless differed by
   up to *d* = 51 on reconstruction and 183 on liveness as released, and the
   whole gap decomposed into five implementation details (a bias form, a
   warmup, two initialisation choices, a gradient projection), none in
   either paper's equations. With all five matched: null on every metric.
2. **With sampling on, the damage is the reparameterisation, not the KL.**
   Removing the KL entirely recovered 3.9 % of the gap to a deterministic
   baseline; the other 96.1 % was sampling. No β in {10⁻⁴ … 1} gave a working
   model.
3. **The mechanism is selection churn under noise, and it is a property of
   hard-*k* selection.** At training-time noise TopK's selection is close to
   random (Jaccard 0.075–0.29 between two passes, chance 0.067). Across a
   six-point noise-scale sweep FVE rose monotonically with that Jaccard
   (Spearman ρ = 1; *r* = +0.952 TopK, +0.945 BatchTopK). For JumpReLU's soft
   learned threshold the naive coupling (*r* = +0.977) was confounded by a
   2.7× swing in achieved L0; at matched L0 it reversed (*r* = −0.55, n = 7).
4. **One-line implementation details are effects of *d* = 7–175.** As
   one-factor contrasts, on each detail's most-affected metric: a ReLU on the
   posterior mean (*d* = 32.5), the initial decoder scale (30.3), the
   decoder-gradient projection (15.3), the initial weight distribution (7.3),
   and on six pilot seeds the bias form (175) and a KL warmup (8.6).
5. **The field cannot see effects of that size.** All ten SAE-methods papers
   in the bibliography train one seed per configuration; a one-seed design
   has a sigma ceiling of 0.
6. **On the preprint's own Pythia checkpoints, SCR and TPP disagree** about
   whether the vSAE's advantage is explained by its smaller live dictionary,
   and the disagreement survived every robustness check run on it. It is
   reported as unresolved.

---

## 1. Fixed variance: the null model, and what separated two implementations of it

*Experiment E1. Arms `e1_penalty` (TopK + ½·β‖f‖², β = 1) against successive
generations of `e1_vsae_ref` (vSAE, `var_flag = 0`, β = 1).*

**The identity.** At σ = 1 the vSAE's KL term is ½‖μ‖² and its forward pass
is deterministic, so a fixed-variance vSAE is a TopK SAE with an L2 penalty.
This was verified as algebra first: both trainers return the same loss,
511.895264, on a shared batch (the dated record); `check_loss_identity.py`
now re-checks it with one set of weights loaded into both trainers —
agreement term by term to float32 precision (relative difference ≤ 3 × 10⁻⁷)
at initialisation and at a trained `e1_penalty` checkpoint. Every checkpoint in the preprint's analysis
directory is named `_fixed_var`, which the training script emits exactly when
`var_flag = 0`.

**The pilot found them very different.** At six seeds the two arms separated
at *d* = 37–59 on the original `features_used` metric — a metric that turned
out to saturate at this dictionary size (`ERRATA.md`) — and at *d* ≈ 13–16 on
reconstruction and the pre-registered liveness thresholds once the metric was
replaced. Three stacked configuration mismatches were removed first: the
vSAE ramped its KL over 1,000 steps while the penalty arm applied its penalty
flat from step 0; the vSAE used an untied decoder bias and no pre-bias while
the penalty arm centred its input on a tied one; and the vSAE initialised its
decoder at 0.1× unit norm.

**The code diff was then enumerated and frozen.** Reading the two trainers
and their training scripts against each other produced fifteen differences
and no more: two matched by configuration, two run as measured factors, two
confirmed no-ops by measurement, seven static no-ops (unreachable branches or
algebraically inert given the configuration), one — the RNG consumption
order — treated as noise by the seed-permutation design itself, and one, the
initial weight draw, run as the closing factor. Fixing the factor set from
the diff *before* the closing arm ran is what makes the ladder confirmatory
rather than a search for agreement.

Exactly one of the fifteen was a difference in the *objective*: the vSAE
penalises clamp(z, −10, 10) where the penalty arm penalises f unclamped.
Measured over 20,000 activations and every seed of all five E1 arms
(*GPU*, `read_penalty_clamp.py`, cached in `penalty_clamp_results.json`), the
largest pre-clamp activation observed anywhere was 0.215 against a clamp at
10 — 46× headroom, never binding. (An earlier reading over three of the five
arms reported 0.194.) (Read in the
saved-checkpoint space rather than training space the maximum is 4.96 and the
headroom looks like 2×; the clamp acts in training space.) The identity is
unqualified and the entire measured gap lay in parameterisation,
initialisation and the update rule, not the objective.

**The ladder.** Each generation matched one more factor. *d* is
`e1_penalty` minus the arm.

| vSAE arm | factors matched (cumulative) | FVE | frac. recovered | < 0.1× k/d | < 0.5× k/d |
|---|---|---|---|---|---|
| `e1_vsae_ref` | KL warmup, bias form | −5.7 | −8.5 | −2.8 | **+13.0** |
| `e1_vsae_ref_unitinit` | + decoder init scale | **+16.5** | +17.7 | +0.8 ns | +0.2 ns |
| `e1_vsae_ref_gradproj` | + gradient projection | +3.8 | +4.9 | −2.4 | −3.7 |
| `e1_vsae_ref_fullmatch` | + initial weight draw | −0.3 ns | −0.8 ns | −0.1 ns | −0.6 ns |

Every intermediate generation was significant on at least one metric family,
and each factor traded against the last: matching the decoder's initial
scale closed the liveness gap and opened reconstruction (*d* = 16.5);
matching the gradient projection closed 78.9 % of the FVE gap and 74.9 % of
the `frac_recovered` gap and re-opened liveness. Only the final row, where the
code diff is exhausted, was null on every metric — at the same power that
detected every earlier generation at 5σ. FVE differed by −0.0003
(*p* = 0.39), `frac_recovered` by −0.0004 (*p* = 0.07), `frac_alive` was
identical to six decimals, and both liveness thresholds were non-significant
and agreed.

Two of the factors deserve to be named:

* **The decoder-gradient projection.** `vsae_topk.py` imported
  `remove_gradient_parallel_to_decoder_directions` and never called it,
  while still renormalising the decoder every step — the radial gradient
  component was applied and then undone. As a factor on its own (`unitinit`
  vs `gradproj`) it was *d* = −14.3 on FVE and *d* = −15.3 on
  `frac_recovered`, and it moved the vSAE *away from* the null model on
  liveness (0.1816 → 0.2197 at the loose threshold, against `e1_penalty`'s
  0.1836) even though the null model had had the projection all along. The
  same update-rule change had opposite-signed effects in the two
  implementations: an interaction, not a missing match.
* **The initial weight distribution.** Normalised Gaussian versus normalised
  uniform columns, identical in every summary statistic anyone would report
  about a random 512-dimensional direction, was predicted in writing to be
  null. It was *d* = −4.7 on FVE and *d* = −7.3 on `frac_recovered` on its
  own.

**What this licenses** is the decomposition, not a bare equivalence verdict.
A failure to detect a difference is not a demonstration of its absence in the
formal sense — a pre-specified margin would be needed for that, and the
project's open decision on that margin was never taken because at 13 seeds
the across-seed SDs are ~10⁻³ and essentially any non-zero difference clears
5σ. But the margin is no longer load-bearing for which way the result goes:
the residual differences (0.0003 FVE, 0.006 live fraction) are two orders of
magnitude smaller than the gaps that opened and closed at every rung.

*Record: notebook RESULTS addenda 1, 2, 4, 5.*

## 2. Sampling on: it is the reparameterisation, not the KL

*Experiment E2. `var_flag = 1` arms with a learned log-variance
(`log_var_init = −2`).*

**Where these numbers are measured.** Every number in §2–§3 is measured in
the regime the models were trained in (*GPU*, `read_training_regime.py`,
cached in `falsification/training_regime_results.json`): each checkpoint is
converted back by `scale_biases(1/norm_factor)` and fed activations divided by
the re-estimated norm_factor (25.47). A saved sampled checkpoint fed raw
activations runs at σ/norm_factor ≈ σ/25.5 of its training noise, because the
save-time rescaling multiplies μ by norm_factor and leaves log σ² alone
(`ERRATA.md` §2). An earlier version of this file, and of the paper, measured
the saved checkpoints that way; those numbers are kept, beside these, in the
paper's appendix table "The noise sweeps as saved and in the training regime"
and in `reproduce.py`'s output.


**No β gives a working model.** A pre-registered two-stage design — a
single-seed pilot over β ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}, then 13 confirmatory
seeds at the selected β on seeds disjoint from the pilot — found FVE falling
monotonically from 0.329 at β = 10⁻⁴ to zero by β = 10⁻², never approaching
the deterministic baseline's 0.900 (as saved: 0.458 → 0.0001). The
confirmatory arm `e2_confirm` (β = 10⁻⁴) characterises a model at FVE ≈ 0.33,
and its liveness numbers
were recorded as confounded before they were read.

**The control that separated the two explanations** was sampling on with
β = 0, so the KL contributes exactly zero to the loss (`e2_sampling_only`,
proposed after the pilot and reported as exploratory):

| Configuration | FVE |
|---|---|
| baseline, deterministic | 0.900239 ± 0.0006 |
| σ learned, β = 10⁻⁴ (`e2_confirm`) | 0.329065 ± 0.0007 |
| σ learned, β = 0 (`e2_sampling_only`) | 0.351120 ± 0.0006 |

Of the 0.5712 gap, removing the KL recovered 3.9 %; the remaining 96.1 % was
attributable to the reparameterisation itself. (As saved, after the
`scale_biases` log-variance fix, the split was 93.6 / 6.4 of a 0.4420 gap;
before that fix it was reported as 94.7 / 5.3. `ERRATA.md`.)

**The learned σ does not collapse.** Read correctly (*GPU*,
`read_learned_sigma.py` with the bias correction), `e2_confirm` and
`e2_sampling_only` settle at mean log σ² ≈ −2.6 to −2.7, σ ≈ 0.27 — nowhere
near the reparameterisation clamp's floor of −6 (σ = 0.0498). The
deterministic SAE is not the variational SAE's optimum; the first reading of
this, which said it was, was a measurement artifact (`ERRATA.md`).

**Most of the gap is the initial noise scale.** `e2_sigma_low_init`
(`log_var_init = −8`, clamped to −6 from step 0; otherwise identical to
`e2_sampling_only`) reached FVE 0.7494 ± 0.0003 against `e2_sampling_only`'s
0.351 — 73 % of the gap to baseline closed (84 % as saved) — and a real
residual of 0.151 FVE remained, at the 5.03σ permutation floor. The residual
was then explained by the mechanism in §3.

*Record: addenda 3 (retracted), 7, 8, 9; REMEDIATION F6.*

## 3. The mechanism: selection churn under noise, for hard-*k* selection only

**Pre-flight.** TopK selection is an argmax over pre-activations; noise of
scale σ flips the selected set whenever the gap between the *k*-th and
(*k*+1)-th largest pre-activation is smaller than the noise. Measured on a
converged deterministic checkpoint (*GPU*, `read_preact_gap.py`) that gap has
median 0.0001 and 99th percentile 0.0006 in training space. Even the clamp's
floor σ (0.0498) sits 80–540× above it, so no achievable noise scale keeps
sampling below the boundary gap. The original prediction — that FVE and
Jaccard would show a knee together at a threshold σ — was therefore revised,
in writing, before the sweep ran: no knee is reachable on this architecture.

**Selection Jaccard** is the overlap between the TopK sets of two independent
stochastic forward passes on the same token, averaged over 8,192 tokens at
the converged checkpoint (*GPU*, `read_a{2,3,4}_dose_response.py`; chance
level is 0.067 for k = 256, d = 2048). Across training (5 seeds,
15 checkpoints from step 0 to 10,000; *GPU*, `read_trajectory_regime.py`),
`e2_sampling_only`'s selection never leaves chance in the regime it trains
in (0.067 → 0.086), and `e2_sigma_low_init`'s rises only to 0.292
(0.067 → 0.292); both endpoints match §3.1's table. Measured as saved, on raw
activations, the same checkpoints read 0.069 → 0.811 and 0.431 → 0.952
(`read_selection_jaccard.py`), which is what an earlier version reported. An earlier reading that TopK was not padding
its selection with μ = 0 features (0 % of selections) was also made as saved
and does not hold in training: on `a2_sigma_init_m3` seed 1, 71 % of each
pass's selections lie outside μ's own top-256 and 24 % have μ = 0, selected
on noise alone (*GPU*, `read_selection_example.py`; 5 % and 0 % as saved).

### 3.1 TopK: a smooth dose-response, and FVE rises monotonically with Jaccard

Six noise scales, 13 seeds each (log σ²_init ∈ {−1, −2, −3, −4, −5, −8}, the
last clamped to −6). Training regime. Deterministic baseline FVE =
0.9002 ± 0.0006.

| log σ²_init | σ | FVE | Jaccard |
|---|---|---|---|
| −1 | 0.6065 | 0.2517 ± 0.0006 | 0.0750 ± 0.0001 |
| −2 | 0.3679 | 0.3511 ± 0.0006 | 0.0862 ± 0.0002 |
| −3 | 0.2231 | 0.4708 ± 0.0005 | 0.1094 ± 0.0004 |
| −4 | 0.1353 | 0.6107 ± 0.0005 | 0.1686 ± 0.0005 |
| −5 | 0.0821 | 0.7200 ± 0.0004 | 0.2624 ± 0.0005 |
| −8 (clamped) | 0.0498 | 0.7494 ± 0.0003 | 0.2919 ± 0.0006 |

Both curves rise smoothly and monotonically with no knee, as the pre-flight
predicted. FVE and Jaccard agree in rank order at every point (Spearman
ρ = 1; Pearson *r* = +0.9521 over the six arm means). The relationship is
monotone rather than linear: at the three noisiest points Jaccard sits near
chance (0.075–0.109) while FVE still climbs 0.25 → 0.47. Even at the clamp
floor selection is far from stable (Jaccard 0.292) and FVE is 0.151 below
baseline. As saved the same checkpoints give Jaccard 0.77–0.95 and
*r* = +0.9993. Figure: `workshop/figs/a2_dose_response.pdf`.

### 3.2 BatchTopK: a global selection budget does not decouple them

BatchTopK selects the global top-(*k*·batch) activations over a flattened
batch rather than exactly *k* per token — the natural candidate for "more
noise-robust", since one token's noise-promoted feature can be compensated by
another's demoted one. The identical sweep on `VSAEBatchTopK` (matched d, k,
hook point and schedule; two saved-checkpoint bugs found and fixed en route,
`ERRATA.md`) gave *r* = +0.9455, rank order again identical at every point
(as saved: +0.9979). Training regime; deterministic baseline FVE =
0.9508 ± 0.0003.

| log σ²_init | σ | FVE | Jaccard |
|---|---|---|---|
| −1 | 0.6065 | 0.2459 ± 0.0005 | 0.0758 ± 0.0004 |
| −2 | 0.3679 | 0.3599 ± 0.0026 | 0.0826 ± 0.0011 |
| −3 | 0.2231 | 0.4833 ± 0.0022 | 0.0968 ± 0.0011 |
| −4 | 0.1353 | 0.6232 ± 0.0008 | 0.1311 ± 0.0005 |
| −5 | 0.0821 | 0.7366 ± 0.0006 | 0.1881 ± 0.0005 |
| −8 (clamped) | 0.0498 | 0.7692 ± 0.0004 | 0.2101 ± 0.0006 |

If anything the more elastic budget was slightly *more* exposed: as a
fraction of its own deterministic baseline, BatchTopK recovered less than
TopK at every one of the six matched scales (80.9 % vs 83.2 % at the clamp
floor). The mechanism does not depend on strict per-token selection; it looks
like a property of hard, cardinality-constrained selection under noise in
general. (`frac_recovered` is not reported for this trainer: its global
budget miscounts under the evaluation harness's 3-D activations,
`ERRATA.md`.) Figure: `workshop/figs/a3_batchtopk_dose_response.pdf`.

### 3.3 JumpReLU: the coupling is confounded by achieved sparsity, and reverses when controlled

JumpReLU replaces the hard cardinality constraint with a per-feature learned
threshold and a soft L0-target loss — discrete but not hard-*k*, the sharpest
available test of whether the mechanism needs hard-*k* selection specifically.
No training script had ever exercised the variational JumpReLU trainer;
building one surfaced three bugs, one of which (the gate applied *before*
sampling, so noise could never change the selected set) would have made the
experiment structurally meaningless (`ERRATA.md`). With all three fixed and
verified (mean Jaccard between repeated passes on one token 0.35 after the
fix, 1.0 before), the same sweep gave *r* = +0.9770 naively. Training regime;
deterministic baseline FVE = 0.9213 ± 0.0002, L0 = 269.5 (target 256).

| log σ²_init | σ | L0 | FVE | Jaccard |
|---|---|---|---|---|
| −1 | 0.6065 | 121.7 | 0.3401 ± 0.0012 | 0.2628 ± 0.0030 |
| −2 | 0.3679 | 202.7 | 0.4671 ± 0.0015 | 0.3199 ± 0.0020 |
| −3 | 0.2231 | 331.4 | 0.6349 ± 0.0011 | 0.5526 ± 0.0029 |
| −4 | 0.1353 | 268.1 | 0.7073 ± 0.0008 | 0.7186 ± 0.0102 |
| −5 | 0.0821 | 269.0 | 0.7707 ± 0.0006 | 0.7216 ± 0.0047 |
| −8 (clamped) | 0.0498 | 270.3 | 0.7848 ± 0.0005 | 0.7090 ± 0.0041 |

The naive number is not a replication. Achieved L0 swung from 121.7 to 331.4
— a 2.7× range, non-monotonic in σ (under half the target at the two
noisiest points, 30 % over it at σ = 0.22). TopK and BatchTopK hold sparsity
exactly fixed at every grid point; JumpReLU does not, and *r*(FVE, L0) =
+0.8054 over the same six points. Only three of the six points have L0 in
the band [240, 320] fixed as comparable to the target (log σ²_init ∈ {−4, −5,
−8}), too few for a correlation; across them Jaccard is flat (0.709–0.722)
while FVE rises 0.707 → 0.785.

**The follow-up densified the L0-matched region.** Four points
(log σ²_init ∈ {−3.5, −4.5, −5.5, −6.0}, 5 seeds each — the under-powered
quantity was grid coverage, not per-point precision) bring the band to seven
points: *r*(FVE, Jaccard) = −0.5544, *r*(FVE, L0) = −0.7251 (n = 7). Jaccard
stays within 0.708–0.724 while FVE rises 0.679 → 0.785. The full ten-point
naive coupling is *r* = +0.9610, carried by the three points outside the
band. (As saved — effective noise σ/25.5 — L0 swung 36.8 → 290.8, the naive
coupling was +0.9334, and the eight-point band gave +0.5026.)

**What this licenses.** The FVE-vs-churn coupling does not hold for a soft,
no-*k* selection mechanism: at matched sparsity its sign is wrong. The more
basic finding is that a gradient-learned threshold's achieved *sparsity
level* is itself not noise-robust, a failure mode hard-*k* architectures
cannot exhibit. Seven points, four at 5 seeds, is still a small-n
correlation; this is not a formal null. Figures:
`workshop/figs/a4_jumprelu_dose_response.pdf`, `a4_followup_dose_response.pdf`.

*Record: addenda 8, 16, 17, 18, 20, 23.*

## 4. The ReLU that only one trainer applied

*Experiment E3. `e3_masked_kl` (KL masked to the selected features, no ReLU,
as in the preprint's equations) against `e3_masked_kl_relu` (the same with
`F.relu(mu)`, as in the released `vsae_topk.py`).*

The masked-KL trainer was built to test the preprint's stated feature-death
mechanism directly, and differed from the released trainer by the ReLU as
well as by the mask. Rather than patch one to match the other, the ReLU
became a flag (`relu_mu`, default `False` so no existing checkpoint changed)
and both were run:

| Threshold | no ReLU | ReLU | *d* |
|---|---|---|---|
| < 0.1× k/d | 0.1315 ± 0.0074 | 0.0287 ± 0.0033 | +19.3 |
| < 0.5× k/d | 0.4655 ± 0.0113 | 0.3081 ± 0.0092 | +15.3 |

Both thresholds agree, so this is a robust effect and not a shape change:
the ReLU cut the near-dead population 4.6× at the tight threshold. Had either
trainer been patched to match the other — routine practice when reconciling
two implementations of "the same" model — every subsequent number would have
inherited a *d* ≈ 15–19 effect misattributed to the KL mask. The mask's own
effect on feature death was not separately measured against the unmasked
trainer, because that comparison would have been confounded by the ReLU and
by the rest of the E1 code diff.

*Record: the battery's main results (E3) and its first addendum; REMEDIATION F9b.*

## 5. Implementation variance, and whether the field can see it

### 5.1 The catalogue

Each detail as a one-factor contrast — two arms that differ in it alone —
|*d*| on the four pre-registered metrics (`reproduce.py`'s Table 7):

| Detail | Seeds | FVE | frac. rec. | < 0.1× | < 0.5× | σ |
|---|---|---|---|---|---|---|
| Bias form (pre-encoder bias or not) | 6 v 13 | 39.7 | 31.7 | 121.7 | 174.5 | 4.13 (ceiling) |
| KL warmup (1,000 steps or none) | 6 v 6 | 6.9 | 2.9 | 8.6 | 2.9 | 3.07 (ceiling) |
| Initial decoder scale (0.1 or 1) | 13 v 13 | 23.1 | 30.3 | 2.8 † | 18.0 † | 5.03–5.16 |
| Decoder-gradient projection | 13 v 13 | 14.3 | 15.3 | 2.4 | 5.0 | 4.48–5.16 |
| Initial weight distribution | 13 v 13 | 4.7 | 7.3 | 2.2 | 3.6 | 4.31–5.16 |
| `F.relu(mu)` | 13 v 13 | 32.5 | 24.0 | 19.3 | 15.3 | 5.03–5.16 |

† the two liveness thresholds disagree in direction (a shape change). Every
entry has *p* ≤ 0.05. The first two rows come from the archived six-seed
pilot arms (`archive/e1_vsae_ref_{aprilmode,klwarmup1000}`), the only arms in
which those two details differ alone. None is a hyperparameter anyone tunes,
reports or ablates; each is one trainer's default, one flag or line of code,
and in neither paper's loss equation. An earlier version of this table
printed gaps to the null model for three entries (`ERRATA.md` §4).

### 5.2 The projection effect does not generalise to a plain TopK SAE

Every arm in §1 carries an activation penalty. Whether the projection is a
generic unreported factor in TopK training or an interaction with the
penalty was tested directly: the trainer behind the clean baseline hardcoded
the projection, a flag was added (`project_decoder_grad`, default `True`),
and `claim4_baseline_noproj` — `baseline` with the projection off — ran at 13
seeds.

| Metric | projection on | projection off | *d* (*p*) |
|---|---|---|---|
| FVE | 0.900159 ± 0.0006 | 0.900009 ± 0.0005 | +0.3 (0.50) |
| frac. recovered | 0.964336 ± 0.0005 | 0.964065 ± 0.0006 | +0.5 (0.22) |
| < 0.1× k/d | 0.754169 ± 0.0013 | 0.753794 ± 0.0014 | +0.3 (0.53) |
| < 0.5× k/d | 0.754357 ± 0.0014 | 0.754056 ± 0.0013 | +0.2 (0.63) |

Nothing moved, at the power that detected *d* = 14.3 (FVE) and 15.3
(frac. recovered) for the same one-line change inside the vSAE family. The effect is real and undocumented
but scoped to models carrying an activation penalty — consistent with an
interaction between the projection and the penalty's gradient, not with the
projection mattering to decoder renormalisation on its own.

### 5.3 The seed-count survey

A two-sided seed-permutation test's smallest attainable *p* is 2/C(2n, n):

| n / group | 1 | 2 | 3 | 5 | 6 | 10 | 13 | 20 |
|---|---|---|---|---|---|---|---|---|
| sigma ceiling | 0.00 | 0.97 | 1.64 | 2.65 | 3.07 | 4.40 | 5.21 | 6.75 |

Thirteen seeds per group is the first *n* whose ceiling clears 5σ; six seeds
cannot express more than 3.07σ however large the effect; one seed cannot
express any significance at all for a claim of the form "architecture A
beats architecture B". Every SAE-methods paper cited in the paper's related
work — a sampling frame fixed by the bibliography, not curated — was searched
for the number of independently seeded runs per configuration: **ten of ten
report exactly one** (Cunningham et al. 2023; Bricken et al. 2023; Templeton
et al. 2024; Gao et al. 2024; Rajamanoharan et al. 2024, both papers;
Bussmann et al. 2024; Karvonen et al. 2025; Marks et al. 2024; Lu et al.
2025). The one exception found anywhere, Bricken et al.'s second
independently seeded transformer, serves a qualitative universality check
and its own n = 2 sits below *p* < 0.05 regardless. Two studies outside the
survey (Paulo & Belrose 2025, ~30 % feature overlap across seeds; Gerasimov
et al. 2026) confirm the instability a single-seed design cannot see.
Evaluation-level uncertainty (bootstrap CIs over tokens or raters) does
appear in that literature; seed-level uncertainty does not.

*Record: addenda 22, 25; `falsification/seed_count_survey_results.json`.*

## 6. The Pythia checkpoints: SCR and TPP disagree, and the disagreement is the finding

*Experiment E4. The preprint's two recovered Pythia-70M checkpoints (layer 3,
d = 8192): a TopK baseline with AuxK α = 0.03125 and its vSAE (β = 1, fixed
variance, AuxK off, 1,474 / 8,192 live). Single seed each side, so
descriptive by design; no permutation test is possible.*

The preprint read the vSAE's higher SCR and TPP as evidence of a more
disentangled feature space. But both metrics reward *selective* ablation,
and selectivity is easier with fewer live features; the vSAE had 18 % of its
dictionary alive against the baseline's 90 %. The size-matched design masks
the baseline down to a grid of N features by usage frequency (its best foot
forward — a random subset reconstructs badly and is beaten trivially, which
the test suite demonstrates would manufacture a false positive), scores each
point, and compares the vSAE's own unmasked score against the curve at
N = 1,474.

| Metric (`bias_in_bios`, professor/nurse) | vSAE | baseline curve at N = 1,474 | margin | verdict |
|---|---|---|---|---|
| SCR | 0.1017 | 0.0201 | +0.082 | not explained by size |
| TPP (5 classes) | 0.1055 | 0.1905 | −0.085 | explained by size |

Same two checkpoints, same dataset, same masking grid, opposite verdicts.
Both held under the random-subset bracket too (SCR +0.133, TPP −0.051). The
checks that followed, in order:

* **Not an averaging artifact.** Re-derived once per SAEBench ablation
  threshold (N = 2, 5, 10, 20), SCR's verdict held at all four and TPP's at
  all four (TPP decisive only at N ≤ 10, a near-tie at N = 20).
* **Not test-set noise.** A 200-draw bootstrap of the cached test set put
  SCR's mean margin at +0.082 [+0.024, +0.150] and TPP's at −0.086
  [−0.090, −0.081]. SCR's interval is ~15× wider and its per-threshold margin
  clears zero only at N = 2; TPP's side is the sturdier of the two. The full
  bootstrap that also resamples the train set and re-derives node effects
  every draw (7.2 h) widened SCR's interval to +0.088 [+0.008, +0.171] without
  crossing zero, and SCR's N = 2 and N = 5 feature selection was bit-identical
  between the two bootstraps.
* **Not one dataset or one pair.** Across SAEBench's three other
  `bias_in_bios` pairs and four `amazon_reviews` pairs, SCR said "not
  explained by size" on 7 of 8 points (mean margin +0.072; the one reversal,
  Books/CDs, was a near-tie against the random reference). TPP said
  "explained by size" on both datasets via the usage-ranked reference, with
  its random bracket going to a tie on `amazon`.
* **Not "masked versus trained small".** A TopK SAE trained from scratch at
  d = 1,474 scored 0.020127 on SCR against the masked curve's 0.020127 on the
  original point (6 × 10⁻⁸ apart) and 0.2025 on TPP against 0.1905. Extended
  to the other seven points, the trained-small reference shifted by 0.075 on
  average (up to 0.138) and flipped one point's sign — a real per-pair effect
  — but the aggregate margin was unchanged (+0.079 vs +0.072) and 6 of 8
  points still said "not explained".
* **Not a usage-rank artifact.** The prediction that SCR's top-effect
  features sit low in the baseline's usage ranking and TPP's high — which
  would have explained why the baseline's masked curve is flat for SCR and
  rising for TPP — was stated with its falsifier in advance and falsified:
  both metrics' top-effect features sat in the same 0.55–0.73 usage-percentile
  band at every N.

What was not controlled: single seed each side; the AuxK confound (the
baseline had it, the vSAE did not), which the size control cannot rule in or
out; and the two metrics' different constructions (SCR is a ratio with a
small, noisy denominator; TPP a plain accuracy difference). This is the
preprint's own methodological failure reproduced fresh — its Global section
and its Conclusion reached opposite verdicts on the same hypothesis because
nothing in the method could combine heterogeneous evidence — and it is
reported as unresolved rather than adjudicated by picking a metric.

*Record: addenda 10–15, 19, 21, 24.*

## 7. Exploratory: no single liveness–reconstruction frontier

Not pre-registered; every arm was trained for another purpose and no
hypothesis about the shape was recorded before plotting.

**Across the nine replicated fixed-variance arms** — which sample nothing,
so the measurement-regime issue of §2–§3 does not touch them — better
reconstruction goes with *more* near-dead features: arm-mean Spearman
ρ = +0.90 (*p* = 0.0009) and +0.78 (*p* = 0.013) at the two thresholds, the
same direction at both (`frontier.py`'s loader; `reproduce.py` checks it).
So "architecture A reconstructs better *and* has fewer dead features" is not
a doubly supported claim; it asserts A is off the frontier, which is rarer
and stronger than either half, and is the shape of claim the preprint made.

**What this replaces.** The first reading pooled 11 replicated arms
(ρ = −0.14, *p* = 0.69) and read the near-zero as two opposite-signed
relationships cancelling, with the eight working models sloping up
(ρ = +0.86, *p* = 0.007 and +0.69, *p* = 0.058). That pooling includes
sampled arms whose FVE was measured on saved checkpoints at 1/25 of their
training noise and whose liveness describes the deterministic mean code
rather than sampled selection (`ERRATA.md` §2); they place the collapsed end
of the curve; the current 13-arm pool gives ρ = +0.07 / +0.09 and is not
reported. Figure: `workshop/figs/frontier.pdf` (the 8-arm version).

*Record: addendum 6.*

## 8. What was predicted and turned out wrong

Recorded predictions that the data contradicted, kept because they are the
evidence that the design could say no:

* The closing E1 factor (the initial weight draw) was predicted null. It was
  *d* = −4.7 / −7.3.
* The A2 sweep was designed to find a σ threshold where FVE and Jaccard knee
  together. The pre-flight revised this before the run; the data showed a
  smooth curve with no knee.
* The learned σ was first read as fully collapsed to the clamp floor and
  eval-time noise as harmless (0.000012 FVE). Both were artifacts of a
  save-time bug; corrected, σ ≈ 0.27 and the noise costs reconstruction.
* A proxy estimate that correcting that bug would move reported FVE by
  ≈ 0.12 did not survive the official re-evaluation (≈ 0.003–0.006).
* The `gradproj` arm was speculated to have "bought reconstruction and paid
  in liveness", i.e. moved along the frontier. It was dominated outright by
  three other arms; `fullmatch` bought both.
* The usage-rank explanation for the SCR/TPP disagreement was stated as a
  falsifiable prediction and falsified.
* The A4 four-point reading was doubted as a possible small-n artifact; the
  densified grid reproduced it.
* TopK was read as "not padding its selection with μ = 0 features", and FVE
  as tracking Jaccard "almost exactly" (*r* = +0.9993). Both were measured on
  saved checkpoints at 1/25 of the training noise. In the training regime 24 %
  of selections have μ = 0 and the coupling is monotone but *r* = +0.95
  (`ERRATA.md` §2).

## 9. Pre-registered and not run

**E0, the pipeline negative control** — split the 13 `baseline` seeds into two
arbitrary groups, run the real metric pipeline, and ask the framework to
validate "group A is better organised" — was never run. It is the only check
that the real pipeline is exchangeable across seeds; every seed-permutation
*p*-value in this document assumes it is. Error control of the *procedure*
was established separately by simulation (`METHODS.md`).
