# Methods

The statistical design behind `RESULTS.md`: what the framework is, why the
unit of analysis is the training seed, what that costs, what the simulations
settled before any GPU time was spent, and the battery as it was
pre-registered. The pre-registration is reproduced in its original form —
that is what makes it one — with outcomes noted beside it rather than edited
in. The dated working version is `notebook/PROJECT.md`.

## 1. The problem

Interpretability makes claims of the form "architecture A produces better
features than architecture B", validated informally: run a benchmark suite,
read the numbers, write a conclusion. That process has two failure modes,
and the preprint this repository extends exhibits both.

**No principled aggregation.** The preprint ran core metrics, SCR, TPP, t-SNE
and feature visualisation and combined them by narrative. Its Global section
concluded the dispersion hypothesis was confirmed; its Conclusion concluded
it was rejected. Nothing in the method could have adjudicated that.

**Sub-hypotheses the main null does not imply.** The vSAE outscored the
baseline on SCR and TPP, read as evidence of a more disentangled feature
space. But both metrics reward *selective* ablation, selectivity is easier
with fewer live features, and the vSAE had 18 % of its dictionary alive
against the baseline's 90 %. "No better organisation" being true does not
imply "no higher SCR", so the result carried no evidence about organisation.

POPPER (Huang et al., 2025) supplies the missing machinery: an implication
check (their Assumption 1) and sequential aggregation of e-values with Type-I
control under optional stopping. This repository instantiates it where the
hard part is different. POPPER's difficulty is proposing good falsification
tests over a static database; here it is producing a valid *p*-value at all,
because the randomness in an SAE comparison is the training seed and seeds
are expensive.

## 2. The framework (`falsification/`)

**E-values.** A *p*-value is calibrated to an e-value by e = κ·p^(κ−1), which
has E[e] = 1 under the null for any κ ∈ (0, 1). Evidence from a sequence of
tests is the product E = ∏ eᵢ, a non-negative supermartingale under the
null, so the rule "reject when E ≥ 1/α" controls Type-I error at α *however
the analyst decides how many tests to run and when to stop*. Fisher's
combined test cannot do this; `test_fisher_combination_would_fail_where_e_values_hold`
demonstrates the inflation. Pre-registered: **α = 0.1, κ = 0.3** (§4).

**The implication check.** A falsification test carries evidence about the
main hypothesis only if the main null implies the test's sub-null. Nearly
every SAE metric co-varies with a nuisance — live-feature count, L0,
reconstruction quality — so in this setting the implication check is
confounder control for benchmark metrics. Confounders are declared on
`FalsificationTest`; a test with any listed as uncontrolled is *excluded*
from the product, not down-weighted, because the implication assumption is
binary. `worked_example.py` applies this to the preprint's own numbers and
excludes both headline wins before any evidence accrues.

**Unit of analysis.** A claim about an *architecture* is a claim about the
distribution of models that training produces, so its unit is the training
run and the natural instrument is an exact permutation test over seeds.
Token-level tests describe two specific checkpoints; with a million
evaluation tokens they report *p* ≈ 10⁻³⁰⁰ for differences that are pure
seed noise. `paired_token_test` refuses architecture-level use unless the
narrower scope is acknowledged in the call.

| function | unit | supports a claim about |
|---|---|---|
| `seed_permutation_test` | training run | an architecture |
| `monotone_trend_test` | sweep condition | a dose-response (one run per condition) |
| `subsample_null_test` | random sub-dictionary | a score, controlling dictionary size |
| `paired_token_test` | evaluation token | two specific checkpoints |

**Conventions that are load-bearing.** Monte Carlo permutation *p*-values
use (count + 1)/(n_perm + 1); the naive form is anti-conservative and can
return *p* = 0, which maps to an infinite e-value. Every test returns
`p_floor`, the smallest *p* the design could have produced, and `exact`,
whether the null distribution was enumerated or sampled. A *p* sitting at
its floor means the design ran out, not the evidence.

## 3. Power: the combinatorial floor

An exact two-group permutation test over *n* seeds per group has C(2n, n)
assignments. Its smallest attainable two-sided *p* is 2/C(2n, n), one-sided
1/C(2n, n), regardless of the effect size. (The one-sided and two-sided
branches of this function were once swapped, making every floor 2× too
pessimistic; `ERRATA.md`.)

| seeds / group | C(2n, n) | min two-sided *p* | sigma ceiling | e at κ = 0.3 (one-sided floor) |
|---|---|---|---|---|
| 3 | 20 | 0.100 | 1.64 | 2.44 |
| 5 | 252 | 0.0079 | 2.65 | 14.39 |
| 6 | 924 | 0.00216 | 3.07 | 35.73 |
| 8 | 12,870 | 1.6 × 10⁻⁴ | 3.78 | 225.85 |
| 13 | 10,400,600 | 1.9 × 10⁻⁷ | 5.21 | — |

Validation at α = 0.1 needs E ≥ 10. Five seeds per group can validate on a
single maximally significant test at κ = 0.3 (`seeds_required(0.1, 0.3, 1)`
= 5); the confirmatory battery ran at **13 seeds per arm** after a six-seed
generation hit the 3.07σ ceiling on every comparison. Above 200,000
assignments the test falls back to Monte Carlo, whose floor is 1/(n_perm+1);
the 100,000-draw default caps evidence at 4.42σ, so every 13-seed
comparison in this repository uses 4 M draws (vectorised, ~1 s), floor
5.16σ. When the Monte Carlo floor binds rather than the design floor, the
reported floor is the binding one.

The same arithmetic applies to a single-run-per-condition sweep: the
preprint's β dose-response is monotone over six orders of magnitude, but the
exact trend test over four conditions floors at 1/4! = 0.042, e = 2.45
against a threshold of 10. Even a perfect dose-response does not validate
without replication. At ~1 min per gelu-1l run on the RTX 3080, seeds were
never the binding cost; the 10⁶-sample feature-usage analysis at ~6.5 min per
checkpoint was.

## 4. What the simulation settled before any run

`falsification/simulate.py`, no GPU; all three results are locked into the
test suite.

**Reusing runs across tests is valid but spends the margin.** Several tests
reading metrics off the same runs are dependent — exactly what POPPER's
Assumption 2 guards against. Empirical Type-I at α = 0.1, 5 seeds/group: two
independent tests 0.002; three fully correlated 0.050; five at correlation
0.95–1.0, 0.092–0.095. Validity holds; redundant tests take the rate to the
edge of α. Rule: few, genuinely different tests.

**Choosing test order by peeking breaks Type-I control.** With ten candidate
metrics, running the most significant first gives Type-I = 0.123 > α against
0.020 with the order fixed in advance. Optional stopping covers *when you
stop*, never *which test you reach for next*. Rule: the battery and its
order were written down before any seeded run was inspected.

**κ = 0.3 maximises power while staying under α.** Worst-case Type-I (five
fully redundant tests) against power at 5 seeds/group:

| κ | Type-I (worst) | power, d = 1.0 | d = 1.5 | d = 2.0 |
|---|---|---|---|---|
| 0.2 | 0.066 | 0.27 | 0.63 | 0.90 |
| **0.3** | **0.086** | **0.29** | **0.66** | **0.92** |
| 0.4 | 0.095 | 0.27 | 0.63 | 0.91 |
| 0.5 | 0.095 | 0.19 | 0.53 | 0.85 |
| 0.7 | 0.061 | 0.01 | 0.08 | 0.27 |

**Power is the binding constraint.** Probability of validating a true effect
with two pre-specified tests at α = 0.1 (κ = 0.5; κ = 0.3 is slightly better):

| seeds / group | d = 0.5 | d = 1.0 | d = 1.5 | d = 2.0 |
|---|---|---|---|---|
| 4 | 0.01 | 0.07 | 0.24 | 0.50 |
| 5 | 0.03 | 0.20 | 0.54 | 0.86 |
| 6 | 0.06 | 0.32 | 0.73 | 0.96 |
| 8 | 0.10 | 0.48 | 0.91 | 0.99 |
| 10 | 0.14 | 0.61 | 0.96 | 1.00 |

A one-SD effect is not reliably detectable even at ten seeds per group.
Effects expected to be huge — the degeneracy, β-driven feature death — were
fine at five or six seeds; any subtle claim about "feature organisation" was
out of reach at this scale, and the design says so rather than running an
underpowered arm and reporting a null.

## 5. Measurement conventions

**Liveness: two pre-registered thresholds, both always reported.** The
preprint's `features_used` ("selected at least once") saturates at d = 2048:
TopK selects exactly *k* of *d* features per sample, so the mean selection
frequency is *k/d* by construction and essentially every feature fires
somewhere in 10⁶ samples. It was replaced by the fraction of features
selected in fewer than *r*·(*k/d*) of samples at *r* = 0.1 and *r* = 0.5,
computed from exact per-feature selection counts. Both thresholds were fixed
after the pilot exposed the saturation but from the arithmetic of the design,
not from observed effect sizes; the confirmatory battery ran on fresh seeds
regardless. A liveness result is robust only if both agree in direction; a
disagreement is itself the finding, and twice caught a distribution *shape*
change that one threshold alone would have reported as a clean effect.

**Reconstruction.** `frac_variance_explained` (FVE) and `frac_recovered`
(fraction of the language-model loss recovered relative to zero-ablation)
from each trainer's own `evaluate()` at the end of training, at a batch size
that fits the 10 GB card (`ERRATA.md` for why the default did not).

**Effect sizes.** Cohen's *d* with the pooled SD taken as the mean of the two
groups' across-seed SDs. At 13 seeds those SDs are ~10⁻³, so *d* is large for
differences that may or may not matter; it is an effect size, not a verdict.

**Matching, and when to stop.** E1's arms were matched factor by factor, and
"keep matching until the arms agree" is a garden of forking paths with a
pre-registered metric attached. The rule adopted was: *the factor set is the
code diff*. The diff between the two trainers was enumerated by reading and
frozen at fifteen items before the closing arm ran; nothing could be added
without a code difference to justify it, and two late candidates that looked
alarming (the dead-feature rule, the ±10 penalty clamp) were closed by
measurement rather than by training runs.

**Exploratory versus confirmatory.** Arms added after results were seen
(`e2_sampling_only`, `e2_sigma_low_init`, the A2–A4 sweeps, the frontier,
Claims #4–#5) are reported as such and were never entered into the evidence
product. The 5-seed A4 follow-up is reported at its lower power.

## 6. Training configuration

Every gelu-1l arm: `blocks.0.hook_resid_post`, d_model = 512, dictionary
d = 2048 (4×), k = 256, AuxK α = 1/32, lr 8 × 10⁻⁴, 10,000 steps, activations
normalised to unit mean squared norm during training and the biases rescaled
by `norm_factor` at save time. Arms are trained by `falsification/run_arm.py`
from a frozen `BASE` config with per-arm overrides, one directory per seed.
The Pythia-70M checkpoints (E4) are the preprint's own: layer 3
(`blocks.3.hook_resid_post`), d = 8192 (16×), k = 256, recovered rather than
retrained.

## 7. The pre-registration

The battery as written before any seeded run was inspected. Seed counts say
6; the confirmatory battery ran at 13 after the six-seed generation hit the
combinatorial floor. Outcomes are in `RESULTS.md`; where a design was changed
after seeing results, the change is noted and the arm marked exploratory.

**E0 — pipeline negative control. 10 runs.** Train ten TopK SAEs with
identical config and different seeds, run the real metric pipeline, split
into two arbitrary groups of five and ask the framework to validate "group A
is better organised than group B". Null by construction. This does not
measure Type-I error (re-splits of a fixed set are uniform by construction);
it tests the plumbing — whether the real metrics, computed by the real
pipeline, are exchangeable across seeds. A metric that depends on
checkpoint filename order, a shared data-loading order, or a cached artifact
leaking between runs would show up here and nowhere else.
*Outcome: never run. The 13 `baseline` seeds exist and are analysed; the
split test was not done.*

**E1 — the degeneracy claim. 6 runs.** Train TopK + explicit (β/2)‖f‖²
penalty. Under the degeneracy this should be indistinguishable from the
fixed-variance vSAE at matched β. This is an equivalence claim: a failure to
reject is not evidence of equivalence; report as TOST with a pre-specified
margin, or as a power statement.
*Outcome: confirmed as a decomposition, not a bare equivalence — five matched
factors, null on every metric at the last rung. The equivalence margin was
never specified (§5 of `RESULTS.md`, §1).*

**E2 — is it variational at all? 6 runs.** Train with `var_flag = 1`, the
experiment the preprint claims to have run and did not. Diagnostic: the
learned σ; if it collapses toward 0 the degeneracy is the optimum rather than
an accident. Selection rule for β, written into `run_arm.py` before any pilot
result existed: one seed (101) at each β ∈ {10⁻⁴ … 1}; select the largest β
whose FVE is within 0.02 of the baseline mean, else the smallest β tried;
confirm at seeds disjoint from the pilot so no checkpoint contributes to both
selection and inference.
*Outcome: no β qualified, the fallback selected 10⁻⁴; sampling, not the KL,
was the damage; σ did not collapse (after a bug was corrected).*

**E3 — the masked-KL intervention. 6 runs.** `vsae_topk_masked_kl`, testing
the preprint's stated death mechanism. Beware the ReLU confound: either patch
one trainer to match the other or report the comparison as confounded.
*Outcome: the ReLU was run as a two-level factor instead; it is
d = +19.3 / +15.3 on its own.*

**E4 — size-matched SCR/TPP control. No training.** Measure SCR and TPP as
a function of dictionary size for the baseline, masking it to its top-N
most-used features (its best foot forward — a *random* subset was the
originally planned reference and would have rigged the test in the vSAE's
favour, which `test_random_subset_null_would_falsely_confirm_the_hypothesis`
demonstrates), and place the vSAE's own score on the curve at N = 1,474.
Single seed each side, so descriptive.
*Outcome: SCR not explained by size, TPP explained by size; the disagreement
survived every check and is reported unresolved.*

**Order and additions.** The battery and its order were fixed here before
any seeded run was inspected. Adding an arm after seeing results is
permitted only if reported as exploratory and excluded from the evidence
product.

## 8. What the framework can and cannot warrant

Type-I control is over the randomness the *p*-values model. Seed-permutation
tests control error over training seeds; they say nothing about bias shared
by every run — a buggy trainer, a mislabelled hook point, a save-time
transformation applied to the wrong parameter. Several of those were found
in this codebase (`ERRATA.md`), none by the statistics, all by reading the
code against the numbers. Non-rejection is not evidence of a null. And
e-values require conditionally valid *p*-values: the next test is chosen
without looking at the data it will be computed on.
