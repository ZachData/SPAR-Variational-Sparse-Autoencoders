# PROJECT.md — Falsification-based validation of sparse autoencoder claims

## Status

| | |
|---|---|
| Stage | Reframing after the workshop deadline was dropped |
| Framework | `falsification/` implemented, 23 tests green, Type-I control verified |
| Blocking | Seeded training runs (local GPU only) |
| Prior artifact | arXiv preprint; workshop draft on `claude/vae-workshop-paper-condensing-zumu6b` |

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

| seeds/group | distinct splits | min attainable p | e-value (κ=0.5) |
|---|---|---|---|
| 3 | 10 | 1.0e-01 | 1.58 |
| 4 | 35 | 2.9e-02 | 2.96 |
| 5 | 126 | 7.9e-03 | 5.61 |
| 6 | 462 | 2.2e-03 | 10.75 |
| 8 | 6,435 | 1.6e-04 | 40.11 |

Validation at α=0.1 needs aggregate E ≥ 10. So **5 seeds per group cannot validate
on a single test no matter how large the effect** — it maxes out at e=5.61. Either
6 seeds for one test, or 5 seeds with two independent tests
(`seeds_required()` computes this).

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

## Experiments

Every arm is gelu-1l, `blocks.0.hook_resid_post`, d=2048, k=256, auxk=1/32,
lr=8e-4, 10k steps — the existing sweep configuration — varying only the stated
intervention and the seed.

### E0 — Pipeline negative control. 10 runs.
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
Train TopK SAE + explicit `(β/2)·||f||²` activation penalty, 6 seeds. Under the
degeneracy (`CLAUDE.md`, landmine 1) this should be **indistinguishable** from the
fixed-variance vSAE at matched β.

Note this is an equivalence claim, so a failure to reject is not evidence of
equivalence. Report it as a TOST-style equivalence test with a pre-specified
margin, or as a power statement ("we could have detected a difference of size X").
Do not report a null result as confirmation.

### E2 — Is it variational at all? 6 runs.
Train with `var_flag=1`, 6 seeds — the experiment the preprint claims to have run
and did not. Diagnostic of interest is the learned σ: if it collapses toward 0, the
degeneracy is the *optimum* rather than an implementation accident, which is a
substantially stronger result.

### E3 — The masked-KL intervention. 6 runs.
`vsae_topk_masked_kl`, 6 seeds. Tests the preprint's stated death mechanism
directly. **Beware the confound in landmine 2**: that trainer omits the `F.relu(mu)`
that `vsae_topk.py` applies, so either patch one to match the other or report the
comparison as confounded.

### E4 — Size-matched SCR/TPP control. No training.
Restrict the trained baseline dictionary to a random subset matching the vSAE's
live count, recompute SCR/TPP, repeat ≥1000 times → `subsample_null_test`. Turns
the preprint's excluded tests into admissible ones by controlling the confounder.
Cheapest high-value experiment available; do it first.

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
  fraction would we consider a real departure from degeneracy?
