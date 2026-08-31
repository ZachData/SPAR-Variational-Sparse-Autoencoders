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

## Experiments

Every arm is gelu-1l, `blocks.0.hook_resid_post`, d=2048, k=256, auxk=1/32,
lr=8e-4, 10k steps — the existing sweep configuration — varying only the stated
intervention and the seed.

### E0 — Type-I calibration (the negative control). 10 runs.
Train 10 TopK SAEs with **identical config, different seeds**. Split into two
arbitrary groups of 5 and ask the framework to validate "group A learns a better
organised feature space than group B". Ground truth: null, by construction. Repeat
over many random splits and measure the empirical false-validation rate against α.

This is POPPER's permuted-negatives design adapted to SAEs, and it is the
experiment that makes the whole thing credible. Without it we are asserting error
control rather than demonstrating it. These 10 runs also serve as the baseline arm
for E1–E3, so the marginal cost of the rest is low.

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
- **α and κ.** Currently α=0.1, κ=0.5 (POPPER's defaults). κ=0.3 extracts more
  evidence from small p-values and would relax the seed budget; it must be fixed
  before looking at results.
- **Equivalence margin for E1.** Needs to be pre-specified. What difference in live
  fraction would we consider a real departure from degeneracy?
