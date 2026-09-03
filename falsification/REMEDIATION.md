# REMEDIATION — tracking the fixes from the 2026-09-02 sweep

Companion to `falsification/FINDINGS_2026-09-02.md`, which records *what is wrong*
and the evidence. This file tracks *what is being done about it*.

**Framing, so this does not drift.** The 30-run sweep produced e-values of 22–36
per test against a threshold of 10, with effect sizes d = 24–855. There is no
significance problem and no seed shortage. Every remaining issue is about
**validity** — whether a significant number means what it appears to mean. Nothing
in this file should be changed in order to make a result come out; the only
legitimate reasons to touch the design here are (a) a comparison is confounded,
(b) a metric cannot resolve the thing it stands for, or (c) code is broken.

**Pre-registration status.** All 30 existing runs are hereby treated as a **pilot**.
They were instrumentation-finding and they found five instrumentation bugs. The
corrected battery is pre-registered on what the pilot revealed about *measurement*
— metric resolution, penalty schedules, loader defects — and never on observed
effect sizes. Confirmatory arms run with fresh seeds.

---

## Status board

| id | issue | owner | cost | status |
|---|---|---|---|---|
| F1 | `min_p_floor` / `min_attainable_p` one-sided/two-sided swapped | claude | mins | **DONE** |
| F1b | Regenerate power tables in PROJECT.md / RUNBOOK.md | claude | mins | **DONE** |
| F7a | Expose `kl_warmup_steps` on the vSAE configs | claude | mins | **DONE** |
| F8a | Use exact per-feature counts, not the coarse histogram | claude | mins | **DONE** (no code change needed) |
| F7b | Choose how E1's two arms are schedule-matched | author | done | **DECIDED — option (a), applied** |
| F9a | `from_pretrained` for the masked-KL class | claude (review) | done | **DONE — review wanted** |
| F10a | Masked-KL `evaluate_model()` died on missing `dict_class` | claude | mins | **DONE** |
| F10b | Masked-KL script never imported `math` | claude | mins | **DONE** |
| F9b | Resolve the `F.relu(mu)` mismatch — which trainer moves? | **author** | — | **DECISION NEEDED** |
| F6 | E2 beta-selection rule for the `var_flag=1` regime | **author** | 2-stage | **DECISION NEEDED** |
| F8b | Pre-register the sparsity-relative liveness threshold | **author** | — | **DECISION NEEDED** |
| F9c | Verify masked-KL loader through the real analyzer | claude | done | **DONE** (6/6, 0 failures) |
| F7c | Match the bias parameterisation in E1 (`use_april_update_mode`) | **author** | 45 min | **DECISION NEEDED** |
| — | Confirmatory battery, fresh seeds | after above | ~3.75 h | BLOCKED on the 4 decisions |

---

## F1 — floor functions had one-sided and two-sided swapped

Exact enumeration over `N = C(n_a+n_b, n_a)` assignments attains **1/N one-sided**
and **2/N two-sided**; both functions reported the mirror image. A one-sided seed
test could return a p-value below its own reported floor.

**Why it mattered beyond tidiness.** It flips a headline planning claim. PROJECT.md
states "5 seeds per group cannot validate on a single test no matter how large the
effect". At the pre-registered κ=0.3, one-sided:

* reported (buggy) floor 1/126 → e = 8.85 → cannot validate
* corrected floor 1/252 → e = **14.4** → **validates**

The claim was true at κ=0.5 (the superseded value) and false at κ=0.3 with the
correct floor. Fixed in both modules; the strict `xfail` pinning it is now a
positive test.

## F1b — corrected planning tables

Attainable evidence, κ=0.3, threshold E ≥ 10:

| seeds/grp | p (1-sided) | e | p (2-sided) | e |
|---|---|---|---|---|
| 5 | 3.97e-03 | 14.4 | 7.94e-03 | 8.9 |
| 6 | 1.08e-03 | 35.7 | 2.16e-03 | 22.0 |
| 8 | 7.77e-05 | 225.9 | 1.55e-04 | 139.0 |
| 10 | 5.41e-06 | 1457.9 | 1.08e-05 | 897.5 |

**Planning consequence: seeds are no longer the binding constraint.** At the
measured ~1 min/run, 6 seeds/arm already delivers 2–3.6× the required evidence.
The real cost is the 1M-sample analysis at ~6.5 min/checkpoint — 6.5× the cost of
the run it measures. Optimise that, not the seed count.

## F7a — `kl_warmup_steps` exposed

The vSAE multiplies its KL by `kl_scale`, a linear ramp over
`int(0.1 * steps)` = 1000 steps; the activation penalty is constant from step 0.
So E1's two arms applied their nominally-matched penalty on different schedules.
The field was not reachable from `run_arm.py`. It now is, defaulting to `None`
(existing behaviour preserved).

## F8a — exact per-feature counts were already being saved

**Planned as a code change; turned out to need none.** I had recorded that the
analyzer saved only a 50-bin frequency histogram whose 0.02-wide first bin could
not separate "never selected" from "selected rarely". That was wrong: the `.npz`
already contains `feature_selection_counts`, one exact integer per feature. The
binning limit never existed, and **no re-analysis is required** — the exact data
for all 24 checkpoints has been on disk since the sweep.

`report_summaries.py` now reads those counts directly. What that buys, immediately:

| arm | min selection count (of 1,000,192) | fraction < 0.1x k/d |
|---|---|---|
| baseline | 101-210 | 0.7529-0.7554 |
| e1_penalty | 4,765-10,424 | 0.0005-0.0010 |
| e1_vsae_ref | 0 | 0.4165-0.4287 |
| e2_learned_var | 0-4 | 0.9961-0.9990 |

All four arms separate cleanly with small across-seed spread. Note `e1_penalty`:
its least-used feature still fires 4,765 times, so the L2 activation penalty does
not kill features at all — it pushes the whole dictionary toward uniform usage.
The opposite of the feature death the degeneracy story predicts.

---

## F9a / F10 — masked-KL made loadable (please review)

Three stacked defects, each hiding the next: no `dict_class` in the saved config,
no `from_pretrained` on the class, and no `import math` in the training script.
All three fixed; the 6 E3 checkpoints were retrained so their config is written by
the corrected code rather than hand-repaired.

**The one judgement call I made, flagged for review.** `vsae_topk.py`'s
`from_pretrained` defaults `normalize_decoder=True`, which rescales decoder weights
on load. Mine defaults to **False**, so a checkpoint loads exactly as saved — for
analysis we want the trained model, not a rescaled one. (The sibling's rescaling
already emits `Warning: Could not normalize decoder weights: Normalization changed
model output` when loading `e1_vsae_ref`, which is its own thing to look at.) If
you want parity with the sibling instead, flip the default.

**Verified.** An ad-hoc FVE harness gave 0.5812 for `e1_vsae_ref` against the
0.8888 its own evaluation reported — the *check* was wrong (wrong data
distribution or normalisation), not the loader, which is why it was not trusted.
Verification instead went through the real analyzer: **all 6 E3 checkpoints
analysed, 0 failures**, and E3's own evaluation now yields FVE 0.8939 ± 0.0008,
`frac_recovered` 0.9570 ± 0.0007 — a healthy model, in line with the other arms.
30/30 checkpoints now have summaries.

E3 remains **confounded** regardless, pending F9b.

## Decisions that are the author's, with the one-line change for each

### F7b — how should E1's arms be schedule-matched? **DECIDED: option (a)**

Applied 2026-09-02: `"kl_warmup_steps": 0` added to the `e1_vsae_ref` overrides in
`run_arm.py`, so both arms apply their penalty at constant full strength from step
0. Verified in the run log — `kl_scale` is now `1.0000` at step 0, where it was
`0.0000` before. The 6 arm runs were retrained and re-analysed.

The pre-correction runs are preserved under `archive/e1_vsae_ref_klwarmup1000/`,
so the size of the confound is itself measurable rather than merely asserted:
`python falsification/compare_e1.py --archive` scores them, and
`python falsification/compare_e1.py` scores the matched pair.

The two options, for the record:

* **(a) `kl_warmup_steps=0` on `e1_vsae_ref`** — both penalties constant from step 0.
  Tests the *loss functions* for equivalence, treating warmup as a training-schedule
  artifact irrelevant to the degeneracy claim. **Recommended**, and one line:
  add `"kl_warmup_steps": 0` to the `e1_vsae_ref` overrides in `run_arm.py`.
* **(b) add a matching ramp to the penalty trainer** — tests the *training procedures*
  for equivalence. Requires a trainer change, not just config.

Until one is chosen, E1 carries `confounders_uncontrolled=("kl_warmup_steps",)` and
is excluded from the evidence product.

### F7c — match the bias parameterisation? (new, from F7b's result)
With the schedule matched, E1's arms still differ at d = 39-129. The loss functions
were then shown to be **numerically identical** (511.895264 both, six decimals), so
the difference is architectural, and it is the bias form:

* `top_k_with_feature_penalty`: `relu(encoder(x - b_dec))`, `decoder(f) + b_dec`
* `vsae_topk` in april mode (what every vSAE arm used): no pre-bias at all,
  untied `decoder.bias` instead

**Recommended:** set `"use_april_update_mode": False` on `e1_vsae_ref`, which makes
it subtract `self.bias` pre-encoder and matches the TopK form. One line in
`run_arm.py`, ~45 min to retrain and re-analyse. Caveat: the non-april path is the
less exercised of the two, so it wants a canary run before committing all six
seeds. Until this is settled E1 carries
`confounders_uncontrolled=("use_april_update_mode",)`.

### F9b — which trainer moves for `F.relu(mu)`?
`vsae_topk.py` applies it, `vsae_topk_masked_kl.py` does not, and the preprint's
equations show no ReLU. Patching the masked trainer to add it matches the
released code; removing it from `vsae_topk.py` matches the paper. These give
different experiments. Not guessable from the code — it depends which claim E3 is
meant to test.

### F6 — E2's beta-selection rule
`kl_coeff` is not a shared scale across `var_flag`: at 0 the KL is `0.5‖mu‖²`; at 1
the variance term contributes ~220 of a ~225 total loss. A single "matched beta"
point compares two different interventions, and at beta=1.0 the model
posterior-collapses (mu → 1e-3, FVE = 0.0001) in all 6 seeds.

Proposed two-stage design, needing the author's sign-off **before** the pilot is
looked at:
1. 1 seed per beta over {1e-4, 1e-3, 1e-2, 1e-1, 1} at `var_flag=1`.
2. Selection rule fixed in advance, e.g. *"the largest beta whose FVE is within
   0.02 of baseline"*.
3. 6 confirmatory seeds at the selected beta, seeds **disjoint** from the pilot.

### F8b — the liveness threshold
`features_used` saturates because mean selection frequency is exactly `k/d = 0.125`
by construction. That is knowable a priori, so re-specifying the metric on sparsity
arithmetic is legitimate — but the threshold must be fixed now, before it is
applied. Proposed: **"selected in fewer than 0.1·(k/d) of samples"**. Defined
relative to the design's own sparsity, so it transfers between d=2048 and d=8192;
`features_used` does not, which is why the preprint's d=8192 numbers never exposed
this.

---

## Verification

```bash
python -m pytest falsification/tests/ -q      # 93 passed as of 2026-09-02
python falsification/preflight.py
python falsification/report_summaries.py --table
```

## Changed so far

| file | change |
|---|---|
| `falsification/permutation.py` | `min_p_floor` branches un-swapped |
| `falsification/evalues.py` | `min_attainable_p` branches un-swapped |
| `falsification/tests/test_permutation.py` | `xfail` promoted to real assertions; floor-attainment tests added |
| `falsification/tests/test_evalues.py` | table values corrected; kappa=0.3 reversal pinned |
| `falsification/report_summaries.py` | `liveness()` uses exact counts + sparsity-relative threshold |
| `PROJECT.md` | floor table + 5-seed claim corrected; run-cost note; test count |
| `training_scripts/train_vsae_topk.py` | `kl_warmup_steps` exposed and plumbed |
| `training_scripts/train_vsae_topk_masked_kl.py` | same |

Nothing is committed; all of it is working-tree on `claude/falsification-framework`.
