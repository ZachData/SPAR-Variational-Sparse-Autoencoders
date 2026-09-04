# HANDOFF

Read this first. `CLAUDE.md` has the repo's standing landmines; this file has
**current state, what is established, and what to do next**.

Last updated: 2026-09-03, end of the gradient-projection session. Branch
`claude/falsification-framework`. Working tree clean, nothing running, GPU idle.
`git log --oneline -10` shows the session's commits, newest first.

---

## State in one paragraph

The falsification battery is **complete at 13 seeds per arm** — 9 arms, 117
checkpoints, 0 failures — and reaches **5 sigma** on every comparison. E2 and E3
have landed. **E1 has not, and the reason is now precise rather than vague**: its
verdict depends on which implementation asymmetries you match, and matching the
newest one (the decoder-gradient projection) closed 79% of the reconstruction gap
while opening a liveness gap that had been closed. E1's remaining questions are
both decisions, not measurements. What remains is one cheap experiment (E4), one
optional factor, and two human decisions that block the write-up.

---

## What is established

### E1 — confirmed on the pre-registered construct, but only for one of the arms

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

**...but only on the arm that leaves the decoder-gradient projection unmatched.**
That was the next factor, it has now been run, and it changes the picture — see the
section below. The E1 verdict as it stands:

| arm | reconstruction (FVE gap) | liveness (pre-registered) |
|---|---|---|
| `e1_vsae_ref_unitinit` (projection off) | 0.0181, d = +16.5 | **indistinguishable** (p = 0.114 / 0.614) |
| `e1_vsae_ref_gradproj` (projection on) | **0.0038**, d = +3.8 | **distinguishable** (d = −2.4 / −3.7, both agreeing) |

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

---

## Next steps, in priority order

### 1. Two decisions, and they block the write-up — human only

**Compute cannot supply either.** Both are recorded in `PROJECT.md`, open decisions.

**(a) Which implementation is E1's claim about?** Three asymmetries have been found
after the pilot and run as factors; a fourth is identified (below). Matching them
one at a time is honest as far as it goes — each contribution is measured, and two
of them turned up effects that would otherwise have been misattributed — but "keep
matching until the arms agree" is a garden of forking paths with a pre-registered
metric attached. Either the claim is about the **objectives** (in which case the
projection is a nuisance factor and the unprojected arm is a legitimate vSAE), or
about the **released implementations** (in which case no asymmetry should have been
matched at all). Those readings license different arms and currently give different
verdicts. Record the choice before running another factor.

**(b) E1's equivalence margin.** Unchanged in substance and now sharper: at 13 seeds
the across-seed SDs are ~1e-3, so almost any non-zero difference clears 5 sigma, and
E1's verdict has already flipped on a live-fraction difference of 0.036. The margin
must be a statement about what difference would **matter**, and it can no longer be
chosen innocently.

### 2. E4 — no training required

The size-matched SCR/TPP control. `RUNBOOK.md` section 1 has the design; the missing
piece is the `scorer(keep_indices)` function that masks the dictionary and calls
SAEBench. Everything else (usage counts from the baseline summaries) is on disk.

### 3. Optional — the dead-feature tracking factor, ~30 min, blocked on decision (a)

The two trainers define "fired" differently, and it is the only part of the loss
that targets liveness — the construct that now separates the arms:

* `top_k.py:543` — `did_fire[top_indices_BK.flatten()] = True`: **selected** by
  TopK, whatever its value.
* `vsae_topk.py:888` — `active_features = (sparse_features_BF.sum(0) > 0)`:
  selected **and strictly positive**.

Both encoders ReLU before selecting (`top_k.py:183`, `vsae_topk.py:290`), so values
are non-negative and the rules differ on exactly one case: a feature TopK selects at
value 0, which resets the counter in one trainer and not the other. The vSAE
therefore declares features dead sooner and gives them more AuxK pressure. Whether
that case is common enough to explain d = −3.7 is unmeasured.

**Do not run it before decision (a).** Under the "objectives" reading it is out of
scope entirely; under the "implementations" reading so were the three factors
already run. Running it first and deciding afterwards is exactly the failure mode
this framework exists to prevent.

### 4. Optional — extend the E2 beta grid downward (1e-5, 1e-6)

The control makes this much less interesting: if 94.7% of the damage is the sampling,
no smaller beta recovers a healthy model. It would close the question formally. It
would be a **new pre-registration**, not a continuation of this one.

---

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

**Flags that leave no trace in the weights need `config.json`.** `relu_mu` and
`decoder_init_scale` change behaviour but not parameter shapes, so they cannot be
recovered from a state dict. Both are written into the trainer's `config` property and
read back in `utils.load_dictionary`. Any new factor of this kind needs the same.

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
| `falsification/RESULTS_2026-09-03.md` | all measured results; addendum 1 is the 13-seed/5σ rerun, addendum 2 the gradient projection |
| `falsification/FINDINGS_2026-09-02.md` | the five instrumentation bugs the pilot exposed |
| `falsification/REMEDIATION.md` | fix tracking + the four author decisions and their rationale |
| `RUNBOOK.md` | commands, arm table, E4 design |
| `PROJECT.md` | pre-registration, power accounting, open decisions |

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
