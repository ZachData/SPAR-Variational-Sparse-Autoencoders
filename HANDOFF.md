# HANDOFF

Read this first. `CLAUDE.md` has the repo's standing landmines; this file has
**current state, what is established, and what to do next**.

Last updated: 2026-09-03, end of session. Branch `claude/falsification-framework`.
Working tree clean, nothing running, GPU idle. `git log --oneline -8` shows the
session's commits, newest first; this file was added by the last of them.

---

## State in one paragraph

The falsification battery is **complete at 13 seeds per arm** — 8 arms, 104
checkpoints, 0 failures — and reaches **5 sigma** on every comparison. Three of the
four experiments have landed. E1's degeneracy claim is **confirmed** on the metric it
was pre-registered against. E2 has produced a stronger result than it was designed
for. E3 turned a confound into a measured factor. What remains is one cheap
experiment, one piece of code to write, and one decision only a human can make.

---

## What is established

### E1 — the degeneracy claim is CONFIRMED on the pre-registered construct

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

**Still open on E1:** reconstruction. `frac_variance_explained` differs by 0.0181
(d = +16.5, 5.16σ) with the init matched — and, awkwardly, that gap is *larger* than
with the init unmatched (0.0059). Liveness and reconstruction point in opposite
directions here; only liveness is pre-registered. See "Next steps" #1 for the
leading explanation.

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

### 1. The gradient-projection asymmetry — cheapest, highest value, ~30 min

`vsae_topk.py` **imports `remove_gradient_parallel_to_decoder_directions` and never
calls it** (grep count: 0). `top_k.py` calls it every step, before renormalising the
decoder to unit norm. Renormalising without projecting out the parallel gradient
makes the optimiser fight the constraint — the radial component is applied and then
immediately undone, a per-feature change in effective learning rate.

This is the leading candidate for E1's remaining reconstruction gap, and it looks
like an oversight rather than a design choice.

**Do it as a factor, not a silent fix** — the same pattern as `relu_mu` and
`decoder_init_scale`, both of which paid off. Add a config flag (default preserving
current behaviour), add an arm, run 13 seeds, analyse, compare. Precedent to copy:
`decoder_init_scale` in `dictionary_learning/trainers/vsae_topk.py` and the
`e1_vsae_ref_unitinit` entry in `falsification/run_arm.py`.

### 2. E4 — no training required

The size-matched SCR/TPP control. `RUNBOOK.md` section 1 has the design; the missing
piece is the `scorer(keep_indices)` function that masks the dictionary and calls
SAEBench. Everything else (usage counts from the baseline summaries) is on disk.

### 3. E1's equivalence margin — human decision, blocks the write-up

**Compute cannot supply this.** E1 is an *equivalence* claim, and more power makes
equivalence *harder* to declare: at 13 seeds E1 clears 5.16σ on a reconstruction
difference of 0.0181, and 5.03σ on 0.0059. Across-seed SDs are ~0.001, so almost any
non-zero difference will be "significant". The margin must be a statement about what
difference would **matter**, chosen independently of the measured numbers — and it
can no longer be chosen innocently, since those numbers are now known. `PROJECT.md`
still lists it as open.

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

**Timings, measured.** Training ≈ 1 min/run. Analysis ≈ 1.1 min/checkpoint at 1M
samples (was 6 min before `update_histograms` was vectorised — 59 of 60 output arrays
verified bit-identical after that change). A full 13-seed arm is ≈ 13 min train +
≈ 15 min analyse.

---

## Where the detail lives

| file | what it holds |
|---|---|
| `falsification/RESULTS_2026-09-03.md` | all measured results, incl. the 13-seed/5σ addendum |
| `falsification/FINDINGS_2026-09-02.md` | the five instrumentation bugs the pilot exposed |
| `falsification/REMEDIATION.md` | fix tracking + the four author decisions and their rationale |
| `RUNBOOK.md` | commands, arm table, E4 design |
| `PROJECT.md` | pre-registration, power accounting, open decisions |

## Verify the environment is sane

```bash
python falsification/preflight.py                 # says which environment you are in
python -m pytest falsification/tests/ -q          # 93 passing; must stay green
python falsification/run_arm.py --check           # validates every arm without torch
python falsification/report_summaries.py --table  # cross-arm liveness
```

Note `preflight.py` still passes on a machine that cannot train — it verifies wiring
by reading training-script source as text rather than importing it, so missing
packages slip through (FINDINGS item 5). Importing each module through
`run_arm.load_training_module()` would close that gap.
