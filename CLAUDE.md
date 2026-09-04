# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this repository is

A fork of [`dictionary_learning`](https://github.com/saprmarks/dictionary_learning)
extended with variational sparse autoencoders (vSAEs), plus a vendored copy of
SAEBench and a falsification framework for validating claims about SAEs.

Two lines of work live here:

1. **`dictionary_learning/`, `training_scripts/`, `analysis_scripts/`** — vSAE
   architectures and the experiments behind the arXiv preprint.
2. **`falsification/`** — sequential falsification with e-values, used to decide
   what the experiments in (1) actually license. See `PROJECT.md`.

## Landmines — read before touching the vSAE code

These have already caused incorrect claims in a written paper. Verify against the
code before repeating any of them.

**`var_flag=0` means there is no sampling at all.** In
`dictionary_learning/trainers/vsae_topk.py::encode`, the reparameterisation is
gated on `var_flag == 1`; otherwise `z = mu`. Every checkpoint in
`comprehensive_histogram_analysis/` is named `_fixed_var`, which
`train_vsae_topk.py:275` emits exactly when `var_flag == 0`. So every evaluated
"variational" model is deterministic. With `sigma = 1` fixed the KL also reduces
to `0.5 * ||mu||^2`, i.e. a plain L2 penalty on the activations. **The evaluated
vSAE is a TopK SAE with an L2 activation penalty.** Do not describe these runs as
variational, stochastic, or as testing a posterior.

**`vsae_topk.py` applies `F.relu(mu)`; `vsae_topk_masked_kl.py` does not.** The
two trainers therefore differ by more than the KL mask. Any comparison between
them confounds the mask with the ReLU. The preprint's equations show no ReLU.
The masked trainer now carries a `relu_mu` flag (default `False`, so every
existing checkpoint is unchanged), and E3 runs as two arms — `e3_masked_kl`
(no ReLU, matching the preprint) and `e3_masked_kl_relu` (ReLU, matching the
released code) — so the ReLU's contribution is measured rather than assumed.
`relu_mu` changes no parameter shape and so **cannot be recovered from a state
dict**; `config.json` is the only record of which arm a checkpoint belongs to.

**The baseline had AuxK on and the vSAE had it off.** The headline Pythia
comparison is `auxk0.03125` (baseline) against `aux0` (vSAE). AuxK is the standard
dead-feature revival mechanism, so that comparison confounds the KL term with the
absence of the standard remedy. The gelu-1l beta sweep holds AuxK fixed at 1/32 and
is not affected.

**Config values in the preprint do not match the code.** Code is authoritative:
layer 3 (`blocks.3.hook_resid_post`), not layer 0; `total_steps = 10000`, not
20,000; Pythia dictionaries are `d=8192` (16x `d_model=512`) while gelu-1l is
`d=2048` (4x). The preprint's SAE loss equation includes an L1 term that
`top_k.py` does not have — TopK enforces sparsity architecturally.

**Metric names are not self-explanatory.** `features_used` in the
`comprehensive_summary_*.json` files counts dictionary entries selected at least
once over the streamed sample; it is sample-size dependent. The preprint quotes
two different dead-feature numbers from two different measurements (1,227/6,970
from sae_vis histograms; 1,474/7,379 from the 1M-sample analysis). Prefer the
1M-sample numbers and say which measurement you used.

**Two p-value floors are combinatorial, and both have bitten.** `min_p_floor` /
`min_attainable_p` once had their one-sided and two-sided branches swapped; that is
**fixed** (F1), and the `xfail` that pinned it is now a positive test. What remains
is not a bug but a property to plan around:

* The **seed** floor is `2/C(2n,n)` two-sided. 6 seeds/group cannot beat 3.07 sigma
  however large the effect; 13 seeds/group is the first n reaching 5 sigma.
* Above `_EXACT_ENUMERATION_LIMIT` (200k assignments) the test silently falls back
  to **Monte Carlo**, whose floor is `1/(n_perm+1)`. The 100k default caps evidence
  at 4.42 sigma *regardless of effect size*. Pass a larger `n_perm` (it is
  vectorised; 4M draws take ~1s) whenever n > 8 per group.

A p-value sitting exactly at one of these floors means **the design ran out, not the
evidence**. Check `result["exact"]` and `result["p_floor"]` before reporting.

**`frac_recovered = 0.0` in a summary file usually means an OOM, not a result.**
`loss_recovered()` OOMs at the default eval batch size on a 10GB card, the failure
is swallowed by an `except ... continue`, and the run reports success with the
cross-entropy metrics written as NaN. Every committed checkpoint shows that
signature. `falsification/run_arm.py` now evaluates at batch 2 x 48, which fixes it.

## Environment

- **Two environments, and it matters which you are in.** Run
  `python falsification/preflight.py` to find out.
  - *Remote/web sessions* have no GPU and no torch; `nvidia-smi` and
    `import torch` both fail. Available work: reading code, the `falsification/`
    package, analysis of committed `comprehensive_summary_*.json` files, figure
    generation, writing.
  - *Local sessions* on the RTX 3080 (10GB) can train. Use `./run_overnight.sh`
    for sweeps and `falsification/run_arm.py` for single runs; never hand-edit
    `create_full_config()`, because `get_experiment_name()` omits the seed and
    seeds will silently overwrite one another.
- `numpy`, `scipy`, `matplotlib`, `pytest` install cleanly with pip when needed.
- Training targets bfloat16 on 10GB; buffer settings in the training scripts are
  tuned for that and are easy to OOM if raised.

## Commands

```bash
# Falsification framework tests (no GPU needed; these must stay green)
python -m pytest falsification/tests/ -q

# Apply the framework to the data already committed here
python falsification/worked_example.py

# Regenerate the beta dose-response figure from committed JSONs
python workshop/make_fig_beta.py     # if the workshop/ docs are present

# Training (LOCAL GPU ONLY). Use run_arm.py -- do NOT hand-edit
# create_full_config(); get_experiment_name() omits the seed, so seeds
# written that way silently overwrite one another.
python falsification/run_arm.py --check              # validate all arms, no torch
python falsification/run_arm.py --arm baseline --seed 1
./run_overnight.sh --hours 10                        # the seeded arms
./run_e2_pilot.sh                                    # E2 stage 1 + selection rule

# Feature-usage measurement. run_analysis.sh handles the per-seed output dir
# (the analyzer names outputs after the checkpoint dir, which omits the seed)
# and re-analyses only checkpoints whose summary is older than their ae.pt.
./run_analysis.sh
./run_analysis.sh --force                            # ~1.2 min x every run on disk

# Cross-arm tables and the E1 comparison across its confound generations
python falsification/report_summaries.py --table
python falsification/compare_e1.py --ref current|aprilmode|klwarmup
```

**Read `PROJECT.md` first.** It is the living document: current state, what is
established, the prioritised next steps, the pre-registration and the open
decisions. It absorbed the former `HANDOFF.md` on 2026-09-03.

## Conventions

- **Statistics.** Any new statistical test goes in `falsification/permutation.py`
  with a test in `falsification/tests/`. Monte Carlo permutation p-values must use
  `(count + 1) / (n_perm + 1)`; the naive form is anti-conservative and can emit
  `p = 0`, which maps to an infinite e-value. Every test returns `p_floor` — report
  it, because an underpowered design cannot be rescued by its result.
- **Unit of analysis.** A claim about an *architecture* requires a permutation test
  over *training seeds*. Token-level tests answer questions about two specific
  checkpoints only; `paired_token_test` refuses architecture-level use unless the
  narrower scope is explicitly acknowledged.
- **Confounders.** Record them on `FalsificationTest`. A test with
  `confounders_uncontrolled` is excluded from the evidence product rather than
  down-weighted, because the implication assumption it violates is binary.
- Checkpoint directory names encode the config and are parsed by the analysis
  scripts — keep the existing naming scheme when adding runs.

## Working style for this repo

Claims here are checked against code and data, not against the preprint. When the
preprint and the code disagree, the code wins and the discrepancy gets written
down. Several conclusions in the published version did not survive that check, and
the value of the current work comes from having caught them.
