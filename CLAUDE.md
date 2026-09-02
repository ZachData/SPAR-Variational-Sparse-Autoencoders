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

# Training (LOCAL GPU ONLY). Configs are hardcoded in create_full_config();
# there are no CLI flags for model/beta/seed -- edit the function.
python training_scripts/train_vsae_topk.py --config full
python training_scripts/train_topk.py --config full

# Feature-usage measurement after a run
python analysis_scripts/online_histogram_analyzer.py \
  --model-path <checkpoint_dir> --n-samples 1000000 --no-individual
```

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
