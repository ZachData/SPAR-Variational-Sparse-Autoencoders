# RUNBOOK — commands for when GPU access returns

Copy-pasteable. Ordered so failures surface early and cheaply.
Config values come from `PROJECT.md`; the landmines they avoid are in `CLAUDE.md`.

**Pre-registered before any run is inspected:** α = 0.1, κ = 0.3, two falsification
tests per arm, 6 seeds per group. Do not change these after seeing results
(`falsification/simulate.py` shows why: peeking at test order pushed Type-I to
0.123 against α = 0.1).

---

## 0. Setup and pre-flight (2 minutes)

```bash
cd ~/path/to/SPAR-Variational-Sparse-Autoencoders
git fetch origin
git checkout claude/falsification-framework
git pull

# CPU-only; must stay green
python -m pytest falsification/tests/ -q

# Fails loudly rather than 20 minutes into a run
python falsification/preflight.py
```

Do not start training until pre-flight prints `Pre-flight clear.`

---

## 1. E4 — size-matched SCR/TPP control (NO TRAINING; do this first)

The cheapest high-value experiment: it needs existing checkpoints, not new runs,
and it converts the preprint's two excluded results into admissible evidence.

Restrict the trained baseline dictionary to a random subset matching the vSAE's
live-feature count (1474 of 8192), recompute SCR and TPP, repeat >= 1000 times.

```bash
python analysis_scripts/online_histogram_analyzer.py \
  --model-path comprehensive_histogram_analysis/TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto \
  --n-samples 1000000 --no-individual
```

Then feed the resulting null distribution to `subsample_null_test`:

```python
from falsification.permutation import subsample_null_test
res = subsample_null_test(observed_score=VSAE_SCR, baseline_scores_subsampled=SUBSAMPLE_SCORES)
print(res["p_value"], res["p_floor"])
```

**Note:** the subsampling harness itself is not written yet — it is the one piece
of E4 still to build. Tell me and I will write it; it needs no GPU to develop.

---

## 2. Seeded training arms

Configs are **hardcoded** in `create_full_config()`. There are no CLI flags for
model, beta, or seed — edit the function, run, repeat per seed.

Shared base for every arm (the existing sweep point, so results are comparable):

```python
model_name = "gelu-1l"
layer = 0
hook_name = "blocks.0.hook_resid_post"
dict_size_multiple = 4.0      # d = 2048
k_fraction = 0.125            # k = 256
total_steps = 10000
lr = 8e-4
auxk_alpha = 1/32
var_flag = 0
seed = <1..6>                 # THE ONLY FIELD THAT CHANGES WITHIN AN ARM
```

### E0 — baseline, 10 seeds (`training_scripts/train_topk.py`)
Identical config, seeds 1–10. Doubles as the baseline arm for E1–E3.
```bash
for s in 1 2 3 4 5 6 7 8 9 10; do
  # edit seed = $s in create_full_config(), then:
  python training_scripts/train_topk.py --config full
done
```

### E1 — degeneracy control, 6 seeds (`training_scripts/train_topk.py`)
Same as E0 plus `activation_penalty = 1.0` (newly added to `TopKTrainingConfig`;
defaults to 0.0 so every other arm is unaffected).

**Verify on the first run before launching all six:** confirm
`activation_penalty_loss` appears in the logged losses and is non-zero. If it is
absent or zero, the penalty is not wired through and the arm is worthless.

### E2 — is it variational at all?, 6 seeds (`training_scripts/train_vsae_topk.py`)
Base config plus `kl_coeff = 1.0`, `var_flag = 1`. Checkpoints should be named
`_learned_var`, not `_fixed_var` — if they say `fixed_var`, the flag did not take.
Also record the learned sigma (`get_kl_diagnostics` logs `variance_mean`).

### E3 — masked-KL, 6 seeds (`training_scripts/train_vsae_topk_masked_kl.py`)
Base config plus `kl_coeff = 1.0`. **Confounded as-is:** this trainer omits the
`F.relu(mu)` that `vsae_topk.py` applies. Either patch one to match the other
first, or report the comparison as confounded. Do not skip this decision.

---

## 3. Measure every checkpoint

```bash
for ckpt in experiments/*/; do
  python analysis_scripts/online_histogram_analyzer.py \
    --model-path "$ckpt" --n-samples 1000000 --no-individual
done
```

Live-feature count is `feature_usage_summary.features_used` in each emitted
`comprehensive_summary_*.json`.

---

## 4. Analyse

```bash
python falsification/worked_example.py     # sanity: framework still runs
python -m pytest falsification/tests/ -q   # must stay green
```

Then per arm, with κ = 0.3 as pre-registered:

```python
from falsification.evalues import FalsificationTest, SequentialFalsifier
from falsification.permutation import seed_permutation_test

f = SequentialFalsifier(main_hypothesis="<the arm's hypothesis>", alpha=0.1, kappa=0.3)
res = seed_permutation_test(arm_values_per_seed, baseline_values_per_seed)
f.add(FalsificationTest(
    name="live-feature fraction",
    null_hypothesis="...", alt_hypothesis="...",
    p_value=res["p_value"],
    unit_of_analysis=res["unit_of_analysis"], n_units=res["n_units"],
    confounders_controlled=("auxk_alpha", "k", "dict_size", "lr", "steps", "seed"),
))
print(f.report())
```

---

## Power, so you know what you can conclude

κ = 0.3, two pre-specified tests, α = 0.1. `d` is the effect size in across-seed
standard deviations.

| seeds/group | d=0.5 | d=1.0 | d=1.5 | d=2.0 |
|---|---|---|---|---|
| 5 | 0.07 | 0.29 | 0.66 | 0.92 |
| 6 | 0.11 | 0.43 | 0.82 | 0.98 |
| 10 | 0.19 | 0.71 | 0.98 | 1.00 |

Effects we expect to be large (degeneracy, β-driven feature death) are fine at 6
seeds. **A one-SD effect is not reliably detectable even at 10 seeds.** Estimate
`d` from the existing single-seed data before committing to an arm; if it is
below ~1.5, say so rather than running it underpowered and reporting a null.

---

## Reporting back

Paste the `model_info` and `feature_usage_summary` blocks from each
`comprehensive_summary_*.json`, and I will run the analysis and fold the numbers
into the paper.
