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

Measure SCR/TPP as a function of dictionary size for the baseline, then place the
vSAE's score on that curve. **Do not use a random size-matched subset as the
reference** — it is a weak null that the vSAE beats trivially, manufacturing a
false positive. The reference is the baseline's top-N most-used features.

```bash
python analysis_scripts/online_histogram_analyzer.py \
  --model-path comprehensive_histogram_analysis/TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto \
  --n-samples 1000000 --no-individual
```

Then build the size-response curve and read off the verdict:

```python
from falsification.size_control import size_response_curve, verdict

# usage_counts: per-feature selection counts from the baseline's
# comprehensive_summary_*.json. scorer(keep_indices) -> SCR score, which is the
# one piece still to write: it masks the dictionary and calls SAEBench.
points = size_response_curve(
    usage_counts, scorer, n_grid=(500, 1000, 1474, 3000, 5000, 7379), n_draws=20
)
print(verdict(observed_score=VSAE_SCR, observed_n=1474, points=points))
```

`mask_dictionary()` handles the weight masking (note encoder rows and decoder
columns index features on *different* axes — it requires you to say which).

**Still to write:** the `scorer` closure that calls SAEBench on a masked
dictionary. Everything else is implemented and tested.

---

## 2. Seeded training arms — one command

`run_overnight.sh` drives everything. It is safe to interrupt and re-run:
completed runs are skipped via their `RUN_COMPLETE.json` marker.

```bash
./run_overnight.sh --dry-run          # review the plan; no GPU needed
nohup ./run_overnight.sh --hours 10 > sweep.out 2>&1 &
```

Monitor:
```bash
tail -f logs/sweep_*/sweep.log
column -t logs/sweep_*/summary.tsv
```

It gates on framework tests, arm-config validation and preflight before touching
the GPU; runs arms in priority order so the most valuable work finishes first;
refuses to start a run that cannot fit the remaining budget; and logs and skips
individual failures rather than aborting.

**Do NOT edit `create_full_config()` by hand.** That was the old workflow and it
cannot be automated. Worse, `get_experiment_name()` omits the seed, so running
seeds that way makes every seed of an arm overwrite the previous one and you end
up with a single checkpoint. `falsification/run_arm.py` gives each run its own
`save_dir`; use it, or `run_overnight.sh` which calls it.

Single arm at a time, if you prefer manual control:
```bash
python falsification/run_arm.py --check                      # validate all arms, no torch
python falsification/run_arm.py --arm baseline --seed 1 --dry-run
python falsification/run_arm.py --arm baseline --seed 1
```

Arms, in the priority order the sweep uses:

| arm | script | what it is |
|---|---|---|
| `baseline` | `train_topk.py` | clean TopK, `activation_penalty=0.0` (E0) |
| `e2_learned_var` | `train_vsae_topk.py` | `var_flag=1` — the genuinely variational model (E2) |
| `e1_penalty` | `train_topk.py` | TopK + L2 penalty matched to the vSAE beta (E1) |
| `e1_vsae_ref` | `train_vsae_topk.py` | fixed-variance vSAE that E1 should reproduce |
| `e3_masked_kl` | `train_vsae_topk_masked_kl.py` | masked-KL (E3) — **confounded, see below** |

**Verify the first `e1_penalty` run before trusting the arm:** confirm
`activation_cost` appears in the logged losses and is non-zero. If it is absent or
zero the penalty is not reaching the trainer and the arm is worthless.

**E3 is confounded as it stands.** `vsae_topk_masked_kl.py` omits the `F.relu(mu)`
that `vsae_topk.py` applies, so a masked-vs-unmasked comparison mixes the KL mask
with the ReLU. It is last in priority for that reason. Patch one trainer to match
the other before drawing any conclusion from it.

## 3. Measure every checkpoint

```bash
./run_analysis.sh            # every completed run; skips ones already analysed
./run_analysis.sh --force    # re-analyse regardless
```

Live-feature count is `feature_usage_summary.features_used` in each emitted
`comprehensive_summary_*.json`.

**Do not use the plain `for ckpt in experiments/*/` loop that used to be here.** It
was wrong in three ways (`falsification/FINDINGS_2026-09-02.md`, item 0), and the
first was silent:

1. The analyzer writes to `<output-dir>/<model_name>/`, and `model_name` comes from
   the checkpoint directory, which `get_experiment_name()` builds **without the
   seed**. Every seed of an arm therefore resolves to the same output path, so with
   the default `--output-dir` each seed overwrote the last and you were left with
   one summary per arm instead of six. This is the landmine from `CLAUDE.md`
   resurfacing at the one step where `run_arm.py`'s per-seed `save_dir` does not
   protect you. Pass `--output-dir` per seed.
2. `--model-path` must be the `trainer_0/` directory that actually holds `ae.pt`
   and `config.json`, not its parent.
3. The script needed a `sys.path` bootstrap to import `dictionary_learning` when
   invoked from the repo root (now added), and `seaborn` must be installed.

Single checkpoint, if you need one by hand:

```bash
python analysis_scripts/online_histogram_analyzer.py \
  --model-path experiments/<arm>/seed<N>/<ckpt>/trainer_0 \
  --output-dir experiments/<arm>/seed<N> \
  --n-samples 1000000 --no-individual
```

`--n-samples 1000000` for every checkpoint without exception: `features_used` is
sample-size dependent and cross-arm comparability depends on it.

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
