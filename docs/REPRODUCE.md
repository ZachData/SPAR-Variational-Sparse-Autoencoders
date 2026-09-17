# Reproducing the paper's numbers

Every figure and every numeric table in `workshop/mechanism_paper.tex` is
regenerated from data committed in this repository by one command, on a CPU:

```bash
pip install numpy scipy matplotlib          # the only dependencies
python reproduce.py --check                 # tables to stdout, figures to workshop/figs/,
                                            # exit 1 if any headline number drifts from the paper
python reproduce.py --write docs/reproduced_tables.md   # keep the tables as Markdown
./workshop/build_paper.sh                   # rebuild the PDF (fetches `tectonic` on first use)
```

`--check` compares 125 numbers against what the paper prints (every table
cell that is a measurement, every correlation quoted in the text, the E1 gap
closures and verdict statistics) at the paper's own printed precision. It
takes about 45 s; the permutation tests are the slow part.

## What is committed, and what is not

| Committed | Where | Size |
|---|---|---|
| Every training run's metadata: `config.json`, `experiment_config.json`, `evaluation_results.json` (+ `_corrected.json` where it exists), `RUN_COMPLETE.json` | `experiments/<arm>/seed<n>/…` | 2.4 MB, 1,940 files |
| Every analysed run's `comprehensive_summary_*.json` and `all_histograms_*.npz` (the exact per-feature selection counts behind the liveness thresholds) | same | 16 MB, 170 runs |
| The preprint-era analysis of the original checkpoints | `comprehensive_histogram_analysis/` | 0.8 MB |
| GPU measurements cached as JSON: selection Jaccard per checkpoint, SAEBench SCR/TPP scores, the seed-count survey | `falsification/*_results.json` | < 1 MB |
| The compiled paper | `workshop/mechanism_paper.pdf` | |

Not committed: the 437 checkpoints (`ae.pt`, 2.3 GB — a release archive is
planned; see `README.md`), the per-run histogram panels (`.png`) and logs.

## Environment

* Reproducing the paper: any Python ≥ 3.10 with `numpy`, `scipy`,
  `matplotlib`. No torch.
* Training or re-measuring checkpoints: the local RTX 3080 (10 GB) with
  `/usr/bin/python3` (torch 2.10.0+cu128, nnsight 0.7.0, transformer_lens;
  `pip install -r requirements.txt`). Training targets bfloat16 on 10 GB and
  the buffer settings OOM easily if raised.
  A bare `python` on that machine is a CPU-only miniforge and will not work.
  `python falsification/preflight.py` tells you which environment you are in.
* `pytest falsification/tests/ -q` (115 tests, CPU) must stay green.

## Where each figure and table comes from

"Run data" means `experiments/<arm>/seed*/*/evaluation_results.json`,
preferring `evaluation_results_corrected.json` when present (the re-evaluation
of pre-fix `var_flag=1` checkpoints with the `scale_biases` bug corrected —
`falsification/reeval_var_flag1.py`, `ERRATA.md`), plus `all_histograms_*.npz` for
the two liveness thresholds (`falsification/report_summaries.py::liveness`).
Effect sizes are Cohen's *d* with the pooled SD taken as the mean of the two
groups' SDs; *p*-values are two-sided seed-permutation tests
(`falsification/permutation.py`) with 4 M Monte Carlo draws.

| Paper | Arms | Data | `reproduce.py` | Equivalent script |
|---|---|---|---|---|
| Table 1 — E1 ladder | `e1_penalty` vs `e1_vsae_ref`, `_unitinit`, `_gradproj`, `_fullmatch` | run data | `table_e1_ladder` | `falsification/compare_arms.py e1_penalty <arm>` |
| §3.3–3.4 text — gap closures (78.9 %, 74.9 %), factor sizes (−14.3; −4.7, −7.3), verdict | same | run data | `table_e1_ladder` | `compare_arms.py e1_vsae_ref_unitinit e1_vsae_ref_gradproj`, `… e1_vsae_ref_gradproj e1_vsae_ref_fullmatch` |
| Table 2 — KL vs reparameterisation split | `baseline`, `e2_confirm`, `e2_sampling_only` | run data (corrected) | `table_e2_split` | — |
| Table 3, Figure 1 — TopK dose-response | `a2_sigma_init_m{1,3,4,5}`, `e2_sampling_only`, `e2_sigma_low_init`; baseline `baseline` | FVE: run data (corrected); Jaccard: `falsification/a2_dose_response_results.json` | `tables_dose_response`, `figures` | `falsification/read_a2_dose_response.py` (GPU; `--fig-only` re-plots) |
| Table 4, Figure 2 — BatchTopK | `a3_batchtopk_*` | FVE: run data; Jaccard: `a3_dose_response_results.json` | same | `read_a3_dose_response.py` |
| Table 5, Figure 3 — JumpReLU | `a4_jumprelu_*` (six points) | FVE, L0: run data; Jaccard: `a4_dose_response_results.json` | same | `read_a4_dose_response.py` |
| §4.4.2, Figure 4 — ten-point JumpReLU grid | + `a4_jumprelu_sigma_init_m{3_5,4_5,5_5,6}` (5 seeds) | FVE, L0: run data; Jaccard: `a4_followup_results.json` | same | `read_a4_followup.py` |
| Table 6 — `F.relu(mu)` | `e3_masked_kl` vs `e3_masked_kl_relu` | run data (npz) | `table_e3` | `compare_arms.py e3_masked_kl e3_masked_kl_relu` |
| Table 7 — implementation-detail summary | — | Tables 1 and 6 | `table_variance_summary` | — |
| Table 8 — projection off in a plain TopK SAE | `baseline` vs `claim4_baseline_noproj` | run data | `table_claim4` | `compare_arms.py baseline claim4_baseline_noproj` |
| Table 9 — sigma ceiling by seeds/group | — | formula 2/C(2n,n) | `table_seed_ceiling` | `falsification/seed_count_survey.py` |
| Table 10 — seeds per configuration in the literature | — | `falsification/seed_count_survey_results.json` | `table_seed_survey` | `seed_count_survey.py` |
| Table 11 — SCR vs TPP on the Pythia pair | `e4_pythia_baseline`, `e4_pythia_vsae` | `falsification/e4_scr_results.json`, `e4_tpp_results.json`, `e4_size_matched_baseline_results.json` | `table_e4` | `run_e4_scr.py`, `run_e4_tpp.py`, `score_e4_size_matched_baseline.py` (GPU + SAEBench) |
| Table 12 — the 15-item code diff | — | static (a reading of the two trainers) | — | — |

Two figures in `workshop/figs/` are not in the mechanism paper.
`beta_sweep.pdf` (the preprint-era β sweep, `workshop/make_fig_beta.py`, from
`comprehensive_histogram_analysis/`) is regenerated by `reproduce.py`.
`frontier.pdf` is the exploratory liveness/reconstruction frontier of RESULTS
addendum 6 over the 8 arms analysed at the time; `python
falsification/frontier.py` redraws it over every analysed arm on disk (11
now), so it is left as committed.

## Numbers that need a GPU and a checkpoint

These are quoted in the paper from measurements on the saved weights and are
not recomputed by `reproduce.py`; the scripts are in `falsification/` and
each states its inputs.

| Number | Script |
|---|---|
| Selection Jaccard per checkpoint (cached in the `*_dose_response_results.json` above) | `read_a2_dose_response.py`, `read_a3_…`, `read_a4_…`, `read_a4_followup.py` |
| Pre-activation gap at the k-th/(k+1)-th boundary (median 0.0001, 99th pct 0.0006) | `read_preact_gap.py` |
| Largest pre-clamp activation (0.194 over 20,000 activations) | `read_penalty_clamp.py` |
| Learned posterior sigma of the E2 arms | `read_learned_sigma.py` (apply the `ERRATA.md` bias correction by hand) |
| Mean Jaccard 0.35 vs 1.0 before/after the JumpReLU gate fix | `dictionary_learning/trainers/vsae_jump_relu.py` (a direct forward-pass check, described in `ERRATA.md`) |
| SCR/TPP scores (cached in `e4_*_results.json`) | `run_e4_scr.py`, `run_e4_tpp.py`, `score_e4_size_matched_baseline.py` |

## Re-running the training (GPU)

Everything below needs the RTX 3080 environment. Runs are ~1 min each for
gelu-1l; the 10⁶-sample feature-usage analysis is ~6.5 min per checkpoint
and is the binding cost.

```bash
python -m pytest falsification/tests/ -q      # must stay green first
python falsification/preflight.py             # fails loudly rather than 20 min into a run
python falsification/run_arm.py --check       # validates every arm's config, no torch

# One run. Each arm x seed gets its own directory under experiments/<arm>/seed<n>/;
# completed runs are skipped via their RUN_COMPLETE.json.
python falsification/run_arm.py --arm baseline --seed 1 --dry-run
python falsification/run_arm.py --arm baseline --seed 1

# The seeded batteries (all resumable; they gate on tests + preflight first)
./run_overnight.sh --dry-run                  # the confirmatory arms, priority order
nohup ./run_overnight.sh --hours 10 > sweep.out 2>&1 &
./run_e2_pilot.sh                             # E2 stage 1 (5 betas, seed 101) + selection rule
./falsification/run_a2_sweep.sh               # A2 TopK sigma sweep, 4 arms x 13 seeds
./falsification/run_a3_sweep.sh               # A3 BatchTopK
./falsification/run_a4_sweep.sh               # A4 JumpReLU

# Feature-usage analysis (per-seed output dir handled; serial -- two analysers OOM the card)
./run_analysis.sh                             # only checkpoints whose summary is older than ae.pt
./run_analysis.sh --force

# Then the checkpoint readers listed above, and
python falsification/compare_arms.py <arm_a> <arm_b>     # any two arms, 4M-draw permutation test
python falsification/report_summaries.py --table         # every analysed run
```

Never hand-edit `create_full_config()` in a training script to change a seed:
`get_experiment_name()` omits the seed, so runs written that way overwrite one
another. `run_arm.py` and `run_analysis.sh` exist to prevent exactly that.
The E4 scorers (`run_e4_scr.py`, `run_e4_tpp.py`) cache activations by file
name only: never `--smoke` a dataset/pair you are about to run for real
(`ERRATA.md`).

Using the framework on a new comparison, with the pre-registered α and κ:

```python
from falsification.evalues import FalsificationTest, SequentialFalsifier
from falsification.permutation import seed_permutation_test

f = SequentialFalsifier(main_hypothesis="<the claim>", alpha=0.1, kappa=0.3)
res = seed_permutation_test(arm_values_per_seed, baseline_values_per_seed, n_perm=4_000_000)
f.add(FalsificationTest(
    name="live-feature fraction",
    null_hypothesis="...", alt_hypothesis="...",
    p_value=res["p_value"],
    unit_of_analysis=res["unit_of_analysis"], n_units=res["n_units"],
    confounders_controlled=("auxk_alpha", "k", "dict_size", "lr", "steps", "seed"),
))
print(f.report())                 # and read res["p_floor"] / res["exact"] before quoting p
```
