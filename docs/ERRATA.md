# Errata

Corrections to the arXiv preprint, bugs found in this codebase, and results
that were reported and then retracted, written for someone deciding how much
to trust any of it. Everything here was found by checking claims against the
code and the data rather than against the paper; when the two disagreed, the
code won and the discrepancy was written down. The dated record, with the
order in which each item was found, is in `notebook/` (the F-numbers are
`notebook/REMEDIATION.md`'s; "addendum *N*" is `notebook/RESULTS_2026-09-03.md`).

## 1. Corrections to the preprint

**The evaluated models were not variational.** The reparameterisation in
`vsae_topk.py::encode` is gated on `var_flag == 1`; otherwise z = μ, with no
sampling at all. Every checkpoint in `comprehensive_histogram_analysis/` is
named `_fixed_var`, which the training script emits exactly when
`var_flag == 0`. With σ = 1 fixed, the KL term reduces to ½‖μ‖², a plain L2
penalty on the activations. Every model the preprint evaluated is therefore a
deterministic TopK SAE with an L2 activation penalty, and none of its
descriptions of those runs as stochastic, variational, or as testing a
posterior is correct. The experiment the preprint describes — `var_flag = 1`
— was run for the first time in this repository (E2), and no β in the
preprint's grid produced a working model.

**The headline comparison confounded the KL with AuxK.** The Pythia
comparison is a baseline with `auxk_alpha = 0.03125` against a vSAE with
`aux0`. AuxK is the standard dead-feature revival mechanism, so the vSAE's
higher dead-feature count is confounded with the absence of the standard
remedy. The gelu-1l β sweep held AuxK fixed at 1/32 and is not affected. E4
controlled for dictionary size but could not control for this; the two
confounds are independent.

**Configuration values in the preprint do not match the code.** Code is
authoritative: the Pythia hook is layer 3 (`blocks.3.hook_resid_post`), not
layer 0; `total_steps = 10000`, not 20,000; the Pythia dictionaries are
d = 8192 (16 × d_model = 512) while gelu-1l is d = 2048 (4×). The gelu-1l
arms in this repository *are* at layer 0 — the layer-3 correction is about
the Pythia checkpoints only. The preprint's SAE loss equation includes an L1
term that `top_k.py` does not have; TopK enforces sparsity architecturally.

**The SCR/TPP reading did not follow from the design.** The vSAE's higher
SCR and TPP were read as evidence of a more disentangled feature space, but
both metrics reward selective ablation and selectivity is easier with fewer
live features (18 % alive against 90 %). On a size-matched design the two
metrics disagree (`RESULTS.md` §6). The preprint's Global section and its
Conclusion also reached opposite verdicts on the dispersion hypothesis; the
method had no rule for combining heterogeneous evidence.

**The dead-feature numbers came from two different measurements.** The
preprint quotes 1,227 / 6,970 (from sae_vis histograms) and 1,474 / 7,379
(from the 10⁶-sample analysis) for the same two checkpoints. This
repository uses the 10⁶-sample numbers and says so. `features_used` in the
summary files counts entries selected at least once over the streamed sample
and is sample-size dependent; at d = 2048 it saturates and carries no signal
(§3 below).

## 2. Bugs in the code, and what they touched

Each entry: what was wrong, what it affected, its status.

**The save-time rescaling corrupted the log-variance bias** (`vsae_topk.py`,
also `vsae_batch_topk.py`). Training normalises activations to unit mean
squared norm and scales the biases back up by `norm_factor` (≈ 25.5 for
gelu-1l layer 0) when saving, so a saved model is correct on raw
activations. Until 2026-09-04 the same multiplication was also applied to
`var_encoder.bias` — wrong, because a log-variance is not on the additive
x/μ axis and clamp-then-exp is not scale-homogeneous. It drove every saved
`var_flag = 1` checkpoint's log σ² to appear fully clamp-collapsed regardless
of what was learned (`log_var_init = −2.0` saved as −51.5). *Affected:* the
first reading of E2's learned σ ("the posterior collapses completely", "noise
is harmless at eval time" — both retracted, §4), and every `var_flag = 1`
checkpoint saved before the fix. *Status:* fixed by rescaling
`var_encoder.weight` instead, which preserves log σ²'s value on raw
activations. Checkpoints on disk from before the fix are corrected on read
by dividing both `var_encoder.weight` and `.bias` by a re-estimated
`norm_factor`; the official evaluation was re-run on all 37 of them
(`falsification/reeval_var_flag1.py`, written to
`evaluation_results_corrected.json`, which the readers prefer). The same bug
was found again in `vsae_batch_topk.py` on 2026-09-11 and fixed the same way
before any BatchTopK arm ran. `e2_sigma_low_init` is unaffected either way:
its init already sits below the clamp floor.

**`norm_factor` is not recorded in any checkpoint's `config.json`.**
`trainSAE` writes it into `trainer.config`, but every trainer's `config` is a
property that builds a fresh dict, so the write lands on a temporary. It
affects every arm identically and confounds nothing, but any analysis in
training space must re-estimate it as √(mean ‖x‖²), the estimator
`get_norm_factor` uses. Skipping that rescales every activation by ~25×;
`read_penalty_clamp.py` shows the pattern. *Status:* unchanged; documented.

**The decoder-gradient projection was imported and never called**
(`vsae_topk.py`). `remove_gradient_parallel_to_decoder_directions` was
imported but not applied, while the decoder was still renormalised to unit
norm every step. Not a crash — a silent difference in the update rule worth
*d* = 14.3 on reconstruction inside the vSAE family and null in a plain TopK
SAE (`RESULTS.md` §1, §5.2). *Status:* exposed as a flag
(`project_decoder_grad`) on both trainers, default preserving each
checkpoint's original behaviour.

**`F.relu(mu)` in one trainer and not the other** (`vsae_topk.py` vs
`vsae_topk_masked_kl.py`). Neither paper's equations show it. Worth
*d* = +19.3 / +15.3 on liveness on its own (`RESULTS.md` §4). *Status:*
exposed as `relu_mu` on the masked trainer, default `False`. It changes no
parameter shape, so it cannot be recovered from a state dict; `config.json`
is the only record of which arm a checkpoint belongs to.

**The masked-KL trainer's checkpoints could not be loaded** (F9a/F10). Its
`config` property omitted `dict_class`, `layer`, `lm_name` and
`submodule_name`, so `load_dictionary` raised `KeyError` and the trainer's
own post-training evaluation died on the same key. Six E3 checkpoints were
trained but unanalysable until fixed. *Status:* fixed.

**All loss terms were discarded whenever wandb was off.** `log_stats()`
forwarded the loss dict only to the wandb queue; with `use_wandb = False`
(what `run_arm.py` sets) nothing was logged, and the canary check for E1's
penalty term was unsatisfiable as written. *Status:* fixed; the verbose
branch prints every term.

**Evaluation OOMed silently and wrote NaN.** `loss_recovered()` at the
default evaluation batch size OOMs on a 10 GB card; the failure was swallowed
by an `except … continue` and the run reported success with the
cross-entropy metrics as NaN. Every early checkpoint shows the signature
`frac_recovered = 0.0`. *Status:* `run_arm.py` evaluates at batch 2 × 48.

**Seeds overwrote each other, twice.** `get_experiment_name()` omits the
seed, so training via `create_full_config()` writes every seed of an arm to
the same directory; `run_arm.py` gives each seed its own `save_dir`. The
analysis step then reintroduced the collision: the histogram analyser names
its output after the checkpoint directory, which is identical across seeds.
*Status:* `run_analysis.sh` handles the per-seed output directory; found
before any data was lost.

**`VSAEJumpReLU` had three compounding bugs**, found and fixed on 2026-09-12
before any arm ran (no training script had ever exercised it). (1) The
threshold received zero gradient: the gate was computed with a
non-differentiable comparison, so the `nn.Parameter` could never move from
its init. Fixed by routing through the straight-through estimator the
repository's own non-variational `jumprelu.py` already defines. (2) There
was no L0-target sparsity term, so nothing pushed the threshold toward any
sparsity level; one was added mirroring `JumpReluTrainer`'s. (3) The gate was
applied *before* sampling, so noise could perturb the value of an
already-selected feature but never change the selected set — which would
have made the A4 experiment structurally unable to observe the selection
churn it exists to measure. Fixed by gating the noisy z; verified by mean
Jaccard 0.35 between repeated passes on one token after the fix, 1.0 before.
Side effect: the KL is now computed on the dense, ungated μ, matching
`vsae_topk.py`'s convention. `target_l0` is a *soft* target; achieved L0 must
be checked against it before trusting any sparsity-matched comparison.

**`VSAEBatchTopK`'s global budget is wrong under the evaluation harness's
3-D activations.** The trainer selects the global top-(k · `z.size(0)`)
activations. During training z is pre-flattened to [tokens, d], so that is
correct; `loss_recovered_transformer_lens` calls the model inside a hook on
unflattened [batch, seq, d] activations, where `z.size(0)` is the sequence
count, undershooting the budget by ~ctx_len. `frac_recovered` and
`loss_reconstructed` for any `VSAEBatchTopK` checkpoint are therefore
unreliable (catastrophically negative); `frac_variance_explained`, computed
on the buffer's flattened batches, is not. *Status:* not fixed; A3 reports
FVE and Jaccard only.

**`train_vsae_batchtopk.py` derived its schedule at construction time.**
`ExperimentConfig.__post_init__` computes `warmup_steps`,
`sparsity_warmup_steps` and `decay_start_step` from `total_steps` once;
`run_arm.py`'s override then sets `total_steps = 10000` on a config built at
25,000 without re-running it, leaving `decay_start_step = 20000` — past the
end of the run. *Status:* the A3 arms pin all three explicitly; the TopK
script's default happens to be 10,000 already, which is luck.

**The E4 activation cache was loaded if the file existed, regardless of the
config that built it.** A `--smoke` run (`test_set_size = 50`) followed by a
full run on the same dataset/pair silently reused the smoke-sized cache,
visible only as suspiciously exact small-fraction accuracies (0.9583… =
23/24) and an undersized cache file. Caught by comparing cache sizes before
the numbers were trusted; the contaminated run was deleted and redone.
*Status:* documented in both scorers; never smoke a pair you are about to
run for real.

**The p-value floor had its one-sided and two-sided branches swapped** (F1).
`min_p_floor` / `min_attainable_p` returned 2/C(2n, n) for one-sided and
1/C(2n, n) for two-sided, making every one-sided floor 2× too pessimistic;
the earlier claim that "five seeds per group cannot validate on a single
test" was an artefact of it. *Status:* fixed; the `xfail` that pinned it is a
positive test. The floors that remain are properties, not bugs: 6 seeds
cannot beat 3.07σ two-sided, 13 is the first *n* reaching 5σ, and above
200,000 assignments the test falls back to Monte Carlo with floor
1/(n_perm + 1) — 4.42σ at the 100,000 default, which is why every 13-seed
comparison here uses 4 M draws.

**`preflight.py` passed on a machine that could not train.** It checked
imports, not CUDA. *Status:* fixed. Separately, on the training machine bare
`python` resolves to a CPU-only conda base without nnsight; training needs
`/usr/bin/python3`.

## 3. Metrics that were replaced

**`features_used` saturates.** TopK selects exactly *k* of *d* features per
sample, so mean selection frequency is *k/d* by construction and essentially
every feature fires at least once in 10⁶ samples; at d = 2048 the
pre-registered primary metric carried no signal (F8). Replaced by two
sparsity-relative thresholds (< 0.1× and < 0.5× of *k/d*), both always
reported, computed from the exact per-feature counts the analyser already
saved in its `.npz` (F8a) — no binning limit. The thresholds were fixed after
the saturation was seen, from the arithmetic of the design, and the
confirmatory battery ran on fresh seeds (`METHODS.md` §5).

## 4. Results reported and then retracted

Kept because a record that only contains what survived is not a record.

* **"The posterior collapses completely; eval-time noise is harmless"**
  (addendum 3, 2026-09-03). Mean log σ² read as −66.9 against a clamp at −6,
  100 % of 41 M values at the floor, and turning sampling off at eval
  recovered 0.000012 FVE. Both were the `scale_biases` bug (§2). Corrected:
  mean log σ² ≈ −2.6 to −2.7 (σ ≈ 0.27), zero values at the floor, and the
  noise costs reconstruction.
* **"Correcting the bug moves FVE by ≈ 0.12"** (addendum 8). A proxy
  estimate from an uncommitted script. The official `evaluate()` re-run
  (addendum 9) moved reported FVE by 0.003–0.006, and its absolute numbers
  matched neither side of the proxy's comparison. The qualitative claim it
  supported stands; the magnitude does not.
* **The first Jaccard-overlap read** found selection stability moving the
  *opposite* way to reconstruction. It was made on the not-yet-known-to-be
  corrupted checkpoints and is superseded by the corrected read (addendum 8).
* **The E1 pilot's *d* = 37–59** was on the saturating `features_used`
  metric and three unmatched configuration items. It was never a finding.
* **The claim that 5 seeds/group cannot validate on a single test** rested
  on the swapped floor (F1). At the pre-registered κ = 0.3, five can.
* **The prediction that the initial weight draw would be null**, and the
  speculation that `gradproj` had moved *along* the frontier, were recorded
  and were wrong (`RESULTS.md` §8).
* **The mechanism paper's first draft** placed the gelu-1l arms at layer 3.
  They are at layer 0; layer 3 is the Pythia checkpoints. Caught against
  `experiments/baseline/*/config.json` before the draft went further.
* **The paper's Table 6** carried the superseded 6-seed E3 numbers
  (*d* = +20.8 / +19.8) while claiming 13 seeds; corrected to the 13-seed
  values (+19.3 / +15.3) on first compilation. Its Table 3 quoted
  `e2_sampling_only`'s uncorrected FVE while Table 2 quoted the corrected
  one; and its E2 gap was written as 0.4410 where the arithmetic gives
  0.4420. All three were caught by `reproduce.py --check` and fixed.

## 5. Modifications to vendored SAEBench

`SAEBench-main/` is SAEBench v0.4.2. SAEBench's own evaluation code is
unmodified except for one import path updated for a newer `sae_lens`
(`sae_lens.toolkit.pretrained_saes_directory` →
`sae_lens.loading.pretrained_saes_directory`, in
`sae_bench_utils/general_utils.py` and `sae_selection_utils.py`). Everything
else added lives beside it: `dictionary_learning_wrapper.py` (adapts this
repository's dictionaries to SAEBench's SAE interface, including
`_normalize_decoder_weights()` and support for the vSAE trainers),
`sae_bench/custom_saes/vsae_topk_sae.py`, `test_wrapper.py`, and edits to
the `custom_saes` runner and visualiser for the preprint's own evaluation.
The E4 size control (`falsification/size_control.py`, `run_e4_scr.py`,
`run_e4_tpp.py`, `e4_local_sae.py`) drives SAEBench's internals from outside
the package. `git log -- SAEBench-main/` lists the exact changes.
