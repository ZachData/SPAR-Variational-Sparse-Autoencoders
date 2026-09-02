# GPU runs — launch these first, they train while I write

> **READ `02_KEY_FINDING.md` FIRST.** Every evaluated model used `var_flag=0`,
> which means `z = mu` and **no sampling ever happened**. The evaluated "vSAE" is a
> TopK SAE with an L2 activation penalty. This reorders the runs below: the
> experiment the preprint claims to have run (an actually-stochastic vSAE) has not
> been run at all, and it is now Run D — arguably the most important of the four.

---

## Run D — turn the sampling ON (`var_flag=1`)  ← NEW, top priority alongside A

**Why:** this is the experiment the paper is ostensibly about. With `var_flag=1`
the encoder learns `log_var`, `reparameterize()` actually fires, and the KL uses
the full `0.5*(mu^2 + sigma^2 - 1 - log sigma^2)`. Only then is there a posterior,
and only then can the dispersive-pressure hypothesis be tested at all.

Match the existing sweep point so it is one variable against a known number
(gelu-1l, beta=1, d=2048, k=256, auxk=1/32 -> **1013/2048 alive**, fixed-var):

```python
# training_scripts/train_vsae_topk.py, create_full_config()
model_name = "gelu-1l",
layer = 0,
hook_name = "blocks.0.hook_resid_post",
dict_size_multiple = 4.0,     # d = 2048
k_fraction = 0.125,           # k = 256
total_steps = 10000,
lr = 8e-4,
kl_coeff = 1.0,
auxk_alpha = 1/32,
var_flag = 1,                 # <-- THE POINT. checkpoint will say _learned_var
```

```bash
python training_scripts/train_vsae_topk.py --config full
python analysis_scripts/online_histogram_analyzer.py \
  --model-path <ckpt_D> --n-samples 1000000 --no-individual
```

**Read out:** live features vs 1013; and if you can, the learned `sigma`
distribution (`get_kl_diagnostics` logs `variance_mean`). Two informative outcomes:
- sigma collapses toward 0 → the model *chooses* determinism, posterior collapse,
  and the fixed-var degeneracy was not an implementation accident but the
  optimum. Strong result.
- sigma stays finite and behaviour differs → there is a real variational effect the
  preprint never measured, and the paper reports the first honest test of it.

Either way we can finally say something true about variational SAEs rather than
about an L2 penalty.

---

## Run A — masked-KL (the falsification experiment)

**Why:** the paper's stated mechanism is that features *unselected* by TopK still
absorb KL pressure and therefore die. `vsae_topk_masked_kl.py` applies the KL only
to the top-k selected features. If the mechanism is right, feature death should
largely disappear. If death persists, our mechanism is wrong and we need to say so.
Either outcome is publishable; that is what makes it worth running first.

**Match it to an existing sweep point** so masked-vs-unmasked is a clean
single-variable contrast. The comparison target already exists:
`VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var` → **1013/2048 alive**.

Edit `create_full_config()` in `training_scripts/train_vsae_topk_masked_kl.py`
(it currently defaults to Pythia d=8192/k=512 — we want the gelu-1l point):

```python
model_name = "gelu-1l",
layer = 0,
hook_name = "blocks.0.hook_resid_post",
dict_size_multiple = 4.0,     # -> d = 4 * 512 = 2048
k_fraction = 0.125,           # -> k = 0.125 * 2048 = 256
total_steps = 10000,          # confirm this matches how the sweep runs were trained
lr = 8e-4,
kl_coeff = 1.0,
auxk_alpha = 1/32,
var_flag = 0,
```

```bash
python training_scripts/train_vsae_topk_masked_kl.py --config full
```

Then measure live features the same way the sweep did:

```bash
python analysis_scripts/online_histogram_analyzer.py \
  --model-path <checkpoint_dir_from_run_A> \
  --n-samples 1000000 --no-individual
```

**Read out:** `feature_usage_summary.features_used` in the emitted
`comprehensive_summary_*.json`.

**Prediction to state before you look** (pre-registering this makes the paper
stronger — write the number down now): masked KL should land substantially above
1013, plausibly near the β=1e-4 level (~1605), since masking removes the penalty
from exactly the features that were dying. If it lands near 1013, the mechanism is
refuted and the honest paper says the death is driven by shrinkage on the
*selected* features instead.

---

## Run B — Pythia vSAE with AuxK on (removes the headline confound)

**Why:** the flagship "82% fewer live features" compares a baseline with
AuxK dead-feature revival **on** (auxk=1/32) against a vSAE with it **off**
(aux0). That is the confound a reviewer finds first.

Edit `create_full_config()` in `training_scripts/train_vsae_topk.py` to match the
existing baseline `TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto`:

```python
model_name = "EleutherAI/pythia-70m-deduped",
layer = 3,
hook_name = "blocks.3.hook_resid_post",
dict_size_multiple = 16.0,    # -> d = 8192
k_fraction = 0.03125,         # -> k = 256   (NOT 0.0625, which gives k=512)
total_steps = 10000,
lr = 8e-4,
kl_coeff = 1.0,
auxk_alpha = 1/32,            # <-- the fix; the published run used 0
var_flag = 0,
```

```bash
python training_scripts/train_vsae_topk.py --config full
python analysis_scripts/online_histogram_analyzer.py \
  --model-path <checkpoint_dir_from_run_B> --n-samples 1000000 --no-individual
```

**Read out:** live features out of 8192, against baseline 7379 and the aux0 vSAE's
1474. Whatever it is, it is the number that belongs in the abstract. If AuxK
recovers most of the gap, our headline shrinks but the paper gets *more* honest and
the β sweep still carries the mechanism. If it does not, the claim is now clean.

---

## Run C — size-matched SCR/TPP control

**Why:** the preprint argues vSAE's SCR/TPP wins show better disentanglement, then
the Conclusion says they "likely reflect reduced dictionary size." Nobody tested
it. This tests it.

**Cheapest version needs no training.** Take the *trained baseline* TopK SAE and
restrict its dictionary to its top-N most-used features, N = the vSAE's live count
(1474), then re-run SCR and TPP. Zero out the other columns of `W_dec` and rows of
`W_enc`, or mask at encode time; feature-usage rankings are already in
`comprehensive_summary_TopK_SAE_pythia70m_*.json` under
`feature_usage_summary`. Then:

```bash
python SAEBench-main/sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py
```
(restricted to the scr/tpp evals)

**Read out:** does the size-matched SAE's SCR/TPP improve toward the vSAE's scores?
- If **yes** → the vSAE "win" is a dictionary-size artifact. This is the paper's
  second headline and a genuine measurement-validity finding about SAEBench.
- If **no** → the vSAE really does have better-separated features, the artifact
  hypothesis is dead, and we report that instead. Also fine, also interesting.

Do not skip this on the grounds that we already believe the answer — the whole
point of the framing is that we tested it.

---

## Run E (cheap, optional) — the honest baseline: plain L2 activation penalty

Since the fixed-var vSAE *is* a TopK SAE + L2 penalty, a TopK SAE trained with an
explicit `+ (beta/2)*||f||^2` term should reproduce the vSAE's numbers to within
noise. If it does, that is the cleanest possible demonstration of the degeneracy —
a one-line code change that makes the whole "variational" framing evaporate. Only
worth it if D and A are already running.

## If you only have time for one

Run **D**, now — not A. The mechanism (Section 3 of the draft) is already carried by
the beta sweep we have in hand, but D is the only run that tests the paper's actual
subject. A is the strongest follow-up: it makes the death mechanism causal rather
than correlational.

## Reporting back

For each run I need: live-feature count, total dictionary size, and the config
actually used. Paste the `comprehensive_summary_*.json` (or just the
`model_info` + `feature_usage_summary` blocks) and I will fold the numbers in.
Every number in my draft that depends on a pending run is marked `\TODO{...}` so
nothing unverified reaches the PDF.
