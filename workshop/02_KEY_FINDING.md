# The finding that should define the paper

**In every model you evaluated, no sampling ever occurred. The "variational" SAE
is, operationally, a TopK SAE with an L2 penalty on its activations.**

## The evidence

`dictionary_learning/trainers/vsae_topk.py`, `encode()`:

```python
mu = self.encoder(x_processed)
mu = F.relu(mu)                      # <-- ReLU is still here

log_var = None
if self.var_flag == 1:
    log_var = self.var_encoder(x_processed)

if training and self.var_flag == 1 and log_var is not None:
    z = self.reparameterize(mu, log_var)
else:
    z = mu                           # <-- deterministic latents
```

Sampling is gated on `var_flag == 1`. Every checkpoint in
`comprehensive_histogram_analysis/` is named `_fixed_var`, and
`train_vsae_topk.py:275` sets that suffix precisely when `var_flag == 0`
(`var_suffix = "_learned_var" if self.config.var_flag == 1 else "_fixed_var"`).
`create_full_config()` sets `var_flag=0`. There is no `learned_var` artifact
anywhere in the repo.

Therefore, in all eight evaluated models: `z = mu = ReLU(W_enc x + b_enc)`, and
with `var_flag=0` the KL reduces (in code, `_compute_kl_loss`) to
`0.5 * ||z||²`.

## What that means

The evaluated vSAE is exactly:

    f     = ReLU(W_enc x + b_enc)          # same encoder as the baseline SAE
    f_topk = TopK(f)                        # same sparsity as the baseline
    x_hat = W_dec f_topk + b_dec            # same decoder
    L     = ||x - x_hat||² + (beta/2)||f||² + auxk

That is a standard TopK SAE with an **L2 activation penalty**. The
reparameterization trick is a no-op. The noise is never drawn. There is no
posterior, no stochasticity, and therefore no "probabilistic area around a
direction" — the mechanism the preprint's Figure 2 illustrates and the entire
dispersive-pressure hypothesis rests on.

## Why this is good news, not bad

Three claims in the preprint are now false as written, and must be corrected
regardless of what we do next:

- Abstract: "replaces deterministic ReLU gating with stochastic sampling." It
  does neither — the ReLU is still applied, and no sampling occurs.
- §2.2: the encoder is given as `mu = W_enc(x - b_dec) + b_enc` with no ReLU.
  Code applies `F.relu`.
- §3/§4: the reparameterization trick is derived over half a page and then
  repeatedly invoked as one of "the two differences between the SAE and vSAE."
  It is not a difference. The *only* operative difference is the L2 penalty.

But correcting them makes the paper **much** stronger, because it explains every
result at once, with no hand-waving:

- **Why feature death is monotonic in beta.** An L2 penalty on activations is
  weight decay on the code. Turn it up, activations shrink, features fall below
  the TopK cut and never recover. The dose-response curve we already have
  (78% -> 64% -> 49% -> 35% live as beta goes 1e-4 -> 100) is exactly the shape
  an activation penalty predicts, and the 8x collapse in activation scale
  (max 43.1 -> 5.5) is its direct signature.
- **Why the dispersive pressure never materialized.** There was no stochasticity
  to disperse anything. The preprint's Global section reports the hypothesis
  confirmed and the Conclusion reports it rejected; the truth is it was never
  tested.
- **Why "variational methods don't help SAEs" was never actually tested.** The
  preprint's final sentence — "variational methods are not a promising direction
  for sparse autoencoder development" — generalizes from an experiment in which
  the variational component was inert.

## The new thesis

> A Gaussian-prior vSAE with fixed unit variance degenerates exactly into a TopK
> SAE with an L2 activation penalty. We show the degeneracy analytically and
> confirm it in code, then show that the resulting penalty causes monotonic
> feature death in beta, and that the SAEBench metrics on which the vSAE appears
> to *win* (SCR, TPP) track the reduced live-feature count rather than any
> improvement in feature quality. Reported benefits of "variational" SAEs must
> therefore be controlled against a plain L2 penalty and a size-matched
> dictionary before they can be attributed to probabilistic structure.

That is a falsifiability + measurement-validity paper about a whole class of
methods, supported by data you already have. It is a substantially better
submission than "we tried it and it lost," and it is honest about what happened.

## What this changes about the runs

It promotes a fourth run to the top of the list: **actually turn the sampling on**
(`var_flag=1`). That is the experiment the preprint claims to have run and did
not. See `01_RUNS.md`.
