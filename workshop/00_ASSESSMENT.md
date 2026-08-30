# Workshop paper assessment (InterpScience @ NeurIPS 2026)

## Verdict on feasibility

**4-5 pages in ~5h: very doable — but not as a compression of the current paper.**

The CFP limit is **5 pages** (short paper), not 4. References and appendices are
free. That is a lot of room. The binding constraint is not length, it is that the
arXiv preprint's argument does not currently survive the review criteria this
workshop is explicitly selecting on:

> "Falsifiability and experimental designs that distinguish mechanisms from artifacts"
> "Measurement validity, identifiability, and evaluation design"

The preprint has *exactly* the raw material to be a strong submission to this
workshop, but it is currently organized as "we tried X, X lost." Reorganized as
"here is a mechanism, here is the dose-response curve, and here is why the metrics
that say X *won* are measuring dictionary size rather than feature quality," it is
a much better fit and a much stronger paper.

## The one-sentence reframe

Current: *We built a vSAE. It underperforms. Variational methods are not promising.*

Proposed: **A Gaussian KL prior acts as L2 shrinkage on the pre-activation, which
kills features monotonically in β; and the SAEBench metrics on which the vSAE
"wins" (SCR/TPP) improve as a side effect of having fewer live features, so those
wins are an artifact of dictionary size, not evidence of better disentanglement.**

That is a measurement-validity claim about SAEBench with a controlled dose-response
behind it. It is publishable at this workshop in a way the current negative result
is not.

---

## What the code has that the paper does not (highest value first)

### 1. A β dose-response sweep. THIS IS THE PAPER.
`comprehensive_histogram_analysis/` contains completed 1M-sample runs at four KL
coefficients with **everything else held fixed** (gelu-1l, d=2048, k=256,
auxk=1/32, lr=8e-4, fixed variance):

| β (kl_coeff) | live features / 2048 | % alive | mean act | max act |
|---|---|---|---|---|
| 1e-4 | 1605 | 78.4% | 1.209 | 43.12 |
| 1e-2 | 1301 | 63.5% | 1.097 | 41.59 |
| 1.0  | 1013 | 49.5% | 0.825 | 18.01 |
| 100  |  724 | 35.4% | 0.201 |  5.53 |

Monotonic in β, over six orders of magnitude, with the auxiliary loss held
constant. And the activation *scale* collapses 8x from β=1e-4 to β=100 — which is
the direct fingerprint of the predicted mechanism, since with σ fixed to 1 the KL
term reduces to (1/2)||μ||², i.e. plain L2 shrinkage on the pre-activation.

The preprint asserts the KL-kills-features mechanism in prose (§Results) but never
shows this curve. It is sitting in the repo, already computed. **This single figure
converts an assertion into a tested mechanism.** It is the difference between a
rejected workshop paper and an accepted one.

### 2. A confound in the headline comparison that we should disclose ourselves
The Pythia-70m headline numbers are:

| model | auxk | live / 8192 | % alive |
|---|---|---|---|
| TopK SAE | **1/32** | 7379 | 90.1% |
| vSAE (β=1) | **0** | 1474 | 18.0% |

The baseline had AuxK dead-feature revival **on**; the vSAE had it **off**
(`aux0` in the checkpoint name; confirmed in `vsae_topk.py`, which skips
`get_auxiliary_loss` entirely when `auxk_alpha == 0`). AuxK (Gao et al. 2024) is
*the* standard dead-feature mitigation. So the flagship "82% fewer living
features" comparison confounds the KL term with the absence of the standard fix.

This is the first thing a good reviewer will catch, and it would sink the paper.
But it is entirely survivable, because the gelu-1l sweep above holds auxk fixed at
1/32 and *still* shows monotonic death in β. So the mechanism claim stands on the
sweep; the Pythia number needs to be either re-run with auxk on, or explicitly
reported as an upper bound with the confound named. Naming it ourselves and
showing the controlled sweep is a strength, not a weakness — it is precisely the
"distinguishing mechanisms from artifacts" the CFP asks for.

### 3. `vsae_topk_masked_kl.py` — the falsification experiment, already written
The preprint's stated mechanism is:

> "Features unselected by the TopK mechanism still experience the pressure, meaning
> that many features will be penalized to the point that they stop activating."

`dictionary_learning/trainers/vsae_topk_masked_kl.py` applies the KL **only to the
top-k selected features** (`kl_per_dim * mask`). That is a direct, surgical test of
that exact claim: mask the KL off the unselected features and feature death should
largely disappear if the mechanism is right.

The code is written but I found no trained checkpoint or result for it. **This is
the highest-value new run available.** One training run (~the same cost as any
other run in the sweep) turns the paper's central mechanistic claim from
"consistent with the data" into "predicted, then confirmed by intervention." That
is a causal/interventional result, which is the workshop's stated core topic.

### 4. Spike-and-slab and Laplace priors — implemented, untested in the paper
`vsae_priors.py` implements gaussian / laplace / exponential / **spike_slab**
priors. The paper only ever tests the isotropic Gaussian, then concludes
"variational methods are not a promising direction for sparse autoencoder
development." That conclusion is far broader than the evidence: a Gaussian prior is
*a priori the wrong prior for a sparse code* — it has no mass at zero and its KL is
pure L2 shrinkage, which is exactly why it kills features. Spike-and-slab is the
textbook sparse prior.

Even without running it, this reframes the conclusion honestly: we falsified
*Gaussian-prior* vSAEs and identified the mechanism, which predicts which priors
could work. That is a much more defensible claim and it opens future work rather
than closing a door we have not actually looked behind.

### 5. Unused analysis machinery
`analysis_scripts/` has ~250KB of tooling the paper never draws on:
`online_gaussianity.py` (Q-Q, KS, Wasserstein-to-normal on the latents — directly
tests whether the posterior actually matches the prior it is being pulled toward),
`analyze_feature_degeneracy.py` (correlation/MI/LSH duplicate-feature detection —
would let us test whether vSAE features are genuinely more independent or just
fewer), `feature_cosine_analysis.py`. The degeneracy tooling is the natural way to
check claim #6 below at low cost if any activations are cached.

---

## Where the paper is weak (in priority order)

1. **SCR/TPP "wins" are uncontrolled for dictionary size.** The paper reports vSAE
   beating SAE on SCR and TPP, argues in §4.1/§4.2 that this shows a genuinely more
   disentangled latent space — and then the Conclusion reverses and says the gains
   "likely reflect reduced dictionary size." The paper contradicts itself, and
   never tests it. The fix is cheap and is the paper's second-best contribution:
   compare against an SAE with a dictionary size matched to the vSAE's *live*
   feature count (or subsample the SAE dictionary to 1474 features). If the SAE
   also improves, the SCR/TPP gain is a size artifact and that is a real, useful
   finding *about SAEBench* — which is squarely on-topic for this workshop.

2. **The Methods section does not match the code.** Three concrete mismatches:
   - Paper says layer 0 (`blocks.0.residpost`); `train_vsae_topk.py` sets
     `layer=3, hook_name="blocks.3.hook_resid_post"` and the result JSONs confirm
     `blocks.3.hook_resid_post`.
   - Paper says "4x MLP width, yielding 2,048 features"; the Pythia runs are
     d=8192 (16x d_model=512), the gelu-1l runs are d=2048 (4x d_model). Two setups
     are being described as one. Also 4x *d_model*, not 4x MLP width.
   - Paper says 20,000 training steps; `total_steps: int = 10000`.
   These are desk-reject-adjacent for a reproducibility-minded reviewer. Must fix.

3. **The stated SAE loss is not the loss that was used.** The paper writes
   L_SAE = ||x - x̂||² + λ||f||₁. The actual TopK baseline in `top_k.py` is
   `total_loss = l2_loss + auxk_alpha * auxk_loss` — no L1 at all (correctly, since
   TopK enforces sparsity architecturally). Presenting an L1 penalty that was never
   used, and then contrasting "linear pressure in the L2 of the SAE" against the
   KL's quadratic pressure, makes a mechanism argument out of a term that isn't
   there. Fix the equations to match the code and the argument actually gets
   *cleaner*: the real contrast is "no penalty on unselected features" vs "quadratic
   penalty on all features."

4. **The two figures of individual features prove nothing.** Two hand-picked
   features (one per model) shown side by side cannot support "a large difference in
   interpretability was not found." Either drop them for space or state plainly that
   they are illustrative. In a 5-page paper, cut them — they cost a full page and
   carry no inferential weight.

5. **The t-SNE "global" section does not support its conclusion.** t-SNE distances
   are not metric-preserving; "more dispersed in t-SNE" cannot establish that
   features are "less correlated in general," and the section reads as confirming
   the dispersion hypothesis while the Conclusion rejects it. Worse, the two models
   have very different live-feature counts, so the embeddings aren't comparable.
   Replace with a direct, defensible measure on the decoder: the distribution of
   pairwise cosine similarities between live decoder directions (tooling already
   exists in `feature_cosine_analysis.py`). If it must stay, it must be labeled as
   qualitative.

6. **The conclusion overreaches.** "Variational methods are not a promising
   direction for sparse autoencoder development" is not supported by one prior on
   one architecture at one layer of one 70M model. Narrow it to the Gaussian prior
   and let the mechanism do the generalizing.

7. **Two internal contradictions to resolve.** §Global concludes the dispersion
   hypothesis is *confirmed*; the Conclusion says it is *rejected*. The abstract
   says the vSAE "excels at feature independence" while the conclusion says those
   gains are an artifact. Pick one line and hold it throughout.

8. **The dead-feature count is reported two different ways.** Paper §Local says
   1,227 vs 6,970 (from the sae_vis max-activation histograms, "82% fewer"); the
   1M-sample analysis says 1,474 vs 7,379 (80% fewer). Use the 1M-sample numbers as
   primary — larger sample, documented config — and drop the other.

9. **Anonymization.** Submission is double-blind. The current draft has names,
   affiliations, a SPAR acknowledgement, and "see preliminary analysis here" with a
   dangling link. All must go, and the repo link must be anonymized or dropped.

---

## What to cut to make room

- Autoencoder/VAE/SAE history (Rumelhart, Hinton-Salakhutdinov, PCA). A workshop
  audience needs none of it. ~0.75 page.
- The reparameterization trick derivation (~0.5 page). One sentence + citation.
- The two individual-feature figures (~1 page).
- The bulleted per-panel walkthroughs of each figure — say what the figure shows
  once, in the caption, and state the inference in the text.
- The "paper is structured as follows" itemize block.
- Related work trimmed to a short paragraph; the 2006 sparse-coding lineage can go.

That is roughly 3 pages recovered, which is more than enough for the new material.

## Proposed 5-page structure

1. **Intro** (0.5p) — polysemanticity → SAEs → the natural idea of a variational
   prior → our claim: the Gaussian KL is L2 shrinkage, it kills features
   monotonically, and the benchmarks that reward the result are measuring size.
2. **Setup** (0.5p) — vSAE definition, the σ=1 collapse KL → (1/2)||μ||², honest
   config table matching the code.
3. **The KL is shrinkage: a dose-response** (1.25p) — the β sweep table/figure,
   live features and activation scale vs β, auxk held fixed. Plus the masked-KL
   intervention if the run lands.
4. **The SCR/TPP wins are a dictionary-size artifact** (1.25p) — size-matched
   control, and what that implies for reading SAEBench numbers generally.
5. **Core benchmarks** (0.5p) — condensed to one figure, reported not narrated.
6. **Discussion / limitations** (0.75p) — the auxk confound stated plainly, the
   Gaussian-prior-only scope, spike-and-slab as the predicted next step.
7. Appendix (free): t-SNE, feature visualizations, full configs.

## Priority order given ~5h

**Must (paper is not submittable without these):**
- Anonymize; fix the three Methods/code mismatches; fix the SAE loss equation.
- Resolve the confirmed/rejected contradiction into one consistent line.
- Build the β dose-response figure from the JSONs already in the repo (no GPU).
- State the auxk confound explicitly in Limitations.

**High value, needs your 3080 (launch these first, they run while we write):**
- (a) `vsae_topk_masked_kl` run at β=1, gelu-1l, k=256, auxk=1/32 — the
  falsification experiment. Highest value per GPU-hour in the whole repo.
- (b) Pythia vSAE β=1 **with auxk=1/32** — removes the headline confound.
- (c) Size-matched SAE control for SCR/TPP (or a subsampled-dictionary eval, which
  may not need training at all).

**Nice to have:** spike-and-slab run; Gaussianity Q-Q on the latents.

If none of the GPU runs land, the paper still works: the β sweep alone carries the
mechanism, and the SCR/TPP artifact can be argued from the live-feature counts we
already have, flagged as a hypothesis rather than a demonstrated result.
