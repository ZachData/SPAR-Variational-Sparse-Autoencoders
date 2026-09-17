# SPAR — Variational Sparse Autoencoders, and what the experiments actually license

A fork of [`dictionary_learning`](https://github.com/saprmarks/dictionary_learning)
extended with variational sparse autoencoders (vSAEs), a vendored copy of
[SAEBench](https://github.com/adamkarvonen/SAEBench), and a sequential
falsification framework (e-values + permutation tests over training seeds)
built to decide which claims from the vSAE preprint survive contact with the
code and the data. Several did not; catching that is the point of the current work.

**This file is the short, dated status page.** It is meant to be bumped every
session. It does not replace `PROJECT.md` (the living document) — when the two
disagree, `PROJECT.md` wins.

---

## Status — 2026-09-17

| | |
|---|---|
| **Branch** | Single-branch repo since 2026-09-17: everything lives on `master`. Work on short-lived feature branches and PR them in; delete on merge. |
| **PR** | [#5](https://github.com/ZachData/SPAR-Variational-Sparse-Autoencoders/pull/5) (mechanism paper + addenda 22–25) merged 2026-09-17; [#6](https://github.com/ZachData/SPAR-Variational-Sparse-Autoencoders/pull/6) is this README. |
| **Tests** | `python -m pytest falsification/tests/ -q` → **115 passed** (CPU only, ~6 s). |
| **Battery** | Complete at 13 seeds/arm, 5σ on every comparison. 48 arm directories under `experiments/`, 0 failed runs. |
| **Paper** | `workshop/mechanism_paper.tex` drafted; every item on its checklist is closed. **Never compiled** — no LaTeX toolchain on this machine. Checked by hand only (citations resolve, braces balance, all `\ref` have `\label`). |
| **Running** | Nothing. |
| **Blocked on** | Nothing (compute or data). Open question is venue / what to do with the paper next. |
| **Environment** | Training runs on the **system `/usr/bin/python3` (3.14)** — torch 2.10.0+cu128, nnsight 0.7.0 in `~/.local`. If `python` resolves to miniforge `base` (CPU-only torch, no nnsight), `preflight.py` fails on `CUDA available`; none of the conda envs (`mets`, `sltdiff`, `vibevoice`) has nnsight either. `preflight.py` was fully green on 2026-09-17 with the system python. |

### What is established (one line each; detail in `PROJECT.md` → *What is established*, and `falsification/RESULTS_2026-09-03.md`)

- **E1** — a fixed-variance vSAE *is* a TopK SAE with an L2 activation penalty (identity verified to 6 decimals). The measured d≈13–16 gap between the two implementations decomposes entirely into 5 optimiser/init details; with all 5 matched the arms are null everywhere.
- **E2** — it is the *sampling*, not the KL: removing the KL entirely recovers only 6.4 % of the FVE gap to baseline; the remaining 93.6 % is the reparameterisation.
- **E3** — the `F.relu(mu)` that only one trainer applies is itself a d≈15–19 effect.
- **A2/A3 (Claim #3)** — sampling-induced FVE damage tracks selection-Jaccard instability almost exactly for hard top-k: TopK r = +0.9993, BatchTopK r = +0.9979, across a 6-point, 13-seed σ-init sweep. Smooth dose-response, no knee.
- **A4 (JumpReLU)** — *not* a clean third point: the naive r = +0.93 is confounded by an 8× swing in achieved L0. L0-matched, r ≈ +0.50 (n=8, addendum 23). The real finding: a soft learned threshold's sparsity *level* isn't noise-robust, a failure hard top-k can't exhibit.
- **E4** — SCR says the vSAE's advantage is *not* explained by dictionary size; TPP says it *is*. Same checkpoints, same grid. Threshold-uniform, survives both bootstraps, replicates across 8 (dataset, pair) points; the size-matched trained-from-scratch baseline confirms "masked vs. trained-small" doesn't flip the aggregate verdict (addenda 21, 24). **The disagreement itself is still unexplained.**
- **Claims #4, #5** — the decoder-gradient projection effect is real but scoped to penalised models (null on plain TopK, addendum 22); 10/10 SAE-methods papers in our bibliography train one seed per config, so none can reach architecture-level significance (addendum 25).

### Next (nothing pre-selected — read `PROJECT.md` → *Next steps* fresh)

1. Compile `workshop/mechanism_paper.tex` somewhere with a LaTeX toolchain and fix whatever the compiler finds.
2. Decide venue / next step for the mechanism paper; the methods paper (Deliverable #1) is still only scoped.
3. Optional, new pre-registration: extend the E2 β grid downward (1e-5, 1e-6) — see `PROJECT.md` Next steps #2 for why it's low priority.

---

## Which document for what

| File | Read it for |
|---|---|
| **`README.md`** (this) | Dated status, what's established, what's next. Bump every session. |
| **`PROJECT.md`** | The living document: Status → Where things stand → What is established → Next steps → pre-registration → open decisions → deliverables. Authoritative. |
| **`CLAUDE.md`** | **Landmines in the vSAE code** — bugs and mismatches that have already produced false claims. Read before touching `dictionary_learning/` or describing any checkpoint. |
| `falsification/RESULTS_2026-09-03.md` | Numbered addenda (currently through **#25**). Every result lands here in full before `PROJECT.md` summarises it. |
| `HANDOFF.md` | Cold-start table of contents with a session-by-session "Done" log. |
| `RUNBOOK.md` | Copy-pasteable commands for when GPU access returns, ordered so failures surface cheaply. |
| `OVERVIEW.md` | Plain-language "what is this project" for a non-specialist. |
| `workshop/` | `mechanism_paper.tex` (current), `paper.tex` (superseded workshop draft), `references.bib`, `figs/`. |

## Repository layout

```
dictionary_learning/     fork of saprmarks/dictionary_learning + vSAE trainers
  trainers/vsae_topk.py            TopK vSAE (applies F.relu(mu) — see CLAUDE.md)
  trainers/vsae_topk_masked_kl.py  masked-KL variant (relu_mu flag, default False)
  trainers/vsae_batch_topk.py      BatchTopK vSAE (frac_recovered unreliable — CLAUDE.md)
  trainers/vsae_jump_relu.py       JumpReLU vSAE (4 bugs fixed 2026-09-12, before any run)
training_scripts/        train_vsae_topk.py, train_vsae_batchtopk.py, train_vsae_jumprelu.py, train_topk.py
analysis_scripts/        feature-usage / histogram analysers (driven by run_analysis.sh)
falsification/           the framework: evalues.py, permutation.py, simulate.py, run_arm.py,
                         the E1–E4 / A1–A4 readers and scorers, tests/, RESULTS_2026-09-03.md
experiments/             one directory per arm, one subdirectory per seed (48 arms)
comprehensive_histogram_analysis/  the preprint's recovered Pythia checkpoints + summaries
SAEBench-main/           vendored SAEBench (SCR / TPP scorers live here)
sae_vis/                 feature visualisation
workshop/                papers, bib, figures
archive/, logs/          old runs and run logs
```

## Commands

```bash
# Which environment am I in? (remote/web sessions have no GPU and no torch)
# Locally: use the system python, not miniforge base — see the Environment row above
python falsification/preflight.py

# Framework tests — CPU only, must stay green
python -m pytest falsification/tests/ -q

# Apply the framework to the committed data
python falsification/worked_example.py

# Training — LOCAL GPU ONLY. Always via run_arm.py; never hand-edit
# create_full_config() (get_experiment_name() omits the seed → seeds overwrite each other)
python falsification/run_arm.py --check            # validate all arms, no torch
python falsification/run_arm.py --arm baseline --seed 1
./run_overnight.sh --hours 10

# Feature-usage measurement (serial: two analysers OOM the 10 GB card)
./run_analysis.sh            # only re-analyses checkpoints whose summary is stale
./run_analysis.sh --force

# Cross-arm tables, frontier, figures
python falsification/report_summaries.py --table
python falsification/frontier.py
python falsification/read_a2_dose_response.py      # (and read_a3_/read_a4_followup)
```

Environment: Python ≥ 3.10, `pip install -r requirements.txt`. Training targets
bfloat16 on a 10 GB RTX 3080; buffer settings are tuned for that and OOM easily
if raised.

## The five landmines you will hit first

Full list, with the evidence, in `CLAUDE.md`. The ones that bite most often:

1. **`var_flag=0` means no sampling at all.** Every `_fixed_var` checkpoint is a deterministic TopK SAE with an L2 penalty. Do not call them variational.
2. **`vsae_topk.py` applies `F.relu(mu)`; the masked-KL trainer does not.** Comparing them confounds the mask with the ReLU; `relu_mu` can't be recovered from a state dict — only `config.json` knows.
3. **Pre-2026-09-04 checkpoints have a corrupted `var_encoder.bias`** (`scale_biases` rescaled the log-variance bias). Divide `var_encoder.{weight,bias}` by `norm_factor` before reading `log_var`; `norm_factor` itself isn't saved and must be re-estimated.
4. **P-value floors are combinatorial.** 6 seeds/group caps at 3.07σ; 13 is the first n reaching 5σ; Monte Carlo fallback caps at `1/(n_perm+1)`. A p at the floor means the design ran out, not the evidence — check `result["p_floor"]`.
5. **`frac_recovered = 0.0` is usually an OOM, not a result**, and `VSAEBatchTopK`'s `frac_recovered` is wrong by construction under `loss_recovered()`'s 3D activations — use `frac_variance_explained` for it.

## Conventions

- New statistical tests go in `falsification/permutation.py` with a test in `falsification/tests/`; Monte Carlo p-values are `(count+1)/(n_perm+1)`; every test returns `p_floor`.
- A claim about an *architecture* needs a permutation test over *training seeds*; token-level tests only compare two specific checkpoints.
- Confounders are recorded on `FalsificationTest`; an uncontrolled one excludes the test from the evidence product, it does not down-weight it.
- Checkpoint directory names encode the config and are parsed by the analysers — keep the naming scheme.
- Code beats preprint. When they disagree, the code wins and the discrepancy gets written down.

## Session log

One line per session, newest first. Detail belongs in `HANDOFF.md` / `RESULTS`.

| Date | What changed |
|---|---|
| 2026-09-17 | PR #5 merged. Branches consolidated to `master` only (`main`, `claude/falsification-framework`, `claude/vae-workshop-paper-condensing-zumu6b`, `claude/fix-decoder-weight-normalization-…` deleted — the last was already superseded by `_normalize_decoder_weights()` in the SAEBench wrapper). 115 tests green, preflight green with `/usr/bin/python3`. This README rewritten as the status page; the 2025-08-22 README repeated claims `CLAUDE.md` refutes. |
| 2026-09-13 | Mechanism paper drafted (`workshop/mechanism_paper.tex`); addenda 22–25: Claim #4 (projection null on plain TopK), A4 n=4 doubt closed, E4 box (5) extended to 8 points, Claim #5 seed-count survey; lit-review pass (Chanin 2026 et al.). |
| 2026-09-12 | A4 JumpReLU run (addendum 20, confounded); 4 bugs fixed in `vsae_jump_relu.py` first. E4 box (5) size-matched baseline (addendum 21) — checklist fully closed. |
| 2026-09-11 | A1 falsified (15), A2 σ-init dose-response (16–17, r=+0.9993), A3 BatchTopK (18, r=+0.9979), E4 box (4) coverage widened (19). |
| 2026-09-09/10 | E4 bootstraps (13–14): the SCR/TPP disagreement is not test-set noise. |
| 2026-09-03/04 | E1 decomposition closed (4–5); `scale_biases` bug found and fixed (8); E2 mechanism corrected and re-measured (7–9). |

## Provenance

Built on `dictionary_learning` (Marks, Karvonen & Mueller, 2024) and SAEBench
(Karvonen et al., 2025). MIT licence, as upstream.
