# HANDOFF — cold-start orientation

Short by design. Read this first, act from `PROJECT.md`. When this file and
`PROJECT.md` disagree, `PROJECT.md` wins — it is the living document and this one
is a table of contents with a heartbeat.

Last touched: 2026-09-09.

## What this repo is

A fork of `dictionary_learning` extended with variational SAEs (vSAEs), plus a
vendored SAEBench copy and a sequential-falsification framework (e-values +
permutation tests) that decides what the vSAE experiments actually license.
Two lines of work:

1. **vSAE architectures + experiments** — `dictionary_learning/`,
   `training_scripts/`, `analysis_scripts/`. Behind the arXiv preprint.
2. **Falsification battery** — `falsification/`. The confirmatory work. This is
   where active effort goes.

## Which doc for what

| File | Use it for |
|---|---|
| `CLAUDE.md` | **Landmines in the vSAE code.** Read before touching `dictionary_learning/` or describing any checkpoint. Several are bugs that already produced false claims. |
| `PROJECT.md` | The living document: Status → Where things stand → What is established → Next steps, then the pre-registration. Cold-start reading order is stated at its top. |
| `falsification/RESULTS_2026-09-03.md` | Numbered addenda (currently through **#12**). Each experiment result lands here in full before `PROJECT.md`'s summary is updated. |
| `RUNBOOK.md` | Copy-pasteable commands, ordered so failures surface cheaply. |
| `OVERVIEW.md` | Plain-language "what is this project" for a non-specialist. |

## Current state (2026-09-09)

- Confirmatory battery **complete at 13 seeds/arm**, 5σ on every comparison.
  11 arms, 153 checkpoints, 0 failures. Framework: 115 tests green.
- **E1, E2, E3 have landed.** E0 pre-registered, never reported.
- **E4 has a first reading** (RESULTS addenda 10–13): SCR and TPP give opposite
  verdicts on the same two Pythia checkpoints; the disagreement is
  threshold-uniform (addendum 12) and survives a test-set bootstrap
  (addendum 13) — though SCR's side of it is ~15× wider and rests on the N=2
  ablation threshold alone. Single seed each side, one dataset/class-pair —
  descriptive by design, not a permutation test.
- **`PROJECT.md` Next steps #0 boxes (1) and (2) are done** (addenda 12–13).
  Boxes (3)–(5) are open and not scoped in code — pick up there.
- Branch `claude/falsification-framework`, pushed to origin, merged to `master`.

### Next task — E4 diagnostics, `PROJECT.md` Next steps #0 boxes (3)–(5)

Boxes (1)–(2) are closed. The remaining three, cheapest first (read
Claims-worth-opening #6 and RESULTS addenda 10–13 before picking one).

**Also available (needs a free GPU):** `e4_bootstrap.py --metric scr
--resample-train --n-grid 1474` — the full (non-conditional) bootstrap that
closes addendum 13's stated caveat about SCR's thin +0.082 margin. The
`--resample-train` path now has 10GB-card memory handling (2026-09-09) but has
**not been run** — the attempt was blocked by a concurrent ~3GB GPU job
(`main.py GeoDeepLearning/*`) on the machine, not by the code. Smoke first
(`--smoke --resample-train --sae-batch-size 32`) to confirm it fits and measure
the per-draw rate, then size `--boot` (est. ~1–2 h at grid `[1474]`).

- **(3)** read which features SCR's/TPP's own effect computation selects
  (`get_effects_per_class_precomputed_acts`) — do the same vSAE features get
  reused across TPP's five classes (overloading) while SCR's top-effect set is
  disjoint (a dedicated axis)? A read of existing artifacts, no new run.
- **(4)** the second SAEBench dataset (`canrager/amazon_reviews_mcauley_1and5`)
  + the other three `bias_in_bios` class pairs — widens coverage, still
  single-seed. Both scorers exist, caches warm for the current pair only.
- **(5)** train a plain TopK baseline from scratch at `dict_size=1474` on
  Pythia-70m layer 3 — removes "masked vs. trained-small" as a confound. A real
  training run (LOCAL GPU), single-seed.

### Done 2026-09-09 — box (2), the bootstrap error bars (addendum 13)

TPP bootstrap ran to completion on the 5-point grid `500 1000 1474 2000 3000`
(200 draws, 5.06 h) after a ~15-line fix to `e4_bootstrap.py`: it now writes
`e4_bootstrap_tpp_results.json.partial` every 10 draws and resumes from it on
restart (draw RNG seeded by draw index, so resume is exact). Both 2026-09-08 TPP
attempts had been lost because the script only wrote on completion.

Result: both `vsae − top_usage@1474` margins clear zero under test-set
resampling — SCR **+0.082 [+0.024, +0.150]**, TPP **−0.086 [−0.090, −0.081]** —
so the disagreement is not a test-set-noise artifact. But SCR's CI is ~15×
wider and per threshold clears zero only at N=2 (N=5/10/20 straddle it); TPP
clears zero at N=2/5/10. Conditional bootstrap (node effects fixed) — CIs are
lower bounds; `--resample-train` (unrun) would widen them.

## What is established (one line each — detail in RESULTS + PROJECT.md)

- **E1:** a fixed-variance vSAE *is* a TopK SAE with an L2 penalty (identity
  verified to 6 decimals). The measured d≈16 / d≈13 gap between the two
  implementations decomposes entirely into 5 optimiser/init details; with all 5
  matched the arms are null everywhere. Code diff enumerated at 15 items, all
  settled.
- **E2:** it is the sampling, not the KL. Removing the KL entirely recovers only
  6.4% of the FVE gap to baseline; the other 93.6% is the reparameterisation.
  Robust to the `scale_biases` correction (addenda 8–9).
- **E3:** the `F.relu(mu)` that only one trainer applies is a d≈15–19 effect,
  run as its own arm rather than patched away.
- **E4:** SCR says the vSAE's advantage is *not* explained by dictionary size;
  TPP says it *is*. Same checkpoints, same grid. The disagreement is the finding
  (thesis Failure 1, reproduced fresh). It is threshold-uniform (addendum 12)
  and survives a test-set bootstrap in both directions (addendum 13) — but
  SCR's margin CI is ~15× wider than TPP's and rests on the N=2 threshold alone.

## Next action

`PROJECT.md` Next steps **#0**, boxes (3)–(5), cheapest first — boxes (1)–(2) are
done (addenda 12–13). (3) read which features SCR's/TPP's own effect computation
selects; (4) the second SAEBench dataset + other 3 class pairs; (5) train a
size-matched baseline from scratch. None are scoped in code — read
Claims-worth-opening #6 in full and RESULTS addenda 10–13 before picking one.

## Environment

`python falsification/preflight.py` tells you which environment you are in.
Local RTX 3080 (10 GB) can train; remote/web sessions have no GPU or torch and
are limited to reading, analysis of committed JSONs, figures, and writing.
Training OOMs easily at 10 GB — use `falsification/run_arm.py`, never hand-edit
`create_full_config()`.
