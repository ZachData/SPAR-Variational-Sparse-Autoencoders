# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this repository is

A fork of `dictionary_learning` extended with variational sparse autoencoders
(vSAEs), a vendored SAEBench, and a falsification framework for validating
claims about SAEs. The science is finished; the repository is being made to
read as finished. `README.md` is the front door and `docs/` is the account:

- `docs/RESULTS.md` — what was established, in final form
- `docs/METHODS.md` — the statistical design and the pre-registration
- `docs/ERRATA.md` — **read before touching `dictionary_learning/` or describing
  any checkpoint.** Every bug and mismatch that has already produced a false
  claim is there: `var_flag=0` means no sampling; the two trainers differ by
  a ReLU as well as the KL mask; the save-time bias rescaling corrupted
  `log_var` in pre-2026-09-04 checkpoints; `norm_factor` is not saved;
  `frac_recovered` is wrong for `VSAEBatchTopK`; the E4 cache ignores its
  config; the p-value floors are combinatorial.
- `docs/REPRODUCE.md` — environment, `reproduce.py`, and the GPU commands
- `docs/notebook/` — the dated working record (PROJECT.md, the RESULTS
  addenda, REMEDIATION), frozen. Code comments that cite "PROJECT.md",
  "RESULTS addendum N" or "REMEDIATION F6" mean these files.
- `FINISHING.md` — the plan for finishing the repository; read its STATUS
  block first in any session that continues that work.

## Environment

- Two environments, and it matters which you are in: `python
  falsification/preflight.py` tells you. Remote/web sessions have no GPU and
  no torch. Local sessions on the RTX 3080 (10 GB) train with
  `/usr/bin/python3` (torch 2.10 cu128 + nnsight in `~/.local`); bare
  `python` there is a CPU-only conda base and will not work.
- `reproduce.py` needs only numpy, scipy, matplotlib.

## Commands

```bash
python -m pytest falsification/tests/ -q        # 115 tests, CPU; must stay green
python reproduce.py --check                     # every paper number from committed data; must exit 0
./workshop/build_paper.sh                       # rebuild workshop/mechanism_paper.pdf
python falsification/run_arm.py --check         # validate all arms, no torch
python falsification/run_arm.py --arm <arm> --seed <n>   # LOCAL GPU ONLY
./run_analysis.sh                               # feature-usage measurement, per-seed output dirs
python falsification/compare_arms.py <a> <b>    # any two arms, 4M-draw permutation test
```

Never hand-edit `create_full_config()` in a training script to change a seed:
`get_experiment_name()` omits the seed and runs written that way overwrite
one another. Use `run_arm.py`.

## Conventions

- **Statistics.** Any new statistical test goes in `falsification/permutation.py`
  with a test in `falsification/tests/`. Monte Carlo permutation p-values must use
  `(count + 1) / (n_perm + 1)`; the naive form is anti-conservative and can emit
  `p = 0`, which maps to an infinite e-value. Every test returns `p_floor` and
  `exact` — report them, because a p sitting at its floor means the design ran
  out, not the evidence. At 13 seeds/group pass `n_perm=4_000_000`; the 100k
  default caps evidence at 4.42σ.
- **Unit of analysis.** A claim about an *architecture* requires a permutation test
  over *training seeds*. Token-level tests answer questions about two specific
  checkpoints only; `paired_token_test` refuses architecture-level use unless the
  narrower scope is explicitly acknowledged.
- **Confounders.** Record them on `FalsificationTest`. A test with
  `confounders_uncontrolled` is excluded from the evidence product rather than
  down-weighted, because the implication assumption it violates is binary.
- **Liveness.** Both pre-registered thresholds (`below_0.1x`, `below_0.5x`) are
  always reported; a result counts only if they agree in direction.
- **Checkpoints.** Directory names encode the config and are parsed by the
  analysis scripts — keep the naming scheme. `relu_mu` and
  `project_decoder_grad` change no parameter shape; `config.json` is the only
  record of which arm a checkpoint belongs to. Prefer
  `evaluation_results_corrected.json` where it exists (`compare_arms.py` does).
- **The paper's numbers are checked, not typed.** If a number in
  `workshop/mechanism_paper.tex` changes, change the matching expectation in
  `reproduce.py` and make `--check` pass; if `--check` fails, the paper is
  wrong until shown otherwise.

## Working style

Claims here are checked against code and data, not against the preprint. When
the preprint and the code disagree, the code wins and the discrepancy gets
written down in `docs/ERRATA.md`. Several conclusions in the published version
did not survive that check, and the value of the work comes from having caught
them.
