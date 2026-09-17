# The notebook

The project's working documents, frozen verbatim on 2026-09-17 when the
science finished. Nothing here is maintained; the finished account is one
level up (`docs/RESULTS.md`, `docs/METHODS.md`, `docs/ERRATA.md`,
`docs/REPRODUCE.md`). These files are kept because the audit trail — what was
predicted, what was found, what was retracted, and in what order — is part of
the contribution.

| File | Was | What it is |
|---|---|---|
| `PROJECT.md` | `/PROJECT.md` | The living document: status, what was established, next steps, the pre-registration, open decisions. Authoritative while the work ran. |
| `RESULTS_2026-09-03.md` | `/falsification/RESULTS_2026-09-03.md` | The confirmatory battery's results and **addenda 1–25**, one per finding, in the order they landed. Code and paper comments that say "RESULTS addendum *N*" mean this file. |
| `FINDINGS_2026-09-02.md` | `/falsification/FINDINGS_2026-09-02.md` | The 2026-09-02 audit: where the documentation, code and data disagreed. |
| `REMEDIATION.md` | `/falsification/REMEDIATION.md` | The fixes for that audit, numbered **F1–F10**; comments that say "F8b" or "F9b" mean this file. |
| `HANDOFF.md` | `/HANDOFF.md` | Cold-start table of contents with a session-by-session log. |
| `RUNBOOK.md` | `/RUNBOOK.md` | The GPU command sequence, superseded by `docs/REPRODUCE.md`. |
| `OVERVIEW.md` | `/OVERVIEW.md` | A plain-language description for a non-specialist. |
| `workshop_00_ASSESSMENT.md`, `workshop_01_RUNS.md`, `workshop_02_KEY_FINDING.md` | `/workshop/0*_*.md` | Planning notes for the superseded workshop draft. |
| `workshop_paper.tex` | `/workshop/paper.tex` | The superseded workshop draft (not built; it expects a NeurIPS style file and `figs/beta_sweep.pdf` next to it). The current paper is `workshop/mechanism_paper.tex`. |

Docstrings in `falsification/` and the trainers still cite these by their
old names ("PROJECT.md Next steps A2", "RESULTS addendum 8", "REMEDIATION.md
F6"). Those references are correct as citations of the record and have been
left alone; resolve them here.

## Session log (from the README as it stood on 2026-09-17)

One line per session, newest first; detail in `HANDOFF.md` and `RESULTS_2026-09-03.md`.

| Date | What changed |
|---|---|
| 2026-09-17 | PR #5 merged. Branches consolidated to `master` only (`main`, `claude/falsification-framework`, `claude/vae-workshop-paper-condensing-zumu6b`, `claude/fix-decoder-weight-normalization-…` deleted — the last was already superseded by `_normalize_decoder_weights()` in the SAEBench wrapper). 115 tests green, preflight green with `/usr/bin/python3`. This README rewritten as the status page; the 2025-08-22 README repeated claims `CLAUDE.md` refutes. |
| 2026-09-13 | Mechanism paper drafted (`workshop/mechanism_paper.tex`); addenda 22–25: Claim #4 (projection null on plain TopK), A4 n=4 doubt closed, E4 box (5) extended to 8 points, Claim #5 seed-count survey; lit-review pass (Chanin 2026 et al.). |
| 2026-09-12 | A4 JumpReLU run (addendum 20, confounded); 4 bugs fixed in `vsae_jump_relu.py` first. E4 box (5) size-matched baseline (addendum 21) — checklist fully closed. |
| 2026-09-11 | A1 falsified (15), A2 σ-init dose-response (16–17, r=+0.9993), A3 BatchTopK (18, r=+0.9979), E4 box (4) coverage widened (19). |
| 2026-09-09/10 | E4 bootstraps (13–14): the SCR/TPP disagreement is not test-set noise. |
| 2026-09-03/04 | E1 decomposition closed (4–5); `scale_biases` bug found and fixed (8); E2 mechanism corrected and re-measured (7–9). |

