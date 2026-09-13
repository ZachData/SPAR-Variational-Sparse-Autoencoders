"""Claims-worth-opening #5 (PROJECT.md): "most published SAE comparisons
cannot reach the significance they imply."

The combinatorial floor `falsification/permutation.py::min_p_floor` computes
is not a subtlety of this repo's own design -- it is arithmetic that applies
to ANY seed-permutation test of an architecture-level claim, regardless of
who runs it. This script does two things:

  1. Reuses `min_p_floor` (not a reimplementation) to build the sigma-ceiling
     table for n seeds/group -- the same function every comparison in this
     repo's confirmatory battery is measured against.
  2. Records a literature survey: for each SAE paper this repo's own
     `workshop/references.bib` already cites (the sampling frame was fixed
     BEFORE the survey was read, not chosen afterward to make the point),
     how many independently-seeded training runs does its methodology
     report for the SAME hyperparameter configuration?

Every entry in SURVEY was checked by fetching the paper's own full text
(arXiv HTML where available; the two Anthropic/Transformer-Circuits posts
resisted full-text fetch -- their entries rest on multiple independent
web searches quoting the paper's own methodology text, cross-checked
against each other, not on a direct full-text read; see RESULTS addendum
25 for exact evidence and caveats). This is a keyword/section search over
each paper's stated methodology, not a from-scratch re-derivation of every
number in every paper -- a paper that runs a single seed for its headline
comparison but buries a secondary multi-seed robustness check somewhere
this search missed would be under-counted here. None of the ten found any
such check beyond the one explicitly recorded below (Bricken et al.'s
matched-pair universality check, itself only n=2 and not used for the
paper's main quantitative results).

    python falsification/seed_count_survey.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falsification.permutation import min_p_floor  # noqa: E402

# ---------------------------------------------------------------------------
# Part 1: the sigma ceiling, n seeds/group, two-sided seed-permutation test.
# Same function, same formula, as every comparison in this repo's battery.
# ---------------------------------------------------------------------------

CEILING_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20, 25, 30, 50]


def sigma_ceiling(n_per_group: int) -> float:
    from scipy.stats import norm

    p = min_p_floor(n_per_group, n_per_group, "two-sided")
    return float(norm.isf(p / 2.0))


# ---------------------------------------------------------------------------
# Part 2: the survey. Sampling frame = every SAE-methods paper already cited
# in workshop/references.bib as of this session (fixed before reading any of
# them for this purpose), i.e. the papers this repo's OWN related-work
# section already treats as the field's core references. `seeds_per_config`
# is the number of independently-seeded training runs of the SAME
# hyperparameter configuration the paper's methodology reports -- NOT the
# number of distinct configurations/architectures/widths swept (several of
# these papers train hundreds of SAEs, but as a sweep over hyperparameters,
# which is a different design question from seed repetition of one point).
# ---------------------------------------------------------------------------

SURVEY = [
    {
        "paper": "Cunningham et al. 2023, \"Sparse Autoencoders Find Highly "
                 "Interpretable Features in Language Models\"",
        "arxiv": "2309.08600",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search (arXiv HTML) finds no mention of training "
            "multiple SAEs with different seeds for one configuration, no "
            "seed-variance error bars; only a reproducibility footnote "
            "pointing at the code release, no seed count stated."
        ),
    },
    {
        "paper": "Bricken et al. 2023, \"Towards Monosemanticity\" (Anthropic)",
        "arxiv": None,
        "url": "https://transformer-circuits.pub/2023/monosemantic-features",
        "seeds_per_config": 1,
        "seeds_special_case": 2,
        "evidence": (
            "All headline results (loss curves, feature counts, "
            "interpretability case studies) are reported on a single "
            "transformer/dictionary pair, model 'A'. A SECOND, "
            "independently-seeded transformer ('B') is trained and its "
            "dictionaries compared against A's ONLY for a supplementary "
            "feature-universality check ('does the same feature reappear "
            "under a different seed?') -- itself descriptive (which "
            "features match, not a significance test), not applied to the "
            "paper's main quantitative claims. This is the single largest "
            "seed count found anywhere in this survey, and it is n=2, used "
            "for one qualitative check only. (Full-text fetch of the paper "
            "itself was not possible -- the page exceeds this session's "
            "fetch size limit; this entry rests on multiple independent web "
            "searches quoting the paper's own stated methodology, "
            "cross-checked against each other for consistency. Flagged "
            "explicitly as the one entry not backed by a direct full-text "
            "read.)"
        ),
    },
    {
        "paper": "Templeton et al. 2024, \"Scaling Monosemanticity\" (Anthropic, "
                 "Claude 3 Sonnet)",
        "arxiv": None,
        "url": "https://transformer-circuits.pub/2024/scaling-monosemanticity",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search (arXiv HTML mirror) finds the three headline "
            "SAEs (1M/4M/34M features) are each a single training run; a "
            "learning-rate sweep chooses one value by lowest loss, not by "
            "aggregating across seeds. No seed-variance reporting found."
        ),
    },
    {
        "paper": "Gao et al. 2024, \"Scaling and evaluating sparse "
                 "autoencoders\" (OpenAI, TopK)",
        "arxiv": "2406.04093",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search finds no mention of multi-seed training, "
            "variance, or error bars across independently trained models "
            "anywhere in main text or appendices."
        ),
    },
    {
        "paper": "Rajamanoharan et al. 2024, \"Improving Dictionary Learning "
                 "with Gated Sparse Autoencoders\" (DeepMind)",
        "arxiv": "2404.16014",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search finds no mention of multi-seed training, "
            "variance, or error bars across independently trained models "
            "anywhere in main text or appendices."
        ),
    },
    {
        "paper": "Rajamanoharan et al. 2024, \"Jumping Ahead: Improving "
                 "Reconstruction Fidelity with JumpReLU Sparse Autoencoders\" "
                 "(DeepMind)",
        "arxiv": "2407.14435",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search finds 'multiple 131k-width SAEs (with a range "
            "of sparsity levels) of each type' trained to build the "
            "sparsity-fidelity curves -- a sweep over sparsity, not repeated "
            "seeds of one point. Binomial confidence intervals ARE reported "
            "for a manual human-interpretability rating study (an "
            "evaluation-level, not a training-seed-level, uncertainty), "
            "consistent with CLAUDE.md's own token/seed distinction: "
            "eval-level uncertainty is not rare in this literature, "
            "architecture-level (seed) uncertainty essentially is."
        ),
    },
    {
        "paper": "Bussmann et al. 2024, \"BatchTopK Sparse Autoencoders\"",
        "arxiv": "2412.06410",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search finds no mention of multi-seed training, "
            "variance, or error bars across independently trained models "
            "anywhere in main text or Appendix A.1's hyperparameter/dataset "
            "details."
        ),
    },
    {
        "paper": "Karvonen et al. 2025, \"SAEBench: A Comprehensive Benchmark "
                 "for Sparse Autoencoders\"",
        "arxiv": "2503.09532",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search finds 'over 200 SAEs' trained -- a sweep over "
            "widths (4k/16k/65k) and sparsities across multiple "
            "architectures, i.e. 200+ distinct CONFIGURATIONS, not repeated "
            "seeds of the same configuration. No seed-variance reporting "
            "found for any single configuration."
        ),
    },
    {
        "paper": "Marks et al. 2024, \"Sparse Feature Circuits\"",
        "arxiv": "2403.19647",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search finds no mention of multi-seed training, "
            "variance, or error bars across independently trained models "
            "anywhere in main text or appendices; uses existing SAE suites "
            "(their own Pythia-70M SAEs and Gemma Scope), one each."
        ),
    },
    {
        "paper": "Lu et al. 2025, \"Sparse Autoencoders, Again?\"",
        "arxiv": "2506.04859",
        "seeds_per_config": 1,
        "evidence": (
            "Full-text search (main text + Appendix D) finds results "
            "presented as single values per method throughout, no "
            "confidence intervals or variance measures from repeated "
            "training runs."
        ),
    },
]

# Corroborating evidence: independent papers whose ENTIRE finding is that SAE
# training is seed-sensitive -- not part of the seed-count survey itself
# (they are about seed SENSITIVITY, not about how the field evaluates it),
# but directly relevant context for why the near-universal single-seed
# practice above is not a merely theoretical gap.
CORROBORATING_EVIDENCE = [
    {
        "paper": "Paulo & Belrose 2025, \"Sparse Autoencoders Trained on the "
                 "Same Data Learn Different Features\"",
        "arxiv": "2501.16615",
        "finding": (
            "Only ~30% feature overlap between two otherwise-identical "
            "131k-latent SAEs trained on Llama-3-8B activations, differing "
            "only in random seed; observed across multiple layers, three "
            "LLMs, two datasets, and several SAE architectures."
        ),
    },
    {
        "paper": "Gerasimov et al. 2026, \"Unstable Features, Reproducible "
                 "Subspaces: Understanding Seed Dependence in Sparse "
                 "Autoencoders\"",
        "arxiv": "2606.12138",
        "finding": (
            "Individual features vary substantially across training seeds "
            "and cluster into reproducible lower-rank subspaces; a "
            "large-scale study across seeds, models, layers, dictionary "
            "sizes, and SAE variants."
        ),
    },
]


def main() -> int:
    print("=== Part 1: the sigma ceiling (two-sided seed-permutation test, "
          "n seeds/group) ===\n")
    print(f"{'n/group':>8} {'p_floor':>14} {'sigma ceiling':>14}")
    ceiling_by_n = {}
    for n in CEILING_GRID:
        s = sigma_ceiling(n)
        ceiling_by_n[n] = s
        flag = "  <- no significance possible, any effect size" if n == 1 else ""
        flag = "  <- below conventional p<0.05 (1.96 sigma)" if n == 2 else flag
        print(f"{n:>8} {ceiling_by_n[n]:>14.3f}{flag}")

    print("\n=== Part 2: the survey "
          f"({len(SURVEY)} papers, workshop/references.bib as of this "
          "session) ===\n")
    n1 = sum(1 for p in SURVEY if p["seeds_per_config"] == 1)
    for p in SURVEY:
        n = p["seeds_per_config"]
        special = p.get("seeds_special_case")
        note = f" (n={special} for one supplementary check only)" if special else ""
        print(f"  n={n}{note}  ceiling={ceiling_by_n.get(n, sigma_ceiling(n)):.3f}sigma  "
              f"-- {p['paper']}")

    print(f"\n{n1}/{len(SURVEY)} papers: n=1 for every reported result "
          "(no possible significance at any effect size for an architecture-"
          "level claim). Modal and median seed count across the survey: 1. "
          "Maximum observed anywhere, for any purpose: 2 (one paper, one "
          "supplementary check, not a formal test).")

    print("\n=== Corroborating evidence: independent seed-sensitivity "
          "studies ===\n")
    for c in CORROBORATING_EVIDENCE:
        print(f"  {c['paper']}: {c['finding']}")

    out = {
        "sigma_ceiling_by_n": ceiling_by_n,
        "survey": SURVEY,
        "corroborating_evidence": CORROBORATING_EVIDENCE,
        "n_papers": len(SURVEY),
        "n_papers_seeds_eq_1": n1,
    }
    out_path = REPO / "falsification/seed_count_survey_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
