"""Is there one liveness-reconstruction frontier?

PROJECT.md, "Claims worth opening" #2. Every arm has (reconstruction, liveness)
at 13 seeds and the data has been on disk since the battery finished. The
question it answers is not about any one arm:

* If every arm lies on **one tight curve**, then "the vSAE has more dead
  features" and "the vSAE reconstructs worse" are not two findings but one, and a
  paper reporting both as independent evidence is double-counting. That is
  Failure 2 of the thesis -- metrics co-varying with a nuisance -- demonstrated on
  our own battery instead of argued in the abstract.
* If arms sit **off** the curve, the frontier gives a principled meaning to
  "better": above the curve, not merely higher on one axis.

**This is exploratory.** It is not part of the pre-registered battery, every arm
in it was trained for another purpose, and no hypothesis about the shape of this
relationship was recorded before the points were plotted. It is reported as a
description of data already collected, and nothing here is licensed to carry a
confirmatory p-value.

Four decisions that shape what comes out, recorded so they can be argued with:

* **The liveness axis is the near-dead fraction, not `frac_alive`.**
  `frac_alive` is 1.000000 in every arm including the collapsed ones -- a feature
  that fires once in a million tokens counts as alive -- so it has no variance to
  correlate. The two pre-registered sparsity-relative thresholds are used instead,
  and BOTH are reported, because the pre-registration's rule is that a
  disagreement between them is itself the finding (F8b).
* **The unit of analysis for a cross-arm claim is the arm, not the checkpoint.**
  Thirteen seeds of one arm are thirteen draws of one architecture, not thirteen
  draws of "an architecture", so a correlation over 136 checkpoints is
  pseudo-replicated and its p-value is not valid for a claim about the set of
  architectures (CLAUDE.md, unit of analysis). Both are printed; the arm-mean row
  is the one to quote, and the checkpoint-level row is labelled.
* **No regime threshold is chosen.** The relationship is not monotone, so any
  single correlation over the whole range is a summary of two different things.
  Rather than pick a cut and be accused of tuning it, every cut is scanned: for
  each possible split between adjacent arms the correlation above and below is
  reported, so the reader sees the sign flip and how little it depends on where
  the line goes. A cut is used afterwards, for the figure and for the
  "working models" subset, and it is descriptive -- it sits in a 0.34-wide gap
  with no arm in it, and the scan shows every cut in that gap agrees.
* **Log-ish y axis with `linthresh` at the measurement resolution.** The
  near-dead fraction spans 0 to 0.9995 and five checkpoints sit at exactly 0, which
  a plain log axis would silently drop. `symlog` with `linthresh = 1/2048` -- one
  feature, the smallest difference the measurement can express -- keeps them on
  the plot and in the right place.

    python falsification/frontier.py
    python falsification/frontier.py --no-fig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falsification.compare_arms import arm_values  # noqa: E402

# d = 2048, so one feature is 1/2048 of the dictionary: the finest difference the
# near-dead fraction can express, and the linear/log crossover on the y axis.
DICT_SIZE = 2048
RESOLUTION = 1.0 / DICT_SIZE

# An arm needs replicate seeds to contribute a mean worth correlating. The E2 beta
# pilot ran one seed per beta by design (stage 1 of a two-stage rule), so those
# points are drawn as context and excluded from every statistic.
MIN_SEEDS_FOR_STATS = 6

THRESHOLDS = ("below_0.1x", "below_0.5x")

# Composite encoding: hue carries the family, marker carries the arm within it.
# Eleven arms is past the point where distinct hues stay separable, and the skill's
# rule is that the ninth series folds into a composite encoding rather than a
# generated hue. Hues are slots 1-4 of the validated default categorical palette,
# in fixed order, and never cycled.
FAMILIES = {
    "baseline":  ("#2a78d6", "no penalty"),
    "e1":        ("#eb6834", "E1: TopK+L2 vs fixed-variance vSAE"),
    "e2":        ("#1baf7a", "E2: sampling on"),
    "e3":        ("#eda100", "E3: masked KL"),
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def family_of(arm: str) -> str:
    if arm.startswith("e1"):
        return "e1"
    if arm.startswith("e2"):
        return "e2"
    if arm.startswith("e3"):
        return "e3"
    return "baseline"


def load() -> list[dict]:
    """Per-arm points, keyed by seed so the two axes are paired within a run."""
    out = []
    for d in sorted((REPO / "experiments").iterdir()):
        if not d.is_dir():
            continue
        vals, _ = arm_values(d)
        fve = vals.get("frac_variance_explained", {})
        if not fve:
            continue
        seeds = set(fve)
        for k in THRESHOLDS:
            seeds &= set(vals.get(k, {}))
        seeds = sorted(seeds)
        if not seeds:
            continue
        out.append({
            "arm": d.name,
            "family": family_of(d.name),
            "seeds": seeds,
            "fve": [fve[s] for s in seeds],
            **{k: [vals[k][s] for s in seeds] for k in THRESHOLDS},
        })
    return out


def spearman(xs, ys):
    from scipy.stats import spearmanr

    r = spearmanr(xs, ys)
    return float(r.statistic), float(r.pvalue)


def cut_scan(rows: list[dict], k: str, min_side: int = 3) -> list[tuple]:
    """Correlation above and below every possible cut between adjacent arms.

    A single "healthy vs collapsed" threshold would be chosen after seeing these
    points, and the choice would carry the result. Scanning every cut removes the
    choice: if the sign of the relationship above the cut is positive for every
    cut that leaves enough arms on each side, that is a property of the data and
    not of a threshold.
    """
    ordered = sorted(rows, key=lambda r: mean(r["fve"]))
    out = []
    for i in range(min_side, len(ordered) - min_side + 1):
        below, above = ordered[:i], ordered[i:]
        if len(above) < min_side:
            continue
        cut = (mean(ordered[i - 1]["fve"]) + mean(ordered[i]["fve"])) / 2.0
        rho_b, _ = spearman([mean(r["fve"]) for r in below], [mean(r[k]) for r in below])
        rho_a, _ = spearman([mean(r["fve"]) for r in above], [mean(r[k]) for r in above])
        out.append((cut, len(below), rho_b, len(above), rho_a))
    return out


def widest_gap(rows: list[dict], above: float) -> tuple[float, float]:
    """The widest gap in arm-mean FVE above a floor -- used only to place the
    descriptive cut and the figure's shading, never to produce a statistic."""
    means = sorted(m for m in (mean(r["fve"]) for r in rows) if m > above)
    gaps = [(b - a, a, b) for a, b in zip(means, means[1:])]
    _, lo, hi = max(gaps)
    return lo, hi


def report(rows: list[dict]) -> dict:
    stat_rows = [r for r in rows if len(r["seeds"]) >= MIN_SEEDS_FOR_STATS]
    # Descriptive only: the widest gap among arms that are not fully collapsed.
    # It places the figure's shading and names the "working models" subset; every
    # statistic that depends on a cut is reported across ALL cuts instead.
    lo, hi = widest_gap(stat_rows, above=0.01)
    split = (lo + hi) / 2.0

    print(f"{'arm':<24}{'n':>3}{'FVE':>12}{'<0.1x k/d':>12}{'<0.5x k/d':>12}")
    print("-" * 66)
    for r in sorted(rows, key=lambda r: -mean(r["fve"])):
        n = len(r["seeds"])
        tag = "" if n >= MIN_SEEDS_FOR_STATS else "   (1 seed, context only)"
        print(f"{r['arm']:<24}{n:>3}{mean(r['fve']):>12.4f}"
              f"{mean(r['below_0.1x']):>12.4f}{mean(r['below_0.5x']):>12.4f}{tag}")

    print("\n== The relationship over the whole range ==\n")
    for k in THRESHOLDS:
        am_x = [mean(r["fve"]) for r in stat_rows]
        am_y = [mean(r[k]) for r in stat_rows]
        cl_x = [v for r in stat_rows for v in r["fve"]]
        cl_y = [v for r in stat_rows for v in r[k]]
        rho_a, p_a = spearman(am_x, am_y)
        rho_c, _ = spearman(cl_x, cl_y)
        print(f"   {k:<12} arm-mean rho = {rho_a:+.3f}  (p = {p_a:.3g}, "
              f"n = {len(stat_rows)} arms)"
              f"   [checkpoint-level rho = {rho_c:+.3f}, pseudo-replicated]")
    print("\n   Read alone this says the two metrics are unrelated. They are not --\n"
          "   the scan below shows it is two opposite-signed relationships cancelling.\n")

    print("== Every possible cut, so no threshold is doing the work ==\n")
    signs = {}
    for k in THRESHOLDS:
        print(f"   {k}")
        print(f"      {'cut on FVE':>12}{'n below':>9}{'rho below':>11}"
              f"{'n above':>9}{'rho above':>11}")
        rows_scan = cut_scan(stat_rows, k)
        for cut, nb, rb, na, ra in rows_scan:
            print(f"      {cut:>12.4f}{nb:>9}{rb:>+11.3f}{na:>9}{ra:>+11.3f}")
        signs[k] = (all(rb < 0 for _, _, rb, _, _ in rows_scan),
                    all(ra > 0 for _, _, _, _, ra in rows_scan))
        print()
    for k, (neg_below, pos_above) in signs.items():
        print(f"   {k}: rho below the cut is negative at every cut: {neg_below}; "
              f"rho above is positive at every cut: {pos_above}")

    working = [r for r in stat_rows if mean(r["fve"]) > split]
    print(f"\n== Working models only ({len(working)} arms above FVE {split:.3f}) ==\n")
    for k in THRESHOLDS:
        rho, p = spearman([mean(r["fve"]) for r in working],
                          [mean(r[k]) for r in working])
        print(f"   {k:<12} arm-mean rho = {rho:+.3f}  (p = {p:.3g}, n = {len(working)})")

    fam = [r for r in stat_rows if r["arm"] == "e1_penalty" or r["arm"].startswith("e1_vsae")]
    if len(fam) >= 3:
        print(f"\n== E1 family alone ({len(fam)} arms, identical objective, only "
              f"optimiser/init details differ) ==\n")
        for k in THRESHOLDS:
            rho, p = spearman([mean(r["fve"]) for r in fam], [mean(r[k]) for r in fam])
            print(f"   {k:<12} arm-mean rho = {rho:+.3f}  (p = {p:.3g}, n = {len(fam)})")

    # Dominance. On a genuine frontier every pair TRADES: better on one axis means
    # worse on another. A pair where one arm is better on all three axes is a pair
    # no frontier can accommodate, and the dominated arm is strictly wasteful.
    def dominance(subset):
        out = []
        for a in subset:
            for b in subset:
                if a is b:
                    continue
                if (mean(a["fve"]) > mean(b["fve"])
                        and all(mean(a[k]) < mean(b[k]) for k in THRESHOLDS)):
                    out.append((a["arm"], b["arm"]))
        return out

    print("\n== Dominance: better FVE and fewer near-dead at BOTH thresholds ==\n")
    for label, subset in (("all replicated arms", stat_rows),
                          (f"working models only (FVE > {split:.3f})", working)):
        dom = dominance(subset)
        pairs = len(subset) * (len(subset) - 1) // 2
        print(f"   {label}: {len(dom)} of {pairs} pairs dominated "
              f"({100 * len(dom) / pairs:.0f}%) -- {pairs - len(dom)} trade")
        for a, b in dom:
            print(f"      {a}  dominates  {b}")
        print()

    return {"split": split, "gap": (lo, hi), "working": [r["arm"] for r in working]}


# Short labels for the zoom panel. The full arm names are in the printed table;
# on the plot what matters is which factor each arm adds.
ZOOM_LABELS = {
    "baseline": "baseline (no penalty)",
    "e3_masked_kl": "masked KL",
    "e3_masked_kl_relu": "masked KL + ReLU",
    "e1_vsae_ref": "vSAE ref",
    "e1_vsae_ref_fullmatch": "+ init draw (fully matched)",
    "e1_penalty": "TopK + L2",
    "e1_vsae_ref_gradproj": "+ grad projection",
    "e1_vsae_ref_unitinit": "+ init scale",
}
# Hand-placed label offsets in points, per panel -- the arms sit in different
# places at the two thresholds, so one set of offsets cannot serve both.
# e1_penalty and _fullmatch land on top of each other (which IS E1's result), so
# their labels are pushed apart rather than left to overlap.
LABEL_OFFSETS = {
    "below_0.1x": {
        "baseline": (-8, 6), "e3_masked_kl": (-8, 6), "e3_masked_kl_relu": (-8, 6),
        "e1_vsae_ref": (8, -2), "e1_vsae_ref_fullmatch": (8, 5),
        "e1_penalty": (8, -13), "e1_vsae_ref_gradproj": (8, 3),
        "e1_vsae_ref_unitinit": (4, -14),
    },
    "below_0.5x": {
        "baseline": (-8, 5), "e3_masked_kl": (-8, 5), "e3_masked_kl_relu": (-8, 7),
        "e1_vsae_ref": (8, -3), "e1_vsae_ref_fullmatch": (9, 4),
        "e1_penalty": (9, -16), "e1_vsae_ref_gradproj": (-4, 10),
        "e1_vsae_ref_unitinit": (2, -15),
    },
}

# Axis limits per threshold. Set from the data rather than shared, because the
# loose threshold's smallest value is 0.088 and a shared scale would leave two
# thirds of that panel empty.
YLIM = {
    "below_0.1x": {"full": (-RESOLUTION / 2, 2.4), "zoom": (-RESOLUTION / 2, 1.5)},
    "below_0.5x": {"full": (0.05, 2.4), "zoom": (0.06, 1.15)},
}


def figure(rows: list[dict], split: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    marker_of, per_family = {}, {}
    for r in sorted(rows, key=lambda r: r["arm"]):
        f = r["family"]
        marker_of[r["arm"]] = MARKERS[per_family.get(f, 0) % len(MARKERS)]
        per_family[f] = per_family.get(f, 0) + 1

    stat_rows = [r for r in rows if len(r["seeds"]) >= MIN_SEEDS_FOR_STATS]
    working = [r for r in stat_rows if mean(r["fve"]) > split]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    for row, k in enumerate(THRESHOLDS):
        # ---- left: the whole range, where the sign reverses -------------------
        ax = axes[row][0]
        ax.axvspan(-0.03, split, color="#000000", alpha=0.045, lw=0)
        for r in sorted(rows, key=lambda r: r["arm"]):
            colour = FAMILIES[r["family"]][0]
            replicated = len(r["seeds"]) >= MIN_SEEDS_FOR_STATS
            ax.scatter([mean(r["fve"])], [mean(r[k])], s=46 if replicated else 20,
                       color=colour, marker=marker_of[r["arm"]],
                       alpha=1.0 if replicated else 0.45,
                       edgecolors="white", linewidths=0.8 if replicated else 0,
                       zorder=3)
        ax.set_yscale("symlog", linthresh=RESOLUTION)
        ax.set_ylim(*YLIM[k]["full"])
        ax.set_xlim(-0.03, 0.97)
        if k == "below_0.1x":
            ax.axhline(RESOLUTION, color="#52514e", lw=0.7, ls=":", zorder=1)
        ax.annotate("collapsed", (split / 2, 1.55), ha="center", fontsize=8,
                    color="#52514e")
        ax.annotate("working", ((split + 0.97) / 2, 1.55), ha="center", fontsize=8,
                    color="#52514e")
        ax.annotate("rho < 0", (split / 2, 0.30), ha="center", fontsize=9,
                    color="#52514e", style="italic")
        ax.annotate("rho > 0", ((split + 0.97) / 2, 0.30), ha="center", fontsize=9,
                    color="#52514e", style="italic")
        ax.set_ylabel(f"near-dead fraction\n({k.replace('below_', '< ')} of k/d)")
        ax.set_title("Full range" if row == 0 else "", fontsize=10, color="#52514e")

        # ---- right: the regime where comparisons are actually made ------------
        ax = axes[row][1]
        xs = [mean(r["fve"]) for r in working]
        ys = [mean(r[k]) for r in working]
        rho, p = spearman(xs, ys)
        for r in working:
            colour = FAMILIES[r["family"]][0]
            mx, my = mean(r["fve"]), mean(r[k])
            ax.scatter(r["fve"], r[k], s=10, color=colour, alpha=0.30,
                       marker=marker_of[r["arm"]], linewidths=0, zorder=2)
            ax.errorbar(mx, my, xerr=stdev(r["fve"]),
                        yerr=[[min(stdev(r[k]), my)], [stdev(r[k])]],
                        fmt=marker_of[r["arm"]], color=colour, ms=8,
                        mec="white", mew=1.0, elinewidth=1.2, capsize=0, zorder=3)
            dx, dy = LABEL_OFFSETS[k].get(r["arm"], (6, 6))
            ax.annotate(ZOOM_LABELS.get(r["arm"], r["arm"]), (mx, my),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.5,
                        color="#52514e", ha="right" if dx < 0 else "left")
        ax.set_yscale("symlog", linthresh=RESOLUTION)
        ax.set_ylim(*YLIM[k]["zoom"])
        ax.set_xlim(0.820, 0.914)
        if k == "below_0.1x":
            ax.axhline(RESOLUTION, color="#52514e", lw=0.7, ls=":", zorder=1)
        ax.annotate(f"Spearman rho = {rho:+.2f}  (p = {p:.3f}, n = 8 arms)",
                    (0.03, 0.93), xycoords="axes fraction", fontsize=8.5,
                    color="#0b0b0b")
        ax.set_title("Working models only — better reconstruction buys MORE "
                     "near-dead features" if row == 0 else "",
                     fontsize=10, color="#52514e")

        for ax in axes[row]:
            ax.grid(alpha=0.22, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("fraction of variance explained")

    axes[0][0].annotate("one feature of 2048", (0.0, RESOLUTION),
                        xytext=(3, 5), textcoords="offset points",
                        fontsize=7, color="#52514e")

    handles = [Line2D([], [], marker="o", color=c, ls="", ms=7, label=lab)
               for c, lab in FAMILIES.values()]
    handles.append(Line2D([], [], marker="o", color="#52514e", ls="", ms=5,
                          alpha=0.45, label="1 seed (E2 beta pilot, context only)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("No single liveness–reconstruction frontier: the relationship "
                 "reverses sign between regimes", fontsize=12, y=1.0)
    fig.tight_layout()
    out = REPO / "workshop" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    # Omit the PDF CreationDate so regenerating the figure is byte-reproducible.
    fig.savefig(out / "frontier.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out / "frontier.png", dpi=180, bbox_inches="tight")
    print(f"\nWrote {out / 'frontier.pdf'} and {out / 'frontier.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fig", action="store_true", help="statistics only")
    args = ap.parse_args()

    rows = load()
    if not rows:
        print("no analysed arms found under experiments/")
        return 1
    res = report(rows)
    if not args.no_fig:
        figure(rows, res["split"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
