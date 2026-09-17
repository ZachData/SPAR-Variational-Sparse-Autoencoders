"""The one dose-response figure the paper uses four times (TopK, BatchTopK,
JumpReLU, JumpReLU densified): reconstruction (FVE) and selection stability
(Jaccard) against the reparameterisation noise scale, plus the FVE-vs-Jaccard
scatter with its Pearson r.

Shared by `read_a{2,3,4}_dose_response.py` (which measure and then plot) and
`reproduce.py` (which re-plots from the cached `*_results.json`), so the paper's
figures have exactly one definition. Sized at the paper's full text width
(6.5in) so fonts render at their nominal size instead of being scaled down.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

# Categorical slots 1 and 2 of the validated default palette (adjacent-pair
# CVD-safe), with marker shape as the secondary encoding. Text stays in ink.
FVE_COLOR = "#2a78d6"
JACCARD_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    return cov / (sx * sy)


def dose_response_figure(rows: list[dict], baseline_fve: float | None, *,
                         arch: str, out_stem: Path, show_l0: bool = False,
                         highlight: set[float] | None = None) -> float:
    """Write `<out_stem>.pdf` and `.png`; return the Pearson r(FVE, Jaccard).

    `rows` are the reader's per-grid-point dicts (`sigma`, `fve_mean`, `fve_std`,
    `jaccard_mean`, `jaccard_std`, optionally `l0_mean`, `n_seeds`). `show_l0`
    labels each scatter point with its achieved L0 as well as sigma -- the
    JumpReLU confound the paper's text is about. `highlight` marks a subset of
    `log_var_init` values (the densifying follow-up points) with open markers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    rows = sorted(rows, key=lambda r: -r["sigma"])
    sigma = [r["sigma"] for r in rows]
    fve = [r["fve_mean"] for r in rows]
    fve_sd = [r["fve_std"] for r in rows]
    jac = [r["jaccard_mean"] for r in rows]
    jac_sd = [r["jaccard_std"] for r in rows]
    r_fj = pearson(fve, jac)
    highlight = highlight or set()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.9))

    # Left: both quantities are fractions in [0, 1], so they share one axis.
    ax.errorbar(sigma, fve, yerr=fve_sd, fmt="o-", color=FVE_COLOR, ms=5, lw=1.6,
                capsize=2, label="FVE")
    ax.errorbar(sigma, jac, yerr=jac_sd, fmt="s-", color=JACCARD_COLOR, ms=5, lw=1.6,
                capsize=2, label="selection Jaccard")
    if baseline_fve is not None:
        ax.axhline(baseline_fve, color=INK_SECONDARY, lw=0.8, ls=":")
        below = jac[0] > baseline_fve      # JumpReLU's Jaccard sits above the line
        ax.annotate(f"baseline FVE = {baseline_fve:.3f}",
                    (sigma[0], baseline_fve), xytext=(0, -9 if below else 4), ha="left",
                    textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.invert_xaxis()
    # One tick per grid point; the default log locator labels only 1e-1.
    ticks = sorted(set(round(x, 3) for x in sigma), reverse=True)   # -8 and -6 share sigma
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in ticks], fontsize=7,
                       rotation=45 if len(ticks) > 6 else 0)
    ax.minorticks_off()
    ax.set_xlabel("noise scale $\\sigma$ (log; noise decreases $\\rightarrow$)")
    ax.set_ylabel("value")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(f"{arch}: FVE and Jaccard vs. $\\sigma$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: the coupling itself.
    for r in rows:
        new = r["log_var_init"] in highlight
        bx.errorbar([r["jaccard_mean"]], [r["fve_mean"]], xerr=[r["jaccard_std"]],
                    yerr=[r["fve_std"]], fmt="o", ms=5.5, color=INK,
                    mfc="white" if new else INK, capsize=2)
    labels = []
    for r in rows:
        label = f"$\\sigma$={r['sigma']:.2f}"
        if show_l0 and r.get("l0_mean") is not None:
            label += f", $L_0$={r['l0_mean']:.0f}"
        if r["log_var_init"] < -6:      # reparameterize() clamps log_var to [-6, 2]
            label += " (clamped)"
        labels.append(label)
    if show_l0:
        # JumpReLU's points bunch within a few thousandths of Jaccard, so
        # in-place labels collide. Stack them in a column at the left of the
        # panel, in FVE order, with a leader line to each point.
        y_lo, y_hi = 0.3, 1.0
        gap = 0.052 * (y_hi - y_lo)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["fve_mean"])
        ys = []
        for i in order:
            y = rows[i]["fve_mean"]
            if ys:
                y = max(y, ys[-1] + gap)
            ys.append(y)
        overflow = ys[-1] - (y_hi - 0.03)
        if overflow > 0:
            ys = [y - overflow for y in ys]
        x_col = min(jac) + 0.12 * (max(jac) - min(jac))
        for i, y in zip(order, ys):
            bx.annotate(labels[i], (rows[i]["jaccard_mean"], rows[i]["fve_mean"]),
                        xytext=(x_col, y), textcoords="data", ha="left", va="center",
                        fontsize=7, color=INK_SECONDARY,
                        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.5,
                                    "shrinkA": 0, "shrinkB": 4})
    else:
        # Alternate label sides so near-coincident low-noise points stay legible.
        for i, r in enumerate(rows):
            side = (5, -3, "left") if i % 2 == 0 else (-5, 4, "right")
            bx.annotate(labels[i], (r["jaccard_mean"], r["fve_mean"]), xytext=side[:2],
                        ha=side[2], textcoords="offset points", fontsize=7, color=INK_SECONDARY)
    if baseline_fve is not None:
        bx.axhline(baseline_fve, color=INK_SECONDARY, lw=0.8, ls=":")
    bx.set_xlabel("selection Jaccard at convergence")
    bx.set_ylabel("fraction of variance explained")
    bx.set_ylim(0.3, 1.0)
    # The r annotation goes wherever the points are not: top-left for the
    # hard-k architectures, bottom-right for JumpReLU's column-labelled panel.
    pos, ha = ((0.97, 0.40), "right") if show_l0 else ((0.03, 0.93), "left")
    bx.annotate(f"Pearson $r$ = {r_fj:+.4f}  ($n$ = {len(rows)} grid points)",
                pos, xycoords="axes fraction", fontsize=8, color=INK, ha=ha)
    if highlight:
        bx.annotate("open markers: 5-seed densifying points", (pos[0], pos[1] + 0.07),
                    xycoords="axes fraction", fontsize=7.5, color=INK_SECONDARY, ha=ha)
    bx.set_title(f"{arch}: FVE against Jaccard")
    bx.grid(alpha=0.25, lw=0.6)
    bx.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # No CreationDate: regenerating an identical figure leaves the file unchanged.
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return r_fj
