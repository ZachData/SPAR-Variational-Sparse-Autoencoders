#!/usr/bin/env python3
"""The explanatory figures of workshop/mechanism_paper.tex, built from the data
committed in this repository (the same loaders `reproduce.py` checks the paper's
numbers with). No GPU, no torch.

    python workshop/make_paper_figs.py        # writes workshop/figs/fig_*.{pdf,png}

`reproduce.py --figures` runs this too. The four dose-response figures are drawn
by `falsification/dose_response_figure.py`; this file draws everything else:

  fig_e1_ladder        Sec. 3  the decomposition ladder, per seed and as d
  fig_e2_split         Sec. 4.1 sampling, not the KL, costs the reconstruction
  fig_selection_churn  Sec. 4.2 how noise before Top-K changes the selected set
  fig_synthesis        Sec. 4.5 the three selection mechanisms on common axes
  fig_impl_variance    Sec. 5  six one-line details, and the projection null
  fig_seed_ceiling     Sec. 5.2 the strongest evidence n seeds can express
  fig_e4               Sec. 6  SCR and TPP on the same two checkpoints

fig_selection_churn reads falsification/selection_example_results.json, cached
by the GPU script falsification/read_selection_example.py.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import reproduce as R  # noqa: E402

FIGS = REPO / "workshop" / "figs"
FALS = REPO / "falsification"

# Validated default palette (dataviz skill, references/palette.md), light mode.
# Slots 1-3 validate all-pairs, so they are safe for scatter and small multiples;
# marker shape is the secondary encoding everywhere colour carries identity.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.titlelocation": "left", "axes.titleweight": "bold",
    "pdf.fonttype": 42, "lines.linewidth": 1.6,
})


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def save(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGS / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote workshop/figs/{stem}.{{pdf,png}}")


def jitter(n, width=0.22, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


def strip(ax, x, vals, color, filled=True, ms=3.2):
    ax.plot(x + jitter(len(vals), seed=int(x * 7) + 1), vals, "o", ms=ms, color=color,
            mfc=color if filled else "white", mew=0.8, alpha=0.9, zorder=3)
    ax.plot([x - 0.32, x + 0.32], [mean(vals)] * 2, color=INK, lw=1.4, zorder=4)


# --------------------------------------------------------------------------
# Sec. 3: E1 ladder
# --------------------------------------------------------------------------

# Rung names as the caption defines them: R0 = vSAE as released, then one
# factor matched per rung (init scale, gradient projection, weight draw).
E1_ARMS = [("e1_penalty", "null"),
           ("e1_vsae_ref", "R0"),
           ("e1_vsae_ref_unitinit", "R1"),
           ("e1_vsae_ref_gradproj", "R2"),
           ("e1_vsae_ref_fullmatch", "R3")]


def fig_e1_ladder():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, metric, lab in [(axes[0], "frac_variance_explained", "fraction of variance explained"),
                            (axes[1], "below_0.5x", "near-dead fraction ($<0.5\\times k/d$)")]:
        null = R.per_seed("e1_penalty", metric)
        lo, hi = min(null), max(null)
        ax.axhspan(lo, hi, color=GRID, lw=0, zorder=0)
        for i, (a, _) in enumerate(E1_ARMS):
            strip(ax, i, R.per_seed(a, metric), INK2 if i == 0 else BLUE, filled=i != 0)
        ax.set_xticks(range(len(E1_ARMS)))
        ax.set_xticklabels([l for _, l in E1_ARMS], fontsize=7.5)
        ax.set_ylabel(lab)
        style(ax)
        ax.tick_params(axis="x", length=0)
    axes[0].set_title("(a) FVE, per seed")
    axes[1].set_title("(b) Liveness, per seed")

    # (c) d against the null model, all four pre-registered metrics.
    ax = axes[2]
    metrics = [("frac_variance_explained", "FVE", BLUE, "o"), ("frac_recovered", "frac. rec.", AQUA, "s"),
               ("below_0.1x", "$<0.1\\times$", ORANGE, "^"), ("below_0.5x", "$<0.5\\times$", YELLOW, "D")]
    xs = range(1, len(E1_ARMS))
    for j, (m, lab, col, mk) in enumerate(metrics):
        cs = [R.compare("e1_penalty", a, m) for a, _ in E1_ARMS[1:]]
        ds = [c["d"] for c in cs]
        dx = [x + (j - 1.5) * 0.07 for x in xs]     # dodge so coincident markers stay visible
        ax.plot(dx, ds, "-", color=col, lw=1.3, zorder=2)
        for x, c in zip(dx, cs):
            sig = c["p"] <= 0.05
            ax.plot(x, c["d"], mk, ms=5, color=col, mfc=col if sig else "white", mew=1.1, zorder=3)
        ax.plot([], [], mk + "-", color=col, ms=4.5, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l for _, l in E1_ARMS[1:]], fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Cohen's $d$ (null model $-$ rung)")
    ax.set_title("(c) $d$ vs. null model")
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.8)
    ax.set_ylim(-11, 24)
    ax.annotate("open marker: $p>0.05$", (0.97, 0.03), xycoords="axes fraction", ha="right",
                fontsize=6.5, color=INK2)
    style(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_e1_ladder")


# --------------------------------------------------------------------------
# Sec. 4.1: sampling, not the KL
# --------------------------------------------------------------------------

def fig_e2_split():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    base = R.regime_vals("baseline", "fve")
    b = mean(base)

    # (a) FVE vs beta with sampling on. Pilots are one seed each; beta=1e-4 and
    # beta=0 are the 13-seed confirmatory arms. beta=0 sits on its own tick.
    betas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    pil = [R.regime_vals(f"e2_beta_pilot_{x:g}", "fve")[0] for x in betas]
    pos = {x: i + 1 for i, x in enumerate(betas)}
    ax.axhline(b, color=INK2, lw=0.8, ls=":")
    ax.annotate(f"deterministic baseline {b:.3f}", (5, b), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7, color=INK2)
    ax.plot([pos[x] for x in betas], pil, "o-", color=BLUE, mfc="white", mew=1.1, ms=5, lw=1.3,
            label="pilot (1 seed each)")
    for a, x in [("e2_sampling_only", 0), ("e2_confirm", pos[1e-4])]:
        v = R.regime_vals(a, "fve")
        ax.plot(x + jitter(len(v), 0.12, seed=x + 3), v, "o", ms=2.8, color=BLUE, alpha=0.8, zorder=3)
    ax.plot([], [], "o", color=BLUE, ms=3, label="confirmatory (13 seeds)")
    ax.axvline(0.5, color=GRID, lw=0.8)
    ax.set_xticks([0] + [pos[x] for x in betas])
    ax.set_xticklabels(["0\n(no KL)", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"])
    ax.set_xlabel("KL coefficient $\\beta$   (sampling on, $\\sigma$ learned)")
    ax.set_ylabel("fraction of variance explained")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlim(-0.5, 5.4)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_title("(a) No $\\beta$ recovers the baseline")
    style(ax)

    # (b) where the gap goes: baseline -> beta=0 -> beta=1e-4, per seed.
    arms = [("baseline", "deterministic\nbaseline", INK2), ("e2_sampling_only", "sampling on\n$\\beta=0$", BLUE),
            ("e2_confirm", "sampling on\n$\\beta=10^{-4}$", BLUE)]
    m = {}
    for i, (a, lab, col) in enumerate(arms):
        v = R.regime_vals(a, "fve")
        m[a] = mean(v)
        strip(bx, i, v, col, filled=True, ms=3)
    gap = m["baseline"] - m["e2_confirm"]
    rep = m["baseline"] - m["e2_sampling_only"]
    kl = m["e2_sampling_only"] - m["e2_confirm"]

    def bracket(x, y0, y1, text, side=1):
        bx.annotate("", (x, y0), (x, y1), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9,
                                                       "shrinkA": 0, "shrinkB": 0})
        bx.annotate(text, (x + 0.08 * side, (y0 + y1) / 2), ha="left" if side > 0 else "right",
                    va="center", fontsize=7, color=INK)
    bracket(0.55, m["baseline"], m["e2_sampling_only"],
            f"reparameterisation\n{rep:.3f} ({100 * rep / gap:.1f}%)")
    bx.plot([0, 0.55], [m["baseline"]] * 2, color=MUTED, lw=0.6, ls=":")
    bx.plot([0.55, 1], [m["e2_sampling_only"]] * 2, color=MUTED, lw=0.6, ls=":")
    bracket(2.45, m["e2_sampling_only"], m["e2_confirm"], f"KL\n{kl:.3f}\n({100 * kl / gap:.1f}%)")
    bx.plot([1, 2.45], [m["e2_sampling_only"]] * 2, color=MUTED, lw=0.6, ls=":")
    bx.plot([2, 2.45], [m["e2_confirm"]] * 2, color=MUTED, lw=0.6, ls=":")
    bx.set_xticks(range(3))
    bx.set_xticklabels([l for _, l, _ in arms], fontsize=7)
    bx.tick_params(axis="x", length=0)
    bx.set_xlim(-0.5, 3.1)
    bx.set_ylim(0.28, 0.95)
    bx.set_ylabel("fraction of variance explained")
    bx.set_title(f"(b) The {gap:.3f} gap, split")
    style(bx)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_e2_split")


# --------------------------------------------------------------------------
# Sec. 4.2: selection churn, illustrated and then measured
# --------------------------------------------------------------------------

def fig_selection_churn(dose):
    """Real data from falsification/selection_example_results.json
    (`read_selection_example.py`, GPU): the a2_sigma_init_m3 seed-1 checkpoint in
    the regime it was trained in (panel b adds the same checkpoint as saved, on
    raw activations). Values are in training-space units."""
    ex = json.loads((FALS / "selection_example_results.json").read_text())
    reg = ex["regimes"]["training"]
    saved = ex["regimes"]["raw"]
    t, k = reg["token"], reg["k"]
    fig = plt.figure(figsize=(6.6, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1, 1], hspace=0.62, wspace=0.38)
    ax, cx, bx = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) one token, the features ranked around the k-th place.
    scale = 1e3
    ranks = np.array(t["ranks"]) + 1                 # 1-based: rank k is the last one Top-k keeps
    mu = np.array(t["mu"]) * scale
    ax.hlines(mu, ranks - 0.36, ranks + 0.36, color=INK2, lw=1.6, label="$\\mu$ (deterministic)")
    for j, col, mk, off in [(1, BLUE, "o", -0.17), (2, ORANGE, "s", 0.17)]:
        z = np.array(t[f"z{j}"]) * scale
        sel = np.array(t[f"selected{j}"])
        ax.plot(ranks[sel] + off, z[sel], mk, color=col, ms=4, label=f"pass {j}: selected")
        ax.plot(ranks[~sel] + off, z[~sel], mk, color=col, mfc="white", ms=4, mew=0.9)
    allz = np.concatenate([mu, np.array(t["z1"]) * scale, np.array(t["z2"]) * scale])
    pad = 0.08 * (allz.max() - allz.min())
    lo, hi = allz.min() - 2 * pad, allz.max() + pad
    ax.set_ylim(lo, hi)
    flips = ranks[np.array(t["selected1"]) != np.array(t["selected2"])]
    ax.plot(flips, [lo + 0.6 * pad] * len(flips), "^", color=INK, ms=3.5, label="selected by one pass only")
    ax.axvline(k + 0.5, color=INK2, lw=0.8, ls="--")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.text(k + 0.8, hi - 0.1 * pad, f"rank $k$ = {k}: the last one\nTop-$k$ keeps without noise",
            fontsize=6.4, color=INK2, va="top")
    ax.set_xlim(ranks[0] - 0.7, ranks[-1] + 0.7)
    ax.set_xticks(ranks[::2])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("feature's rank by $\\mu$ for this token")
    ax.set_ylabel("pre-activation ($\\times10^{-3}$)")
    ax.legend(frameon=False, loc="lower left", ncol=4, fontsize=6.5, handlelength=1.2, columnspacing=1.0,
              bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    n_in = sum(1 for r in t["flip_ranks"] if t["ranks"][0] <= r <= t["ranks"][-1])
    ax.set_title(f"(a) One real token: {t['n_flips'] // 2} of the {k} features one pass keeps, "
                 f"the other drops (Jaccard {t['jaccard']:.3f})", pad=16)
    style(ax)

    # (b) where in the ranking churn happens, over all tokens.
    for rg, col, ls, lab in [(reg, BLUE, "-", "training regime"), (saved, INK2, "--", "as saved (raw acts)")]:
        fr = np.array(rg["flip_rate_by_rank"])
        r = np.arange(1, len(fr) + 1)
        cx.plot(r, fr, color=col, lw=1.3, ls=ls, label=f"{lab}: Jaccard {rg['mean_jaccard']:.2f}")
    cx.fill_between(r, 0, np.array(reg["flip_rate_by_rank"]), color=BLUE, alpha=0.15, lw=0)
    cx.axvline(k + 0.5, color=INK2, lw=0.8, ls=":")
    cx.set_xscale("log")
    cx.set_xlim(1, 2048)
    cx.set_ylim(0, 0.6)
    cx.set_xticks([1, 16, k, 2048])
    cx.set_xticklabels(["1", "16", str(k), "2048"])
    cx.minorticks_off()
    cx.set_xlabel("feature's rank by $\\mu$ (log)")
    cx.set_ylabel("P(selected by one pass only)")
    cx.set_title(f"(b) Where churn happens, {ex['n_tokens']:,} tokens")
    cx.legend(frameon=False, loc="upper right", fontsize=6.2, handlelength=1.8)
    style(cx)

    # (c) Measured scales on one log axis: the k-th/(k+1)-th gap of a converged
    # deterministic TopK (read_preact_gap.py, quoted in Sec. 4.2 and run_arm.py)
    # against every sigma the sweep could reach.
    g = json.loads((FALS / "preact_gap_results.json").read_text())   # read_preact_gap.py, GPU
    gap = {"median": g["percentiles"]["50"], "99th pct.": g["percentiles"]["99"], "max": g["max"]}
    sig = sorted({round(r["sigma"], 4) for r in dose["a2"]})
    bx.set_xscale("log")
    bx.set_xlim(5e-5, 1.2)
    bx.set_ylim(-0.2, 2.2)
    bx.axvspan(5e-5, gap["max"], color=GRID, lw=0)
    for i, (lab, v) in enumerate(gap.items()):
        bx.plot(v, 1.55, "|", ms=14, mew=1.6, color=INK2)
        bx.annotate(lab, (v, 1.55), xytext=(0, 10 if i != 1 else -17), textcoords="offset points",
                    ha="center", fontsize=6.3, color=INK2)
    bx.text(6e-5, 1.98, "$k$-th/($k$+1)-th gap, converged TopK", ha="left", fontsize=7, color=INK)
    bx.plot(sig, [0.55] * len(sig), "o", color=BLUE, ms=5)
    bx.annotate("clamp floor\n$\\sigma=0.0498$", (sig[0], 0.55), xytext=(0, -24), textcoords="offset points",
                ha="center", fontsize=6.5, color=INK2)
    bx.text(0.17, 0.9, "noise scales in the sweep", ha="center", fontsize=7, color=INK)
    bx.annotate("", (gap["99th pct."], 1.15), (sig[0], 1.15),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.8})
    bx.text(math.sqrt(gap["99th pct."] * sig[0]), 1.2, "$\\gtrsim 80\\times$",
            ha="center", va="bottom", fontsize=7, color=INK)
    bx.set_yticks([])
    bx.spines[["left", "top", "right"]].set_visible(False)
    bx.set_xlabel("scale (training space, log)")
    bx.set_title("(c) Measured scales")
    save(fig, "fig_selection_churn")


# --------------------------------------------------------------------------
# Sec. 4.5: three selection mechanisms on common axes
# --------------------------------------------------------------------------

ARCHS = [("a2", "TopK", BLUE, "o"), ("a3", "BatchTopK", ORANGE, "s"), ("a4", "JumpReLU", AQUA, "^")]


def fig_synthesis(dose):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55))
    rows10, _ = dose["a4_followup"]
    data = {"a2": dose["a2"], "a3": dose["a3"], "a4": rows10}
    for key, name, col, mk in ARCHS:
        rows = sorted(data[key], key=lambda r: -r["sigma"])
        b = dose["baselines"][key]
        s = [r["sigma"] for r in rows]
        frac = [100 * r["fve_mean"] / b for r in rows]
        axes[0].plot(s, frac, mk + "-", color=col, ms=4, lw=1.3, label=name)
        axes[1].plot([r["jaccard_mean"] for r in rows], frac, mk, color=col, ms=4.5, label=name)
        if key == "a4":
            axes[2].plot(s, [r["l0_mean"] for r in rows], mk + "-", color=col, ms=4, lw=1.3, label=name)
    ax = axes[0]
    ax.set_xscale("log"); ax.invert_xaxis(); ax.minorticks_off()
    ax.set_xticks([0.6, 0.3, 0.1, 0.05]); ax.set_xticklabels(["0.6", "0.3", "0.1", "0.05"])
    ax.set_xlabel("noise scale $\\sigma$ (decreasing $\\rightarrow$)")
    ax.set_ylabel("FVE, % of own baseline")
    ax.set_ylim(20, 100)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) Cost vs. noise")
    style(ax, "both")

    bx = axes[1]
    for key, name, col, mk in ARCHS[:2]:
        rr = sorted(data[key], key=lambda x: x["jaccard_mean"])
        if True:
            xs = np.array([x["jaccard_mean"] for x in rr])
            ys = np.array([100 * x["fve_mean"] / dose["baselines"][key] for x in rr])
            fit = np.polyfit(xs, ys, 1)
            bx.plot(xs, np.polyval(fit, xs), color=col, lw=0.9, alpha=0.7)
    r_t = R.pearson([x["fve_mean"] for x in data["a2"]], [x["jaccard_mean"] for x in data["a2"]])
    r_b = R.pearson([x["fve_mean"] for x in data["a3"]], [x["jaccard_mean"] for x in data["a3"]])
    bx.axvline(256 / (2 * 2048 - 256), color=MUTED, lw=0.8, ls=":")
    bx.text(0.075, 21, "chance", fontsize=6.3, color=INK2, ha="left", va="bottom")
    bx.annotate(f"TopK $r={r_t:+.4f}$\nBatchTopK $r={r_b:+.4f}$", (0.34, 23), fontsize=6.8, color=INK)
    bx.annotate("JumpReLU at matched $L_0$:\nJaccard flat, FVE rises",
                (0.70, 86), xytext=(0.27, 92), fontsize=6.6, color=INK, ha="left", va="bottom",
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.5, "shrinkB": 3})
    bx.set_xlabel("selection Jaccard")
    bx.set_ylabel("FVE, % of own baseline")
    bx.set_ylim(20, 100)
    bx.set_xlim(0.0, 0.8)
    bx.set_title("(b) Cost vs. churn")
    style(bx, "both")

    cx = axes[2]
    lo, hi = json.loads((FALS / "a4_followup_results.json").read_text())["l0_band"]
    cx.axhspan(lo, hi, color=GRID, lw=0, zorder=0)
    cx.axhline(256, color=INK2, lw=1.0, ls="--", zorder=1)
    cx.annotate("TopK, BatchTopK:\n$L_0=k=256$ by\nconstruction", (0.09, 256), xytext=(0.15, 120),
                fontsize=6.5, color=INK2, ha="left",
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.5, "shrinkB": 2})
    cx.text(0.045, hi + 6, "$L_0$-matched band", fontsize=6.4, color=INK2, ha="right", va="bottom")
    cx.set_xscale("log"); cx.invert_xaxis(); cx.minorticks_off()
    cx.set_xticks([0.6, 0.3, 0.1, 0.05]); cx.set_xticklabels(["0.6", "0.3", "0.1", "0.05"])
    cx.set_xlabel("noise scale $\\sigma$ (decreasing $\\rightarrow$)")
    cx.set_ylabel("achieved $L_0$ (active features/token)")
    cx.set_ylim(0, 380)
    cx.legend(frameon=False, loc="lower left")
    cx.set_title("(c) Achieved sparsity")
    style(cx, "both")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_synthesis")


# --------------------------------------------------------------------------
# Sec. 5: implementation variance
# --------------------------------------------------------------------------

def fig_impl_variance():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw={"width_ratios": [1.7, 1]})
    C = R.compare
    metrics = [("frac_variance_explained", "FVE", "o"), ("frac_recovered", "frac. rec.", "s"),
               ("below_0.1x", "$<0.1\\times$", "^"), ("below_0.5x", "$<0.5\\times$", "D")]
    labels = {"archive/e1_vsae_ref_aprilmode": "bias form", "archive/e1_vsae_ref_klwarmup1000": "KL warmup",
              "e1_vsae_ref": "decoder init scale", "e1_vsae_ref_unitinit": "gradient projection",
              "e1_vsae_ref_gradproj": "init weight distribution", "e3_masked_kl": "F.relu(mu)"}
    rows = [(labels[a] + (" (6 seeds)" if a.startswith("archive/") else ""), [C(a, b_, m) for m, _, _ in metrics])
            for _, a, b_ in R.IMPL_FACTORS]
    rows.append(("gradient projection\n(penalty-free TopK)",
                 [C("baseline", "claim4_baseline_noproj", m) for m, _, _ in metrics]))
    for i, (lab, cs) in enumerate(rows):
        y = len(rows) - 1 - i
        null = i == len(rows) - 1
        col = INK2 if null else BLUE
        ds = [max(abs(c["d"]), 0.05) for c in cs]
        ax.plot([min(ds), max(ds)], [y, y], color=col, lw=2.0, alpha=0.35, solid_capstyle="round")
        for c, d, (_, _, mk) in zip(cs, ds, metrics):
            sig = c["p"] <= 0.05
            ax.plot(d, y, mk, ms=5, color=col, mfc=col if sig else "white", mew=1.0, zorder=3)
    for _, lab, mk in metrics:
        ax.plot([], [], mk, color=INK2, ms=4.5, label=lab)
    ax.axvline(0.8, color=MUTED, lw=0.8, ls=":")
    ax.text(0.8, len(rows) - 0.45, " $d=0.8$", fontsize=6.3, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 400)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.8)
    ax.get_yticklabels()[0].set_color(INK2)
    ax.set_xlabel("$|d|$, one-factor contrast (log)")
    ax.set_title("(a) One flag or line of code each")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3, ncol=2, handletextpad=0.3, columnspacing=0.8)
    style(ax, "x")
    ax.tick_params(axis="y", length=0)

    # (b) the largest one, per seed: no overlap between arms.
    for j, (m, lab) in enumerate([("below_0.1x", "$<0.1\\times k/d$"), ("below_0.5x", "$<0.5\\times k/d$")]):
        for k_, (a, col, filled) in enumerate([("e3_masked_kl", ORANGE, False), ("e3_masked_kl_relu", BLUE, True)]):
            strip(bx, j * 2.6 + k_, R.per_seed(a, m), col, filled=filled, ms=3)
    bx.set_xticks([0, 1, 2.6, 3.6])
    bx.set_xticklabels(["no\nReLU", "ReLU", "no\nReLU", "ReLU"], fontsize=6.8)
    bx.tick_params(axis="x", length=0)
    bx.text(0.5, 0.56, "$<0.1\\times k/d$", ha="center", fontsize=7, color=INK)
    bx.text(3.1, 0.56, "$<0.5\\times k/d$", ha="center", fontsize=7, color=INK)
    bx.set_ylim(0, 0.6)
    bx.set_ylabel("near-dead fraction of features")
    bx.set_title("(b) F.relu(mu), per seed")
    style(bx)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_impl_variance")


# --------------------------------------------------------------------------
# Sec. 5.2: seed ceiling
# --------------------------------------------------------------------------

def fig_seed_ceiling():
    survey = json.loads((FALS / "seed_count_survey_results.json").read_text())
    ns = np.arange(1, 21)
    ceil = [R.sigma_of(min(2.0 / math.comb(2 * n, n), 1.0)) for n in ns]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.axhline(5, color=INK2, lw=0.8)
    ax.text(20.6, 4.85, "5$\\sigma$", va="top", ha="right", fontsize=7.5, color=INK2)
    for nperm, lab in [(100_000, "Monte Carlo cap, 100k permutations"), (4_000_000, "Monte Carlo cap, 4M (this paper)")]:
        s = R.sigma_of(1 / (nperm + 1))
        ax.axhline(s, color=MUTED, lw=0.8, ls=":")
        ax.text(1.0, s + 0.08, f"{lab}: {s:.2f}$\\sigma$", fontsize=6.6, color=INK2, va="bottom")
    ax.plot(ns, ceil, "o-", color=BLUE, ms=4, lw=1.4)
    n1 = survey["n_papers_seeds_eq_1"]
    ax.annotate(f"$n=1$: {n1} of {survey['n_papers']} surveyed\nSAE-methods papers (0$\\sigma$)",
                (1, ceil[0]), xytext=(2.6, 0.55), fontsize=7, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6, "shrinkB": 4})
    ax.annotate("$n=6$: $3.07\\sigma$", (6, ceil[5]), xytext=(6.6, 2.2), fontsize=7, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6, "shrinkB": 4})
    ax.plot(13, ceil[12], "o", ms=8, mfc="none", mec=INK, mew=1.2)
    ax.annotate("$n=13$: this paper", (13, ceil[12]), xytext=(14.2, 3.6), fontsize=7, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6, "shrinkB": 5})
    ax.set_xlim(0.5, 20.8)
    ax.set_ylim(0, 7.2)
    ax.set_xticks([1, 2, 3, 5, 6, 10, 13, 15, 20])
    ax.set_xlabel("training seeds per arm, $n$")
    ax.set_ylabel("most significance expressible ($\\sigma$)")
    ax.set_title("The most evidence a seed-permutation test can report, at any effect size")
    style(ax, "both")
    fig.tight_layout()
    save(fig, "fig_seed_ceiling")


# --------------------------------------------------------------------------
# Sec. 6: SCR vs TPP
# --------------------------------------------------------------------------

def _curve(d, strategy):
    pts = sorted((c["n_features"], c["mean"], min(c["scores"]), max(c["scores"]))
                 for c in d["baseline_curve"] if c["strategy"] == strategy)
    return [np.array(x) for x in zip(*pts)]


def fig_e4():
    scr = json.loads((FALS / "e4_scr_results.json").read_text())
    tpp = json.loads((FALS / "e4_tpp_results.json").read_text())
    sm = json.loads((FALS / "e4_size_matched_baseline_results.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    for ax, d, name, tag in [(axes[0, 0], scr, "SCR", "scr"), (axes[0, 1], tpp, "TPP", "tpp")]:
        n, m, lo, hi = _curve(d, "random")
        ax.fill_between(n, lo, hi, color=GRID, lw=0, label="baseline, random subsets (range of 10)")
        ax.plot(n, m, color=MUTED, lw=1.0)
        n, m, _, _ = _curve(d, "top_usage")
        ax.plot(n, m, "o-", color=BLUE, ms=3.5, lw=1.4, label="baseline, top-usage mask")
        v = d["vsae"]
        ax.plot(v["live_n"], sm[tag]["score"], "D", color=AQUA, ms=5, mec="white", mew=0.6, zorder=4,
                label="Top-K trained at $d=1474$")
        ax.plot(v["live_n"], v["score"], "*", color=ORANGE, ms=12, mec="white", mew=0.6, zorder=5,
                label="vSAE (1,474 live features)")
        ax.annotate("", (v["live_n"], v["top_usage_reference_at_vsae_n"]), (v["live_n"], v["score"]),
                    arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.8, "shrinkA": 5, "shrinkB": 3})
        mg = d["verdict"]["top_usage"]["margin"]
        verdict = "not explained by size" if mg > 0 else "explained by size"
        ax.annotate(f"margin {mg:+.3f}\n{verdict}", (v["live_n"], (v["score"] + v["top_usage_reference_at_vsae_n"]) / 2),
                    xytext=(9, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("baseline dictionary masked to $N$ features (log)")
        ax.set_ylabel(f"{name} score")
        ax.set_title(f"({'a' if tag == 'scr' else 'b'}) {name}: vSAE {'above' if mg > 0 else 'below'} the size curve")
        style(ax, "both")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=1.4)
    axes[0, 0].set_ylim(-0.09, 0.2)
    axes[0, 1].set_ylim(-0.01, 0.25)

    # (c) bootstrap 95% intervals on the margin, per threshold and the mean.
    ax = axes[1, 0]
    boots = [("SCR, test-set bootstrap", "e4_bootstrap_scr_results.json", "scr_metric_threshold_{}_vs_top_usage", ORANGE, "o"),
             ("SCR, + train resample", "e4_bootstrap_scr_resample_train_results.json",
              "scr_metric_threshold_{}_vs_top_usage", ORANGE, "s"),
             ("TPP, test-set bootstrap", "e4_bootstrap_tpp_results.json",
              "tpp_threshold_{}_total_metric_vs_top_usage", BLUE, "o")]
    labels = ["mean", "$N$=2", "5", "10", "20"]
    for j, (lab, f, key, col, mk) in enumerate(boots):
        mm = json.loads((FALS / f).read_text())["margins_at_vsae_n"]
        keys = ["mean_vs_top_usage"] + [key.format(t) for t in (2, 5, 10, 20)]
        for i, k in enumerate(keys):
            s = mm[k]
            x = i + (j - 1) * 0.22
            ax.plot([x, x], [s["p2.5"], s["p97.5"]], color=col, lw=1.4, alpha=0.8)
            ax.plot(x, s["mean"], mk, color=col, ms=4.2, mfc=col if mk == "o" else "white", mew=1.0)
        ax.plot([], [], mk + "-", color=col, ms=4, mfc=col if mk == "o" else "white", label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0.5, color=GRID, lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_xlabel("features ablated by the metric")
    ax.set_ylabel("vSAE margin over size curve\n(mean, 95% bootstrap interval)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.3)
    ax.set_ylim(-0.25, 0.45)
    ax.set_title("(c) The sign flip survives resampling")
    style(ax)

    # (d) SCR across the 8 (dataset, pair) points, masked vs trained-small reference.
    ax = axes[1, 1]
    pts = []
    for f in sorted(glob.glob(str(FALS / "e4_scr_results*.json"))):
        d = json.loads(Path(f).read_text())
        suffix = Path(f).name[len("e4_scr_results"):]
        smf = FALS / ("e4_size_matched_baseline_results" + suffix)
        s = json.loads(smf.read_text())["scr"]
        ds = "bios" if "bias_in_bios" in d["dataset"] else "amazon"
        pair = "/".join(x.split("_")[0] for x in d["column1_vals"])
        pts.append((f"{ds}: {pair}", d["verdict"]["top_usage"]["margin"], s["vsae_minus_trained_small"]))
    pts.sort(key=lambda p: (p[0].startswith("amazon"), p[0]))
    for i, (lab, a, b) in enumerate(pts):
        y = len(pts) - 1 - i
        ax.plot([a, b], [y, y], color=MUTED, lw=0.8, zorder=1)
        ax.plot(a, y, "o", color=BLUE, ms=4.5, zorder=3)
        ax.plot(b, y, "D", color=AQUA, ms=4.2, mfc="white", mew=1.1, zorder=3)
    ax.plot([], [], "o", color=BLUE, ms=4, label=f"vs. masked curve (mean {mean(p[1] for p in pts):+.3f})")
    ax.plot([], [], "D", color=AQUA, mfc="white", ms=4, label=f"vs. Top-K trained at 1474 (mean {mean(p[2] for p in pts):+.3f})")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([p[0] for p in pts][::-1], fontsize=6.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SCR margin ($>0$: not explained by size)")
    ax.legend(frameon=False, loc="lower left", fontsize=6, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1)
    ax.set_xlim(-0.16, 0.24)
    ax.set_title("(d) SCR across 8 class pairs", pad=24)
    style(ax, "x")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, "fig_e4")


def main():
    R.QUIET = True
    dose = R.tables_dose_response()
    fig_e1_ladder()
    fig_e2_split()
    fig_selection_churn(dose)
    fig_synthesis(dose)
    fig_impl_variance()
    fig_seed_ceiling()
    fig_e4()


if __name__ == "__main__":
    main()
