#!/usr/bin/env python3
"""Regenerate every figure and every numeric table in workshop/mechanism_paper.tex
from the data committed in this repository. No GPU, no torch, no checkpoints:
only numpy, scipy and matplotlib.

    python reproduce.py               # all tables to stdout, all figures to workshop/figs/
    python reproduce.py --tables      # tables only
    python reproduce.py --figures     # figures only
    python reproduce.py --check       # also compare every headline number against
                                      # what the paper prints; exit 1 on a mismatch
    python reproduce.py --write docs/reproduced_tables.md

What each number is computed from is written next to it below and summarised
in docs/REPRODUCE.md. Two kinds of source:

* `experiments/<arm>/seed<n>/<run>/` -- every training run's own
  `evaluation_results.json` (the trainer's `evaluate()` at the end of training;
  `evaluation_results_corrected.json` is preferred when present, see
  `falsification/compare_arms.py::_corrected_or_original`) and the
  `all_histograms_*.npz` behind `comprehensive_summary_*.json`, whose exact
  per-feature selection counts give the two pre-registered liveness thresholds
  (`falsification/report_summaries.py::liveness`).
* `falsification/*_results.json` -- measurements that needed the checkpoints and
  a GPU when they were made (selection Jaccard between two stochastic forward
  passes; SAEBench SCR/TPP scores) and are cached here. The scripts that produced
  them are named at each table.

Effect sizes are Cohen's d with the pooled SD taken as the mean of the two
group SDs, exactly as `compare_arms.py` prints them; p-values are two-sided
seed-permutation tests (`falsification/permutation.py`) with 4M Monte Carlo
draws where exact enumeration is impossible (13 v 13), so the floor is
1/(4e6+1) = 5.16 sigma rather than the 100k default's 4.42.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from falsification.compare_arms import arm_values, effective_floor  # noqa: E402
from falsification.dose_response_figure import dose_response_figure, pearson  # noqa: E402
from falsification.permutation import seed_permutation_test  # noqa: E402

EXP = REPO / "experiments"
FALS = REPO / "falsification"
FIGS = REPO / "workshop" / "figs"

N_PERM = 4_000_000

# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

_ARM_CACHE: dict[str, dict] = {}


def arm(name: str) -> dict[str, dict[int, float]]:
    """Per-seed metrics for one arm: FVE, frac_recovered, frac_alive, below_0.1x,
    below_0.5x (the last two need the .npz)."""
    if name not in _ARM_CACHE:
        # "archive/<arm>": the superseded six-seed pilot arms Table 7 reads.
        root = REPO / name if name.startswith("archive/") else EXP / name
        if not root.is_dir():
            sys.exit(f"missing experiments/{name}")
        vals, _ = arm_values(root)
        _ARM_CACHE[name] = vals
    return _ARM_CACHE[name]


def per_seed(name: str, metric: str) -> list[float]:
    v = arm(name).get(metric)
    if not v:
        sys.exit(f"experiments/{name} has no {metric} (missing evaluation_results.json "
                 "or all_histograms_*.npz?)")
    return [v[s] for s in sorted(v)]


def l0_per_seed(name: str) -> list[float]:
    out = {}
    for f in sorted((EXP / name).glob("seed*/*/evaluation_results.json")):
        seed = int(f.parts[-3][4:])
        out[seed] = json.loads(f.read_text())["l0"]
    return [out[s] for s in sorted(out)]


def sigma_of(p: float) -> float:
    from scipy.stats import norm
    return float(norm.isf(p / 2.0))


def compare(arm_a: str, arm_b: str, metric: str) -> dict:
    """mean/sd of both arms, diff = a - b, d = diff / mean(sd_a, sd_b), and the
    two-sided seed-permutation p with its binding floor."""
    va, vb = per_seed(arm_a, metric), per_seed(arm_b, metric)
    sa, sb = stdev(va), stdev(vb)
    diff = mean(va) - mean(vb)
    pooled = (sa + sb) / 2
    res = seed_permutation_test(va, vb, alternative="two-sided", n_perm=N_PERM)
    floor = effective_floor(res)
    return {"mean_a": mean(va), "sd_a": sa, "mean_b": mean(vb), "sd_b": sb,
            "n_a": len(va), "n_b": len(vb), "diff": diff,
            "d": diff / pooled if pooled > 0 else float("inf"),
            "p": res["p_value"], "floor": floor, "at_floor": res["p_value"] <= floor,
            "sigma": sigma_of(res["p_value"]), "sigma_floor": sigma_of(floor)}


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

OUT: list[str] = []
CHECKS: list[tuple[str, float, float, float]] = []   # (label, got, expected, tol)
QUIET = False


def emit(s: str = "") -> None:
    if QUIET:
        return
    print(s)
    OUT.append(s)


def table(title: str, source: str, header: list[str], rows: list[list[str]]) -> None:
    emit(f"\n### {title}\n")
    emit(f"_Source: {source}_\n")
    emit("| " + " | ".join(header) + " |")
    emit("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        emit("| " + " | ".join(r) + " |")


def check(label: str, got: float, expected: float, tol: float) -> None:
    CHECKS.append((label, got, expected, tol))


def pm(m: float, s: float, mp: int = 4, sp: int = 4) -> str:
    return f"{m:.{mp}f} ± {s:.{sp}f}"


def dstr(c: dict) -> str:
    """d with its p; `ns` marks p > 0.05, `*` a p sitting at the Monte Carlo floor."""
    flag = "*" if c["at_floor"] else (" ns" if c["p"] > 0.05 else "")
    return f"{c['d']:+.1f} (p={c['p']:.2g}{flag})"


def pstr(c: dict) -> str:
    return f"{c['p']:.2e}" + ("*" if c["at_floor"] else "")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

E1_METRICS = [("frac_variance_explained", "FVE"), ("frac_recovered", "frac. rec."),
              ("below_0.1x", "<0.1× k/d"), ("below_0.5x", "<0.5× k/d")]


def table_e1_ladder() -> None:
    ladder = [("e1_vsae_ref", "KL warmup, bias form"),
              ("e1_vsae_ref_unitinit", "+ decoder init scale"),
              ("e1_vsae_ref_gradproj", "+ gradient projection"),
              ("e1_vsae_ref_fullmatch", "+ initial weight draw")]
    expected = {  # Table 1 of the paper, d against e1_penalty
        "e1_vsae_ref": (-5.7, -8.5, -2.8, 13.0),
        "e1_vsae_ref_unitinit": (16.5, 17.7, 0.8, 0.2),
        "e1_vsae_ref_gradproj": (3.8, 4.9, -2.4, -3.7),
        "e1_vsae_ref_fullmatch": (-0.3, -0.8, -0.1, -0.6),
    }
    rows = []
    for a, factors in ladder:
        cells = [f"`{a}`", factors]
        for i, (m, _) in enumerate(E1_METRICS):
            c = compare("e1_penalty", a, m)
            cells.append(dstr(c))
            check(f"Table 1 {a} {m} d", c["d"], expected[a][i], 0.15)
        rows.append(cells)
    table("Table 1 — E1's decomposition ladder (d of each generation against `e1_penalty`, 13 seeds/arm)",
          "experiments/e1_*/seed*/*/evaluation_results.json + all_histograms_*.npz; "
          "`compare_arms.py e1_penalty <arm>`",
          ["vSAE arm", "factors matched (cumulative)"] + [f"{lab} (d)" for _, lab in E1_METRICS], rows)
    emit("\n`*` = p at the Monte Carlo floor (5.16σ): the permutation budget ran out, not the evidence.")

    # Numbers the §3 text quotes
    cf = compare("e1_penalty", "e1_vsae_ref_fullmatch", "frac_variance_explained")
    cr = compare("e1_penalty", "e1_vsae_ref_fullmatch", "frac_recovered")
    ca = compare("e1_penalty", "e1_vsae_ref_fullmatch", "frac_alive")
    cl = compare("e1_penalty", "e1_vsae_ref_fullmatch", "below_0.5x")
    emit(f"\nVerdict row (`e1_penalty` vs `e1_vsae_ref_fullmatch`): FVE differs by {cf['diff']:+.4f} "
         f"(d={cf['d']:+.1f}, p={cf['p']:.2f}), frac_recovered by {cr['diff']:+.4f} "
         f"(d={cr['d']:+.1f}, p={cr['p']:.2f}), frac_alive is {ca['mean_a']:.6f} vs {ca['mean_b']:.6f}, "
         f"and the loose-threshold live fraction differs by {abs(cl['diff']):.3f}.")
    check("E1 verdict FVE diff", cf["diff"], -0.0003, 0.00006)
    check("E1 verdict FVE p", cf["p"], 0.39, 0.02)
    check("E1 verdict frac_rec p", cr["p"], 0.07, 0.01)

    emit("\nThe gradient-projection factor on its own (`e1_vsae_ref_unitinit` vs `e1_vsae_ref_gradproj`):")
    fve_u = compare("e1_penalty", "e1_vsae_ref_unitinit", "frac_variance_explained")
    fve_g = compare("e1_penalty", "e1_vsae_ref_gradproj", "frac_variance_explained")
    rec_u = compare("e1_penalty", "e1_vsae_ref_unitinit", "frac_recovered")
    rec_g = compare("e1_penalty", "e1_vsae_ref_gradproj", "frac_recovered")
    closed_fve = 1 - fve_g["diff"] / fve_u["diff"]
    closed_rec = 1 - rec_g["diff"] / rec_u["diff"]
    emit(f"  {100*closed_fve:.1f}% of the FVE gap closes ({fve_u['diff']:.4f} → {fve_g['diff']:.4f}), "
         f"{100*closed_rec:.1f}% of frac_recovered's ({rec_u['diff']:.4f} → {rec_g['diff']:.4f}).")
    check("gradproj FVE gap closed %", 100 * closed_fve, 78.9, 0.15)
    check("gradproj frac_rec gap closed %", 100 * closed_rec, 74.9, 0.15)
    ug = compare("e1_vsae_ref_unitinit", "e1_vsae_ref_gradproj", "frac_variance_explained")
    emit(f"  As a factor: FVE d={ug['d']:+.1f} ({ug['sigma']:.2f}σ)")
    check("unitinit vs gradproj FVE d", ug["d"], -14.3, 0.15)
    lv = compare("e1_vsae_ref_unitinit", "e1_vsae_ref_gradproj", "below_0.5x")
    pen = mean(per_seed("e1_penalty", "below_0.5x"))
    emit(f"  Liveness (<0.5× k/d): {lv['mean_a']:.4f} (off) → {lv['mean_b']:.4f} (on) against e1_penalty's {pen:.4f}.")
    check("unitinit <0.5x", lv["mean_a"], 0.1816, 0.0001)
    check("gradproj <0.5x", lv["mean_b"], 0.2197, 0.0001)
    check("e1_penalty <0.5x", pen, 0.1836, 0.0001)

    emit("\nThe initial-weight-draw factor on its own (`e1_vsae_ref_gradproj` vs `e1_vsae_ref_fullmatch`):")
    for m, lab, exp in [("frac_variance_explained", "FVE", -4.7), ("frac_recovered", "frac_recovered", -7.3)]:
        c = compare("e1_vsae_ref_gradproj", "e1_vsae_ref_fullmatch", m)
        emit(f"  {lab}: d={c['d']:+.1f} (p={pstr(c)})")
        check(f"gradproj vs fullmatch {lab} d", c["d"], exp, 0.15)


# --------------------------------------------------------------------------
# Section 4: the sampled arms, measured in the regime they were trained in
# --------------------------------------------------------------------------
#
# Every sampled (var_flag=1) checkpoint runs at sigma/norm_factor of its
# training noise when fed raw activations, because scale_biases() scales mu by
# norm_factor at save time and leaves log sigma^2 alone (docs/ERRATA.md §2).
# Tables 2-5 are therefore read from falsification/training_regime_results.json
# (`read_training_regime.py`, GPU), which measures every checkpoint both in the
# training regime (primary) and as saved, on the same tokens and noise draws.
# The numbers an earlier version of the paper printed -- the as-saved
# measurement, FVE from each run's evaluate() and Jaccard from the A2-A4
# readers -- are kept in Table 13 (appendix) and still checked below.

REGIME = FALS / "training_regime_results.json"
_REGIME: dict | None = None
SWEEP_BASELINE = {"topk": "baseline", "batchtopk": "a3_batchtopk_baseline", "jumprelu": "a4_jumprelu_baseline"}
L0_BAND = (240.0, 320.0)       # falsification/read_a4_followup.py, fixed before the follow-up ran
SIX = (-1.0, -2.0, -3.0, -4.0, -5.0, -8.0)


def regime_arm(name: str) -> dict:
    global _REGIME
    if _REGIME is None:
        _REGIME = {a["arm"]: a for a in json.loads(REGIME.read_text())["arms"]}
    return _REGIME[name]


def regime_vals(name: str, metric: str, regime: str = "training") -> list[float]:
    return [s[regime][metric] for s in regime_arm(name)["seeds"]]


def regime_rows(sweep: str, regime: str = "training", six_only: bool = True) -> list[dict]:
    """One row per grid point: sigma, FVE, Jaccard, L0 (mean and SD over seeds)."""
    if _REGIME is None:
        regime_arm("baseline")
    rows = []
    for a in _REGIME.values():
        if a["sweep"] != sweep or a["log_var_init"] is None:
            continue
        if six_only and a["log_var_init"] not in SIX:
            continue
        f, j, l = (regime_vals(a["arm"], m, regime) for m in ("fve", "jaccard", "l0"))
        rows.append({"log_var_init": a["log_var_init"], "arm": a["arm"], "sigma": a["sigma"],
                     "n_seeds": len(f), "fve_mean": mean(f), "fve_std": stdev(f),
                     "jaccard_mean": mean(j), "jaccard_std": stdev(j), "l0_mean": mean(l), "l0_std": stdev(l)})
    rows.sort(key=lambda r: -r["log_var_init"])
    return rows


def table_e2_split() -> None:
    arms = [("baseline", "Baseline (deterministic)"),
            ("e2_confirm", "σ learned, β=1e-4 (`e2_confirm`)"),
            ("e2_sampling_only", "σ learned, β=0 (`e2_sampling_only`)")]
    rows, m = [], {}
    for a, lab in arms:
        v = regime_vals(a, "fve")
        m[a] = mean(v)
        rows.append([lab, pm(mean(v), stdev(v), 6, 4), str(len(v))])
    table("Table 2 — removing the KL recovers a small fraction of the gap (FVE in the training regime, 13 seeds/arm)",
          "falsification/training_regime_results.json (`read_training_regime.py`, GPU)",
          ["Configuration", "FVE", "n"], rows)
    gap = m["baseline"] - m["e2_confirm"]
    kl_share = (m["e2_sampling_only"] - m["e2_confirm"]) / gap
    emit(f"\nTotal gap {gap:.4f}; removing the KL recovers {100*kl_share:.1f}%, "
         f"the reparameterisation accounts for the remaining {100*(1-kl_share):.1f}%.")
    check("Table 2 baseline FVE", m["baseline"], EXP_E2["baseline"], 1e-6)
    check("Table 2 e2_confirm FVE", m["e2_confirm"], EXP_E2["e2_confirm"], 1e-6)
    check("Table 2 e2_sampling_only FVE", m["e2_sampling_only"], EXP_E2["e2_sampling_only"], 1e-6)
    check("E2 total gap", gap, EXP_E2["gap"], 0.0001)
    check("E2 KL share %", 100 * kl_share, EXP_E2["kl_share"], 0.06)
    pil = [(b, mean(regime_vals(f"e2_beta_pilot_{b:g}", "fve"))) for b in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    emit("β pilot (1 seed each), training-regime FVE: " + ", ".join(f"β={b:g}: {v:.3f}" for b, v in pil))
    check("E2 pilot FVE at beta=1e-4", pil[0][1], 0.329, 0.0006)
    check("E2 pilot FVE at beta=1", pil[-1][1], 0.000, 0.0006)

    # As saved (Table 13): each run's own evaluate() on raw activations.
    ms = {a: mean(per_seed(a, "frac_variance_explained")) for a, _ in arms}
    gap_s = ms["baseline"] - ms["e2_confirm"]
    share_s = (ms["e2_sampling_only"] - ms["e2_confirm"]) / gap_s
    emit(f"As saved (Table 13): FVE {ms['e2_confirm']:.6f} (β=1e-4), {ms['e2_sampling_only']:.6f} (β=0); "
         f"gap {gap_s:.4f}, KL share {100*share_s:.1f}%.")
    check("Table 13 e2_confirm FVE as saved", ms["e2_confirm"], 0.458146, 1e-6)
    check("Table 13 e2_sampling_only FVE as saved", ms["e2_sampling_only"], 0.486276, 1e-6)
    check("Table 13 E2 KL share % as saved", 100 * share_s, 6.4, 0.06)


def dose_table(title: str, rows: list[dict], baseline: str, with_l0: bool) -> float:
    hdr = ["log σ²_init", "σ (effective)"] + (["L0"] if with_l0 else []) + ["FVE", "Jaccard", "n"]
    body = []
    for r in rows:
        lv = f"{r['log_var_init']:.1f}" + (" (clamped)" if r["log_var_init"] < -6 else "")
        cells = [lv, f"{r['sigma']:.4f}"]
        if with_l0:
            cells.append(f"{r['l0_mean']:.1f}")
        cells += [pm(r["fve_mean"], r["fve_std"]), pm(r["jaccard_mean"], r["jaccard_std"]), str(r["n_seeds"])]
        body.append(cells)
    table(title, "falsification/training_regime_results.json (`read_training_regime.py`, GPU), training regime",
          hdr, body)
    bf = regime_vals(baseline, "fve")
    line = f"\nDeterministic baseline (`{baseline}`): FVE {pm(mean(bf), stdev(bf), 4, 4)}"
    if with_l0:
        line += f", L0 = {mean(regime_vals(baseline, 'l0')):.1f}"
    emit(line)
    return mean(bf)


def tables_dose_response() -> dict:
    out = {"baselines": {}}
    chance = 256 / (2 * 2048 - 256)
    for key, sweep, name, tno in [("a2", "topk", "TopK", 3), ("a3", "batchtopk", "BatchTopK", 4)]:
        rows = regime_rows(sweep)
        b = dose_table(f"Table {tno} — {name}: the noise-scale dose-response, training regime, 13 seeds/point",
                       rows, SWEEP_BASELINE[sweep], False)
        fve, jac = [x["fve_mean"] for x in rows], [x["jaccard_mean"] for x in rows]
        r = pearson(fve, jac)
        rho = spearman(fve, jac)
        emit(f"Pearson r(FVE, Jaccard) = {r:+.4f}, Spearman ρ = {rho:+.2f} over {len(rows)} points; "
             f"chance Jaccard {chance:.4f}.")
        exp = EXP_DOSE[key]
        check(f"{name} r(FVE,Jaccard)", r, exp["r"], 0.00006)
        check(f"{name} Spearman", rho, 1.0, 1e-9)
        check(f"{name} baseline FVE", b, exp["baseline"], 0.00006)
        for x in rows:
            check(f"Table {tno} FVE at {x['log_var_init']}", x["fve_mean"], exp["rows"][x["log_var_init"]][0], 0.00006)
            check(f"Table {tno} Jaccard at {x['log_var_init']}", x["jaccard_mean"], exp["rows"][x["log_var_init"]][1], 0.00006)
        out[key] = rows
        out["baselines"][key] = b
    fa, fb = ({x["log_var_init"]: x["fve_mean"] / out["baselines"][k] for x in out[k]} for k in ("a2", "a3"))
    emit("FVE as a fraction of each architecture's own deterministic baseline:")
    emit("| log σ²_init | TopK | BatchTopK |\n|---|---|---|")
    for lv in SIX:
        emit(f"| {lv:.1f} | {100*fa[lv]:.1f}% | {100*fb[lv]:.1f}% |")
    check("BatchTopK below TopK at every point", sum(fb[lv] < fa[lv] for lv in SIX), 6, 0)
    check("A2 clamp-floor % of baseline", 100 * fa[-8.0], 83.2, 0.06)
    check("A3 clamp-floor % of baseline", 100 * fb[-8.0], 80.9, 0.06)

    # ---- JumpReLU
    rows = regime_rows("jumprelu")
    b = dose_table("Table 5 — JumpReLU: the same sweep plus achieved L0 (target 256), training regime, 13 seeds/point",
                   rows, "a4_jumprelu_baseline", True)
    out["baselines"]["a4"] = b
    fve, jac, l0 = ([x[k] for x in rows] for k in ("fve_mean", "jaccard_mean", "l0_mean"))
    r_fj, r_fl = pearson(fve, jac), pearson(fve, l0)
    emit(f"Naive r(FVE, Jaccard) = {r_fj:+.4f}; r(FVE, L0) = {r_fl:+.4f} over the six points; "
         f"L0 spans {min(l0):.1f}–{max(l0):.1f}.")
    exp = EXP_DOSE["a4"]
    check("A4 naive r(FVE,Jaccard)", r_fj, exp["r"], 0.00006)
    check("A4 r(FVE,L0)", r_fl, exp["r_l0"], 0.00006)
    check("A4 baseline FVE", b, exp["baseline"], 0.00006)
    check("A4 baseline L0", mean(regime_vals("a4_jumprelu_baseline", "l0")), 269.5, 0.06)
    for x in rows:
        e = exp["rows"][x["log_var_init"]]
        check(f"Table 5 FVE at {x['log_var_init']}", x["fve_mean"], e[0], 0.00006)
        check(f"Table 5 Jaccard at {x['log_var_init']}", x["jaccard_mean"], e[1], 0.00006)
        check(f"Table 5 L0 at {x['log_var_init']}", x["l0_mean"], e[2], 0.06)
    six_band = [x for x in rows if L0_BAND[0] <= x["l0_mean"] <= L0_BAND[1]]
    emit(f"Six-point grid inside the L0 band {L0_BAND}: {[x['log_var_init'] for x in six_band]} "
         f"({len(six_band)} points -- too few for a correlation).")
    check("A4 six-point grid points in band", len(six_band), 3, 0)
    out["a4"] = rows

    # ---- the ten-point grid (four densifying points at 5 seeds)
    rows10 = regime_rows("jumprelu", six_only=False)
    new = {-3.5, -4.5, -5.5, -6.0}
    body = [[f"{x['log_var_init']:.1f}" + (" (new)" if x["log_var_init"] in new else ""), f"{x['sigma']:.4f}",
             pm(x["l0_mean"], x["l0_std"], 1, 1), pm(x["fve_mean"], x["fve_std"]),
             pm(x["jaccard_mean"], x["jaccard_std"]), str(x["n_seeds"])] for x in rows10]
    table("A4 follow-up — the ten-point JumpReLU grid, training regime (four densifying points at 5 seeds)",
          "falsification/training_regime_results.json", ["log σ²_init", "σ", "L0", "FVE", "Jaccard", "n"], body)
    fve, jac, l0 = ([x[k] for x in rows10] for k in ("fve_mean", "jaccard_mean", "l0_mean"))
    r10_fj, r10_fl = pearson(fve, jac), pearson(fve, l0)
    m = [x for x in rows10 if L0_BAND[0] <= x["l0_mean"] <= L0_BAND[1]]
    mf, mj, ml = ([x[k] for x in m] for k in ("fve_mean", "jaccard_mean", "l0_mean"))
    rb_fj, rb_fl = pearson(mf, mj), pearson(mf, ml)
    emit(f"\nTen-point grid: r(FVE, Jaccard) = {r10_fj:+.4f}, r(FVE, L0) = {r10_fl:+.4f}.")
    emit(f"L0 band {L0_BAND} → {len(m)} points ({[x['log_var_init'] for x in m]}): "
         f"r(FVE, Jaccard) = {rb_fj:+.4f}, r(FVE, L0) = {rb_fl:+.4f}; Jaccard {min(mj):.3f}–{max(mj):.3f} "
         f"while FVE {min(mf):.3f}–{max(mf):.3f}.")
    check("A4 10-point r(FVE,Jaccard)", r10_fj, exp["r10"], 0.00006)
    check("A4 10-point r(FVE,L0)", r10_fl, exp["r10_l0"], 0.00006)
    check("A4 band n", len(m), 7, 0)
    check("A4 band r(FVE,Jaccard)", rb_fj, exp["rb"], 0.00006)
    check("A4 band r(FVE,L0)", rb_fl, exp["rb_l0"], 0.00006)
    check("A4 band Jaccard min", min(mj), 0.708, 0.0006)
    check("A4 band Jaccard max", max(mj), 0.724, 0.0006)
    check("A4 band FVE min", min(mf), 0.679, 0.0006)
    check("A4 band FVE max", max(mf), 0.785, 0.0006)
    out["a4_followup"] = (rows10, new)
    table_as_saved()
    return out


def table_as_saved() -> None:
    """Table 13: the same sweeps as saved -- the published measurement, FVE from
    each run's evaluate() and Jaccard from read_a{2,3,4}_dose_response.py, both on
    raw activations, i.e. at sigma/norm_factor -- beside the training regime."""
    body = []
    for key, sweep, name, cache, rexp in [("a2", "topk", "TopK", "a2_dose_response_results.json", 0.9993),
                                          ("a3", "batchtopk", "BatchTopK", "a3_dose_response_results.json", 0.9979),
                                          ("a4", "jumprelu", "JumpReLU", "a4_dose_response_results.json", 0.9334)]:
        data = {r["arm"]: r for r in json.loads((FALS / cache).read_text())["rows"]}
        tr = {x["arm"]: x for x in regime_rows(sweep)}
        fs, js = [], []
        for arm, x in sorted(tr.items(), key=lambda kv: -kv[1]["log_var_init"]):
            f = mean(per_seed(arm, "frac_variance_explained"))
            j = data[arm]["jaccard_mean"]
            fs.append(f); js.append(j)
            body.append([name, f"{x['log_var_init']:.1f}", f"{f:.4f}", f"{j:.4f}", f"{x['fve_mean']:.4f}",
                         f"{x['jaccard_mean']:.4f}"])
        r = pearson(fs, js)
        check(f"Table 13 {name} r(FVE,Jaccard) as saved", r, rexp, 0.00006)
        body.append([name, "r(FVE, Jaccard)", f"{r:+.4f}", "", f"{EXP_DOSE[key]['r']:+.4f}", ""])
    table("Table 13 — the noise sweeps as saved (raw activations, effective σ/norm_factor) and in the training regime",
          "as saved: experiments/*/evaluation_results[_corrected].json + falsification/a{2,3,4}_dose_response_results.json; "
          "training: falsification/training_regime_results.json",
          ["Architecture", "log σ²_init", "FVE as saved", "Jaccard as saved", "FVE training", "Jaccard training"], body)
    # The reader's own as-saved measurement against the published one (appendix text).
    jd, fd, bd = [], [], []
    for key, cache in [("topk", "a2"), ("batchtopk", "a3"), ("jumprelu", "a4")]:
        data = {r["arm"]: r for r in json.loads((FALS / f"{cache}_dose_response_results.json").read_text())["rows"]}
        jd += [abs(regime_arm(a)["raw_jaccard_mean"] - data[a]["jaccard_mean"]) for a in data]
    for a in json.loads(REGIME.read_text())["arms"]:
        if a["published_fve_mean"] is not None:
            (bd if a["log_var_init"] is None else fd).append(a["raw_fve_mean"] - a["published_fve_mean"])
    fd += bd
    emit(f"\nReader as saved vs published: Jaccard within {max(jd):.4f}; FVE {min(fd):+.4f} to {max(fd):+.4f}; "
         f"deterministic baselines within {max(abs(x) for x in bd):.4f}.")
    check("as-saved Jaccard agreement <= 0.003", max(jd), 0.0025, 0.0005)
    check("as-saved FVE diff min", min(fd), -0.013, 0.0006)
    check("as-saved FVE diff max", max(fd), 0.002, 0.0006)
    check("baselines agree to 0.0003", max(abs(x) for x in bd), 0.0003, 0.00006)
    check("JumpReLU as-saved L0 at -1", mean(l0_per_seed("a4_jumprelu_sigma_init_m1")), 36.8, 0.06)
    ex = json.loads((FALS / "selection_example_results.json").read_text())
    tr, sv = ex["regimes"]["training"], ex["regimes"]["raw"]
    emit(f"\nSelection example (`read_selection_example.py`, {ex['arm']} seed {ex['seed']}): Jaccard "
         f"{tr['mean_jaccard']:.3f} training / {sv['mean_jaccard']:.3f} as saved; selected features outside mu's "
         f"own top-k {100*tr['frac_selected_outside_mu_topk']:.0f}% / {100*sv['frac_selected_outside_mu_topk']:.0f}%; "
         f"with mu = 0 {100*tr['frac_selected_with_mu_zero']:.0f}% / {100*sv['frac_selected_with_mu_zero']:.0f}%.")
    check("example Jaccard training", tr["mean_jaccard"], 0.109, 0.0006)
    check("example Jaccard as saved", sv["mean_jaccard"], 0.868, 0.0006)
    check("example outside top-k % training", 100 * tr["frac_selected_outside_mu_topk"], 71, 0.6)
    check("example mu=0 % training", 100 * tr["frac_selected_with_mu_zero"], 24, 0.6)
    check("example outside top-k % as saved", 100 * sv["frac_selected_outside_mu_topk"], 5, 0.6)
    check("example token Jaccard training", tr["token"]["jaccard"], 0.133, 0.0006)
    nf = json.loads(REGIME.read_text())["norm_factor"]
    emit(f"\nnorm_factor re-estimated by read_training_regime.py: {nf:.2f}")
    check("norm_factor", nf, 25.47, 0.006)


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    return pearson(ranks(xs), ranks(ys))


# The printed numbers of Tables 2-5 and the Section 4 text (training regime).
EXP_E2 = {"baseline": 0.900239, "e2_confirm": 0.329065, "e2_sampling_only": 0.351120, "gap": 0.5712, "kl_share": 3.9}
EXP_DOSE = {
    "a2": {"r": 0.9521, "baseline": 0.9002,
           "rows": {-1.0: (0.2517, 0.0750), -2.0: (0.3511, 0.0862), -3.0: (0.4708, 0.1094),
                    -4.0: (0.6107, 0.1686), -5.0: (0.7200, 0.2624), -8.0: (0.7494, 0.2919)}},
    "a3": {"r": 0.9455, "baseline": 0.9508,
           "rows": {-1.0: (0.2459, 0.0758), -2.0: (0.3599, 0.0826), -3.0: (0.4833, 0.0968),
                    -4.0: (0.6232, 0.1311), -5.0: (0.7366, 0.1881), -8.0: (0.7692, 0.2101)}},
    "a4": {"r": 0.9770, "r_l0": 0.8054, "baseline": 0.9213,
           "rows": {-1.0: (0.3401, 0.2628, 121.7), -2.0: (0.4671, 0.3199, 202.7), -3.0: (0.6349, 0.5526, 331.4),
                    -4.0: (0.7073, 0.7186, 268.1), -5.0: (0.7707, 0.7216, 269.0), -8.0: (0.7848, 0.7090, 270.3)},
           "r10": 0.9610, "r10_l0": 0.7492, "rb": -0.5544, "rb_l0": -0.7251},
}


def table_e3() -> None:
    rows = []
    for m, lab, exp in [("below_0.1x", "<0.1× k/d", 19.3), ("below_0.5x", "<0.5× k/d", 15.3)]:
        c = compare("e3_masked_kl", "e3_masked_kl_relu", m)
        rows.append([lab, pm(c["mean_a"], c["sd_a"]), pm(c["mean_b"], c["sd_b"]),
                     f"{c['d']:+.1f}", pstr(c), f"{c['sigma']:.2f}"])
        check(f"Table 6 {lab} d", c["d"], exp, 0.15)
        check(f"Table 6 {lab} no-ReLU mean", c["mean_a"], {19.3: 0.1315, 15.3: 0.4655}[exp], 0.00006)
        check(f"Table 6 {lab} ReLU mean", c["mean_b"], {19.3: 0.0287, 15.3: 0.3081}[exp], 0.00006)
        if m == "below_0.1x":
            emit(f"\nRatio of near-dead populations at the tight threshold: {c['mean_a']/c['mean_b']:.2f}×")
            check("E3 ratio", c["mean_a"] / c["mean_b"], 4.6, 0.06)
    table("Table 6 — the unremarked `F.relu(mu)` (13 seeds/arm)",
          "experiments/e3_masked_kl{,_relu}/seed*/*/all_histograms_*.npz; `compare_arms.py e3_masked_kl e3_masked_kl_relu`",
          ["Threshold", "no-ReLU (preprint equations)", "ReLU (released code)", "d", "p", "σ"], rows)


def table_claim4() -> None:
    rows = []
    exp = {"frac_variance_explained": (0.3, 0.50), "frac_recovered": (0.5, 0.22),
           "below_0.1x": (0.3, 0.53), "below_0.5x": (0.2, 0.63)}
    for m, lab in E1_METRICS:
        c = compare("baseline", "claim4_baseline_noproj", m)
        rows.append([lab, pm(c["mean_a"], c["sd_a"], 6, 4), pm(c["mean_b"], c["sd_b"], 6, 4),
                     f"{c['d']:+.1f} (p={c['p']:.2f})"])
        check(f"Table 8 {lab} d", c["d"], exp[m][0], 0.15)
        check(f"Table 8 {lab} p", c["p"], exp[m][1], 0.015)
    table("Table 8 — the projection turned off in a penalty-free TopK SAE (13 seeds/arm)",
          "experiments/{baseline,claim4_baseline_noproj}/seed*/*/evaluation_results.json + all_histograms_*.npz; "
          "`compare_arms.py baseline claim4_baseline_noproj`",
          ["Metric", "projection on (baseline)", "projection off", "d (p)"], rows)


def table_seed_ceiling() -> None:
    ns = [1, 2, 3, 5, 6, 10, 13, 20]
    exp = [0.00, 0.97, 1.64, 2.65, 3.07, 4.40, 5.21, 6.75]
    cells = []
    for n, e in zip(ns, exp):
        p = 2.0 / math.comb(2 * n, n)
        s = sigma_of(min(p, 1.0))
        cells.append(f"{s:.2f}")
        check(f"Table 9 sigma ceiling n={n}", s, e, 0.006)
    table("Table 9 — the sigma ceiling of a two-sided seed-permutation test, 2/C(2n,n)",
          "formula (`falsification/permutation.py::min_p_floor`); also cached in "
          "falsification/seed_count_survey_results.json",
          ["n/group"] + [str(n) for n in ns], [["sigma ceiling"] + cells])


def table_seed_survey() -> None:
    data = json.loads((FALS / "seed_count_survey_results.json").read_text())
    rows = [[s["paper"].split(",")[0], str(s["seeds_per_config"]) + (" *" if s.get("note") else "")]
            for s in data["survey"]]
    table("Table 10 — independently-seeded runs per configuration in the surveyed SAE papers",
          "falsification/seed_count_survey_results.json (`seed_count_survey.py`)",
          ["Paper", "seeds/configuration"], rows)
    emit(f"\n{data['n_papers_seeds_eq_1']} of {data['n_papers']} report exactly one.")
    check("Table 10 n_papers", data["n_papers"], 10, 0)
    check("Table 10 seeds==1", data["n_papers_seeds_eq_1"], 10, 0)


def table_e4() -> None:
    scr = json.loads((FALS / "e4_scr_results.json").read_text())
    tpp = json.loads((FALS / "e4_tpp_results.json").read_text())
    sm = json.loads((FALS / "e4_size_matched_baseline_results.json").read_text())
    rows = []
    for name, d, exp in [("SCR", scr, (0.1017, 0.0201, 0.082)), ("TPP", tpp, (0.1055, 0.1905, -0.085))]:
        v = d["vsae"]
        margin = d["verdict"]["top_usage"]["margin"]
        expl = d["verdict"]["top_usage"]["explained_by_size"]
        rows.append([name, f"{v['score']:.4f}", f"{v['top_usage_reference_at_vsae_n']:.4f}",
                     f"{margin:+.3f} ({'explained' if expl else 'not explained'} by size)"])
        check(f"Table 11 {name} vSAE", v["score"], exp[0], 0.00006)
        check(f"Table 11 {name} reference", v["top_usage_reference_at_vsae_n"], exp[1], 0.00006)
        check(f"Table 11 {name} margin", margin, exp[2], 0.0006)
    table(f"Table 11 — SCR and TPP on the same two Pythia-70M checkpoints ({scr['dataset']}, "
          f"{'/'.join(scr['column1_vals'])})",
          "falsification/e4_scr_results.json, e4_tpp_results.json (`run_e4_scr.py`, `run_e4_tpp.py`, GPU + SAEBench)",
          ["Metric", "vSAE score", f"baseline curve at N={scr['vsae']['live_n']} (top_usage)", "margin"], rows)
    emit(f"\nTrained-from-scratch {sm['dict_size']}-feature baseline vs the masked curve "
         f"(falsification/e4_size_matched_baseline_results.json): SCR differs by "
         f"{sm['scr']['trained_small_minus_masked_top_usage']:.1e}, TPP by {sm['tpp']['trained_small_minus_masked_top_usage']:+.3f}.")


# Table 7: each implementation detail as a ONE-FACTOR contrast -- two arms that
# differ in that detail only. (An earlier version printed gaps to e1_penalty
# for the warmup/bias and init-scale rows, which measure what remains after a
# factor is matched, not the factor; docs/ERRATA.md §4.) The bias form and the
# KL warmup were matched by configuration before the 13-seed battery, so their
# contrasts come from the archived six-seed pilot arms, at a lower ceiling.
IMPL_FACTORS = [
    ("Bias form (tied pre-encoder bias vs. none)", "archive/e1_vsae_ref_aprilmode", "e1_vsae_ref"),
    ("KL warmup (1000 steps vs. none)", "archive/e1_vsae_ref_klwarmup1000", "archive/e1_vsae_ref_aprilmode"),
    ("Initial decoder scale (0.1 vs. 1)", "e1_vsae_ref", "e1_vsae_ref_unitinit"),
    ("Decoder-gradient projection (off vs. on)", "e1_vsae_ref_unitinit", "e1_vsae_ref_gradproj"),
    ("Initial decoder-weight distribution", "e1_vsae_ref_gradproj", "e1_vsae_ref_fullmatch"),
    ("`F.relu(mu)` (absent vs. present)", "e3_masked_kl", "e3_masked_kl_relu"),
]
IMPL_EXPECTED = {   # |d| for FVE, frac. rec., <0.1x, <0.5x, then the sigma range
    "archive/e1_vsae_ref_aprilmode": (39.7, 31.7, 121.7, 174.5, 4.13, 4.13),
    "archive/e1_vsae_ref_klwarmup1000": (6.9, 2.9, 8.6, 2.9, 3.07, 3.07),
    "e1_vsae_ref": (23.1, 30.3, 2.8, 18.0, 5.03, 5.16),
    "e1_vsae_ref_unitinit": (14.3, 15.3, 2.4, 5.0, 4.48, 5.16),
    "e1_vsae_ref_gradproj": (4.7, 7.3, 2.2, 3.6, 4.31, 5.16),
    "e3_masked_kl": (32.5, 24.0, 19.3, 15.3, 5.03, 5.16),
}


def checks_text() -> None:
    """Numbers the paper's prose quotes outside the tables, wherever committed
    data, or a cached GPU measurement, can derive them."""
    import glob
    from statistics import mean as _m
    # Section 2: the loss identity (check_loss_identity.py, GPU).
    li = json.loads((FALS / "loss_identity_results.json").read_text())["cases"]
    check("loss identity: cases", len(li), 2, 0)
    check("loss identity: max relative difference <= 3e-7", float(max(c["rel_diff"] for c in li.values()) <= 3e-7), 1.0, 0)
    check("loss identity: penalty term nonzero", float(min(c["topk_terms"]["activation_cost"] for c in li.values()) > 0), 1.0, 0)
    # Section 3.2: the +/-10 clamp never binds (read_penalty_clamp.py, GPU, all five E1 arms).
    pc = json.loads((FALS / "penalty_clamp_results.json").read_text())
    check("clamp: arms measured", len(pc["arms"]), 5, 0)
    check("clamp: max pre-clamp activation", pc["max_over_arms"], 0.215, 0.0006)
    check("clamp: headroom x", pc["clamp"] / pc["max_over_arms"], 46, 0.6)
    # Section 4.2: the k/(k+1) pre-activation gap (read_preact_gap.py, GPU).
    g = json.loads((FALS / "preact_gap_results.json").read_text())
    check("gap median", g["percentiles"]["50"], 0.0001, 0.00005)
    check("gap p99", g["percentiles"]["99"], 0.0006, 0.00005)
    check("gap max", g["max"], 0.0014, 0.00005)
    check("clamp floor / p99 gap", 0.0498 / g["percentiles"]["99"], 80, 1)
    check("clamp floor / median gap", 0.0498 / g["percentiles"]["50"], 540, 5)
    # Discussion: exploratory liveness-reconstruction rho over the replicated
    # fixed-variance arms (falsification/frontier.py's loader, deterministic arms only).
    from scipy.stats import spearmanr
    from falsification.frontier import load as frontier_load
    det = [r for r in frontier_load() if len(r["fve"]) >= 6 and not r["arm"].startswith(("e2", "a2", "a3", "a4"))]
    check("liveness note: deterministic arms", len(det), 9, 0)
    for k, er, ep in (("below_0.1x", 0.90, 0.0009), ("below_0.5x", 0.78, 0.013)):
        rho, pv = spearmanr([mean(r["fve"]) for r in det], [mean(r[k]) for r in det])
        check(f"liveness note rho {k}", rho, er, 0.006)
        check(f"liveness note p {k}", pv, ep, 0.0006 if ep > 0.001 else 0.00006)
    # Section 4 derived values
    b, s0, c = (mean(regime_vals(a, "fve")) for a in ("baseline", "e2_sampling_only", "e2_confirm"))
    check("E2 KL recovers (FVE)", s0 - c, 0.022, 0.0006)
    check("E2 sampling costs (FVE)", b - s0, 0.549, 0.0006)
    check("A2 clamp-floor FVE gap", mean(regime_vals("baseline", "fve")) - mean(regime_vals("e2_sigma_low_init", "fve")), 0.151, 0.0006)
    check("chance Jaccard", 256 / (2 * 2048 - 256), 0.0667, 0.00006)
    # Appendix A floors
    check("MC cap 100k sigma", sigma_of(1 / 100_001), 4.42, 0.006)
    check("MC floor 4M p", 1 / 4_000_001, 2.5e-7, 1e-9)
    check("n=6 exact floor p", 2 / math.comb(12, 6), 0.00216, 0.000006)
    check("first equal-n Monte Carlo n", min(n for n in range(2, 30) if math.comb(2 * n, n) > 200_000), 11, 0)
    # Section 6 diagnostics
    def boot(f, key):
        return json.loads((FALS / f).read_text())["margins_at_vsae_n"][key]
    s1 = boot("e4_bootstrap_scr_results.json", "mean_vs_top_usage")
    t1 = boot("e4_bootstrap_tpp_results.json", "mean_vs_top_usage")
    s2 = boot("e4_bootstrap_scr_resample_train_results.json", "mean_vs_top_usage")
    for lab, got, e in [("SCR boot mean", s1["mean"], 0.082), ("SCR boot lo", s1["p2.5"], 0.024), ("SCR boot hi", s1["p97.5"], 0.150),
                        ("TPP boot mean", t1["mean"], -0.086), ("TPP boot lo", t1["p2.5"], -0.090), ("TPP boot hi", t1["p97.5"], -0.081),
                        ("SCR full boot mean", s2["mean"], 0.088), ("SCR full boot lo", s2["p2.5"], 0.008), ("SCR full boot hi", s2["p97.5"], 0.171)]:
        check(f"Section 6 {lab}", got, e, 0.0006)
    check("Section 6 SCR interval / TPP interval", (s1["p97.5"] - s1["p2.5"]) / (t1["p97.5"] - t1["p2.5"]), 15, 0.5)
    scr_pts = []
    for f in sorted(glob.glob(str(FALS / "e4_scr_results*.json"))):
        d = json.loads(Path(f).read_text())
        sm = json.loads((FALS / ("e4_size_matched_baseline_results" + Path(f).name[len("e4_scr_results"):])).read_text())["scr"]
        scr_pts.append((d["verdict"]["top_usage"]["margin"], sm["vsae_minus_trained_small"], sm["trained_small_minus_masked_top_usage"]))
    check("Section 6 SCR not explained (masked)", sum(m > 0 for m, _, _ in scr_pts), 7, 0)
    check("Section 6 SCR not explained (trained small)", sum(t > 0 for _, t, _ in scr_pts), 6, 0)
    check("Section 6 mean margin masked", _m(m for m, _, _ in scr_pts), 0.072, 0.0006)
    check("Section 6 mean margin trained small", _m(t for _, t, _ in scr_pts), 0.079, 0.0006)
    check("Section 6 mean |shift|", _m(abs(x) for _, _, x in scr_pts), 0.075, 0.0006)
    check("Section 6 max |shift|", max(abs(x) for _, _, x in scr_pts), 0.138, 0.0006)
    sm = json.loads((FALS / "e4_size_matched_baseline_results.json").read_text())
    check("Section 6 trained-small SCR", sm["scr"]["score"], 0.020127, 0.0000006)
    check("Section 6 trained-small TPP", sm["tpp"]["score"], 0.2025, 0.00006)
    alive = json.loads(next((EXP / "e4_pythia_baseline_size_matched").glob("seed*/*/evaluation_results.json")).read_text())["frac_alive"]
    check("Section 6 trained-small frac alive %", 100 * alive, 65.5, 0.06)
    ta = json.loads((FALS / "e4_tpp_results_canrager_amazon_reviews_mcauley_1and5.json").read_text())["verdict"]
    check("Section 6 TPP amazon top-usage explained", float(ta["top_usage"]["explained_by_size"]), 1.0, 0)
    check("Section 6 TPP amazon random margin ~ 0", abs(ta["random"]["margin"]), 0.0004, 0.0001)


def table_variance_summary() -> None:
    rows = []
    for lab, a, b in IMPL_FACTORS:
        cs = [compare(a, b, m) for m, _ in E1_METRICS]
        exp = IMPL_EXPECTED[a]
        for i, c in enumerate(cs):
            check(f"Table 7 {lab} {E1_METRICS[i][1]} |d|", abs(c["d"]), exp[i], 0.06)
            check(f"Table 7 {lab} {E1_METRICS[i][1]} p <= 0.05", float(c["p"] <= 0.05), 1.0, 0)
        sig = [c["sigma"] for c in cs]
        check(f"Table 7 {lab} min sigma", min(sig), exp[4], 0.006)
        check(f"Table 7 {lab} max sigma", max(sig), exp[5], 0.006)
        n = f"{cs[0]['n_a']} v {cs[0]['n_b']}"
        live = [cs[2]["d"], cs[3]["d"]]
        flag = " †" if live[0] * live[1] < 0 else ""
        rows.append([lab, n] + [f"{abs(c['d']):.1f}" for c in cs[:2]] +
                    [f"{abs(live[0]):.1f}{flag}", f"{abs(live[1]):.1f}{flag}", f"{min(sig):.2f}–{max(sig):.2f}"])
    table("Table 7 — six implementation details, each as a one-factor contrast (|d|)",
          "experiments/ and archive/ run data; `compare_arms.py <a> <b>`. † = the two liveness thresholds disagree in direction",
          ["Detail", "seeds", "FVE", "frac. rec.", "<0.1× k/d", "<0.5× k/d", "σ"], rows)
    # Section 3.1: the vSAE as released (archived pilot arm) against the null model.
    for m, e in [("frac_variance_explained", 51.3), ("frac_recovered", 37.1), ("below_0.1x", 182.7), ("below_0.5x", 56.9)]:
        c = compare("e1_penalty", "archive/e1_vsae_ref_klwarmup1000", m)
        check(f"Section 3.1 as-released {m} |d|", abs(c["d"]), e, 0.06)
        check(f"Section 3.1 as-released {m} sigma", c["sigma"], 4.13, 0.006)
    check("Table 7 init-scale liveness thresholds disagree",
          float(compare("e1_vsae_ref", "e1_vsae_ref_unitinit", "below_0.1x")["d"]
                * compare("e1_vsae_ref", "e1_vsae_ref_unitinit", "below_0.5x")["d"] < 0), 1.0, 0)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figures(dose: dict) -> None:
    emit("\n## Figures\n")
    b = dose["baselines"]
    r = dose_response_figure(dose["a2"], b["a2"], arch="TopK", out_stem=FIGS / "a2_dose_response")
    emit(f"- workshop/figs/a2_dose_response.{{pdf,png}} (Figure 6; r = {r:+.4f})")
    r = dose_response_figure(dose["a3"], b["a3"], arch="BatchTopK", out_stem=FIGS / "a3_batchtopk_dose_response")
    emit(f"- workshop/figs/a3_batchtopk_dose_response.{{pdf,png}} (Figure 7; r = {r:+.4f})")
    r = dose_response_figure(dose["a4"], b["a4"], arch="JumpReLU", out_stem=FIGS / "a4_jumprelu_dose_response",
                             show_l0=True)
    emit(f"- workshop/figs/a4_jumprelu_dose_response.{{pdf,png}} (Figure 8; naive r = {r:+.4f})")
    rows10, new = dose["a4_followup"]
    r = dose_response_figure(rows10, b["a4"], arch="JumpReLU (ten points)",
                             out_stem=FIGS / "a4_followup_dose_response", show_l0=True, highlight=new)
    emit(f"- workshop/figs/a4_followup_dose_response.{{pdf,png}} (Figure 9; naive r = {r:+.4f})")

    # The explanatory figures (E1 ladder, E2 split, selection churn, synthesis,
    # implementation variance, seed ceiling, E4); same loaders as the tables above.
    # The two diagrams (fig_studymap.tex, fig_pipeline.tex) are TikZ, not data.
    script = "workshop/make_paper_figs.py"
    res = subprocess.run([sys.executable, script], cwd=REPO, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"{script} failed:\n{res.stderr}")
    for line in res.stdout.splitlines():
        emit(f"- {line.removeprefix('wrote ')} (`{script}`)")

    # Not in the mechanism paper, but in workshop/figs/: the preprint-era beta
    # sweep, from the committed comprehensive_histogram_analysis/ summaries.
    # (workshop/figs/frontier.pdf is deliberately left alone: falsification/
    # frontier.py draws every analysed arm on disk, and the committed figure is
    # the 8-arm version RESULTS addendum 6 describes.)
    script = "workshop/make_fig_beta.py"
    res = subprocess.run([sys.executable, script], cwd=REPO, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"{script} failed:\n{res.stderr}")
    emit(f"- workshop/figs/beta_sweep.{{pdf,png}} (`{script}`; not in the mechanism paper)")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="tables only")
    ap.add_argument("--figures", action="store_true", help="figures only")
    ap.add_argument("--check", action="store_true",
                    help="compare every headline number against the paper's printed value; exit 1 on mismatch")
    ap.add_argument("--write", type=Path, help="also write the tables as Markdown to this file")
    args = ap.parse_args()
    do_tables = args.tables or not args.figures
    do_figures = args.figures or not args.tables

    emit("# Reproduced tables and figures — workshop/mechanism_paper.tex\n")
    emit("Generated by `python reproduce.py` from the committed run metadata; see docs/REPRODUCE.md.")

    if do_tables:
        table_e1_ladder()
        table_e2_split()
    global QUIET
    QUIET = not do_tables            # the figures need the dose-response rows either way
    dose = tables_dose_response()
    QUIET = False
    if do_tables:
        table_e3()
        table_variance_summary()
        checks_text()
        table_claim4()
        table_seed_ceiling()
        table_seed_survey()
        table_e4()
    if do_figures:
        figures(dose)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("\n".join(OUT) + "\n")
        print(f"\nwrote {args.write}")

    if args.check:
        bad = [(l, g, e, t) for l, g, e, t in CHECKS if abs(g - e) > t]
        print(f"\n--check: {len(CHECKS) - len(bad)} of {len(CHECKS)} paper numbers reproduced")
        for l, g, e, t in bad:
            print(f"  MISMATCH {l}: computed {g:.6g}, paper says {e:.6g} (tol {t:g})")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
