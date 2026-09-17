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
        root = EXP / name
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


def table_e2_split() -> None:
    arms = [("baseline", "Baseline (deterministic)"),
            ("e2_confirm", "σ learned, β=1e-4 (`e2_confirm`)"),
            ("e2_sampling_only", "σ learned, β=0 (`e2_sampling_only`)")]
    rows, m = [], {}
    for a, lab in arms:
        v = per_seed(a, "frac_variance_explained")
        m[a] = mean(v)
        rows.append([lab, pm(mean(v), stdev(v), 6, 4), str(len(v))])
    table("Table 2 — removing the KL recovers a small fraction of the gap (FVE, 13 seeds/arm)",
          "experiments/{baseline,e2_confirm,e2_sampling_only}/seed*/*/evaluation_results[_corrected].json",
          ["Configuration", "FVE", "n"], rows)
    gap = m["baseline"] - m["e2_confirm"]
    kl_share = (m["e2_sampling_only"] - m["e2_confirm"]) / gap
    emit(f"\nTotal gap {gap:.4f}; removing the KL recovers {100*kl_share:.1f}%, "
         f"the reparameterisation accounts for the remaining {100*(1-kl_share):.1f}%.")
    check("Table 2 baseline FVE", m["baseline"], 0.900159, 1e-6)
    check("Table 2 e2_confirm FVE", m["e2_confirm"], 0.458146, 1e-6)
    check("Table 2 e2_sampling_only FVE", m["e2_sampling_only"], 0.486276, 1e-6)
    check("E2 total gap", gap, 0.4420, 0.0001)
    check("E2 KL share %", 100 * kl_share, 6.4, 0.06)


def dose_rows(cache: Path, with_l0: bool = False) -> tuple[list[dict], float | None, float | None]:
    """Grid rows: sigma + Jaccard from the cache, FVE (and L0) re-read from each
    run's evaluation_results[_corrected].json. Returns (rows, baseline_fve, baseline_l0)."""
    data = json.loads(cache.read_text())
    rows = []
    for r in data["rows"]:
        fve = per_seed(r["arm"], "frac_variance_explained")
        if len(fve) != r["n_seeds"]:
            sys.exit(f"{r['arm']}: {len(fve)} evaluation files vs {r['n_seeds']} seeds in {cache.name}")
        row = {"log_var_init": r["log_var_init"], "arm": r["arm"], "sigma": r["sigma"],
               "n_seeds": len(fve), "fve_mean": mean(fve), "fve_std": stdev(fve),
               "jaccard_mean": r["jaccard_mean"], "jaccard_std": r["jaccard_std"]}
        if with_l0:
            l0 = l0_per_seed(r["arm"])
            row["l0_mean"], row["l0_std"] = mean(l0), stdev(l0)
        rows.append(row)
    rows.sort(key=lambda r: -r["log_var_init"])
    return rows, data.get("baseline_fve_mean"), data.get("baseline_l0_mean")


def dose_table(title: str, source: str, rows: list[dict], baseline: str, with_l0: bool) -> None:
    bfve = per_seed(baseline, "frac_variance_explained")
    hdr = ["log σ²_init", "σ (effective)"] + (["L0"] if with_l0 else []) + ["FVE", "Jaccard", "n"]
    body = []
    for r in rows:
        lv = f"{r['log_var_init']:.1f}" + (" (clamped)" if r["log_var_init"] < -6 else "")
        cells = [lv, f"{r['sigma']:.4f}"]
        if with_l0:
            cells.append(f"{r['l0_mean']:.1f}")
        cells += [pm(r["fve_mean"], r["fve_std"]), pm(r["jaccard_mean"], r["jaccard_std"]), str(r["n_seeds"])]
        body.append(cells)
    table(title, source, hdr, body)
    line = f"\nDeterministic baseline (`{baseline}`): FVE {pm(mean(bfve), stdev(bfve), 6, 4)}"
    if with_l0:
        l0 = l0_per_seed(baseline)
        line += f", L0 = {mean(l0):.1f}"
    emit(line)
    return mean(bfve)


A2_EXPECTED = {-1.0: (0.3766, 0.7657), -2.0: (0.4863, 0.8117), -3.0: (0.6140, 0.8700),
               -4.0: (0.7427, 0.9180), -5.0: (0.8169, 0.9453), -8.0: (0.8338, 0.9523)}
A3_EXPECTED = {-1.0: (0.3434, 0.6969), -2.0: (0.4787, 0.7546), -3.0: (0.6179, 0.8348),
               -4.0: (0.7553, 0.8912), -5.0: (0.8391, 0.9222), -8.0: (0.8588, 0.9311)}
A4_EXPECTED = {-1.0: (0.3967, 0.9637, 36.8), -2.0: (0.5346, 0.9635, 98.2), -3.0: (0.7461, 0.9779, 290.8),
               -4.0: (0.7979, 0.9913, 247.2), -5.0: (0.8522, 0.9883, 261.2), -8.0: (0.8632, 0.9867, 265.0)}


def tables_dose_response() -> dict:
    out = {}
    # ---- A2, TopK
    rows, _, _ = dose_rows(FALS / "a2_dose_response_results.json")
    b_topk = dose_table("Table 3 — TopK: the noise-scale dose-response, 13 seeds/point",
                        "FVE: experiments/{a2_sigma_init_*,e2_sampling_only,e2_sigma_low_init}/…/evaluation_results[_corrected].json; "
                        "Jaccard: falsification/a2_dose_response_results.json (`read_a2_dose_response.py`, GPU)",
                        rows, "baseline", False)
    r = pearson([x["fve_mean"] for x in rows], [x["jaccard_mean"] for x in rows])
    emit(f"Pearson r(FVE, Jaccard) over the {len(rows)} grid points: {r:+.4f}")
    floor = rows[-1]
    emit(f"Residual at the clamp floor: FVE gap {b_topk - floor['fve_mean']:.3f}, Jaccard {floor['jaccard_mean']:.3f}.")
    check("A2 r(FVE,Jaccard)", r, 0.9993, 0.00006)
    for x in rows:
        check(f"Table 3 FVE at {x['log_var_init']}", x["fve_mean"], A2_EXPECTED[x["log_var_init"]][0], 0.00006)
        check(f"Table 3 Jaccard at {x['log_var_init']}", x["jaccard_mean"], A2_EXPECTED[x["log_var_init"]][1], 0.00006)
    out["a2"] = rows
    topk_frac = {x["log_var_init"]: x["fve_mean"] / b_topk for x in rows}

    # ---- A3, BatchTopK
    rows, _, _ = dose_rows(FALS / "a3_dose_response_results.json")
    b_batch = dose_table("Table 4 — BatchTopK: the same sweep, 13 seeds/point",
                         "FVE: experiments/a3_batchtopk_*/…/evaluation_results.json; "
                         "Jaccard: falsification/a3_dose_response_results.json (`read_a3_dose_response.py`, GPU)",
                         rows, "a3_batchtopk_baseline", False)
    r = pearson([x["fve_mean"] for x in rows], [x["jaccard_mean"] for x in rows])
    emit(f"Pearson r(FVE, Jaccard) over the {len(rows)} grid points: {r:+.4f}")
    check("A3 r(FVE,Jaccard)", r, 0.9979, 0.00006)
    check("A3 baseline FVE", b_batch, 0.9509, 0.00006)
    for x in rows:
        check(f"Table 4 FVE at {x['log_var_init']}", x["fve_mean"], A3_EXPECTED[x["log_var_init"]][0], 0.00006)
        check(f"Table 4 Jaccard at {x['log_var_init']}", x["jaccard_mean"], A3_EXPECTED[x["log_var_init"]][1], 0.00006)
    emit("FVE as a fraction of each architecture's own deterministic baseline:")
    emit("| log σ²_init | TopK | BatchTopK |\n|---|---|---|")
    for x in rows:
        emit(f"| {x['log_var_init']:.1f} | {100*topk_frac[x['log_var_init']]:.1f}% | {100*x['fve_mean']/b_batch:.1f}% |")
    check("A3 clamp-floor % of baseline", 100 * rows[-1]["fve_mean"] / b_batch, 90.3, 0.06)
    check("A2 clamp-floor % of baseline", 100 * topk_frac[-8.0], 92.6, 0.06)
    out["a3"] = rows

    # ---- A4, JumpReLU
    rows, _, _ = dose_rows(FALS / "a4_dose_response_results.json", with_l0=True)
    b_jump = dose_table("Table 5 — JumpReLU: the same sweep plus achieved L0 (target 256), 13 seeds/point",
                        "FVE, L0: experiments/a4_jumprelu_*/…/evaluation_results.json; "
                        "Jaccard: falsification/a4_dose_response_results.json (`read_a4_dose_response.py`, GPU)",
                        rows, "a4_jumprelu_baseline", True)
    fve = [x["fve_mean"] for x in rows]
    jac = [x["jaccard_mean"] for x in rows]
    l0 = [x["l0_mean"] for x in rows]
    r_fj, r_fl = pearson(fve, jac), pearson(fve, l0)
    emit(f"Naive r(FVE, Jaccard) = {r_fj:+.4f}; r(FVE, L0) = {r_fl:+.4f} over the six points.")
    check("A4 naive r(FVE,Jaccard)", r_fj, 0.9334, 0.00006)
    check("A4 r(FVE,L0)", r_fl, 0.9494, 0.00006)
    check("A4 baseline FVE", b_jump, 0.9210, 0.00006)
    check("A4 baseline L0", mean(l0_per_seed("a4_jumprelu_baseline")), 269.4, 0.06)
    for x in rows:
        e = A4_EXPECTED[x["log_var_init"]]
        check(f"Table 5 FVE at {x['log_var_init']}", x["fve_mean"], e[0], 0.00006)
        check(f"Table 5 Jaccard at {x['log_var_init']}", x["jaccard_mean"], e[1], 0.00006)
        check(f"Table 5 L0 at {x['log_var_init']}", x["l0_mean"], e[2], 0.06)
    matched = [x for x in rows if x["log_var_init"] in (-3.0, -4.0, -5.0, -8.0)]
    r_fj4 = pearson([x["fve_mean"] for x in matched], [x["jaccard_mean"] for x in matched])
    r_fl4 = pearson([x["fve_mean"] for x in matched], [x["l0_mean"] for x in matched])
    emit(f"L0-matched four points (log σ²_init ∈ {{-3,-4,-5,-8}}, L0 ∈ [{min(x['l0_mean'] for x in matched):.0f}, "
         f"{max(x['l0_mean'] for x in matched):.0f}]): r(FVE, Jaccard) = {r_fj4:+.4f}, r(FVE, L0) = {r_fl4:+.4f}.")
    check("A4 4-point r(FVE,Jaccard)", r_fj4, 0.6297, 0.00006)
    check("A4 4-point r(FVE,L0)", r_fl4, -0.5467, 0.00006)
    out["a4"] = rows

    # ---- A4 follow-up (ten points)
    data = json.loads((FALS / "a4_followup_results.json").read_text())
    rows10, _, _ = dose_rows(FALS / "a4_followup_results.json", with_l0=True)
    band = tuple(data["l0_band"])
    new = {r["log_var_init"] for r in data["new_rows"]}
    body = []
    for x in rows10:
        body.append([f"{x['log_var_init']:.1f}" + (" (new)" if x["log_var_init"] in new else ""),
                     f"{x['sigma']:.4f}", pm(x["l0_mean"], x["l0_std"], 1, 1), pm(x["fve_mean"], x["fve_std"]),
                     pm(x["jaccard_mean"], x["jaccard_std"]), str(x["n_seeds"])])
    table("A4 follow-up — the ten-point JumpReLU grid (four densifying points at 5 seeds)",
          "FVE, L0: experiments/a4_jumprelu_*/…/evaluation_results.json; Jaccard: "
          "falsification/a4_followup_results.json (`read_a4_followup.py`, GPU)",
          ["log σ²_init", "σ", "L0", "FVE", "Jaccard", "n"], body)
    fve = [x["fve_mean"] for x in rows10]
    jac = [x["jaccard_mean"] for x in rows10]
    l0 = [x["l0_mean"] for x in rows10]
    r10_fj, r10_fl = pearson(fve, jac), pearson(fve, l0)
    m8 = [x for x in rows10 if band[0] <= x["l0_mean"] <= band[1]]
    r8_fj = pearson([x["fve_mean"] for x in m8], [x["jaccard_mean"] for x in m8])
    r8_fl = pearson([x["fve_mean"] for x in m8], [x["l0_mean"] for x in m8])
    emit(f"\nFull ten-point grid: r(FVE, Jaccard) = {r10_fj:+.4f}, r(FVE, L0) = {r10_fl:+.4f}.")
    emit(f"L0-matched band L0 ∈ [{band[0]:.0f}, {band[1]:.0f}] → {len(m8)} points "
         f"(L0 ∈ [{min(x['l0_mean'] for x in m8):.0f}, {max(x['l0_mean'] for x in m8):.0f}]): "
         f"r(FVE, Jaccard) = {r8_fj:+.4f}, r(FVE, L0) = {r8_fl:+.4f}.")
    check("A4 follow-up 10-point r(FVE,Jaccard)", r10_fj, 0.9341, 0.00006)
    check("A4 follow-up 10-point r(FVE,L0)", r10_fl, 0.9245, 0.00006)
    check("A4 follow-up n matched", len(m8), 8, 0)
    check("A4 follow-up 8-point r(FVE,Jaccard)", r8_fj, 0.5026, 0.00006)
    check("A4 follow-up 8-point r(FVE,L0)", r8_fl, -0.4704, 0.00006)
    p6 = next(x for x in rows10 if x["log_var_init"] == -6.0)
    check("A4 follow-up -6.0 FVE", p6["fve_mean"], 0.8632, 0.00006)
    check("A4 follow-up -6.0 FVE sd", p6["fve_std"], 0.0010, 0.00006)
    check("A4 follow-up -6.0 L0", p6["l0_mean"], 265.1, 0.06)
    check("A4 follow-up -6.0 Jaccard", p6["jaccard_mean"], 0.9868, 0.00006)
    out["a4_followup"] = (rows10, new)
    out["baselines"] = {"a2": b_topk, "a3": b_batch, "a4": b_jump}
    return out


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


def table_variance_summary() -> None:
    """Table 7 collects effect sizes that are all computed above; recompute the
    ones it quotes so the table stands on its own."""
    rows = [
        ["KL warmup schedule + bias parameterisation", "liveness (loose)",
         f"{abs(compare('e1_penalty', 'e1_vsae_ref', 'below_0.5x')['d']):.1f}"],
        ["Initial decoder scale", "FVE",
         f"{abs(compare('e1_penalty', 'e1_vsae_ref_unitinit', 'frac_variance_explained')['d']):.1f}"],
        ["Decoder-gradient-projection omission", "FVE",
         f"{abs(compare('e1_penalty', 'e1_vsae_ref_gradproj', 'frac_variance_explained')['d']):.1f}–"
         f"{abs(compare('e1_vsae_ref_unitinit', 'e1_vsae_ref_gradproj', 'frac_variance_explained')['d']):.1f}"],
        ["Initial decoder-weight distribution", "FVE, frac. recovered",
         f"{abs(compare('e1_vsae_ref_gradproj', 'e1_vsae_ref_fullmatch', 'frac_variance_explained')['d']):.1f}, "
         f"{abs(compare('e1_vsae_ref_gradproj', 'e1_vsae_ref_fullmatch', 'frac_recovered')['d']):.1f}"],
        ["`F.relu(mu)` presence/absence", "liveness",
         f"{abs(compare('e3_masked_kl', 'e3_masked_kl_relu', 'below_0.5x')['d']):.1f}–"
         f"{abs(compare('e3_masked_kl', 'e3_masked_kl_relu', 'below_0.1x')['d']):.1f}"],
    ]
    table("Table 7 — the implementation details and their measured |d| (13 seeds/arm)",
          "Tables 1 and 6 above", ["Detail", "Metric moved", "|d|"], rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figures(dose: dict) -> None:
    emit("\n## Figures\n")
    b = dose["baselines"]
    r = dose_response_figure(dose["a2"], b["a2"], arch="TopK", out_stem=FIGS / "a2_dose_response")
    emit(f"- workshop/figs/a2_dose_response.{{pdf,png}} (Figure 1; r = {r:+.4f})")
    r = dose_response_figure(dose["a3"], b["a3"], arch="BatchTopK", out_stem=FIGS / "a3_batchtopk_dose_response")
    emit(f"- workshop/figs/a3_batchtopk_dose_response.{{pdf,png}} (Figure 2; r = {r:+.4f})")
    r = dose_response_figure(dose["a4"], b["a4"], arch="JumpReLU", out_stem=FIGS / "a4_jumprelu_dose_response",
                             show_l0=True)
    emit(f"- workshop/figs/a4_jumprelu_dose_response.{{pdf,png}} (Figure 3; naive r = {r:+.4f})")
    rows10, new = dose["a4_followup"]
    r = dose_response_figure(rows10, b["a4"], arch="JumpReLU (ten points)",
                             out_stem=FIGS / "a4_followup_dose_response", show_l0=True, highlight=new)
    emit(f"- workshop/figs/a4_followup_dose_response.{{pdf,png}} (Figure 4; naive r = {r:+.4f})")

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
