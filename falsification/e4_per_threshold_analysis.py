"""E4 -- does the SCR/TPP size verdict hold at every ablation threshold, or is it
an artifact of averaging across `n_values`? (PROJECT.md Next steps #0 item 1,
Claims-worth-opening #6a.)

`run_e4_scr.py` / `run_e4_tpp.py` report one verdict from the mean of the four
`n_values=[2,5,10,20]` per-run scores. That is the same shape-behind-a-scalar
risk the two-threshold liveness rule (F8b) exists to catch. Now that both
runners store the full per-threshold dict at every baseline grid point
(`baseline_curve[i]["per_threshold"]`), redo the verdict once per threshold:
build the `top_usage` size-response curve at that threshold alone, interpolate
to the vSAE's live count, and compare the vSAE's own score at that threshold.

Prints, per metric and threshold: the vSAE score, the interpolated `top_usage`
and `random` references at n=1474, the margins, and the per-threshold verdict --
then whether those verdicts agree with each other and with the mean-based one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _threshold_of(key: str) -> int:
    m = re.search(r"threshold_(\d+)", key)
    if not m:
        raise ValueError(f"no threshold in {key!r}")
    return int(m.group(1))


def _curve_at_threshold(baseline_curve, strategy, thr_key):
    """(xs, ys) for one strategy at one threshold: ys is the mean over draws."""
    pts = [p for p in baseline_curve if p["strategy"] == strategy]
    pts.sort(key=lambda p: p["n_features"])
    xs = np.array([p["n_features"] for p in pts], dtype=float)
    ys = np.array(
        [float(np.mean([d[thr_key] for d in p["per_threshold"]])) for p in pts],
        dtype=float,
    )
    return xs, ys


def analyse(path: Path, label: str) -> dict:
    data = json.loads(path.read_text())
    curve = data["baseline_curve"]
    live_n = data["vsae"]["live_n"]
    vsae_pt = data["vsae"]["per_threshold"]

    thr_keys = sorted(vsae_pt, key=_threshold_of)
    print(f"\n{'=' * 72}\n{label}  (vSAE live_n = {live_n})\n{'=' * 72}")
    header = f"{'thr':>5} {'vSAE':>9} {'top_usage@n':>12} {'margin':>9} {'random@n':>10} {'margin':>9}  verdict"
    print(header)

    rows = []
    for k in thr_keys:
        thr = _threshold_of(k)
        vs = float(vsae_pt[k])

        xs_t, ys_t = _curve_at_threshold(curve, "top_usage", k)
        ref_t = float(np.interp(float(live_n), xs_t, ys_t))
        xs_r, ys_r = _curve_at_threshold(curve, "random", k)
        ref_r = float(np.interp(float(live_n), xs_r, ys_r))

        m_t = vs - ref_t
        m_r = vs - ref_r
        expl = m_t <= 0.0  # "explained by size" == vSAE at or below top_usage
        verdict = "EXPLAINED by size" if expl else "NOT explained"
        print(
            f"{thr:>5} {vs:>9.4f} {ref_t:>12.4f} {m_t:>+9.4f} {ref_r:>10.4f} {m_r:>+9.4f}  {verdict}"
        )
        rows.append((thr, vs, ref_t, m_t, ref_r, m_r, expl))

    mean_expl = data["verdict"]["top_usage"]["explained_by_size"]
    per_thr_expl = [r[6] for r in rows]
    uniform = len(set(per_thr_expl)) == 1
    agrees_mean = all(e == mean_expl for e in per_thr_expl)
    print(
        f"\n  mean-based verdict (from JSON): "
        f"{'EXPLAINED by size' if mean_expl else 'NOT explained'}"
    )
    print(
        f"  per-threshold verdicts uniform? {uniform}   "
        f"(all agree with mean-based? {agrees_mean})"
    )
    if not uniform:
        flip = [r[0] for r in rows if r[6] != mean_expl]
        print(f"  --> thresholds that DISAGREE with the mean-based verdict: {flip}")

    # also: is the top_usage curve monotone-ish in N at each threshold, or flat?
    print("\n  top_usage curve range across the size grid, per threshold:")
    for k in thr_keys:
        xs_t, ys_t = _curve_at_threshold(curve, "top_usage", k)
        print(
            f"    thr {_threshold_of(k):>3}: min {ys_t.min():+.4f}  max {ys_t.max():+.4f}  "
            f"span {ys_t.max() - ys_t.min():.4f}  (N=100 {ys_t[0]:+.4f} -> N=7379 {ys_t[-1]:+.4f})"
        )

    return {"rows": rows, "mean_expl": mean_expl, "uniform": uniform, "agrees_mean": agrees_mean}


def main():
    analyse(REPO / "falsification/e4_scr_results.json", "SCR (professor/nurse)")
    analyse(REPO / "falsification/e4_tpp_results.json", "TPP (5 classes)")


if __name__ == "__main__":
    main()
