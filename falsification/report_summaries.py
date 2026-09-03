"""Collect the reporting blocks from every analysed run.

Prints the `model_info` and `feature_usage_summary` blocks from each
`comprehensive_summary_*.json`, plus a compact cross-arm table.

    python falsification/report_summaries.py             # blocks + table
    python falsification/report_summaries.py --table     # table only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def liveness(summary_path: Path) -> dict | None:
    """Liveness statistics computed from EXACT per-feature selection counts.

    `features_used` ("selected at least once") saturates at this configuration:
    TopK selects exactly k of d features per sample, so mean selection frequency
    is k/d = 0.125 by construction and essentially every feature fires sometimes
    over 1e6 samples. That is a property of the design, knowable before any result
    is seen, which is what makes re-specifying the metric legitimate rather than
    outcome-driven.

    The analyzer already stores `feature_selection_counts` (one exact integer per
    feature) in its .npz, so no re-analysis is needed and there is no binning
    resolution limit -- an earlier version of this function read the 50-bin
    frequency histogram, whose first bin is 0.02 wide and cannot separate "never
    selected" from "selected rarely".

    Returns the sparsity-relative statistic proposed in REMEDIATION.md (F8b):
    the fraction of features selected in fewer than `rel * (k/d)` of samples.
    Defined relative to the design's own sparsity, it transfers between d=2048
    and d=8192; `features_used` does not, which is why the preprint's d=8192
    numbers never exposed this.
    """
    import json as _json

    import numpy as np

    npz = next(summary_path.parent.glob("all_histograms_*.npz"), None)
    if npz is None:
        return None
    data = np.load(npz)
    if "feature_selection_counts" not in data.files:
        return None
    counts = np.asarray(data["feature_selection_counts"], dtype=float)

    summary = _json.loads(summary_path.read_text())
    n_samples = summary.get("processing_info", {}).get("samples_processed")
    info = summary.get("model_info", {})
    k, d = info.get("k_value"), info.get("dict_size")
    if not n_samples or not k or not d:
        return None

    freq = counts / float(n_samples)
    expected = k / d                      # mean selection frequency, by construction
    out = {
        "n_features": int(counts.size),
        "min_count": float(counts.min()),
        "never": float(np.mean(counts == 0)),
        "expected_freq": expected,
    }
    for rel in (0.1, 0.5):
        out[f"below_{rel}x"] = float(np.mean(freq < rel * expected))
    return out


def runs() -> list[tuple[str, int, Path]]:
    found = []
    for path in sorted(REPO.glob("experiments/*/seed*/*/comprehensive_summary_*.json")):
        arm = path.relative_to(REPO).parts[1]
        seed = int(re.search(r"seed(\d+)", str(path)).group(1))
        found.append((arm, seed, path))
    return sorted(found, key=lambda r: (r[0], r[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="table only, omit the blocks")
    args = ap.parse_args()

    found = runs()
    if not found:
        print("No comprehensive_summary_*.json found. Run ./run_analysis.sh first.")
        return 1

    rows = []
    for arm, seed, path in found:
        data = json.loads(path.read_text())
        info = data.get("model_info", {})
        usage = data.get("feature_usage_summary", {})
        proc = data.get("processing_info", {})

        if not args.table:
            print("=" * 74)
            print(f"{arm}  seed={seed}")
            print(f"  {path.relative_to(REPO)}")
            print("=" * 74)
            print("model_info:")
            print(json.dumps(info, indent=2))
            print("feature_usage_summary:")
            print(json.dumps(usage, indent=2))
            print()

        total = usage.get("total_features")
        used = usage.get("features_used")
        rows.append((
            arm, seed, used, total,
            (used / total) if (used is not None and total) else None,
            usage.get("features_never_selected"),
            proc.get("samples_processed") or proc.get("target_samples"),
            liveness(path),
        ))

    print("=" * 74)
    print("Cross-arm summary")
    print("=" * 74)
    print(f"{'arm':<16} {'seed':>4} {'used':>7} {'live frac':>10} {'min count':>10} "
          f"{'<0.1x k/d':>10} {'<0.5x k/d':>10}")
    for arm, seed, used, total, frac, dead, n, lv in rows:
        frac_s = f"{frac:.4f}" if frac is not None else "-"
        if lv:
            print(f"{arm:<16} {seed:>4} {str(used):>7} {frac_s:>10} "
                  f"{lv['min_count']:>10.0f} {lv['below_0.1x']:>10.4f} "
                  f"{lv['below_0.5x']:>10.4f}")
        else:
            print(f"{arm:<16} {seed:>4} {str(used):>7} {frac_s:>10} {'-':>10} "
                  f"{'-':>10} {'-':>10}")

    if any(r[4] == 1.0 for r in rows):
        print("\n  NOTE: features_used saturates at this dictionary size and is\n"
              "  not usable as the live-feature metric here. TopK selects exactly k\n"
              "  of d features per sample, so mean selection frequency is k/d = 0.125\n"
              "  by construction and essentially every feature fires sometimes over\n"
              "  1e6 samples. That is a design property, knowable a priori.\n"
              "  '<0.1x k/d' and '<0.5x k/d' are computed from exact per-feature\n"
              "  counts (feature_selection_counts in the .npz), are defined relative\n"
              "  to the design's own sparsity so they transfer across dictionary\n"
              "  sizes, and do discriminate. See REMEDIATION.md F8b -- the threshold\n"
              "  must be pre-registered before these are used as a falsification test.")

    # features_used is sample-size dependent; comparability requires one value.
    sizes = {r[6] for r in rows if r[6] is not None}
    if len(sizes) > 1:
        print(f"\n  WARNING: mixed --n-samples across runs: {sorted(sizes)}. "
              "features_used is sample-size dependent, so these are NOT comparable.")
    elif sizes:
        print(f"\n  All runs analysed at n_samples = {sizes.pop()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
