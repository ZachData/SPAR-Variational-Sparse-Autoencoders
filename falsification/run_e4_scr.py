"""E4 -- is the vSAE's SCR advantage explained by dictionary size? (PROJECT.md Next steps #0)

Wires `falsification/size_control.py`'s tested-but-scorer-less framework to a
real SAEBench SCR scorer, against the recovered Pythia checkpoints
(`experiments/e4_pythia_baseline/`, `experiments/e4_pythia_vsae/`, both seed 42,
`CLAUDE.md` landmine 3: the baseline has `auxk_alpha=0.03125`, the vSAE has 0).

Design (PROJECT.md E4): mask the baseline's own dictionary down to N features by
usage, at a grid of N spanning the vSAE's live count (1474 of 8192, from the
committed `all_histograms_*.npz`); score SCR at each N; score the vSAE once,
unmasked, at its own natural size; and ask whether the vSAE's score sits above
the baseline's own best-N-features curve (`top_usage` reference -- the `random`
reference is a manufactured false positive, PROJECT.md and
`falsification/tests/test_size_control.py` both demonstrate why).

Restricted to one dataset (bias_in_bios) and one class pair (professor/nurse --
the pair `get_scr_plotting_dict`'s hardcoded "male_professor / female_nurse"
metric name was written for) to keep the grid affordable; `--smoke` shrinks
train/test set size and probe epochs further to validate the pipeline end to
end before spending the full run's time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SAEBENCH = REPO / "SAEBench-main"
for _p in (str(REPO), str(SAEBENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from falsification.e4_local_sae import load_local_topk_sae, load_local_vsae_topk_sae
from falsification.size_control import (
    interpolate_curve,
    size_response_curve,
    verdict,
)

import sae_bench.evals.scr_and_tpp.main as scr_and_tpp  # noqa: E402
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

MODEL_NAME = "pythia-70m-deduped"
DATASET_NAME = "LabHC/bias_in_bios_class_set1"
COLUMN1_VALS = ("professor", "nurse")
RUN_NAME = f"{DATASET_NAME}_scr_{COLUMN1_VALS[0]}_{COLUMN1_VALS[1]}"

BASELINE_DIR = (
    REPO
    / "experiments/e4_pythia_baseline/seed42/"
    "TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto/trainer_0"
)
VSAE_DIR = (
    REPO
    / "experiments/e4_pythia_vsae/seed42/"
    "VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var/trainer_0"
)
BASELINE_NPZ = (
    REPO
    / "comprehensive_histogram_analysis/TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto/"
    "all_histograms_TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto.npz"
)
VSAE_NPZ = (
    REPO
    / "comprehensive_histogram_analysis/VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var/"
    "all_histograms_VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var.npz"
)


def score_from_results(results: dict) -> dict[str, float]:
    """The per-threshold `scr_metric_threshold_N` values `get_scr_plotting_dict` writes."""
    run_results = results[f"{RUN_NAME}_results"]
    return {k: v for k, v in run_results.items() if k.startswith("scr_metric_threshold_")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="tiny config, to validate the pipeline")
    p.add_argument(
        "--n-grid",
        type=int,
        nargs="+",
        default=None,
        help="dictionary sizes to score the baseline at (default: a grid around the vSAE's live count)",
    )
    p.add_argument("--out", default=str(REPO / "falsification/e4_scr_results.json"))
    p.add_argument(
        "--random-draws",
        type=int,
        default=10,
        help="draws of the random-subset reference per grid point (PROJECT.md: report both to bracket the answer)",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    if args.smoke:
        config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            perform_scr=True,
            dataset_names=[DATASET_NAME],
            column1_vals_lookup={DATASET_NAME: [COLUMN1_VALS]},
            n_values=[2, 5],
            train_set_size=200,
            test_set_size=50,
            probe_epochs=3,
            llm_batch_size=64,
            llm_dtype="float32",
        )
        n_grid = args.n_grid or [500, 1474]
    else:
        config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            perform_scr=True,
            dataset_names=[DATASET_NAME],
            column1_vals_lookup={DATASET_NAME: [COLUMN1_VALS]},
            n_values=[2, 5, 10, 20],
            llm_batch_size=512,
            llm_dtype="float32",
        )
        n_grid = args.n_grid or [100, 250, 500, 1000, 1474, 2000, 3000, 5000, 7379]

    print(f"device={device} smoke={args.smoke} n_grid={n_grid}")

    baseline = load_local_topk_sae(BASELINE_DIR, MODEL_NAME, device, dtype)
    vsae = load_local_vsae_topk_sae(VSAE_DIR, MODEL_NAME, device, dtype)
    assert baseline.cfg.hook_name == vsae.cfg.hook_name, (
        baseline.cfg.hook_name,
        vsae.cfg.hook_name,
    )
    print(f"loaded both checkpoints, hook={baseline.cfg.hook_name}")

    usage_counts = np.load(BASELINE_NPZ)["feature_selection_counts"]
    vsae_usage_counts = np.load(VSAE_NPZ)["feature_selection_counts"]
    vsae_live_n = int(np.count_nonzero(vsae_usage_counts))
    print(f"baseline usage counts: {np.count_nonzero(usage_counts)} live of {len(usage_counts)}")
    print(f"vSAE live count: {vsae_live_n}")

    artifacts_folder = str(REPO / "falsification/e4_scr_artifacts")
    os.makedirs(artifacts_folder, exist_ok=True)

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=torch.float32
    )
    print("LLM loaded")

    n_calls = 0
    per_call_thresholds: list[dict[str, float]] = []

    def scorer(keep_indices: np.ndarray) -> float:
        nonlocal n_calls
        n_calls += 1
        baseline.apply_mask(keep_indices)
        results, _ = scr_and_tpp.run_eval_single_sae(
            config, baseline, model, device, artifacts_folder, save_activations=True
        )
        scores = score_from_results(results)
        per_call_thresholds.append(scores)
        score = float(np.mean(list(scores.values())))
        print(f"  scorer call {n_calls}: n={len(keep_indices)} score={score:.4f} ({scores})")
        return score

    points = size_response_curve(
        usage_counts,
        scorer,
        n_grid=n_grid,
        strategies=("top_usage", "random"),
        n_draws=args.random_draws,
        seed=0,
    )

    # `size_response_curve` calls `scorer` in exactly the order it appends points
    # -- for each strategy, for each n, `draws` times (1 for top_usage, n_draws
    # for random) -- so re-walking that order reattaches each call's full
    # per-threshold dict to its grid point. Without this only the mean across
    # n_values survives, which is the shape-behind-a-scalar risk the two-threshold
    # liveness rule (F8b) exists to catch (PROJECT.md Claims-worth-opening #6a).
    _calls = iter(per_call_thresholds)
    per_point_thresholds = [[next(_calls) for _ in pt.scores] for pt in points]
    assert next(_calls, None) is None, "per-call threshold log did not line up with curve points"

    print("scoring vSAE, unmasked, at its own live count")
    vsae_results, _ = scr_and_tpp.run_eval_single_sae(
        config, vsae, model, device, artifacts_folder, save_activations=True
    )
    vsae_scores = score_from_results(vsae_results)
    vsae_score = float(np.mean(list(vsae_scores.values())))
    print(f"  vSAE score={vsae_score:.4f} ({vsae_scores})")

    v_top = verdict(vsae_score, vsae_live_n, points, reference_strategy="top_usage")
    v_rand = verdict(vsae_score, vsae_live_n, points, reference_strategy="random")
    print()
    print("top_usage (decisive):", str(v_top))
    print("random (bracket only):", str(v_rand))

    out = {
        "dataset": DATASET_NAME,
        "column1_vals": list(COLUMN1_VALS),
        "n_values": config.n_values,
        "smoke": args.smoke,
        "baseline_curve": [
            {
                "n_features": pt.n_features,
                "strategy": pt.strategy,
                "scores": pt.scores,
                "mean": pt.mean,
                "per_threshold": thr,
            }
            for pt, thr in zip(points, per_point_thresholds)
        ],
        "vsae": {
            "live_n": vsae_live_n,
            "score": vsae_score,
            "per_threshold": vsae_scores,
            "top_usage_reference_at_vsae_n": interpolate_curve(points, vsae_live_n, "top_usage"),
            "random_reference_at_vsae_n": interpolate_curve(points, vsae_live_n, "random"),
        },
        "verdict": {
            "top_usage": {
                "explained_by_size": v_top.explained_by_size,
                "margin": v_top.margin,
                "reference_score": v_top.reference_score,
            },
            "random": {
                "explained_by_size": v_rand.explained_by_size,
                "margin": v_rand.margin,
                "reference_score": v_rand.reference_score,
            },
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
