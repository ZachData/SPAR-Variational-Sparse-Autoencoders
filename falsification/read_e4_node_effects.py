"""E4 A1 -- which features do SCR and TPP actually select, by usage rank?
(PROJECT.md Next steps A1; Claims-worth-opening #6c.)

Addenda 10/12/13/14 established that SCR says the vSAE's advantage is not
explained by dictionary size while TPP says it is, and that the baseline's own
`top_usage` masking curve is flat/noisy for SCR but clean and near-monotonic for
TPP. This script asks *why* the two curves have such different shapes: does
`top_usage` masking (keep the highest-firing-rate features, discard the rest)
throw away exactly the features SCR's own effect computation would have picked,
while keeping the ones TPP's would have picked?

Stated prediction (PROJECT.md): SCR's top-effect features sit LOW in the usage
ranking; TPP's sit HIGH. Falsifier, stated in advance: if both metrics' top-effect
features sit at comparable usage ranks, the explanation is dead.

No new LLM forward passes, no new probes, no new training -- this drives
`get_all_node_effects_for_one_sae` (the same SAEBench internal `e4_bootstrap.py`
already calls) on the **train-set activations already cached** by
`run_e4_scr.py` / `run_e4_tpp.py` in `falsification/e4_{scr,tpp}_artifacts/`, and
cross-references the resulting per-feature effect ranking against the
usage-rank arrays already committed in `comprehensive_histogram_analysis/`'s
`all_histograms_*.npz`. Runs on both the baseline SAE (whose masking curve is
what addenda 10-11's verdict is read off) and the vSAE (the model the whole
question is about).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SAEBENCH = REPO / "SAEBench-main"
for _p in (str(REPO), str(SAEBENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from falsification.e4_local_sae import load_local_topk_sae, load_local_vsae_topk_sae  # noqa: E402

import sae_bench.evals.scr_and_tpp.main as st  # noqa: E402
import sae_bench.sae_bench_utils.dataset_info as dataset_info  # noqa: E402

MODEL_NAME = "pythia-70m-deduped"
DATASET_NAME = "LabHC/bias_in_bios_class_set1"
COLUMN1_VALS = ("professor", "nurse")
SAE_BATCH_SIZE = 128

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

# The two SCR classes that are actually ablated (get_scr_plotting_dict's `dirs`);
# the third PAIRED_CLASS_KEYS entry ("male_professor / female_nurse") is only an
# eval target, never an ablated class, so it has no "top-effect features" to read.
SCR_CLASSES = ["male / female", "professor / nurse"]
TPP_CLASSES = dataset_info.chosen_classes_per_dataset[DATASET_NAME]
TPP_CLASS_LABELS = {c: dataset_info.profession_int_to_str[int(c)] for c in TPP_CLASSES}

TOP_N_VALUES = [2, 5, 10, 20]


def usage_percentile(usage_counts: np.ndarray) -> np.ndarray:
    """Percentile rank of each feature's usage count, 0 = least used, 1 = most used.

    Ties (mostly among dead features, count 0) get the same, averaged rank --
    `scipy`-free implementation via argsort-of-argsort would break ties
    arbitrarily, so use `np.unique`'s inverse to average within tie groups.
    """
    order = np.argsort(usage_counts, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(usage_counts))
    # average tied ranks (dominated by the mass of dead, count==0 features)
    _, inverse, counts = np.unique(usage_counts, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    avg_rank_per_value = sums / counts
    ranks = avg_rank_per_value[inverse]
    return ranks / (len(usage_counts) - 1)


def load_cache(artifacts_folder: Path, acts_name: str, probes_name: str, device: str):
    acts_path = artifacts_folder / acts_name
    probes_path = artifacts_folder / probes_name
    assert acts_path.exists(), f"missing cache {acts_path} -- run the matching run_e4_*.py first"
    assert probes_path.exists(), f"missing cache {probes_path}"
    acts = torch.load(acts_path, map_location="cpu")
    train_acts = {k: v.clone().contiguous().to(device) for k, v in acts["train"].items()}
    del acts
    with open(probes_path, "rb") as f:
        llm_probes = pickle.load(f)["llm_probes"]
    return train_acts, llm_probes


def read_effects_for_sae(sae, usage_ranks: dict[str, np.ndarray]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {"scr": {}, "tpp": {}}

    scr_train, scr_probes = load_cache(
        REPO / "falsification/e4_scr_artifacts",
        f"{DATASET_NAME}_{COLUMN1_VALS[0]}_{COLUMN1_VALS[1]}_activations.pt".replace("/", "_"),
        f"{DATASET_NAME}_{COLUMN1_VALS[0]}_{COLUMN1_VALS[1]}_probes.pkl".replace("/", "_"),
        device,
    )
    scr_effects = st.get_all_node_effects_for_one_sae(
        sae, scr_probes, SCR_CLASSES, True, scr_train, SAE_BATCH_SIZE
    )
    for cls, effects_F in scr_effects.items():
        out["scr"][cls] = summarise_class(cls, effects_F, usage_ranks)
    del scr_train
    torch.cuda.empty_cache()

    tpp_train, tpp_probes = load_cache(
        REPO / "falsification/e4_tpp_artifacts",
        f"{DATASET_NAME}_activations.pt".replace("/", "_"),
        f"{DATASET_NAME}_probes.pkl".replace("/", "_"),
        device,
    )
    tpp_effects = st.get_all_node_effects_for_one_sae(
        sae, tpp_probes, TPP_CLASSES, False, tpp_train, SAE_BATCH_SIZE
    )
    for cls, effects_F in tpp_effects.items():
        out["tpp"][TPP_CLASS_LABELS[cls]] = summarise_class(cls, effects_F, usage_ranks)
    del tpp_train
    torch.cuda.empty_cache()

    return out


def summarise_class(cls_name: str, effects_F: torch.Tensor, usage_ranks: dict[str, np.ndarray]) -> dict:
    effects = effects_F.detach().float().cpu().numpy()
    n_nonzero = int(np.count_nonzero(effects))
    order = np.argsort(-effects)  # descending; SCR effects already abs'd, TPP's are clamped >= 0
    result = {"n_nonzero_effects": n_nonzero}
    for n in TOP_N_VALUES:
        top_idx = order[: min(n, n_nonzero) if n_nonzero > 0 else 0]
        entry = {
            "feature_indices": [int(i) for i in top_idx],
            "effects": [float(effects[i]) for i in top_idx],
        }
        for arr_name, ranks in usage_ranks.items():
            entry[f"usage_percentile_{arr_name}"] = (
                [float(ranks[i]) for i in top_idx] if len(top_idx) else []
            )
            entry[f"mean_usage_percentile_{arr_name}"] = (
                float(np.mean([ranks[i] for i in top_idx])) if len(top_idx) else float("nan")
            )
        result[f"top_{n}"] = entry
    # also the single top-1 feature's raw usage count, for a quick eyeball check
    if n_nonzero > 0:
        result["top1_feature"] = int(order[0])
        result["top1_effect"] = float(effects[order[0]])
    return result


def aggregate(per_sae: dict, usage_ranks_key: str) -> dict:
    """Mean usage percentile of top-N features, averaged across a metric's classes."""
    agg = {}
    for metric in ("scr", "tpp"):
        agg[metric] = {}
        for n in TOP_N_VALUES:
            vals = [
                cls_result[f"top_{n}"][f"mean_usage_percentile_{usage_ranks_key}"]
                for cls_result in per_sae[metric].values()
            ]
            vals = [v for v in vals if not np.isnan(v)]
            agg[metric][f"top_{n}_mean_usage_percentile"] = float(np.mean(vals)) if vals else None
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(REPO / "falsification/e4_node_effects_results.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.set_grad_enabled(False)

    print(f"device={device}")
    print(f"SCR ablated classes: {SCR_CLASSES}")
    print(f"TPP classes: {TPP_CLASSES} -> {TPP_CLASS_LABELS}")

    baseline_usage = np.load(BASELINE_NPZ)["feature_selection_counts"]
    vsae_usage = np.load(VSAE_NPZ)["feature_selection_counts"]
    usage_ranks = {
        "baseline": usage_percentile(baseline_usage),
        "vsae": usage_percentile(vsae_usage),
    }
    print(f"baseline live features: {np.count_nonzero(baseline_usage)}/{len(baseline_usage)}")
    print(f"vsae live features: {np.count_nonzero(vsae_usage)}/{len(vsae_usage)}")

    baseline_sae = load_local_topk_sae(BASELINE_DIR, MODEL_NAME, device, dtype)
    vsae_sae = load_local_vsae_topk_sae(VSAE_DIR, MODEL_NAME, device, dtype)
    assert baseline_sae.cfg.hook_name == vsae_sae.cfg.hook_name

    results = {}
    for label, sae in (("baseline", baseline_sae), ("vsae", vsae_sae)):
        print(f"\n=== reading node effects for {label} ===")
        per_sae = read_effects_for_sae(sae, usage_ranks)
        results[label] = per_sae
        print(f"--- {label}: mean usage percentile of top-effect features (own dictionary's usage rank) ---")
        agg_own = aggregate(per_sae, label)
        print(json.dumps(agg_own, indent=2))

    out = {
        "dataset": DATASET_NAME,
        "scr_classes": SCR_CLASSES,
        "tpp_classes": TPP_CLASS_LABELS,
        "top_n_values": TOP_N_VALUES,
        "results": {
            label: {
                metric: {cls: res for cls, res in per_metric.items()}
                for metric, per_metric in per_sae.items()
            }
            for label, per_sae in results.items()
        },
        "aggregate": {
            label: {
                "own_usage_rank": aggregate(results[label], label),
                "baseline_usage_rank": aggregate(results[label], "baseline"),
                "vsae_usage_rank": aggregate(results[label], "vsae"),
            }
            for label in results
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n=== summary: mean usage percentile of top-effect features, own dictionary's usage rank ===")
    for label in results:
        print(f"-- {label} --")
        print(json.dumps(out["aggregate"][label]["own_usage_rank"], indent=2))


if __name__ == "__main__":
    main()
