"""E4 -- bootstrap error bars on the SCR/TPP size-response curves.
(PROJECT.md Next steps #0 box (2); Claims-worth-opening #6b.)

Addenda 10-12 read one number per grid point off a single test set. SCR's
`top_usage` size-response curve came out flat and noisy (0.004-0.033 across two
orders of magnitude of N); TPP's came out clean and near-monotonic. #6b asks
whether SCR's flat curve is a *real* absence of size-response or just noise
around zero -- which bears directly on how much weight addendum 10's "not
explained by size" verdict should carry against addendum 11's opposite one.

This resamples the cached **test set** with replacement and re-scores, with no
new LLM or SAE forward passes over new data -- it reuses
`falsification/e4_{scr,tpp}_artifacts/` (cached LLM activations + trained probes)
exactly as they are, and only re-indexes into them. It drives SAEBench's own
scoring internals (`get_all_node_effects_for_one_sae`,
`perform_feature_ablations`, `get_probe_test_accuracy`, `get_scr_plotting_dict`
/ `create_tpp_plotting_dict`) rather than re-implementing them.

Scope, stated plainly:
  * The bootstrap is **conditional on the feature-selection decision.** Node
    effects (which features SCR/TPP chooses to ablate at each threshold) are
    computed once from the full *train* set and held fixed; only test-set
    sampling noise is resampled. `--resample-train` also bootstraps the train
    set (re-derives node effects every draw) -- slower, and a different
    question.
  * Single seed each side, one dataset, one class pair -- every addendum 10-12
    caveat still applies. This puts an error bar on a descriptive number; it
    does not turn it into a permutation test.

The paired quantity that answers #6b is `margin_b = vsae_score_b -
top_usage@1474_b`, computed on the *same* resampled test indices in draw `b`:
its bootstrap CI is what tells us whether the SCR verdict (margin > 0) and the
TPP verdict (margin < 0) survive test-set resampling.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SAEBENCH = REPO / "SAEBench-main"
for _p in (str(REPO), str(SAEBENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from falsification.e4_local_sae import load_local_topk_sae, load_local_vsae_topk_sae  # noqa: E402
from falsification.size_control import select_features  # noqa: E402

import sae_bench.evals.scr_and_tpp.main as st  # noqa: E402
import sae_bench.sae_bench_utils.activation_collection as activation_collection  # noqa: E402
import sae_bench.sae_bench_utils.dataset_info as dataset_info  # noqa: E402
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig  # noqa: E402

MODEL_NAME = "pythia-70m-deduped"
DATASET_NAME = "LabHC/bias_in_bios_class_set1"
COLUMN1_VALS = ("professor", "nurse")

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

FULL_CURVE = [100, 250, 500, 1000, 1474, 2000, 3000, 5000, 7379]


def score_scr(raw_results, llm_clean_accs) -> dict[str, float]:
    d = st.get_scr_plotting_dict(raw_results, llm_clean_accs)
    return {k: float(v) for k, v in d.items() if k.startswith("scr_metric_threshold_")}


def score_tpp(raw_results, llm_clean_accs) -> dict[str, float]:
    overall, _ = st.create_tpp_plotting_dict(raw_results, llm_clean_accs)
    return {k: float(v) for k, v in overall.items() if k.endswith("_total_metric")}


def resample_acts(acts_BLD_by_class: dict, rng: np.random.Generator) -> dict:
    """Draw, per class, `n_c` row indices with replacement from that class's pool."""
    out = {}
    for name, t in acts_BLD_by_class.items():
        n = t.shape[0]
        idx = torch.as_tensor(rng.integers(0, n, size=n), device=t.device)
        out[name] = t.index_select(0, idx)
    return out


def summarise(draws: np.ndarray) -> dict:
    """Percentile summary of a 1-D bootstrap sample."""
    draws = np.asarray(draws, dtype=float)
    return {
        "mean": float(draws.mean()),
        "std": float(draws.std(ddof=1)),
        "p2.5": float(np.percentile(draws, 2.5)),
        "p50": float(np.percentile(draws, 50)),
        "p97.5": float(np.percentile(draws, 97.5)),
        "frac_above_0": float((draws > 0).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metric", choices=["scr", "tpp"], required=True)
    p.add_argument("--boot", type=int, default=200, help="bootstrap resamples")
    p.add_argument(
        "--n-grid",
        type=int,
        nargs="+",
        default=None,
        help="baseline top_usage grid (default: the full addendum-10/11 curve)",
    )
    p.add_argument(
        "--resample-train",
        action="store_true",
        help="also bootstrap the train set (re-derive node effects per draw); slower",
    )
    p.add_argument("--smoke", action="store_true", help="boot=8, grid=[1474], fast sanity check")
    p.add_argument("--sae-batch-size", type=int, default=128,
                   help="SAE inference batch; >256 OOMs the 10GB card on the [B,128,8192] encode intermediate")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    perform_scr = args.metric == "scr"
    boot = 8 if args.smoke else args.boot
    n_grid = args.n_grid or ([1474] if args.smoke else FULL_CURVE)
    out_path = args.out or str(REPO / f"falsification/e4_bootstrap_{args.metric}_results.json")
    partial_path = out_path + ".partial"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    if perform_scr:
        config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            perform_scr=True,
            dataset_names=[DATASET_NAME],
            column1_vals_lookup={DATASET_NAME: [COLUMN1_VALS]},
            n_values=[2, 5, 10, 20],
            llm_batch_size=512,
            llm_dtype="float32",
        )
        chosen_classes = list(dataset_info.PAIRED_CLASS_KEYS.keys())
        artifacts_folder = REPO / "falsification/e4_scr_artifacts"
        acts_name = f"{DATASET_NAME}_{COLUMN1_VALS[0]}_{COLUMN1_VALS[1]}_activations.pt".replace("/", "_")
        probes_name = f"{DATASET_NAME}_{COLUMN1_VALS[0]}_{COLUMN1_VALS[1]}_probes.pkl".replace("/", "_")
        score_fn = score_scr
    else:
        config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            perform_scr=False,
            dataset_names=[DATASET_NAME],
            n_values=[2, 5, 10, 20],
            llm_batch_size=512,
            llm_dtype="float32",
        )
        chosen_classes = dataset_info.chosen_classes_per_dataset[DATASET_NAME]
        artifacts_folder = REPO / "falsification/e4_tpp_artifacts"
        acts_name = f"{DATASET_NAME}_activations.pt".replace("/", "_")
        probes_name = f"{DATASET_NAME}_probes.pkl".replace("/", "_")
        score_fn = score_tpp

    config.sae_batch_size = args.sae_batch_size
    print(f"metric={args.metric} boot={boot} n_grid={n_grid} resample_train={args.resample_train} "
          f"sae_batch_size={config.sae_batch_size}")
    print(f"device={device}")

    acts_path = artifacts_folder / acts_name
    probes_path = artifacts_folder / probes_name
    assert acts_path.exists(), f"missing cache {acts_path} -- run run_e4_{args.metric}.py first"
    assert probes_path.exists(), f"missing cache {probes_path}"

    baseline = load_local_topk_sae(BASELINE_DIR, MODEL_NAME, device, dtype)
    vsae = load_local_vsae_topk_sae(VSAE_DIR, MODEL_NAME, device, dtype)
    assert baseline.cfg.hook_name == vsae.cfg.hook_name

    usage_counts = np.load(BASELINE_NPZ)["feature_selection_counts"]
    vsae_live_n = int(np.count_nonzero(np.load(VSAE_NPZ)["feature_selection_counts"]))
    print(f"vSAE live count: {vsae_live_n}")

    print(f"loading cached activations from {acts_path} ...")
    # Load to CPU first: the .pt on disk is ~2x the tensors' logical size (each
    # was sliced from a bigger buffer at save time and pickle keeps the whole
    # storage), so `.clone()` every tensor to trim it before anything goes to GPU.
    acts = torch.load(acts_path, map_location="cpu")
    train_acts = {k: v.clone().contiguous() for k, v in acts["train"].items()}  # kept on CPU
    # Conditional bootstrap keeps the test set resident on GPU (every draw resamples it).
    # Under --resample-train the GPU also has to hold a ~3GB resampled train copy per draw,
    # so keep test on CPU too and stage each draw's resampled copy (10GB card can't hold both).
    _test_dev = "cpu" if args.resample_train else device
    test_acts = {k: v.clone().contiguous().to(_test_dev) for k, v in acts["test"].items()}
    del acts
    gc.collect()
    with open(probes_path, "rb") as f:
        llm_probes = pickle.load(f)["llm_probes"]
    print(f"  train classes (cpu): {[(k, tuple(v.shape)) for k, v in train_acts.items()]}")
    print(f"  test  classes (gpu): {[(k, tuple(v.shape)) for k, v in test_acts.items()]}")

    torch.set_grad_enabled(False)

    # ----- scoring contexts: (label, sae, keep_indices or None) -----
    mask_rng = np.random.default_rng(0)
    contexts: list[tuple[str, object, np.ndarray | None]] = [("vsae", vsae, None)]
    for n in n_grid:
        contexts.append(
            (f"top_usage_{n}", baseline, select_features(usage_counts, n, "top_usage"))
        )
    # one fixed random-subset reference at the vSAE's own size, to bracket the answer
    contexts.append(
        (f"random_{vsae_live_n}", baseline, select_features(usage_counts, vsae_live_n, "random", mask_rng))
    )

    def node_effects_for(sae, keep, train_by_class):
        if keep is not None:
            sae.apply_mask(keep)
        return st.get_all_node_effects_for_one_sae(
            sae, llm_probes, chosen_classes, perform_scr, train_by_class, config.sae_batch_size
        )

    # Node effects from the full train set, once per context (the conditional-bootstrap default).
    # Train acts live on CPU; stage them on the GPU only for as long as this phase needs.
    # Under --resample-train the full train set is NOT kept resident: a resampled copy is
    # staged per draw instead (train + test + a resampled train copy all on GPU at once
    # OOMs the 10GB card on SCR's 6-class 2000/class train set).
    fixed_node_effects = {}
    if not args.resample_train:
        train_gpu = {k: v.to(device) for k, v in train_acts.items()}
        for label, sae, keep in contexts:
            torch.manual_seed(config.random_seed)
            fixed_node_effects[label] = node_effects_for(sae, keep, train_gpu)
            gc.collect()
            torch.cuda.empty_cache()
        print("precomputed fixed node effects for all contexts")
        del train_gpu
        gc.collect()
        torch.cuda.empty_cache()

    thr_keys = [f"{'scr_metric' if perform_scr else 'tpp'}_threshold_{n}" + ("" if perform_scr else "_total_metric") for n in config.n_values]

    # raw[label]["mean"] -> list[boot], raw[label][thr_key] -> list[boot]
    raw: dict[str, dict[str, list[float]]] = {lab: {"mean": []} for lab, _, _ in contexts}
    for lab in raw:
        for tk in thr_keys:
            raw[lab][tk] = []

    # ----- resume from an interrupted run -----
    # The per-draw RNG is seeded by draw index (default_rng(1000 + b)), so draws
    # are independent of where we resume; a .partial file lets a killed run pick
    # up instead of restarting from zero (both 2026-09-08 attempts were lost this way).
    partial_cfg = {"metric": args.metric, "boot": boot, "n_grid": list(n_grid),
                   "resample_train": args.resample_train}
    start_b = 0
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            prev = json.load(f)
        if prev.get("config") == partial_cfg and set(prev.get("raw", {})) == set(raw):
            raw = prev["raw"]
            start_b = int(prev["done"])
            lens = {len(raw[lab][k]) for lab in raw for k in raw[lab]}
            assert lens == {start_b}, f"corrupt partial: draw counts {lens} != {start_b}"
            print(f"resuming from {partial_path}: {start_b}/{boot} draws already done")
        else:
            print(f"ignoring {partial_path}: config mismatch (was {prev.get('config')})")

    def write_partial(done: int) -> None:
        tmp = partial_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"config": partial_cfg, "done": done, "raw": raw}, f)
        os.replace(tmp, partial_path)

    t0 = time.time()
    for b in range(start_b, boot):
        boot_rng = np.random.default_rng(1000 + b)
        test_b = resample_acts(test_acts, boot_rng)
        if args.resample_train:
            test_b = {k: v.to(device) for k, v in test_b.items()}  # test_acts is on CPU in this mode
        meaned_test_b = activation_collection.create_meaned_model_activations(test_b)
        # resample on CPU, then stage just this draw's copy on the GPU
        train_b = (
            {k: v.to(device) for k, v in resample_acts(train_acts, boot_rng).items()}
            if args.resample_train
            else None
        )

        for label, sae, keep in contexts:
            torch.manual_seed(config.random_seed)  # pair prepare_probe_data's internal RNG across contexts
            if keep is not None:
                sae.apply_mask(keep)

            if args.resample_train:
                node_effects = node_effects_for(sae, keep, train_b)
            else:
                node_effects = fixed_node_effects[label]

            llm_clean_accs = st.get_probe_test_accuracy(
                llm_probes, chosen_classes, meaned_test_b, config.probe_test_batch_size, perform_scr
            )
            raw_results = st.perform_feature_ablations(
                llm_probes,
                sae,
                config.sae_batch_size,
                test_b,
                node_effects,
                config.n_values,
                chosen_classes,
                config.probe_test_batch_size,
                perform_scr,
            )
            scores = score_fn(raw_results, llm_clean_accs)
            missing = [tk for tk in thr_keys if tk not in scores]
            if missing:
                # get_scr_plotting_dict drops a threshold's scr_metric key iff
                # dir1_acc == dir2_acc exactly on this resample -- degenerate, rare.
                print(f"  WARN draw {b} context {label}: missing {missing}; recording nan")
            raw[label]["mean"].append(float(np.mean([scores[tk] for tk in thr_keys if tk in scores])))
            for tk in thr_keys:
                raw[label][tk].append(scores.get(tk, float("nan")))

        del test_b, meaned_test_b, train_b
        if args.resample_train:
            gc.collect()
            torch.cuda.empty_cache()  # the ~3GB per-draw train copy must not fragment the 10GB card
        if (b + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            write_partial(b + 1)  # recoverable checkpoint every 10 draws
        if (b + 1) % max(1, boot // 20) == 0 or b == start_b:
            dt = time.time() - t0
            n_done = b - start_b + 1
            print(f"  boot {b + 1}/{boot}  ({dt:.0f}s, {dt / n_done:.1f}s/draw)")

    # ----- summaries -----
    top_key = f"top_usage_{vsae_live_n}" if vsae_live_n in n_grid else None
    rand_key = f"random_{vsae_live_n}"

    per_context = {}
    for lab in raw:
        per_context[lab] = {stat_key: summarise(raw[lab][stat_key]) for stat_key in raw[lab]}

    margins = {}
    if top_key is not None:
        v = np.array(raw["vsae"]["mean"])
        r_top = np.array(raw[top_key]["mean"])
        r_rand = np.array(raw[rand_key]["mean"])
        margins["mean_vs_top_usage"] = summarise(v - r_top)
        margins["mean_vs_random"] = summarise(v - r_rand)
        for tk in thr_keys:
            vt = np.array(raw["vsae"][tk])
            rt = np.array(raw[top_key][tk])
            margins[f"{tk}_vs_top_usage"] = summarise(vt - rt)

    out = {
        "metric": args.metric,
        "dataset": DATASET_NAME,
        "column1_vals": list(COLUMN1_VALS) if perform_scr else None,
        "n_values": config.n_values,
        "boot": boot,
        "resample_train": args.resample_train,
        "vsae_live_n": vsae_live_n,
        "n_grid": list(n_grid),
        "seconds": time.time() - t0,
        "per_context": per_context,
        "margins_at_vsae_n": margins,
        "raw_draws": {lab: raw[lab] for lab in raw},
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\nwrote {out_path}")

    # ----- console table -----
    print(f"\n{'=' * 78}")
    print(f"{args.metric.upper()} bootstrap ({boot} draws, test-set resample"
          f"{'+train' if args.resample_train else ''}, node effects "
          f"{'re-derived per draw' if args.resample_train else 'fixed'})")
    print(f"{'=' * 78}")
    print(f"{'context':>16}  {'mean score [95% CI]':>34}")
    for lab in raw:
        s = per_context[lab]["mean"]
        print(f"{lab:>16}  {s['p50']:>8.4f}  [{s['p2.5']:+.4f}, {s['p97.5']:+.4f}]")
    if margins:
        print(f"\n  paired margin  vsae - top_usage@{vsae_live_n}  (mean over thresholds):")
        m = margins["mean_vs_top_usage"]
        straddles = m["p2.5"] <= 0.0 <= m["p97.5"]
        print(f"    {m['p50']:+.4f}  [{m['p2.5']:+.4f}, {m['p97.5']:+.4f}]   "
              f"frac_above_0={m['frac_above_0']:.3f}   "
              f"{'STRADDLES 0' if straddles else 'CI excludes 0'}")
        print(f"  paired margin  vsae - random@{vsae_live_n}:")
        m = margins["mean_vs_random"]
        straddles = m["p2.5"] <= 0.0 <= m["p97.5"]
        print(f"    {m['p50']:+.4f}  [{m['p2.5']:+.4f}, {m['p97.5']:+.4f}]   "
              f"{'STRADDLES 0' if straddles else 'CI excludes 0'}")
        print("\n  per-threshold paired margin  vsae - top_usage@{}:".format(vsae_live_n))
        for tk in thr_keys:
            m = margins[f"{tk}_vs_top_usage"]
            straddles = m["p2.5"] <= 0.0 <= m["p97.5"]
            print(f"    {tk:>34}: {m['p50']:+.4f}  [{m['p2.5']:+.4f}, {m['p97.5']:+.4f}]   "
                  f"{'straddles 0' if straddles else 'excludes 0'}")

    del train_acts, test_acts
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
