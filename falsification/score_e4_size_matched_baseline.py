"""E4 box (5) -- score the size-matched TopK baseline (dict_size=1474, trained
from scratch by `train_e4_size_matched_baseline.py`) with SCR and/or TPP, and
compare against the vSAE's own score AND against the masked-from-8192 curve's
`top_usage`/`random` references at n=1474 (`run_e4_scr.py`/`run_e4_tpp.py`'s
already-committed results for the given dataset/pair).

This is the "masked vs. trained-small" confound check (PROJECT.md
Claims-worth-opening #6d): if the trained-small baseline's score is close to
the masked curve's reference at the same N, "masked vs. trained-small" is not
a live confound in the existing verdict. If it differs materially, the
confound is real and the existing masked-curve verdicts need the caveat
sharpened.

No masking involved -- this checkpoint has exactly 1474 dictionary entries and
is scored unmasked, exactly the way the vSAE itself is scored, so the
comparison is a direct three-way one: vSAE vs. masked-from-8192-at-1474 vs.
trained-from-scratch-at-1474.

Generalised with `--dataset`/`--column1-vals` (RESULTS addendum 24, extending
box (5)'s original single-pair check to the other 7 (dataset, pair) points
`run_e4_scr.py --dataset ...`/`run_e4_tpp.py --dataset ...` added in
addendum 19) the same way addendum 19 generalised the masking-curve runners
themselves. TPP has no per-pair concept (it scores all classes in a dataset
at once), so `--skip-tpp` lets a caller score the same trained-small
checkpoint's SCR for several pairs of one dataset without recomputing TPP
each time -- run TPP once per dataset (`--skip-scr`) instead.

Defaults (no flags) reproduce addendum 21 exactly: professor/nurse on
bias_in_bios, both SCR and TPP, writing to
`falsification/e4_size_matched_baseline_results.json`.

    python falsification/score_e4_size_matched_baseline.py
    python falsification/score_e4_size_matched_baseline.py --dataset LabHC/bias_in_bios_class_set1 \
        --column1-vals architect journalist --skip-tpp
    python falsification/score_e4_size_matched_baseline.py \
        --dataset canrager/amazon_reviews_mcauley_1and5 --skip-scr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
SAEBENCH = REPO / "SAEBench-main"
for _p in (str(REPO), str(SAEBENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from falsification.e4_local_sae import load_local_topk_sae  # noqa: E402

import sae_bench.evals.scr_and_tpp.main as scr_and_tpp  # noqa: E402
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

MODEL_NAME = "pythia-70m-deduped"
DEFAULT_DATASET_NAME = "LabHC/bias_in_bios_class_set1"
DEFAULT_COLUMN1_VALS = ("professor", "nurse")  # addenda 10/11's original comparison point

SIZE_MATCHED_DIR = (
    REPO
    / "experiments/e4_pythia_baseline_size_matched/seed42/"
    "TopK_KL_pythia70m_d1474_k256_auxk0.03125_pen0.0_lr_auto/trainer_0"
)


def _prior_path(kind: str, dataset_name: str, column1_vals: tuple[str, str] | None) -> Path:
    """Mirrors run_e4_scr.py / run_e4_tpp.py's own default --out naming, so
    this always reads the masking-curve results the corresponding runner
    already wrote for this exact dataset/pair."""
    is_default_dataset = dataset_name == DEFAULT_DATASET_NAME
    if kind == "scr":
        is_default = is_default_dataset and column1_vals == DEFAULT_COLUMN1_VALS
        if is_default:
            return REPO / "falsification/e4_scr_results.json"
        return REPO / "falsification" / (
            f"e4_scr_results_{dataset_name.replace('/', '_')}_"
            f"{column1_vals[0]}_{column1_vals[1]}.json"
        )
    else:  # tpp
        if is_default_dataset:
            return REPO / "falsification/e4_tpp_results.json"
        return REPO / "falsification" / f"e4_tpp_results_{dataset_name.replace('/', '_')}.json"


def scr_score_from_results(results: dict, run_name: str) -> dict[str, float]:
    run_results = results[f"{run_name}_results"]
    return {k: v for k, v in run_results.items() if k.startswith("scr_metric_threshold_")}


def tpp_score_from_results(results: dict, run_name: str) -> dict[str, float]:
    run_results = results[f"{run_name}_results"]
    return {k: v for k, v in run_results.items() if k.endswith("_total_metric")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--column1-vals", nargs=2, default=list(DEFAULT_COLUMN1_VALS),
                     metavar=("POS", "NEG"), help="SCR class pair; ignored with --skip-scr")
    ap.add_argument("--skip-scr", action="store_true")
    ap.add_argument("--skip-tpp", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_name = args.dataset
    column1_vals = tuple(args.column1_vals)
    is_default = dataset_name == DEFAULT_DATASET_NAME and column1_vals == DEFAULT_COLUMN1_VALS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    sae = load_local_topk_sae(SIZE_MATCHED_DIR, MODEL_NAME, device, dtype)
    print(f"loaded size-matched baseline: d_sae={sae.cfg.d_sae}, k={int(sae.k)}, hook={sae.cfg.hook_name}")
    assert sae.cfg.d_sae == 1474, sae.cfg.d_sae

    model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=device, dtype=torch.float32)
    print("LLM loaded")

    out: dict = {"dict_size": 1474, "k": int(sae.k), "dataset": dataset_name}

    if not args.skip_scr:
        artifacts_folder = str(REPO / "falsification/e4_scr_artifacts")
        scr_run_name = f"{dataset_name}_scr_{column1_vals[0]}_{column1_vals[1]}"
        scr_config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            perform_scr=True,
            dataset_names=[dataset_name],
            column1_vals_lookup={dataset_name: [column1_vals]},
            n_values=[2, 5, 10, 20],
            llm_batch_size=512,
            llm_dtype="float32",
        )
        print(f"\nscoring size-matched baseline on SCR ({column1_vals[0]}/{column1_vals[1]})...")
        scr_results, _ = scr_and_tpp.run_eval_single_sae(
            scr_config, sae, model, device, artifacts_folder, save_activations=True
        )
        scr_per_threshold = scr_score_from_results(scr_results, scr_run_name)
        scr_score = float(sum(scr_per_threshold.values()) / len(scr_per_threshold))
        print(f"  SCR score={scr_score:.4f} ({scr_per_threshold})")

        scr_prior = json.loads(_prior_path("scr", dataset_name, column1_vals).read_text())
        vsae_scr = scr_prior["vsae"]["score"]
        masked_top_usage_scr = scr_prior["vsae"]["top_usage_reference_at_vsae_n"]
        masked_random_scr = scr_prior["vsae"]["random_reference_at_vsae_n"]

        print(f"\n=== SCR ({dataset_name}, {column1_vals}): three-way comparison at N=1474 ===")
        print(f"  vSAE (natural, unmasked):        {vsae_scr:.4f}")
        print(f"  masked-from-8192 (top_usage):    {masked_top_usage_scr:.4f}")
        print(f"  masked-from-8192 (random):       {masked_random_scr:.4f}")
        print(f"  trained-from-scratch (this run): {scr_score:.4f}")
        print(f"  vSAE - trained-from-scratch:     {vsae_scr - scr_score:+.4f}")
        print(f"  trained-from-scratch - masked(top_usage): {scr_score - masked_top_usage_scr:+.4f}")

        out["column1_vals"] = list(column1_vals)
        out["scr"] = {
            "score": scr_score,
            "per_threshold": scr_per_threshold,
            "vsae_score": vsae_scr,
            "masked_top_usage_at_n": masked_top_usage_scr,
            "masked_random_at_n": masked_random_scr,
            "vsae_minus_trained_small": vsae_scr - scr_score,
            "trained_small_minus_masked_top_usage": scr_score - masked_top_usage_scr,
        }

    if not args.skip_tpp:
        tpp_artifacts_folder = str(REPO / "falsification/e4_tpp_artifacts")
        tpp_run_name = f"{dataset_name}_tpp"
        tpp_config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            perform_scr=False,
            dataset_names=[dataset_name],
            n_values=[2, 5, 10, 20],
            llm_batch_size=512,
            llm_dtype="float32",
        )
        print("\nscoring size-matched baseline on TPP...")
        tpp_results, _ = scr_and_tpp.run_eval_single_sae(
            tpp_config, sae, model, device, tpp_artifacts_folder, save_activations=True
        )
        tpp_per_threshold = tpp_score_from_results(tpp_results, tpp_run_name)
        tpp_score = float(sum(tpp_per_threshold.values()) / len(tpp_per_threshold))
        print(f"  TPP score={tpp_score:.4f} ({tpp_per_threshold})")

        tpp_prior = json.loads(_prior_path("tpp", dataset_name, None).read_text())
        vsae_tpp = tpp_prior["vsae"]["score"]
        masked_top_usage_tpp = tpp_prior["vsae"]["top_usage_reference_at_vsae_n"]
        masked_random_tpp = tpp_prior["vsae"]["random_reference_at_vsae_n"]

        print(f"\n=== TPP ({dataset_name}): three-way comparison at N=1474 ===")
        print(f"  vSAE (natural, unmasked):        {vsae_tpp:.4f}")
        print(f"  masked-from-8192 (top_usage):    {masked_top_usage_tpp:.4f}")
        print(f"  masked-from-8192 (random):       {masked_random_tpp:.4f}")
        print(f"  trained-from-scratch (this run): {tpp_score:.4f}")
        print(f"  vSAE - trained-from-scratch:     {vsae_tpp - tpp_score:+.4f}")
        print(f"  trained-from-scratch - masked(top_usage): {tpp_score - masked_top_usage_tpp:+.4f}")

        out["tpp"] = {
            "score": tpp_score,
            "per_threshold": tpp_per_threshold,
            "vsae_score": vsae_tpp,
            "masked_top_usage_at_n": masked_top_usage_tpp,
            "masked_random_at_n": masked_random_tpp,
            "vsae_minus_trained_small": vsae_tpp - tpp_score,
            "trained_small_minus_masked_top_usage": tpp_score - masked_top_usage_tpp,
        }

    if args.out:
        out_path = Path(args.out)
    elif is_default:
        out_path = REPO / "falsification/e4_size_matched_baseline_results.json"
    else:
        suffix = f"_{dataset_name.replace('/', '_')}"
        if not args.skip_scr:
            suffix += f"_{column1_vals[0]}_{column1_vals[1]}"
        out_path = REPO / "falsification" / f"e4_size_matched_baseline_results{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
