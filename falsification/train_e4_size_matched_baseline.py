"""E4 box (5) -- train a size-matched TopK baseline from scratch (PROJECT.md
Claims-worth-opening #6d, "most expensive" of the four SCR/TPP disagreement
diagnostics).

Every point on `run_e4_scr.py`/`run_e4_tpp.py`'s baseline curve is the SAME
8192-wide dictionary (`experiments/e4_pythia_baseline/`) with entries zeroed
out post hoc by usage. The vSAE's 1474 live features were learned together,
with the rest of its capacity never used for anything else -- a dictionary
trained from scratch at `dict_size=1474` might organise its limited capacity
differently (`size_control.py`'s own
`test_random_subset_null_would_falsely_confirm_the_hypothesis` already shows
reference-choice sensitivity matters a lot for this design). This script
removes "masked vs. trained-small" as a live confound by training a real
TopK SAE at exactly the vSAE's live count.

Config is IDENTICAL to the recovered baseline's own
`experiments/e4_pythia_baseline/seed42/.../experiment_config.json` (k=256,
auxk_alpha=0.03125, lr=None/auto-computed, all buffer/schedule settings),
changing ONLY `dict_size_multiple` so `dict_size = int(dict_size_multiple *
512) == 1474` exactly (1474/512 = 2.87890625 is an exact dyadic fraction, no
float-truncation risk). `use_wandb=False` to stay unattended (run_arm.py's own
convention); everything else, including `seed=42`, matches for direct
comparability to both recovered checkpoints.

    python falsification/train_e4_size_matched_baseline.py
    python falsification/train_e4_size_matched_baseline.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training_scripts"))

TARGET_DICT_SIZE = 1474
D_MODEL = 512
SAVE_DIR = str(REPO / "experiments" / "e4_pythia_baseline_size_matched" / "seed42")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit, no training")
    args = ap.parse_args()

    import train_topk  # training_scripts/train_topk.py

    config = train_topk.ExperimentConfig(
        model_name="EleutherAI/pythia-70m-deduped",
        layer=3,
        hook_name="blocks.3.hook_resid_post",
        dict_size_multiple=TARGET_DICT_SIZE / D_MODEL,
        k=256,
        total_steps=10001,
        lr=None,
        warmup_steps=None,
        auxk_alpha=0.03125,
        threshold_beta=0.999,
        threshold_start_step=500,
        n_ctxs=2500,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=192,
        checkpoint_steps=(10000,),
        log_steps=1000,
        save_dir=SAVE_DIR,
        use_wandb=False,  # unattended -- never block on a wandb login prompt
        device="cuda",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
        eval_batch_size=24,
        eval_n_batches=4,
    )

    runner = train_topk.ExperimentRunner(config)
    resolved_dict_size = int(config.dict_size_multiple * D_MODEL)
    assert resolved_dict_size == TARGET_DICT_SIZE, (
        f"dict_size_multiple truncation gave {resolved_dict_size}, expected {TARGET_DICT_SIZE}"
    )
    print(f"dict_size_multiple={config.dict_size_multiple!r} -> dict_size={resolved_dict_size}")
    print(f"save directory will be: {runner.get_save_directory()}")

    if args.dry_run:
        print("dry run, not training")
        return 0

    results = runner.run_training()
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
