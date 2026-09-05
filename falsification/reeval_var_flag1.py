"""Re-evaluate every on-disk var_flag=1 checkpoint with the scale_biases bug corrected.

PROJECT.md Next steps #0 / CLAUDE.md's `var_encoder.bias` landmine / RESULTS
addendum 8. Every `var_flag=1` checkpoint saved before the `scale_biases` fix
(2026-09-04, second session) had its `var_encoder.bias` multiplied by
`norm_factor` at save time -- correct for `encoder.bias`/`decoder.bias`, wrong for
a log-variance bias -- so the standard `evaluate()`/`loss_recovered()` pipeline,
run against these checkpoints as saved, samples with an almost-fully-collapsed
sigma and understates what real (moderate) sampling noise costs. This script
loads each affected checkpoint, applies the correction (divide `var_encoder.bias`
AND `var_encoder.weight` by `norm_factor` -- the same pattern
`read_selection_jaccard.py`'s `mean_jaccard` uses), and re-runs the exact
`evaluate()` call the training scripts use, at the `eval_batch_size` /
`eval_n_batches` / `ctx_len` recorded in each checkpoint's own
`experiment_config.json`, so the corrected numbers are produced by the same
pipeline `compare_arms.py` already knows how to read.

`e2_sigma_low_init` and `e2_sigma_low_init_early` are deliberately excluded:
`log_var_init=-8.0` already sits below the `reparameterize()` clamp floor before
the bug's multiplication is even applied, so their own reported numbers needed no
correction (RESULTS addendum 7/8).

Every checkpoint here is scored on the SAME activation stream (one buffer, drawn
from a freshly-constructed unshuffled generator, consumed continuously across all
checkpoints in one run) -- more comparable across checkpoints than the original
per-run evaluations were, since those each continued from wherever their own
training buffer happened to be.

Corrected results are written to `evaluation_results_corrected.json` alongside
the original `evaluation_results.json`, which is left untouched as the
historical (buggy) record -- this repo corrects forward rather than overwriting
evidence (see `archive/` for retired E1 generations). `compare_arms.py` prefers
the corrected file when present.

    python falsification/reeval_var_flag1.py                  # every var_flag=1 arm
    python falsification/reeval_var_flag1.py --arm e2_confirm  # just one
    python falsification/reeval_var_flag1.py --dry-run         # list checkpoints only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Every arm on disk with var_flag=1 that evaluate()/loss_recovered() has already
# scored on the buggy checkpoint. e2_sigma_low_init(_early) excluded -- see module
# docstring. e2_*_early arms are excluded too: they exist for dense-checkpoint
# Jaccard reads (read_selection_jaccard.py already applies its own correction),
# not for evaluation_results.json comparisons.
ARMS = (
    "e2_learned_var",
    "e2_confirm",
    "e2_sampling_only",
    "e2_beta_pilot_0.0001",
    "e2_beta_pilot_0.001",
    "e2_beta_pilot_0.01",
    "e2_beta_pilot_0.1",
    "e2_beta_pilot_1",
)

EVAL_KEYS = ("frac_variance_explained", "frac_recovered", "frac_alive")


def find_checkpoints(arm: str) -> list[tuple[int, Path]]:
    """seed -> the experiment dir holding trainer_0/ and experiment_config.json."""
    found = {}
    for p in (REPO / "experiments" / arm).glob("seed*/*"):
        if (p / "trainer_0" / "ae.pt").exists() and (p / "experiment_config.json").exists():
            import re
            seed = int(re.search(r"seed(\d+)", str(p)).group(1))
            found[seed] = p
    return [(s, found[s]) for s in sorted(found)]


def build_buffer(device: str, n_ctxs: int):
    from transformer_lens import HookedTransformer

    from dictionary_learning.buffer import TransformerLensActivationBuffer
    from dictionary_learning.utils import hf_dataset_to_generator

    model = HookedTransformer.from_pretrained("gelu-1l", device=device)
    buffer = TransformerLensActivationBuffer(
        data=hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b", split="train", return_tokens=True
        ),
        model=model,
        hook_name="blocks.0.hook_resid_post",
        d_submodule=model.cfg.d_model,
        n_ctxs=n_ctxs,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=192,
        device=device,
    )
    return model, buffer


def correct_and_evaluate(exp_dir: Path, buffer, device: str, norm_factor: float) -> dict:
    from dictionary_learning.evaluation import evaluate
    from dictionary_learning.utils import load_dictionary

    with open(exp_dir / "experiment_config.json") as f:
        cfg = json.load(f)

    vsae, _ = load_dictionary(str(exp_dir / "trainer_0"), device=device)
    if getattr(vsae, "var_flag", 0) != 1:
        raise ValueError(f"{exp_dir} is not a var_flag=1 checkpoint")

    import torch
    with torch.no_grad():
        vsae.var_encoder.bias.div_(norm_factor)
        vsae.var_encoder.weight.div_(norm_factor)

    return evaluate(
        dictionary=vsae,
        activations=buffer,
        batch_size=cfg["eval_batch_size"],
        max_len=cfg["ctx_len"],
        device=device,
        n_batches=cfg["eval_n_batches"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, help="restrict to one arm (default: all)")
    ap.add_argument("--n-ctxs", type=int, default=500,
                    help="buffer size (default 500; eval needs far less than "
                         "training's 2500, so this is cheaper to keep filled)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                     help="list checkpoints that would be re-evaluated and exit")
    args = ap.parse_args()

    arms = (args.arm,) if args.arm else ARMS

    targets = []
    for arm in arms:
        for seed, exp_dir in find_checkpoints(arm):
            targets.append((arm, seed, exp_dir))

    if not targets:
        print("No var_flag=1 checkpoints found for the requested arm(s).")
        return 1

    print(f"{len(targets)} checkpoints to re-evaluate:")
    for arm, seed, exp_dir in targets:
        print(f"  {arm:<24} seed {seed:>3}  {exp_dir.name}")
    print()

    if args.dry_run:
        return 0

    import torch
    from dictionary_learning.training import get_norm_factor

    device = args.device if torch.cuda.is_available() else "cpu"
    model, buffer = build_buffer(device, args.n_ctxs)

    norm_factor = get_norm_factor(buffer, steps=100)
    print(f"\nnorm_factor (recomputed, gelu-1l blocks.0.hook_resid_post): "
          f"{norm_factor:.4f}\n")

    results: dict[str, dict[str, list[float]]] = {arm: {k: [] for k in EVAL_KEYS} for arm in arms}
    before: dict[str, dict[str, list[float]]] = {arm: {k: [] for k in EVAL_KEYS} for arm in arms}

    for arm, seed, exp_dir in targets:
        eval_results = correct_and_evaluate(exp_dir, buffer, device, norm_factor)

        out_path = exp_dir / "evaluation_results_corrected.json"
        with open(out_path, "w") as f:
            json.dump({k: float(v) for k, v in eval_results.items()}, f, indent=2)

        orig_path = exp_dir / "evaluation_results.json"
        orig = json.load(open(orig_path)) if orig_path.exists() else {}

        cells = []
        for k in EVAL_KEYS:
            if k in eval_results:
                results[arm][k].append(eval_results[k])
            if k in orig:
                before[arm][k].append(orig[k])
            cells.append(f"{k}={eval_results.get(k, float('nan')):.4f} "
                         f"(was {orig.get(k, float('nan')):.4f})")
        print(f"{arm:<24} seed {seed:>3}: " + "  ".join(cells))

    print("\nPer-arm means (corrected vs. original evaluation_results.json):")
    header = f"{'arm':<24} " + "  ".join(f"{k:>26}" for k in EVAL_KEYS)
    print(header)
    for arm in arms:
        if not results[arm][EVAL_KEYS[0]]:
            continue
        cells = []
        for k in EVAL_KEYS:
            vc, vb = results[arm][k], before[arm][k]
            mc = mean(vc) if vc else float("nan")
            sc = stdev(vc) if len(vc) > 1 else 0.0
            mb = mean(vb) if vb else float("nan")
            cells.append(f"{mc:.4f}±{sc:.4f} (was {mb:.4f})")
        print(f"{arm:<24} " + "  ".join(f"{c:>26}" for c in cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
