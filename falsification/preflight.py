"""Pre-flight checks to run the moment GPU access returns, before any training.

Every check here has burned time in this project at least once. Run it first;
it takes seconds and fails loudly rather than 20 minutes into a run.

Usage: python falsification/preflight.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Allow `python falsification/preflight.py` from anywhere, not just an installed package.
sys.path.insert(0, str(REPO))
FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False) -> None:
    if condition:
        print(f"  PASS  {name}")
        return
    target = WARNINGS if warn_only else FAILURES
    target.append(f"{name}: {detail}")
    print(f"  {'WARN' if warn_only else 'FAIL'}  {name}   {detail}")


def main() -> int:
    print("Environment")
    try:
        import torch

        check("torch imports", True)
        check(
            "CUDA available",
            torch.cuda.is_available(),
            "training requires a GPU; this is the local-machine check",
        )
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"        device: {torch.cuda.get_device_name(0)}  ({gb:.1f} GB)")
            check(
                "bfloat16 supported",
                torch.cuda.is_bf16_supported(),
                "training configs assume bfloat16",
            )
    except ImportError as exc:
        check("torch imports", False, str(exc))

    print("\nFalsification framework (no GPU needed)")
    for module in ("evalues", "permutation", "simulate"):
        spec = importlib.util.find_spec(f"falsification.{module}")
        check(f"falsification.{module} importable", spec is not None)

    print("\nDegeneracy control (E1) wiring")
    top_k = (REPO / "dictionary_learning/trainers/top_k.py").read_text()
    check(
        "TopK trainer exposes activation_penalty",
        "activation_penalty: float = 0.0" in top_k,
        "E1 has no control arm without it",
    )
    check(
        "penalty applies to pre-TopK activations",
        "activation_penalty > 0" in top_k and "post_relu_acts_BF.pow(2)" in top_k,
        "must penalise unselected features to mirror the vSAE KL",
    )

    print("\nKnown landmines")
    vsae = (REPO / "dictionary_learning/trainers/vsae_topk.py").read_text()
    check(
        "vsae_topk still gates sampling on var_flag == 1",
        "training and self.var_flag == 1" in vsae,
        "if this changed, the degeneracy analysis needs revisiting",
        warn_only=True,
    )
    check(
        "vsae_topk still applies F.relu(mu)",
        "mu = F.relu(mu)" in vsae,
        "the E3 masked-KL comparison is confounded by this ReLU",
        warn_only=True,
    )

    print("\nData already committed")
    summaries = list(
        (REPO / "comprehensive_histogram_analysis").glob("*/comprehensive_summary_*.json")
    )
    check("sweep summaries present", len(summaries) >= 8, f"found {len(summaries)}")

    print("\nDisk")
    stat = os.statvfs(REPO)
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    check(
        "at least 20 GB free",
        free_gb >= 20,
        f"{free_gb:.1f} GB free; ~28 checkpoints plus buffers need room",
        warn_only=True,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} BLOCKING failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"  - {w}")
    if not FAILURES:
        print("Pre-flight clear. Safe to start training.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
