"""Jaccard overlap of TopK-selected indices, measured DURING early training.

PROJECT.md Next steps #1 / Claims-worth-opening #3. RESULTS addendum 3 found the
learned sigma collapses to the reparameterize() clamp floor at convergence and
that sampling on vs. off moves converged FVE by 0.000012 -- so a Jaccard-overlap
read on FINISHED checkpoints is known to be uninformative; the instability, if it
exists, has to show up while `mu` is still small and TopK selection is being
established, and it is gone by the time training ends. Addendum 7 then found that
pinning sigma at the clamp floor from step 0 (`e2_sigma_low_init`) recovers 84% of
`e2_sampling_only`'s FVE gap to baseline but leaves a real 5-sigma residual -- this
script is the direct test of whether that residual is index-selection churn.

`e2_sampling_only_early` and `e2_sigma_low_init_early` (`falsification/run_arm.py`)
are `e2_sampling_only` / `e2_sigma_low_init` with dense intermediate checkpointing
via trainSAE's `save_steps` mechanism (training.py:233): each seed writes
`checkpoints/ae_{step}.pt` at steps
0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2500, 4000, 6000, 8500, plus the
usual final `ae.pt` at 10000.

At each checkpoint, `encode(x, return_topk=True, training=True)` is called TWICE
on the SAME fixed batch of real activations (drawn once and shared across every
arm, seed and step, exactly as read_learned_sigma.py does) -- each call draws an
independent reparameterisation noise `eps`, so the two calls' top-k index sets
differ only through that noise. Per-token Jaccard overlap
(|selected_1 ∩ selected_2| / |selected_1 ∪ selected_2|) averaged over the batch is
the instability measure: 1.0 means sampling never changes the selection (perfectly
stable), 0.0 means the two draws never agree (k/(2k-k)=... well, in the limit of
independent uniform-random k-subsets of d=2048, expected Jaccard is
k/(2d-k) ~= 0.143, not 0 -- printed as a reference line).

IMPORTANT -- every checkpoint here was saved BEFORE the scale_biases fix (RESULTS
addendum 8): var_encoder.bias was being multiplied by norm_factor (~25.75 for this
config) at save time, which is not the correct transformation for a log-variance
bias and corrupts it into a value that saturates the reparameterize() clamp
immediately -- discovered *while building this script*, when a raw first pass at
this exact measurement showed both arms artificially clamp-saturated from step 0.
`mean_jaccard` below undoes it (divides var_encoder.bias AND .weight by
NORM_FACTOR) before reading each checkpoint, reconstructing exactly what a
checkpoint saved with the fixed code would contain. Any future checkpoint trained
with the fixed code needs no such correction.

    python falsification/read_selection_jaccard.py
    python falsification/read_selection_jaccard.py --n-tokens 4096
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARMS = ("e2_sampling_only_early", "e2_sigma_low_init_early")
STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2500, 4000, 6000, 8500, 10000)
# Derived exactly from the ratio of a saved var_encoder.bias to its configured
# log_var_init (e.g. -51.5 / -2.0), consistent across all 5 seeds of both arms.
NORM_FACTOR = 25.75


def checkpoints(arm: str) -> list[tuple[int, Path]]:
    """seed -> the trainer_0 dir holding ae.pt and checkpoints/."""
    found = {}
    for p in (REPO / "experiments" / arm).glob("seed*/*/trainer_0"):
        if (p / "ae.pt").exists():
            found[int(re.search(r"seed(\d+)", str(p)).group(1))] = p
    return [(s, found[s]) for s in sorted(found)]


def checkpoint_path(trainer_dir: Path, step: int) -> Path | None:
    if step == 10000:
        return trainer_dir / "ae.pt"
    p = trainer_dir / "checkpoints" / f"ae_{step}.pt"
    return p if p.exists() else None


def draw_activations(n_tokens: int, device: str):
    """One fixed batch of raw gelu-1l activations, shared by every measurement.

    Raw, not normalised: checkpoints are saved with biases scaled back up by
    norm_factor (CLAUDE.md), so a saved model is only correct on raw activations.
    """
    import torch
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
        d_submodule=512,
        n_ctxs=500,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=2048,
        device=device,
    )
    chunks, got = [], 0
    while got < n_tokens:
        batch = next(buffer)
        chunks.append(batch.to(device))
        got += batch.shape[0]
    del model, buffer
    torch.cuda.empty_cache()
    return torch.cat(chunks)[:n_tokens]


def mean_jaccard(ckpt: Path, acts, device: str) -> float:
    """Mean per-token Jaccard overlap between two independent stochastic
    TopK selections on the same activations, at this checkpoint.

    Every checkpoint this script reads was saved BEFORE the scale_biases fix
    (RESULTS addendum 8): var_encoder.bias was multiplied by norm_factor at save
    time, a bug that has nothing to do with the raw-activation convention every
    other bias here correctly uses (log_var is not on the same additive axis as
    x/mu -- see the fixed scale_biases' docstring in vsae_topk.py). This divides
    var_encoder.bias AND var_encoder.weight by NORM_FACTOR to reconstruct exactly
    what a checkpoint saved with the FIXED code would contain, so plain raw
    activations can be fed in below with no special-casing."""
    import torch

    from dictionary_learning.trainers.vsae_topk import VSAETopK

    ae = VSAETopK.from_pretrained(str(ckpt), device=device, var_flag=1)
    ae.eval()
    k = int(ae.k.item())
    with torch.no_grad():
        ae.var_encoder.bias.div_(NORM_FACTOR)
        ae.var_encoder.weight.div_(NORM_FACTOR)

    overlaps = []
    with torch.no_grad():
        for i in range(0, acts.shape[0], 4096):
            x = acts[i:i + 4096].to(ae.encoder.weight.dtype)
            _, _, _, _, idx1, _ = ae.encode(x, return_topk=True, training=True)
            _, _, _, _, idx2, _ = ae.encode(x, return_topk=True, training=True)
            # k-subsets of the same size: |union| = 2k - |intersection|.
            i1 = torch.zeros(x.shape[0], ae.dict_size, dtype=torch.bool, device=device)
            i2 = torch.zeros_like(i1)
            i1.scatter_(1, idx1, True)
            i2.scatter_(1, idx2, True)
            inter = (i1 & i2).sum(dim=1).float()
            union = 2 * k - inter
            overlaps.append((inter / union).cpu())
    return float(torch.cat(overlaps).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    acts = draw_activations(args.n_tokens, device)
    print(f"Scoring every checkpoint on the same {acts.shape[0]} activations.\n")

    d, k = 2048, 256
    chance = k / (2 * d - k)
    print(f"Reference: two independent uniform-random size-{k} subsets of "
          f"{d} would average Jaccard {chance:.4f}.\n")

    results: dict[str, dict[int, list[float]]] = {arm: {s: [] for s in STEPS} for arm in ARMS}

    for arm in ARMS:
        print(f"=== {arm} ===")
        for seed, trainer_dir in checkpoints(arm):
            row = []
            for step in STEPS:
                path = checkpoint_path(trainer_dir, step)
                if path is None:
                    row.append(None)
                    continue
                j = mean_jaccard(path, acts, device)
                row.append(j)
                results[arm][step].append(j)
            cells = "  ".join(f"{v:.3f}" if v is not None else "  -  " for v in row)
            print(f" seed {seed:>2}: {cells}")
        print()

    header = "step   " + "  ".join(f"{s:>6}" for s in STEPS)
    print(header)
    for arm in ARMS:
        means = [mean(results[arm][s]) if results[arm][s] else None for s in STEPS]
        cells = "  ".join(f"{m:.4f}" if m is not None else "   -  " for m in means)
        print(f"{arm:<24} {cells}")
    print()

    print("mean ± sd per arm per step:")
    for arm in ARMS:
        print(f"  {arm}:")
        for s in STEPS:
            vals = results[arm][s]
            if not vals:
                continue
            sd = stdev(vals) if len(vals) > 1 else 0.0
            print(f"    step {s:>6}: {mean(vals):.4f} ± {sd:.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
