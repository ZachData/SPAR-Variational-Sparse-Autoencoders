"""Does the vSAE's +/-10 clamp on the penalised vector ever bind?

Found while enumerating the E1 code diff (PROJECT.md, "Next steps" #2 step 1).
The two E1 arms penalise the same quantity with the same coefficient, but not
with the same function:

    top_k_with_feature_penalty.py:570   activation_penalty * 0.5 * sum(f^2)
    vsae_topk.py:867                    kl_coeff * 0.5 * sum(clamp(z, -10, 10)^2)

`f` and `z` are the same tensor at var_flag=0 -- relu(encoder(x - b)), the full
pre-TopK vector -- so the objectives coincide exactly while every entry stays
inside the clamp, and diverge above it. Above 10 the vSAE's penalty is flat: the
term contributes a constant and **its gradient is exactly zero**, so a feature
the TopK+L2 arm is still being pushed down on is left alone by the vSAE.

That is a difference in the *objective*, not in the optimiser, which makes it the
only enumerated factor that could touch E1's algebraic claim (the six-decimal
loss identity was checked on one batch; it holds only where the clamp is idle).
So it is worth settling before it is worth running.

It is settled by measurement, not by a training run, exactly as the dead-feature
tracking factor was (PROJECT.md "Next steps" #5): if no activation reaches 10,
the two penalties are the same function on the data and there is nothing to run.

**Scale matters and is easy to get wrong.** Training runs on activations
normalised to unit mean squared norm (`trainSAE(normalize_activations=True)`,
`get_norm_factor`), and the clamp acts in that space. Checkpoints are saved with
their biases scaled back up by `norm_factor`, so a saved model applied to raw
activations returns `norm_factor * mu_train`. `norm_factor` is not recorded in
config.json -- `trainer.config` is a @property, so trainSAE's
`trainer.config["norm_factor"] = ...` writes to a temporary and is discarded --
so it is re-estimated here with the same estimator the trainer used,
sqrt(mean ||x||^2), on the eval batch. Everything below is reported in TRAINING
space.

    python falsification/read_penalty_clamp.py
    python falsification/read_penalty_clamp.py --n-tokens 50000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The clamp in vsae_topk.py:867 (_compute_kl_loss, var_flag=0 branch). Not a
# parameter -- read from the code.
CLAMP_LO, CLAMP_HI = -10.0, 10.0

ARMS = ("e1_penalty", "e1_vsae_ref_unitinit", "e1_vsae_ref_gradproj")


def checkpoints(arm: str) -> list[tuple[int, Path]]:
    found = {}
    for p in (REPO / "experiments" / arm).glob("seed*/*/trainer_0"):
        if (p / "ae.pt").exists():
            found[int(re.search(r"seed(\d+)", str(p)).group(1))] = p
    return [(s, found[s]) for s in sorted(found)]


def draw_activations(n_tokens: int, device: str):
    """One fixed batch of gelu-1l activations, shared by every checkpoint."""
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


def penalised_vector(ae, x):
    """The tensor each trainer's penalty sums over, for either dictionary class.

    Both are the full pre-TopK vector, not the sparse one: `activation_cost` uses
    `post_relu_acts_BF` and `_compute_kl_loss` uses `latent_z`, and both penalise
    unselected features too.
    """
    if hasattr(ae, "var_flag"):                      # VSAETopK
        _, z, _, _ = ae.encode(x, training=False)
        return z
    _, _, _, post_relu = ae.encode(x, return_topk=True)   # AutoEncoderTopK
    return post_relu


def measure(ckpt: Path, acts, norm_factor: float, device: str) -> dict:
    import torch

    from dictionary_learning.utils import load_dictionary

    ae, _ = load_dictionary(str(ckpt), device=device)
    ae.eval()

    n_total = n_over = 0
    v_max = float("-inf")
    pen_raw = pen_clamped = 0.0      # 0.5 * sum(v^2) with and without the clamp
    grad_raw = grad_over = 0.0       # sum|v| overall, and over the clamped entries
    over_per_token = 0.0

    with torch.no_grad():
        for i in range(0, acts.shape[0], 4096):
            x = acts[i:i + 4096]
            # Training space: the saved biases carry norm_factor, so the whole
            # pre-TopK vector scales with it (the encoder weights do not move).
            v = penalised_vector(ae, x).float() / norm_factor
            c = v.clamp(CLAMP_LO, CLAMP_HI)

            n_total += v.numel()
            over = (v > CLAMP_HI) | (v < CLAMP_LO)
            n_over += int(over.sum())
            over_per_token += float(over.sum(dim=1).sum())
            v_max = max(v_max, float(v.max()))

            pen_raw += 0.5 * float(v.pow(2).sum())
            pen_clamped += 0.5 * float(c.pow(2).sum())
            # d/dv of 0.5*v^2 is v, and the clamp zeroes it outside the range.
            grad_raw += float(v.abs().sum())
            grad_over += float(v.abs()[over].sum())

    return {
        "max": v_max,
        "frac_over": n_over / n_total,
        "over_per_token": over_per_token / acts.shape[0],
        "penalty_frac_lost": (pen_raw - pen_clamped) / pen_raw if pen_raw else 0.0,
        "grad_frac_lost": grad_over / grad_raw if grad_raw else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-tokens", type=int, default=20_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()

    import torch

    print(f"vsae_topk.py clamps the penalised vector to "
          f"[{CLAMP_LO}, {CLAMP_HI}]; top_k_with_feature_penalty.py does not.")
    print("Above the clamp the vSAE's penalty is flat and its gradient is zero.\n")

    acts = draw_activations(args.n_tokens, args.device)
    norm_factor = float(torch.sqrt(torch.mean(torch.sum(acts.float() ** 2, dim=1))))
    print(f"Scoring every checkpoint on the same {acts.shape[0]:,} activations.")
    print(f"norm_factor re-estimated as {norm_factor:.4f} "
          f"(sqrt of mean squared norm, the estimator trainSAE used).\n")

    all_clear = True
    for arm in args.arms:
        ck = checkpoints(arm)
        if not ck:
            print(f"{arm}: no checkpoints found\n")
            continue
        print(f"=== {arm}  ({len(ck)} seeds) ===")
        print(f"{'seed':>5} {'max |v|':>9} {'frac>10':>10} {'#>10/tok':>9} "
              f"{'penalty lost':>13} {'grad lost':>10}")
        rows = []
        for seed, path in ck:
            m = measure(path, acts, norm_factor, args.device)
            rows.append(m)
            print(f"{seed:>5} {m['max']:>9.4f} {m['frac_over']:>10.3e} "
                  f"{m['over_per_token']:>9.3f} {m['penalty_frac_lost']:>13.3e} "
                  f"{m['grad_frac_lost']:>10.3e}")
        mx = max(r["max"] for r in rows)
        print(f"{'':>5} {'-' * 60}")
        print(f"  max over seeds: {mx:.4f}   "
              f"headroom to the clamp: {CLAMP_HI - mx:+.4f}")
        if mx >= CLAMP_HI:
            all_clear = False
            print("  ** THE CLAMP BINDS ** -- the two penalties are different "
                  "functions on this data.")
        else:
            print("  The clamp never binds: on this data the two penalties are "
                  "the same function.")
        print()

    if all_clear:
        print("Verdict: no activation in any E1 arm reaches the clamp, so the "
              "clamp is a no-op\nfactor. Do not spend a training run on it.")
    else:
        print("Verdict: the clamp binds. It is a real difference in the "
              "OBJECTIVE, not the optimiser,\nand the six-decimal loss identity "
              "holds only where it is idle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
