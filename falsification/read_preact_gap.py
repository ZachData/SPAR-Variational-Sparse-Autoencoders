"""The k/(k+1) pre-activation gap TopK selection noise has to beat.
(PROJECT.md Next steps A2, run before the sweep so the knee is predicted, not fitted.)

A2's design reasons that reparameterisation noise of scale sigma should flip
TopK's selection whenever it exceeds the gap between the k-th and (k+1)-th
largest pre-activation values for a token, and that the sigma_init dose-response
curve should therefore knee where sigma crosses this gap's typical size. This
script measures that gap distribution on an already-trained, deterministic
baseline checkpoint (`experiments/baseline/seed1`) -- free, no new training --
so the sweep's grid can be chosen with the answer in hand rather than fit to it
after the fact.

Same units discipline as `read_penalty_clamp.py`: training runs on activations
normalised to unit mean squared norm, and `log_var_init` parameterises sigma in
that same normalised space (CLAUDE.md; PROJECT.md Next steps A2), so the gap is
measured there too -- divide the raw pre-TopK vector by the same `norm_factor`
estimator the trainer used (`sqrt(mean ||x||^2)`) before taking the gap. No
further correction is needed: this is a fresh AutoEncoderTopK checkpoint, not a
var_flag=1 one, so the scale_biases var_encoder.bias bug (CLAUDE.md) does not
apply.

    python falsification/read_preact_gap.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CHECKPOINT = (
    REPO
    / "experiments/baseline/seed1/"
    "TopK_KL_gelu1l_d2048_k256_auxk0.03125_pen0.0_lr0.0008/trainer_0"
)

# sigma = exp(0.5 * log_var_init) for A2's planned grid, for a side-by-side
# comparison against the measured gap distribution.
CANDIDATE_LOG_VAR_INITS = (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-tokens", type=int, default=20_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = ap.parse_args()

    import numpy as np
    import torch

    from dictionary_learning.utils import load_dictionary
    from falsification.read_penalty_clamp import draw_activations

    acts = draw_activations(args.n_tokens, args.device)
    norm_factor = float(torch.sqrt(torch.mean(torch.sum(acts.float() ** 2, dim=1))))
    print(f"norm_factor re-estimated as {norm_factor:.4f}")

    ae, _ = load_dictionary(args.checkpoint, device=args.device)
    ae.eval()
    k = int(ae.k)
    print(f"checkpoint: {args.checkpoint}\nk={k}, d={ae.dict_size if hasattr(ae, 'dict_size') else '?'}")

    gaps = []
    with torch.no_grad():
        for i in range(0, acts.shape[0], 4096):
            x = acts[i:i + 4096]
            _, _, _, post_relu = ae.encode(x, return_topk=True)
            v = post_relu.float() / norm_factor
            sorted_desc, _ = torch.sort(v, dim=1, descending=True)
            gaps.append((sorted_desc[:, k - 1] - sorted_desc[:, k]).cpu())
    gaps = torch.cat(gaps).numpy()

    pcts = (1, 5, 25, 50, 75, 95, 99)
    print(f"\nk/(k+1) pre-activation gap over {len(gaps):,} tokens (training/normalised space):")
    print(f"  mean={gaps.mean():.6f}  std={gaps.std():.6f}  max={gaps.max():.6f}")
    for p in pcts:
        print(f"  p{p:<3d}={np.percentile(gaps, p):.6f}")

    print("\ncandidate log_var_init -> sigma=exp(0.5*log_var_init), vs. gap percentiles:")
    print(f"{'log_var_init':>13} {'sigma':>8} {'x median gap':>13} {'x p99 gap':>10}")
    median_gap = float(np.percentile(gaps, 50))
    p99_gap = float(np.percentile(gaps, 99))
    for lvi in CANDIDATE_LOG_VAR_INITS:
        sigma = float(np.exp(0.5 * lvi))
        print(f"{lvi:>13.1f} {sigma:>8.4f} {sigma / median_gap:>13.1f} {sigma / p99_gap:>10.1f}")

    if float(np.exp(0.5 * min(CANDIDATE_LOG_VAR_INITS))) > 10 * p99_gap:
        print(
            "\nEven the clamp-floor sigma is >>10x the p99 gap -- naively this predicts "
            "near-total selection scrambling everywhere on the grid, not a sharp knee "
            "within it. See RESULTS addendum 16."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
