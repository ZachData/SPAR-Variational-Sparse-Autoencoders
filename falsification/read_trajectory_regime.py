"""The step-wise selection-Jaccard trajectory of `read_selection_jaccard.py`
(RESULTS §3: `e2_sampling_only` 0.069 -> 0.811, `e2_sigma_low_init` 0.431 ->
0.952), measured in both regimes.

`read_selection_jaccard.py` fed the saved intermediate checkpoints raw
activations, where a sampled checkpoint runs at sigma/norm_factor of its
training noise (docs/ERRATA.md §2). This reader repeats it on the same
checkpoints (`*_early` arms, 5 seeds, `checkpoints/ae_<step>.pt`) as saved and
in the training regime (`scale_biases(1/norm_factor)`, activations /
norm_factor), on one fixed batch and one noise seed. Every checkpoint predates
the 2026-09-04 scale_biases fix, so each first gets the existing log-variance
correction (var_encoder weight and bias / 25.75).

    /usr/bin/python3 falsification/read_trajectory_regime.py      # LOCAL GPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falsification.read_selection_jaccard import ARMS, STEPS, checkpoint_path, checkpoints  # noqa: E402

PREFIX_NORM_FACTOR = 25.75
OUT = REPO / "falsification" / "trajectory_regime_results.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import torch

    from dictionary_learning.trainers.vsae_topk import VSAETopK
    from falsification.read_penalty_clamp import draw_activations

    torch.manual_seed(args.rng_seed)
    acts = draw_activations(args.n_tokens, args.device)
    nf = float(torch.sqrt(torch.mean(torch.sum(acts.float() ** 2, dim=1))))

    def jaccard(ae, x) -> float:
        k = int(ae.k.item())
        torch.manual_seed(args.rng_seed + 1)
        out = []
        with torch.no_grad():
            for i in range(0, x.shape[0], 1024):
                xb = x[i:i + 1024].to(ae.encoder.weight.dtype)
                _, _, _, _, i1, _ = ae.encode(xb, return_topk=True, training=True)
                _, _, _, _, i2, _ = ae.encode(xb, return_topk=True, training=True)
                a = torch.zeros(xb.shape[0], ae.dict_size, dtype=torch.bool, device=xb.device)
                b = torch.zeros_like(a)
                a.scatter_(1, i1, True)
                b.scatter_(1, i2, True)
                inter = (a & b).sum(1).float()
                out.append((inter / (2 * k - inter)).cpu())
        return float(torch.cat(out).mean())

    results = {"norm_factor": nf, "n_tokens": int(acts.shape[0]), "rng_seed": args.rng_seed, "arms": {}}
    for arm in ARMS:
        per = {"raw": {s: [] for s in STEPS}, "training": {s: [] for s in STEPS}}
        for seed, tdir in checkpoints(arm):
            for step in STEPS:
                p = checkpoint_path(tdir, step)
                if p is None:
                    continue
                for regime in ("raw", "training"):
                    ae = VSAETopK.from_pretrained(str(p), device=args.device, var_flag=1)
                    ae.eval()
                    with torch.no_grad():
                        ae.var_encoder.bias.div_(PREFIX_NORM_FACTOR)
                        ae.var_encoder.weight.div_(PREFIX_NORM_FACTOR)
                    if regime == "training":
                        ae.scale_biases(1.0 / nf)
                        per[regime][step].append(jaccard(ae, acts / nf))
                    else:
                        per[regime][step].append(jaccard(ae, acts))
        results["arms"][arm] = {
            regime: {str(s): {"mean": mean(v), "std": stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
                     for s, v in per[regime].items() if v}
            for regime in per}
        for regime in ("raw", "training"):
            row = results["arms"][arm][regime]
            print(f"{arm:<26} {regime:>8}: " + "  ".join(f"{s}:{row[str(s)]['mean']:.3f}" for s in STEPS if str(s) in row))
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
