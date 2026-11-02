"""The E1 identity, checked numerically: at fixed variance the vSAE's loss and
the TopK+L2 null model's loss are the same function.

Builds the two trainers E1 compares -- `top_k_with_feature_penalty.TopKTrainer`
(activation_penalty = 1) and `vsae_topk.VSAETopKTrainer` (var_flag = 0,
kl_coeff = 1, no KL warmup, the TopK bias form) -- loads ONE set of weights into
both, and evaluates each trainer's own `loss()` on the same batch of real
gelu-1l activations in training space (divided by the re-estimated
norm_factor). Two weight sets: a fresh initialisation, and the trained
`e1_penalty` seed-1 checkpoint converted back to training space. In both, a
fresh trainer has no dead features yet, so the AuxK term is zero on both sides
and the comparison is reconstruction + 0.5 * beta * ||mu||^2 against
reconstruction + 0.5 * lambda * ||f||^2.

Float32 throughout, so agreement is limited by float32, not bfloat16. The
paper's original check (511.895264 on both, six decimals) is in the dated record
and was not scripted; this script is its committed replacement.

    /usr/bin/python3 falsification/check_loss_identity.py      # GPU (CPU works, slowly)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "falsification" / "loss_identity_results.json"
CKPT = REPO / "experiments/e1_penalty/seed1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import torch

    from dictionary_learning.trainers.top_k_with_feature_penalty import (
        TopKConfig, TopKTrainer, TopKTrainingConfig)
    from dictionary_learning.trainers.vsae_topk import (
        VSAETopKConfig, VSAETopKTrainer, VSAETopKTrainingConfig)
    from falsification.read_penalty_clamp import draw_activations

    dev = torch.device(args.device)
    torch.manual_seed(args.rng_seed)
    acts = draw_activations(args.n_tokens, args.device).float()
    nf = float(torch.sqrt(torch.mean(torch.sum(acts ** 2, dim=1))))
    x = acts / nf

    def trainers():
        t = TopKTrainer(model_config=TopKConfig(activation_dim=512, dict_size=2048, k=256,
                                                dtype=torch.float32, device=dev),
                        training_config=TopKTrainingConfig(steps=10_000, lr=8e-4, activation_penalty=1.0),
                        seed=args.rng_seed)
        v = VSAETopKTrainer(model_config=VSAETopKConfig(activation_dim=512, dict_size=2048, k=256, var_flag=0,
                                                        use_april_update_mode=False, dtype=torch.float32,
                                                        device=dev),
                            training_config=VSAETopKTrainingConfig(steps=10_000, lr=8e-4, kl_coeff=1.0,
                                                                   kl_warmup_steps=0),
                            seed=args.rng_seed)
        return t, v

    def copy_into_vsae(t, v):
        with torch.no_grad():
            v.ae.encoder.weight.copy_(t.ae.encoder.weight)
            v.ae.encoder.bias.copy_(t.ae.encoder.bias)
            v.ae.decoder.weight.copy_(t.ae.decoder.weight)
            v.ae.bias.copy_(t.ae.b_dec)

    results = {"n_tokens": int(x.shape[0]), "norm_factor": nf, "rng_seed": args.rng_seed, "cases": {}}
    for case in ("fresh_init", "trained_e1_penalty_seed1"):
        t, v = trainers()
        if case.startswith("trained"):
            sd = torch.load(next(CKPT.glob("*/trainer_0/ae.pt")), map_location=dev)
            with torch.no_grad():
                for name in ("encoder.weight", "encoder.bias", "decoder.weight", "b_dec"):
                    t.ae.state_dict()[name].copy_(sd[name].float())
            t.ae.scale_biases(1.0 / nf)          # saved checkpoints are in raw-activation space
        copy_into_vsae(t, v)
        t.ae.eval(); v.ae.eval()
        with torch.no_grad():
            lt = t.loss(x, step=0, logging=True)
            lv = v.loss(x, step=0, logging=True)
        a, b = float(lt.losses["loss"]), float(lv.losses["loss"])
        results["cases"][case] = {"topk_l2_loss": a, "vsae_loss": b, "abs_diff": abs(a - b),
                                  "rel_diff": abs(a - b) / abs(a),
                                  "topk_terms": {k: lt.losses[k] for k in ("l2_loss", "auxk_loss", "activation_cost")},
                                  "vsae_terms": {k: v_ for k, v_ in lv.losses.items() if isinstance(v_, float)}}
        print(f"{case:>26}: TopK+L2 {a:.6f}   vSAE {b:.6f}   |diff| {abs(a - b):.2e} (rel {abs(a - b) / abs(a):.1e})")
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
