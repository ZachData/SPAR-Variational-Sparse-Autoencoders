"""A real example of selection churn, for the paper's selection-churn figure.

Takes one trained VSAETopK checkpoint from the TopK noise sweep, one fixed batch
of gelu-1l activations, and two independent stochastic forward passes, and
records:

* one representative token (the token whose two-pass Jaccard is closest to the
  batch mean -- a rule, not a pick): its deterministic pre-activation mu and
  both sampled z vectors for the features ranked around the k-th place, and
  which of them each pass selected;
* for every rank r (features ordered by mu, per token), the fraction of tokens
  whose rank-r feature is selected by exactly one of the two passes -- where in
  the ranking the churn happens;
* the batch-mean Jaccard.

Everything is measured in two regimes, because they differ by ~25x in noise:

  raw       the saved checkpoint fed raw activations. This is how
            `read_a2_dose_response.py` measures Tables 3-5's Jaccard and how
            `train_vsae_topk.py`'s `evaluate()` measures their FVE. `scale_biases`
            multiplies mu by norm_factor at save time but deliberately leaves
            log sigma^2 unchanged, so the noise relative to mu is sigma/norm_factor.
  training  `scale_biases(1/norm_factor)` and activations / norm_factor: the
            exact state the model trained in, noise sigma relative to mu.

Values are reported in training-space units (raw values / norm_factor) in both
regimes, so they share an axis with sigma and with `read_preact_gap.py`'s gap.

    /usr/bin/python3 falsification/read_selection_example.py      # LOCAL GPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "falsification" / "selection_example_results.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="a2_sigma_init_m3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--window", type=int, default=16, help="ranks either side of the k-th place to record")
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
    ckpt = next((REPO / "experiments" / args.arm / f"seed{args.seed}").glob("*/trainer_0/ae.pt"))
    cfg = json.loads((ckpt.parent / "config.json").read_text())["trainer"]

    def passes(ae, x, scale):
        """mu, z1, z2 (in training-space units) and the two selection masks."""
        torch.manual_seed(args.rng_seed + 1)
        out = {"mu": [], "z1": [], "z2": [], "s1": [], "s2": []}
        with torch.no_grad():
            for i in range(0, x.shape[0], 4096):
                xb = x[i:i + 4096].to(ae.encoder.weight.dtype)
                _, mu, _, _ = ae.encode(xb, training=False)
                for j in (1, 2):
                    _, z, _, _, idx, _ = ae.encode(xb, return_topk=True, training=True)
                    m = torch.zeros(xb.shape[0], ae.dict_size, dtype=torch.bool, device=xb.device)
                    m.scatter_(1, idx, True)
                    out[f"z{j}"].append((z.float() / scale).cpu())
                    out[f"s{j}"].append(m.cpu())
                out["mu"].append((mu.float() / scale).cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    results = {"arm": args.arm, "seed": args.seed, "checkpoint": str(ckpt.relative_to(REPO)),
               "log_var_init": cfg.get("log_var_init"), "n_tokens": int(acts.shape[0]),
               "norm_factor": nf, "rng_seed": args.rng_seed, "regimes": {}}
    token = None
    for regime in ("raw", "training"):
        ae = VSAETopK.from_pretrained(str(ckpt), device=args.device, var_flag=1)
        ae.eval()
        k = int(ae.k.item())
        if regime == "training":
            ae.scale_biases(1.0 / nf)
            r = passes(ae, acts / nf, 1.0)
        else:
            r = passes(ae, acts, nf)
        inter = (r["s1"] & r["s2"]).sum(1).float()
        jac = inter / (2 * k - inter)
        order = torch.argsort(r["mu"], dim=1, descending=True)            # rank -> feature, per token
        flip = torch.gather(r["s1"] ^ r["s2"], 1, order).float().mean(0)  # P(flip) by mu-rank
        # Of the features a pass selects, how many sit outside mu's own top k, and
        # how many have mu == 0 (ReLU'd off: selected by noise alone).
        in_topk = torch.zeros_like(r["s1"])
        in_topk.scatter_(1, order[:, :k], True)
        outside = float((r["s1"] & ~in_topk).sum(1).float().mean() / k)
        mu_zero = float((r["s1"] & (r["mu"] <= 0)).sum(1).float().mean() / k)
        if token is None:                                                 # chosen once, on the raw regime
            token = int(torch.argmin((jac - jac.mean()).abs()))
        o = order[token]
        lo, hi = k - args.window, k + args.window
        feats = o[lo:hi]
        tok_flips = (r["s1"][token] ^ r["s2"][token]).nonzero().flatten()
        rank_of = torch.empty_like(o)
        rank_of[o] = torch.arange(o.numel())
        flip_ranks = sorted(int(rank_of[f]) for f in tok_flips)
        results["regimes"][regime] = {
            "k": k,
            "mean_jaccard": float(jac.mean()),
            "chance_jaccard": k / (2 * ae.dict_size - k),
            "flip_rate_by_rank": [round(float(v), 6) for v in flip],
            "frac_selected_outside_mu_topk": outside,
            "frac_selected_with_mu_zero": mu_zero,
            "token": {
                "index": token, "jaccard": float(jac[token]),
                "ranks": list(range(lo, hi)),
                "mu": [float(r["mu"][token, f]) for f in feats],
                "z1": [float(r["z1"][token, f]) for f in feats],
                "z2": [float(r["z2"][token, f]) for f in feats],
                "selected1": [bool(r["s1"][token, f]) for f in feats],
                "selected2": [bool(r["s2"][token, f]) for f in feats],
                "n_flips": len(flip_ranks), "flip_ranks": flip_ranks,
                "gap_at_k": float(r["mu"][token, o[k - 1]] - r["mu"][token, o[k]]),
            },
        }
        print(f"{regime:>8}: mean Jaccard {jac.mean():.4f} (chance {k / (2 * ae.dict_size - k):.4f}); "
              f"selected outside mu's top-k {outside:.3f}, with mu=0 {mu_zero:.3f}; token {token}: Jaccard {jac[token]:.4f}, {len(flip_ranks)} flips at ranks "
              f"{flip_ranks[:6]}{'...' if len(flip_ranks) > 6 else ''}")
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {args.out} (norm_factor {nf:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
