"""Re-measure the noise sweeps in the regime the models were trained in.

Every saved checkpoint is converted at save time by `scale_biases(norm_factor)`
so that it runs on raw activations: encoder/decoder biases (and JumpReLU's
threshold) are multiplied by norm_factor (~25.4 for gelu-1l layer 0), so mu
comes out norm_factor times larger, while log sigma^2 is deliberately preserved
(`var_encoder.weight` is divided instead; ERRATA §2). The noise added before
selection therefore stays at sigma while mu grows norm_factor-fold: on raw
activations a saved checkpoint runs at an effective noise of sigma/norm_factor
relative to mu. That is the regime `train_*.py`'s `evaluate()` (FVE, L0) and
`read_a{2,3,4}_dose_response.py` (Jaccard) measured Tables 2-5 in.

This reader measures every sweep checkpoint in BOTH regimes on the same
activations and the same noise draws:

  raw       the saved checkpoint on raw activations (the published regime);
  training  `scale_biases(1/norm_factor)` on activations / norm_factor -- the
            exact state the model trained in, noise sigma relative to mu.

Per checkpoint: FVE (evaluation.py's definition, per 4096-token chunk, with
sampling on as `evaluate()` has it), mean active features per token (L0), and
selection Jaccard between two independent sampled passes (8,192 tokens, as the
A2-A4 readers). The raw-regime FVE is a check on this reader: it should agree
with each run's own evaluation_results[_corrected].json to sampling noise.

Pre-2026-09-04 checkpoints (`e2_*`) first get the existing log-variance
correction (var_encoder weight and bias / 25.75, `read_a2_dose_response.py`).
norm_factor is not saved in any checkpoint (ERRATA §2); it is re-estimated as
sqrt(mean ||x||^2) on the activation batch, the estimator `get_norm_factor` uses.

    /usr/bin/python3 falsification/read_training_regime.py      # LOCAL GPU, ~10 min
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "falsification" / "training_regime_results.json"
PREFIX_NORM_FACTOR = 25.75   # read_a2_dose_response.py / read_selection_jaccard.py

# (sweep, log_var_init or None for a deterministic baseline, arm, needs pre-fix correction)
ARMS = [
    ("topk", None, "baseline", False),
    ("topk", -1.0, "a2_sigma_init_m1", False),
    ("topk", -2.0, "e2_sampling_only", True),
    ("topk", -3.0, "a2_sigma_init_m3", False),
    ("topk", -4.0, "a2_sigma_init_m4", False),
    ("topk", -5.0, "a2_sigma_init_m5", False),
    ("topk", -8.0, "e2_sigma_low_init", True),
    ("batchtopk", None, "a3_batchtopk_baseline", False),
    ("batchtopk", -1.0, "a3_batchtopk_sigma_init_m1", False),
    ("batchtopk", -2.0, "a3_batchtopk_sampling_only", False),
    ("batchtopk", -3.0, "a3_batchtopk_sigma_init_m3", False),
    ("batchtopk", -4.0, "a3_batchtopk_sigma_init_m4", False),
    ("batchtopk", -5.0, "a3_batchtopk_sigma_init_m5", False),
    ("batchtopk", -8.0, "a3_batchtopk_sigma_low_init", False),
    ("jumprelu", None, "a4_jumprelu_baseline", False),
    ("jumprelu", -1.0, "a4_jumprelu_sigma_init_m1", False),
    ("jumprelu", -2.0, "a4_jumprelu_sampling_only", False),
    ("jumprelu", -3.0, "a4_jumprelu_sigma_init_m3", False),
    ("jumprelu", -3.5, "a4_jumprelu_sigma_init_m3_5", False),
    ("jumprelu", -4.0, "a4_jumprelu_sigma_init_m4", False),
    ("jumprelu", -4.5, "a4_jumprelu_sigma_init_m4_5", False),
    ("jumprelu", -5.0, "a4_jumprelu_sigma_init_m5", False),
    ("jumprelu", -5.5, "a4_jumprelu_sigma_init_m5_5", False),
    ("jumprelu", -6.0, "a4_jumprelu_sigma_init_m6", False),
    ("jumprelu", -8.0, "a4_jumprelu_sigma_low_init", False),
    # Table 2's KL arm and the beta pilot (Figure 4a), same TopK trainer.
    ("e2", -2.0, "e2_confirm", True),
    ("e2_pilot", -2.0, "e2_beta_pilot_0.0001", True),
    ("e2_pilot", -2.0, "e2_beta_pilot_0.001", True),
    ("e2_pilot", -2.0, "e2_beta_pilot_0.01", True),
    ("e2_pilot", -2.0, "e2_beta_pilot_0.1", True),
    ("e2_pilot", -2.0, "e2_beta_pilot_1", True),
]


def checkpoints(arm: str) -> list[tuple[int, Path]]:
    found = {}
    for p in (REPO / "experiments" / arm).glob("seed*/*/trainer_0"):
        if (p / "ae.pt").exists():
            found[int(re.search(r"seed(\d+)", str(p)).group(1))] = p
    return [(s, found[s]) for s in sorted(found)]


def published_fve(trainer_dir: Path) -> float | None:
    run = trainer_dir.parent
    for name in ("evaluation_results_corrected.json", "evaluation_results.json"):
        f = run / name
        if f.exists():
            return json.loads(f.read_text())["frac_variance_explained"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=32768, help="tokens for FVE and L0")
    ap.add_argument("--n-jaccard", type=int, default=8192, help="tokens for Jaccard (as the A2-A4 readers)")
    ap.add_argument("--chunk", type=int, default=1024, help="tokens per forward pass: the training and evaluate() batch size")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", nargs="*", help="subset of arm names (default: all)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import torch

    from dictionary_learning.utils import load_dictionary
    from falsification.read_penalty_clamp import draw_activations

    torch.manual_seed(args.rng_seed)
    acts = draw_activations(args.n_tokens, args.device)
    nf = float(torch.sqrt(torch.mean(torch.sum(acts.float() ** 2, dim=1))))
    print(f"norm_factor re-estimated: {nf:.4f} over {acts.shape[0]:,} tokens")

    def forward(ae, x):
        """(x_hat, selection mask) with sampling on, whichever class this is."""
        name = type(ae).__name__
        if name == "VSAETopK":
            sparse, _, _, _ = ae.encode(x, training=True)
            return ae.decode(sparse), sparse != 0
        if name == "VSAEBatchTopK":
            enc = ae.encode(x)
            return ae.decode(enc["sparse_z"]), enc["selection_mask"]
        if name == "VSAEJumpReLU":
            mu, log_var = ae.encode(x)
            gated = ae.select(ae.reparameterize(mu, log_var))
            return ae.decode(gated), gated > 0
        x_hat, f = ae(x, output_features=True)       # deterministic baselines
        return x_hat, f != 0

    def measure(ae, x):
        fves, l0s, jacs = [], [], []
        torch.manual_seed(args.rng_seed + 1)
        with torch.no_grad():
            for i in range(0, x.shape[0], args.chunk):
                xb = x[i:i + args.chunk]
                xd = xb.to(next(ae.parameters()).dtype)
                x_hat, m1 = forward(ae, xd)
                xf, xh = xb.float(), x_hat.float()
                fve = 1 - torch.var(xf - xh, dim=0).sum() / torch.clamp(torch.var(xf, dim=0).sum(), min=1e-8)
                fves.append(float(torch.clamp(fve, 0.0, 1.0)))
                l0s.append(float(m1.float().sum(1).mean()))
                if i < args.n_jaccard:
                    _, m2 = forward(ae, xd)
                    inter = (m1 & m2).sum(1).float()
                    union = (m1 | m2).sum(1).float()
                    jacs.append(torch.where(union > 0, inter / union, torch.ones_like(union)).cpu())
        return mean(fves), mean(l0s), float(torch.cat(jacs).mean())

    out = {"norm_factor": nf, "n_tokens": int(acts.shape[0]), "n_jaccard": args.n_jaccard,
           "chunk": args.chunk, "rng_seed": args.rng_seed, "prefix_norm_factor": PREFIX_NORM_FACTOR,
           "arms": []}
    if Path(args.out).exists() and args.arms:
        out = json.loads(Path(args.out).read_text())
        out["arms"] = [a for a in out["arms"] if a["arm"] not in args.arms]
    t0 = time.time()
    for sweep, lvi, arm, prefix in ARMS:
        if args.arms and arm not in args.arms:
            continue
        seeds = []
        for seed, tdir in checkpoints(arm):
            row = {"seed": seed, "published_fve": published_fve(tdir)}
            for regime in ("raw", "training"):
                ae, _ = load_dictionary(str(tdir), device=args.device)
                ae.eval()
                if prefix:
                    with torch.no_grad():
                        ae.var_encoder.bias.div_(PREFIX_NORM_FACTOR)
                        ae.var_encoder.weight.div_(PREFIX_NORM_FACTOR)
                x = acts
                if regime == "training":
                    ae.scale_biases(1.0 / nf)
                    x = acts / nf
                fve, l0, jac = measure(ae, x)
                row[regime] = {"fve": fve, "l0": l0, "jaccard": jac}
                del ae
            seeds.append(row)
        rec = {"sweep": sweep, "log_var_init": lvi, "arm": arm, "prefix_corrected": prefix,
               "sigma": None if lvi is None else float(torch.exp(torch.tensor(0.5 * max(lvi, -6.0)))),
               "seeds": seeds}
        for regime in ("raw", "training"):
            for m in ("fve", "l0", "jaccard"):
                v = [s[regime][m] for s in seeds]
                rec[f"{regime}_{m}_mean"] = mean(v)
                rec[f"{regime}_{m}_std"] = stdev(v) if len(v) > 1 else 0.0
        pub = [s["published_fve"] for s in seeds if s["published_fve"] is not None]
        rec["published_fve_mean"] = mean(pub) if pub else None
        out["arms"].append(rec)
        print(f"[{time.time() - t0:6.0f}s] {arm:<30} n={len(seeds):>2}  "
              f"FVE pub {rec['published_fve_mean'] or float('nan'):.4f} raw {rec['raw_fve_mean']:.4f} "
              f"train {rec['training_fve_mean']:.4f} | J raw {rec['raw_jaccard_mean']:.4f} "
              f"train {rec['training_jaccard_mean']:.4f} | L0 raw {rec['raw_l0_mean']:.1f} "
              f"train {rec['training_l0_mean']:.1f}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
