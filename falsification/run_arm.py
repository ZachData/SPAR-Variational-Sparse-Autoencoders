"""Run ONE training arm at ONE seed, driven entirely from the command line.

Why this exists: the training scripts hardcode their configuration in
`create_full_config()` with no CLI flags, so the documented workflow was "edit the
function between runs". That cannot be automated, and worse, `get_experiment_name()`
does not include the seed -- so every seed of an arm writes to the SAME directory
and silently overwrites its predecessors. A seeded design run that way yields one
checkpoint, not six.

This driver imports the training module by path, builds its config
programmatically, overrides the fields for the requested arm and seed, and gives
each run its own `save_dir` so nothing collides.

Usage:
    python falsification/run_arm.py --arm baseline --seed 1
    python falsification/run_arm.py --arm e1_penalty --seed 1 --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS_ROOT = REPO / "experiments"

# Shared base: the existing sweep point, so new runs are comparable to committed data.
BASE = dict(
    model_name="gelu-1l",
    layer=0,
    hook_name="blocks.0.hook_resid_post",
    dict_size_multiple=4.0,   # d = 2048
    total_steps=10000,
    lr=8e-4,
    auxk_alpha=1 / 32,
    n_ctxs=2500,
    ctx_len=128,
    refresh_batch_size=12,
    out_batch_size=192,
    use_wandb=False,          # unattended: never block on a wandb login prompt
    # Evaluation-only, and it matters. loss_recovered() tokenises eval_batch_size raw
    # texts and runs a full-vocabulary cross-entropy over them; at the default 24 that
    # asks for ~2.9GB on a 10GB card already holding ~6.7GB of activation buffer, so it
    # OOMs, is swallowed by an except-and-continue, and silently reports loss_original,
    # loss_reconstructed and loss_zero as NaN with frac_recovered = 0.0. Every committed
    # run in this repo shows that signature. 2x48 evaluates the same 96 sequences and
    # fits, recovering frac_recovered ~0.93. Identical for every arm, so cross-arm
    # comparability is unaffected.
    eval_batch_size=2,
    eval_n_batches=48,
)

# Arms in priority order. Each names the training script and its overrides.
ARMS: dict[str, dict[str, Any]] = {
    # E0: clean TopK baseline. activation_penalty=0.0 is essential -- this trainer
    # previously hardcoded a 0.01 penalty that could not be switched off.
    "baseline": {
        "script": "train_topk.py",
        "overrides": {**BASE, "k": 256, "activation_penalty": 0.0},
    },
    # E2: the actually-variational model. The experiment the preprint claims to
    # have run and did not (every released checkpoint used var_flag=0).
    "e2_learned_var": {
        "script": "train_vsae_topk.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 1.0, "var_flag": 1},
    },
    # E1: degeneracy control. TopK plus an L2 penalty matched to the vSAE's beta.
    "e1_penalty": {
        "script": "train_topk.py",
        "overrides": {**BASE, "k": 256, "activation_penalty": 1.0},
    },
    # E1 reference: the fixed-variance vSAE that E1 should reproduce.
    # kl_warmup_steps=0 is load-bearing. The trainer defaults it to int(0.1*steps)
    # = 1000, ramping kl_scale 0 -> 1 over the first 10% of training, while
    # top_k_with_feature_penalty applies its activation_cost at full strength from
    # step 0. Leaving the default made a nominally "matched beta" comparison differ
    # in the schedule of the very quantity being matched, which is the confound
    # recorded as item 7 in FINDINGS_2026-09-02.md. The pre-correction runs are kept
    # under archive/e1_vsae_ref_klwarmup1000/ so the size of that confound stays
    # measurable.
    #
    # use_april_update_mode=False is the second half of the same story. With the
    # warmup matched, the two arms' losses were shown numerically identical
    # (511.895264 both, six decimals) yet the checkpoints still differed at
    # d = 39-129. The residual is architectural: top_k_with_feature_penalty
    # centres its input on a tied pre-bias (`relu(encoder(x - b_dec))`), while
    # vsae_topk in april mode has no pre-bias at all and an untied decoder.bias.
    # That changes what the encoder sees and therefore which features fire --
    # exactly the construct E1 measures. False makes vsae_topk subtract
    # self.bias pre-encoder, matching the TopK form (REMEDIATION.md F7c).
    "e1_vsae_ref": {
        "script": "train_vsae_topk.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 1.0, "var_flag": 0,
                      "kl_warmup_steps": 0, "use_april_update_mode": False},
    },
    # E3: masked-KL, run as a 2-level factor rather than a confound. The masked
    # trainer omits the F.relu(mu) that vsae_topk.py applies unconditionally, so a
    # single E3 arm cannot separate "masking the KL did this" from "the ReLU did
    # this" (CLAUDE.md landmine 2). The preprint's equations show no ReLU and the
    # released code has one; neither is obviously the intended architecture, so
    # both are run and the ReLU's contribution is measured instead of assumed
    # (REMEDIATION.md F9b). e3_masked_kl_relu is the arm that matches vsae_topk.py
    # and is therefore the like-for-like comparison against e1_vsae_ref; plain
    # e3_masked_kl matches the preprint's equations.
    "e3_masked_kl": {
        "script": "train_vsae_topk_masked_kl.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 1.0, "var_flag": 0,
                      "relu_mu": False},
    },
    "e3_masked_kl_relu": {
        "script": "train_vsae_topk_masked_kl.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 1.0, "var_flag": 0,
                      "relu_mu": True},
    },
}

# E2 beta pilot: kl_coeff is NOT a shared scale across var_flag. At var_flag=0 the
# KL reduces to 0.5*||mu||^2, a mild L2 penalty; at var_flag=1 the variance term
# enters and contributes ~220 of a ~225 total loss. So "matched beta" compares two
# different interventions, and at beta=1.0 the model posterior-collapses in all six
# seeds (mu -> 1e-3, FVE = 0.0001) -- E2 as first run cannot distinguish "a
# variational SAE degenerates" from "beta=1.0 is far too large once sampling is on".
#
# Stage 1 is ONE seed per beta, used only to locate a workable beta. The selection
# rule is fixed here, BEFORE the pilot is looked at:
#
#   select the LARGEST beta whose frac_variance_explained is within 0.02 of the
#   baseline arm's mean FVE; if none qualifies, select the smallest beta tried.
#
# Stage 2 then trains 6 confirmatory seeds at the selected beta with seeds DISJOINT
# from the pilot's (the pilot uses seed 101; confirmatory uses 1-6), so no
# checkpoint contributes to both selection and inference (REMEDIATION.md F6).
E2_PILOT_BETAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
E2_PILOT_SEED = 101
E2_SELECTION_FVE_MARGIN = 0.02

for _beta in E2_PILOT_BETAS:
    ARMS[f"e2_beta_pilot_{_beta:g}"] = {
        "script": "train_vsae_topk.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": _beta, "var_flag": 1},
    }
del _beta

# Stage 2. This value was produced by select_e2_beta.py applying the rule above to
# the stage-1 pilot on 2026-09-03; it is written here literally so the confirmatory
# arm is reproducible without re-running the pilot, and the derivation is in
# logs/e2_pilot_*/pilot.log. Do not hand-edit it -- re-run the pilot and the
# selector if the grid or the rule changes.
#
# It came from the FALLBACK branch, and that is the pilot's actual finding: NO beta
# in {1e-4 ... 1} put a var_flag=1 model within 0.02 FVE of baseline (0.900). The
# grid is monotone and never gets close --
#
#   beta   1e-4     1e-3     1e-2     1e-1     1
#   FVE    0.4581   0.2843   0.1367   0.0004   0.0001
#
# -- so at 1e-4 the confirmatory arm characterises a model reconstructing at HALF
# the baseline's FVE, not a healthy one. Any E2 result must be read in that light,
# and a liveness metric on this arm should carry the same reconstruction-collapse
# caveat the pilot's beta=1.0 runs did (FINDINGS_2026-09-02.md item 6).
E2_SELECTED_BETA = 1e-4

ARMS["e2_confirm"] = {
    "script": "train_vsae_topk.py",
    "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": E2_SELECTED_BETA,
                  "var_flag": 1},
}

# E2 diagnostic control: sampling ON, KL entirely OFF.
#
# The stage-1 pilot showed FVE degrading monotonically across four orders of
# magnitude of beta (0.4581 at 1e-4 down to 0.0001 at 1) and never reaching
# baseline's 0.9003. Two very different mechanisms fit that curve equally well and
# the pilot cannot separate them:
#
#   (a) the KL penalty destroys reconstruction, and 1e-4 is still too strong;
#   (b) the reparameterisation destroys reconstruction, and beta is close to
#       irrelevant -- the curve is the KL merely adding insult to injury.
#
# beta = 0 with var_flag = 1 is the one configuration that tells them apart,
# because it is the only point where sampling is on and the KL contributes exactly
# zero to the loss. If FVE recovers toward baseline, (a); if it stays near the
# pilot's 0.46, (b) -- and E2's story is about the architecture, not a penalty.
#
# Read the learned variance too, not just FVE. With no KL there is nothing pushing
# the posterior toward the prior, so the model is free to drive sigma to ~0 and
# make itself deterministic. If it does, that is itself the answer: the model
# escapes sampling the moment it is allowed to, which is (b) stated more sharply.
#
# Six seeds, not one. This control exists to attribute a cause to the ARCHITECTURE,
# and the repo's own rule (CLAUDE.md, "Unit of analysis") is that an
# architecture-level claim needs a permutation test over training seeds. One seed
# would produce a number that cannot be tested. Six costs ~6 minutes.
#
# NOT pre-registered: proposed and run on 2026-09-03 after the stage-1 pilot was
# read. It is a control rather than a selection step, so it does not contaminate
# e2_confirm -- but it is exploratory and must be reported as such.
ARMS["e2_sampling_only"] = {
    "script": "train_vsae_topk.py",
    "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 0.0, "var_flag": 1},
}


def config_fields_static(script: str) -> set[str]:
    """Field names of a training script's ExperimentConfig, WITHOUT importing it.

    The training scripts import torch at module scope, so they cannot be imported
    on a machine without a GPU stack. Parsing the dataclass with `ast` lets us
    validate every arm's overrides on any machine -- catching a renamed or
    misspelled field now rather than three hours into an unattended run.
    """
    import ast

    path = REPO / "training_scripts" / script
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ExperimentConfig":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise ValueError(f"no ExperimentConfig dataclass found in {script}")


def check_arms() -> int:
    """Validate every arm's overrides against its script's config. No torch needed."""
    problems = 0
    for arm, spec in ARMS.items():
        fields = config_fields_static(spec["script"])
        unknown = sorted(set(spec["overrides"]) - fields)
        marker = "OK  " if not unknown else "FAIL"
        print(f"  {marker} {arm:<16} {spec['script']:<28} "
              f"{len(spec['overrides'])} overrides")
        if unknown:
            print(f"       unknown field(s): {unknown}")
            problems += 1
        for required in ("seed", "save_dir"):
            if required not in fields:
                print(f"       MISSING required field {required!r} on config")
                problems += 1
    return problems


def load_training_module(script: str):
    """Import a training script by path (training_scripts/ is not a package)."""
    path = REPO / "training_scripts" / script
    if not path.exists():
        raise FileNotFoundError(f"no such training script: {path}")
    spec = importlib.util.spec_from_file_location(f"_train_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_dir(arm: str, seed: int) -> Path:
    return RESULTS_ROOT / arm / f"seed{seed}"


def is_complete(arm: str, seed: int) -> bool:
    """A run is complete once it has written its done-marker."""
    return (run_dir(arm, seed) / "RUN_COMPLETE.json").exists()


def build_config(arm: str, seed: int):
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    spec = ARMS[arm]
    module = load_training_module(spec["script"])
    config = module.create_full_config()

    unknown = []
    for field, value in spec["overrides"].items():
        if not hasattr(config, field):
            unknown.append(field)
            continue
        setattr(config, field, value)
    if unknown:
        raise SystemExit(
            f"arm {arm!r}: {spec['script']} config has no field(s) {unknown}. "
            "Refusing to run rather than silently training the wrong thing."
        )

    config.seed = seed
    # THE important line: without a per-seed save_dir every seed of an arm lands in
    # the same directory, because get_experiment_name() omits the seed.
    config.save_dir = str(run_dir(arm, seed))
    return module, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=False, choices=sorted(ARMS))
    parser.add_argument("--seed", required=False, type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved config and exit without training")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if already marked complete")
    parser.add_argument("--check", action="store_true",
                        help="statically validate all arm configs and exit (no torch)")
    args = parser.parse_known_args()[0] if "--check" in sys.argv else parser.parse_args()

    if args.check:
        print("Validating arm configs against training-script dataclasses:")
        problems = check_arms()
        print("All arms valid." if not problems else f"{problems} problem(s) found.")
        return 1 if problems else 0

    if args.arm is None or args.seed is None:
        parser.error("--arm and --seed are required unless --check is given")

    if is_complete(args.arm, args.seed) and not args.force:
        print(f"SKIP {args.arm} seed={args.seed} (already complete)")
        return 0

    module, config = build_config(args.arm, args.seed)
    target = run_dir(args.arm, args.seed)

    if args.dry_run:
        from dataclasses import asdict
        print(f"DRY RUN {args.arm} seed={args.seed}")
        print(f"  script:   {ARMS[args.arm]['script']}")
        print(f"  save_dir: {target}")
        print(json.dumps(asdict(config), indent=2, default=str))
        return 0

    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"START {args.arm} seed={args.seed} -> {target}", flush=True)

    runner = module.ExperimentRunner(config)
    results = runner.run_training()

    elapsed = time.time() - started
    from dataclasses import asdict
    (target / "RUN_COMPLETE.json").write_text(json.dumps({
        "arm": args.arm,
        "seed": args.seed,
        "script": ARMS[args.arm]["script"],
        "elapsed_seconds": elapsed,
        "results": results,
        "config": asdict(config),
    }, indent=2, default=str))
    print(f"DONE {args.arm} seed={args.seed} in {elapsed/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
