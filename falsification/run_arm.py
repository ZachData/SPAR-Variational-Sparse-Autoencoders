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
    "e1_vsae_ref": {
        "script": "train_vsae_topk.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 1.0, "var_flag": 0},
    },
    # E3: masked-KL. NOTE: that trainer omits the F.relu(mu) applied by vsae_topk.py,
    # so this comparison is confounded until one is patched to match the other.
    "e3_masked_kl": {
        "script": "train_vsae_topk_masked_kl.py",
        "overrides": {**BASE, "k_fraction": 0.125, "kl_coeff": 1.0, "var_flag": 0},
    },
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
