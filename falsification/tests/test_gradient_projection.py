"""Contracts for the E1 gradient-projection factor.

`top_k.py` calls `remove_gradient_parallel_to_decoder_directions` every step and
then renormalises the decoder to unit norm. `vsae_topk.py` imports the helper and
never calls it, while still renormalising whenever `use_april_update_mode=False` --
the setting both E1 vSAE arms run under. Renormalising without projecting applies
the radial gradient component and then immediately undoes it, so each decoder
column's effective learning rate depends on how radial its gradient happened to be.

`project_decoder_grad` makes that a measured factor rather than a silent fix. These
tests pin the two things that would make the factor meaningless: that the default
preserves the historical behaviour, and that the flag is recoverable from a
checkpoint (it changes no parameter shape, so `config.json` is its only record).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

torch = pytest.importorskip("torch", reason="trainer contracts need torch")


def _load_vsae_topk():
    """Import the trainer without executing `dictionary_learning/__init__.py`.

    That `__init__` pulls in nnsight, circuitsvis, plotly and umap, none of which
    training needs. Loading by path keeps these contracts runnable in an
    analysis-only environment, the same reason `run_arm.load_training_module`
    imports training scripts by path.
    """
    if "dictionary_learning.trainers.vsae_topk" in sys.modules:
        return sys.modules["dictionary_learning.trainers.vsae_topk"]

    for name, path in [
        ("dictionary_learning", REPO / "dictionary_learning"),
        ("dictionary_learning.trainers", REPO / "dictionary_learning" / "trainers"),
    ]:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg

    def load(mod: str, rel: str):
        spec = importlib.util.spec_from_file_location(mod, REPO / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod] = module
        spec.loader.exec_module(module)
        return module

    load("dictionary_learning.dictionary", "dictionary_learning/dictionary.py")
    load("dictionary_learning.trainers.trainer", "dictionary_learning/trainers/trainer.py")
    return load(
        "dictionary_learning.trainers.vsae_topk",
        "dictionary_learning/trainers/vsae_topk.py",
    )


@pytest.fixture(scope="module")
def vsae():
    return _load_vsae_topk()


def _trained_one_step(vsae, project_decoder_grad: bool):
    """One update on the E1 arm's settings, at toy size on CPU in float32."""
    torch.manual_seed(0)
    model_config = vsae.VSAETopKConfig(
        activation_dim=16,
        dict_size=32,
        k=4,
        var_flag=0,
        use_april_update_mode=False,  # as both E1 vSAE arms run
        decoder_init_scale=1.0,
        project_decoder_grad=project_decoder_grad,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    training_config = vsae.VSAETopKTrainingConfig(
        steps=2000, lr=1e-3, kl_coeff=1.0, auxk_alpha=1 / 32
    )
    trainer = vsae.VSAETopKTrainer(
        model_config=model_config, training_config=training_config, seed=0
    )
    torch.manual_seed(1)
    trainer.update(0, torch.randn(64, 16))
    return trainer


def _max_parallel_component(weight) -> float:
    """Largest per-column projection of the gradient onto its decoder direction."""
    unit = weight / weight.norm(dim=0, keepdim=True)
    return (weight.grad * unit).sum(0).abs().max().item()


def test_projection_removes_the_parallel_component(vsae):
    trainer = _trained_one_step(vsae, project_decoder_grad=True)
    assert _max_parallel_component(trainer.ae.decoder.weight) < 1e-5


def test_default_leaves_the_parallel_component_intact(vsae):
    """The default must reproduce every existing checkpoint, so it must NOT project."""
    trainer = _trained_one_step(vsae, project_decoder_grad=False)
    assert _max_parallel_component(trainer.ae.decoder.weight) > 1e-3


def test_the_flag_actually_changes_the_gradient(vsae):
    """Guards against the flag being wired to a no-op."""
    on = _trained_one_step(vsae, project_decoder_grad=True)
    off = _trained_one_step(vsae, project_decoder_grad=False)
    assert not torch.allclose(
        on.ae.decoder.weight.grad, off.ae.decoder.weight.grad, atol=1e-6
    )


def test_config_defaults_to_historical_behaviour(vsae):
    assert vsae.VSAETopKConfig(activation_dim=4, dict_size=8, k=2).project_decoder_grad is False


@pytest.mark.parametrize("project", [True, False])
def test_flag_is_recorded_in_config(vsae, project):
    """Load-bearing: the flag changes no parameter shape, so a state dict cannot
    reveal which E1 arm a checkpoint belongs to. config.json is the only record."""
    trainer = _trained_one_step(vsae, project_decoder_grad=project)
    assert trainer.config["project_decoder_grad"] is project
