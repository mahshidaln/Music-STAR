"""Training entry points for Music-STAR procedure configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music_star.config import (
    MusicStarConfig,
    list_builtin_configs,
    load_builtin_config,
    load_config,
)
from music_star.trainers import (
    AdversarialUniversalTrainer,
    DecoderFinetuner,
    MusicStarMixtureTrainer,
    MusicStarStemTrainer,
    StarLatentTrainer,
    build_trainer,
)
from music_star.utils import LossManager


@dataclass
class TrainerConfig:
    """Minimal training configuration used by compatibility callers.

    Parameters
    ----------
    exp_name : str
        Experiment name.
    checkpoint : pathlib.Path
        Checkpoint root directory.
    recipe : str, optional
        Training recipe name.
    rank : int, optional
        Distributed rank.
    """

    exp_name: str
    checkpoint: Path
    recipe: str = "star_latent"
    rank: int = 0


def load_training_config(config: str | Path | MusicStarConfig) -> MusicStarConfig:
    """Load a training procedure config from a path, bundled name, or object."""

    if isinstance(config, MusicStarConfig):
        loaded = config
    else:
        path = Path(config)
        loaded = load_config(path) if path.exists() else load_builtin_config(str(config))
    supported = {
        "decoder_finetune",
        "embedding_supervised",
        "music_star_mixture_supervised",
        "music_star_stem_supervised",
        "star_latent",
        "universal_adversarial",
    }
    if loaded.recipe not in supported:
        raise ValueError(f"Unsupported training procedure: {loaded.recipe!r}")
    return loaded


def build_trainer_from_config(config: str | Path | MusicStarConfig, **overrides: Any):
    """Build a trainer from a clean procedure config."""

    loaded = load_training_config(config)
    return build_trainer(loaded.to_namespace(**overrides))


def list_training_configs() -> list[str]:
    """List bundled procedure configs, excluding non-training artifacts."""

    return [name for name in list_builtin_configs() if name.startswith("recipe_")]


def train_from_args(args: Any) -> None:
    """Build and run a trainer from an argparse-like object."""

    build_trainer(args).train()


def train_from_config(config: str | Path | MusicStarConfig, **overrides: Any) -> None:
    """Run training from a clean procedure config."""

    build_trainer_from_config(config, **overrides).train()


def main(argv=None) -> None:
    """Run the config-driven training command.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments. When ``None``, ``argparse`` reads from
        ``sys.argv``.
    """

    parser = argparse.ArgumentParser(description="Run a Music-STAR training procedure.")
    parser.add_argument("--config", type=Path, help="Path to a JSON procedure config.")
    parser.add_argument("--config-name", help="Bundled procedure config name.")
    parser.add_argument("--device", help="Override device, for example cpu or cuda.")
    parser.add_argument(
        "--list-configs", action="store_true", help="List bundled training configs."
    )
    parsed = parser.parse_args(argv)

    if parsed.list_configs:
        for name in list_training_configs():
            print(name)
        return

    if not parsed.config and not parsed.config_name:
        raise SystemExit("Provide --config, --config-name, or --list-configs")

    overrides = {}
    if parsed.device:
        overrides["device"] = parsed.device
    train_from_config(parsed.config or parsed.config_name, **overrides)


__all__ = [
    "AdversarialUniversalTrainer",
    "DecoderFinetuner",
    "LossManager",
    "MusicStarMixtureTrainer",
    "MusicStarStemTrainer",
    "StarLatentTrainer",
    "TrainerConfig",
    "build_trainer_from_config",
    "build_trainer",
    "list_training_configs",
    "load_training_config",
    "main",
    "train_from_args",
    "train_from_config",
]
