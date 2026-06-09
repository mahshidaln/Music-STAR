"""Configuration helpers for Music-STAR training and generation recipes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CONFIG_PACKAGE = "music_star.configs"


@dataclass
class MusicStarConfig:
    """Structured representation of a recipe JSON file.

    The original experiments stored most training settings in ``args.pth``.
    These configs make the recipe explicit: which training loop to run, which
    loss family it uses, and which architecture/runtime parameters are needed.
    """

    name: str
    recipe: str
    description: str = ""
    loss: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    optim: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MusicStarConfig:
        """Create a config object from a dictionary.

        Parameters
        ----------
        values : dict[str, Any]
            Parsed JSON configuration values.

        Returns
        -------
        MusicStarConfig
            Structured configuration object.

        Raises
        ------
        ValueError
            If required keys are missing.
        """

        required = {"name", "recipe"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config object to plain Python values.

        Returns
        -------
        dict[str, Any]
            JSON-serializable configuration dictionary.
        """

        return {
            "name": self.name,
            "recipe": self.recipe,
            "description": self.description,
            "loss": self.loss,
            "data": self.data,
            "model": self.model,
            "optim": self.optim,
            "runtime": self.runtime,
            "checkpoint": self.checkpoint,
            "generation": self.generation,
            "metadata": self.metadata,
        }

    def to_namespace(self, **overrides: Any) -> SimpleNamespace:
        """Flatten config sections into an argparse-like namespace."""

        values: dict[str, Any] = {}
        for section in (
            self.data,
            self.model,
            self.optim,
            self.runtime,
            self.checkpoint,
            self.generation,
            {"recipe": self.recipe, "loss": self.loss, "exp_name": self.name},
            overrides,
        ):
            values.update(section)
        return SimpleNamespace(**values)


def load_config(path: str | Path) -> MusicStarConfig:
    """Load a Music-STAR recipe config from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected object config in {path}")
    return MusicStarConfig.from_dict(values)


def save_config(config: MusicStarConfig, path: str | Path) -> None:
    """Write a Music-STAR recipe config as pretty JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def list_builtin_configs() -> list[str]:
    """Return bundled recipe config names."""

    return sorted(
        resource.name
        for resource in resources.files(CONFIG_PACKAGE).iterdir()
        if resource.name.endswith(".json")
    )


def load_builtin_config(name: str) -> MusicStarConfig:
    """Load a bundled config by filename or stem."""

    filename = name if name.endswith(".json") else f"{name}.json"
    resource = resources.files(CONFIG_PACKAGE).joinpath(filename)
    with resource.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected object config in bundled config {filename}")
    return MusicStarConfig.from_dict(values)


__all__ = [
    "MusicStarConfig",
    "list_builtin_configs",
    "load_builtin_config",
    "load_config",
    "save_config",
]
