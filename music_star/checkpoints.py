"""Checkpoint loading helpers for legacy Music-STAR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from music_star.models import Encoder, LegacyEncoder, WaveNet

DEFAULT_MODEL_FILE = "bestmodel_0.pth"


@dataclass
class LoadedModel:
    """A model loaded from a checkpoint with compatibility notes."""

    model: torch.nn.Module
    args: Any
    checkpoint: dict[str, Any]
    notes: list[str]


def load_saved_args(checkpoint_dir: str | Path) -> Any:
    """Load the saved argparse namespace from ``args.pth``."""

    payload = torch.load(Path(checkpoint_dir) / "args.pth", map_location="cpu")
    if isinstance(payload, (list, tuple)):
        return payload[0]
    return payload


def load_checkpoint_state(
    checkpoint_dir: str | Path,
    model_file: str = DEFAULT_MODEL_FILE,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint dict from a checkpoint directory."""

    state = torch.load(Path(checkpoint_dir) / model_file, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(state)!r}")
    return state


def _copy_args(args: Any) -> Any:
    if hasattr(args, "__dict__"):
        return SimpleNamespace(**vars(args))
    raise TypeError(f"Expected argparse-like args object, got {type(args)!r}")


def decoder_args_for_state(
    args: Any, decoder_state: dict[str, torch.Tensor]
) -> tuple[Any, list[str]]:
    """Return args adjusted to match a decoder state dict.

    Some legacy checkpoints have ``args.latent_d`` set to the encoder latent
    dimension while the decoder condition projection was trained with a
    different condition dimension. The state dict is authoritative here.
    """

    patched_args = _copy_args(args)
    notes: list[str] = []
    condition_weight = decoder_state.get("layers.0.condition.weight")
    if condition_weight is None:
        return patched_args, notes

    condition_channels = int(condition_weight.shape[1])
    if getattr(patched_args, "latent_d", None) != condition_channels:
        notes.append(
            f"patched latent_d from {getattr(patched_args, 'latent_d', None)} "
            f"to decoder condition channels {condition_channels}"
        )
        patched_args.latent_d = condition_channels
    return patched_args, notes


def is_legacy_encoder_state(encoder_state: dict[str, torch.Tensor]) -> bool:
    """Return whether an encoder state dict needs the recovered GLU architecture."""

    return any(".extra." in key for key in encoder_state)


def strip_legacy_encoder_extra_keys(
    encoder_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Remove legacy ``DilatedResConv.extra`` weights for explicit fallback use.

    Prefer :class:`music_star.models.LegacyEncoder` when loading recovered
    Music-STAR checkpoints. This helper remains available for experiments that
    intentionally want to compare those weights against the standard encoder.
    """

    stripped = {key: value for key, value in encoder_state.items() if ".extra." not in key}
    removed = len(encoder_state) - len(stripped)
    notes = []
    if removed:
        notes.append(f"ignored {removed} legacy encoder extra parameters")
    return stripped, notes


def load_encoder(
    checkpoint_dir: str | Path,
    model_file: str = DEFAULT_MODEL_FILE,
    state_key: str = "encoder_state",
) -> LoadedModel:
    """Load an encoder state into the matching packaged implementation."""

    args = load_saved_args(checkpoint_dir)
    checkpoint = load_checkpoint_state(checkpoint_dir, model_file)
    encoder_state = checkpoint[state_key]
    notes: list[str] = []

    model: torch.nn.Module
    if is_legacy_encoder_state(encoder_state):
        model = LegacyEncoder(args)
        notes.append("loaded recovered legacy GLU encoder architecture")
    else:
        model = Encoder(args)

    model.load_state_dict(encoder_state, strict=True)
    model.eval()
    return LoadedModel(model=model, args=args, checkpoint=checkpoint, notes=notes)


def load_decoder(
    checkpoint_dir: str | Path,
    model_file: str = DEFAULT_MODEL_FILE,
    state_key: str = "decoder_state",
) -> LoadedModel:
    """Load a WaveNet decoder state into the current implementation."""

    args = load_saved_args(checkpoint_dir)
    checkpoint = load_checkpoint_state(checkpoint_dir, model_file)
    decoder_state = checkpoint[state_key]
    decoder_args, notes = decoder_args_for_state(args, decoder_state)

    model = WaveNet(decoder_args)
    model.load_state_dict(decoder_state, strict=True)
    model.eval()
    return LoadedModel(model=model, args=decoder_args, checkpoint=checkpoint, notes=notes)
