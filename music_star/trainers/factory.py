"""Factory for config-driven trainer construction."""

from __future__ import annotations

from typing import Any

from music_star.trainers.adversarial_universal import AdversarialUniversalTrainer
from music_star.trainers.base import BaseTrainer
from music_star.trainers.decoder_finetuner import DecoderFinetuner
from music_star.trainers.embedding_supervised import EmbeddingSupervisedTrainer
from music_star.trainers.music_star_mixture import MusicStarMixtureTrainer
from music_star.trainers.music_star_stem import MusicStarStemTrainer


def build_trainer(args: Any) -> BaseTrainer:
    """Build a trainer from an argparse-like namespace.

    Parameters
    ----------
    args
        Namespace with a ``recipe`` attribute.

    Returns
    -------
    BaseTrainer
        Concrete trainer for the requested recipe.

    Raises
    ------
    ValueError
        If ``args.recipe`` is unknown.
    """

    recipe = getattr(args, "recipe", None)
    if recipe == "universal_adversarial":
        return AdversarialUniversalTrainer(args)
    if recipe == "decoder_finetune":
        return DecoderFinetuner(args)
    if recipe in {"embedding_supervised", "star_latent"}:
        return EmbeddingSupervisedTrainer(args)
    if recipe == "music_star_mixture_supervised":
        return MusicStarMixtureTrainer(args)
    if recipe == "music_star_stem_supervised":
        return MusicStarStemTrainer(args)
    raise ValueError(f"Unknown training recipe: {recipe!r}")


__all__ = ["build_trainer"]
