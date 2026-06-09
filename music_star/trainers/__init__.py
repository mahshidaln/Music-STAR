"""Trainer implementations for Music-STAR procedures."""

from music_star.trainers.adversarial_universal import AdversarialUniversalTrainer
from music_star.trainers.base import BaseTrainer
from music_star.trainers.decoder_finetuner import DecoderFinetuner
from music_star.trainers.embedding_supervised import EmbeddingSupervisedTrainer, StarLatentTrainer
from music_star.trainers.factory import build_trainer
from music_star.trainers.music_star_mixture import MusicStarMixtureTrainer
from music_star.trainers.music_star_stem import MusicStarStemTrainer

__all__ = [
    "AdversarialUniversalTrainer",
    "BaseTrainer",
    "DecoderFinetuner",
    "EmbeddingSupervisedTrainer",
    "MusicStarMixtureTrainer",
    "MusicStarStemTrainer",
    "StarLatentTrainer",
    "build_trainer",
]
