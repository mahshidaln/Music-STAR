"""Command-line parser for Music-STAR workflows."""

from __future__ import annotations

import argparse
from pathlib import Path


def ratio(value: str) -> float:
    """Parse a split ratio as either ``0.8`` or ``80``."""

    parsed = float(value)
    if parsed > 1:
        parsed /= 100.0
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1, or 0 and 100")
    return parsed


def get_parser() -> argparse.ArgumentParser:
    """Build the main Music-STAR workflow parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for preprocessing and training workflows. Generation is exposed
        by :mod:`music_star.translate`.
    """

    parser = argparse.ArgumentParser(
        prog="music-star",
        description="Music-STAR preprocessing and training tools.",
    )

    parser.add_argument(
        "--operation-mode",
        choices=["preprocess", "train"],
        nargs="+",
        required=True,
        help="Workflow stage(s) to run.",
    )
    parser.add_argument("--universe", action="store_true", help="Train the universe encoder.")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--exp-name", type=str, default="music-star", help="Experiment name.")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints"), help="Checkpoint root."
    )
    parser.add_argument("--pretrained", type=Path, help="Pretrained checkpoint path.")
    parser.add_argument("--load-optimizer", action="store_true", help="Restore optimizer state.")
    parser.add_argument("--save-epoch", action="store_true", help="Save every epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0.98, help="Exponential LR decay.")
    parser.add_argument("--grad-clip", type=float, help="Optional gradient clipping value.")

    parser.add_argument("--data-path", type=Path, help="Raw audio data path.")
    parser.add_argument("--preprocessed-path", type=Path, default=Path("preprocessed"))
    parser.add_argument("--split-path", type=Path, default=Path("splits"))
    parser.add_argument("--h5-path", type=Path, default=Path("h5"))
    parser.add_argument(
        "--segment-length", type=int, default=16000, help="Segment length in samples."
    )
    parser.add_argument("--input-rate", type=int, default=44100, help="Input sample rate.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate.")
    parser.add_argument("--file-type", type=str, default="wav", help="Audio file extension.")
    parser.add_argument("--in-channel", type=int, default=2, help="Input audio channels.")
    parser.add_argument("--out-channel", type=int, default=1, help="Output audio channels.")
    parser.add_argument("--stems", type=int, nargs="+", default=[0, 1, 2], help="Stem ids.")
    parser.add_argument("--train-ratio", type=ratio, default=0.80, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=ratio, default=0.15, help="Validation split ratio.")
    parser.add_argument("--epoch-length", type=int, default=1000, help="Batches per epoch.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--data-aug", action="store_true", help="Enable data augmentation.")
    parser.add_argument("--aug-mag", type=float, default=0.5, help="Pitch augmentation magnitude.")
    parser.add_argument(
        "--h5-dataset-name", type=str, default="wav", help="Dataset name in H5 files."
    )

    parser.add_argument("--latent-d", type=int, default=128, help="Latent size.")
    parser.add_argument("--encoder-channels", type=int, default=128, help="Encoder channels.")
    parser.add_argument("--encoder-blocks", type=int, default=3, help="Encoder blocks.")
    parser.add_argument("--encoder-pool", type=int, default=800, help="Encoder pooling width.")
    parser.add_argument("--encoder-layers", type=int, default=10, help="Layers per encoder block.")
    parser.add_argument("--encoder-func", type=str, default="relu", choices=["relu", "tanh", "glu"])

    parser.add_argument("--blocks", type=int, default=3, help="WaveNet blocks.")
    parser.add_argument("--layers", type=int, default=10, help="WaveNet layers per block.")
    parser.add_argument("--kernel-size", type=int, default=2, help="WaveNet kernel size.")
    parser.add_argument("--skip-channels", type=int, default=256, help="WaveNet skip channels.")
    parser.add_argument(
        "--residual-channels", type=int, default=128, help="WaveNet residual channels."
    )

    parser.add_argument("--rank", default=0, type=int, help="Distributed process rank.")
    parser.add_argument("--world-size", default=1, type=int, help="Distributed world size.")
    parser.add_argument("--master", default="127.0.0.1:29500", help="Distributed TCP master.")
    parser.add_argument("--dist-backend", default="nccl", help="Distributed backend.")
    parser.add_argument("--local-rank", type=int, help="Ignored compatibility option.")

    parser.add_argument("--fft-size", type=int, default=2048, help="FFT size.")
    parser.add_argument("--windows-length", type=int, default=2047, help="Window length.")
    parser.add_argument("--hop-size", type=int, default=80, help="Hop size.")
    return parser
