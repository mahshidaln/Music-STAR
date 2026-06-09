"""Shared trainer utilities and abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_value_

from music_star.data.star_dataset import PairedH5Dataset
from music_star.models import Encoder, LegacyEncoder
from music_star.utils import train_logger


def device_from_args(args: Any) -> torch.device:
    """Return the configured training device.

    Parameters
    ----------
    args
        Namespace that may define ``device``.

    Returns
    -------
    torch.device
        Requested device, or CUDA when available, otherwise CPU.
    """

    requested = getattr(args, "device", None)
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to a training device.

    Parameters
    ----------
    value : torch.Tensor
        Tensor to move.
    device : torch.device
        Destination device.

    Returns
    -------
    torch.Tensor
        Tensor on the destination device.
    """

    return value.to(device=device, non_blocking=device.type == "cuda")


def data_paths(args: Any) -> list[Path]:
    """Return configured training data paths.

    Parameters
    ----------
    args
        Namespace with ``data`` or ``data_paths``.

    Returns
    -------
    list[pathlib.Path]
        Data paths.

    Raises
    ------
    ValueError
        If neither ``data`` nor ``data_paths`` is defined.
    """

    data = getattr(args, "data", None) or getattr(args, "data_paths", None)
    if data is None:
        raise ValueError("Training config must provide data or data_paths")
    if isinstance(data, (str, Path)):
        return [Path(data)]
    return [Path(path) for path in data]


def experiment_path(args: Any) -> Path:
    """Return the experiment checkpoint directory.

    Parameters
    ----------
    args
        Namespace with checkpoint and experiment-name fields.

    Returns
    -------
    pathlib.Path
        Experiment directory.
    """

    checkpoint_root = Path(
        getattr(args, "checkpoint_root", getattr(args, "checkpoint", "checkpoints"))
    )
    exp_name = getattr(args, "exp_name", getattr(args, "expName", "music-star"))
    return checkpoint_root / exp_name


def epochs(args: Any) -> int:
    """Return the number of training epochs.

    Parameters
    ----------
    args
        Namespace that may define ``epochs``.

    Returns
    -------
    int
        Number of epochs.
    """

    return int(getattr(args, "epochs", 100))


def epoch_length(args: Any) -> int:
    """Return the number of batches per epoch.

    Parameters
    ----------
    args
        Namespace with ``epoch_length`` or legacy ``epoch_len``.

    Returns
    -------
    int
        Number of batches per epoch.
    """

    return int(getattr(args, "epoch_length", getattr(args, "epoch_len", 10000)))


def segment_length(args: Any) -> int:
    """Return the audio segment length.

    Parameters
    ----------
    args
        Namespace with ``segment_length`` or legacy ``seq_len``.

    Returns
    -------
    int
        Segment length in samples.
    """

    return int(getattr(args, "segment_length", getattr(args, "seq_len", 16000)))


def paired_loaders(args: Any, source_path: Path, target_path: Path):
    """Build aligned train and validation iterators.

    Parameters
    ----------
    args
        Namespace with loader options.
    source_path : pathlib.Path
        Source split root.
    target_path : pathlib.Path
        Target split root.

    Returns
    -------
    tuple[Iterator, Iterator]
        Train and validation iterators.
    """

    dataset_name = getattr(args, "h5_dataset_name", "wav")
    train_dataset = PairedH5Dataset(
        source_path / "train",
        target_path / "train",
        segment_length(args),
        dataset_name=dataset_name,
        epoch_length=epoch_length(args),
    )
    valid_dataset = PairedH5Dataset(
        source_path / "val",
        target_path / "val",
        segment_length(args),
        dataset_name=dataset_name,
        epoch_length=epoch_length(args),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=max(args.num_workers // 10, 1),
        pin_memory=torch.cuda.is_available(),
    )
    return iter(train_loader), iter(valid_loader)


def legacy_or_standard_encoder(args: Any) -> torch.nn.Module:
    """Build a standard or legacy encoder from args.

    Parameters
    ----------
    args
        Namespace with encoder fields and optional ``legacy_encoder``.

    Returns
    -------
    torch.nn.Module
        Encoder instance.
    """

    return LegacyEncoder(args) if getattr(args, "legacy_encoder", True) else Encoder(args)


class BaseTrainer(ABC):
    """Shared trainer plumbing for Music-STAR recipes.

    Parameters
    ----------
    args
        Argparse-like namespace with training settings.
    """

    checkpoint_keys: tuple[str, ...] = ()

    def __init__(self, args: Any):
        self.args = args
        self.device = device_from_args(args)
        self.exp_path = experiment_path(args)
        self.exp_path.mkdir(parents=True, exist_ok=True)
        self.logger = train_logger(args, self.exp_path)
        self.start_epoch = 0

        torch.manual_seed(int(getattr(args, "seed", 1)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(getattr(args, "seed", 1)))

    def _save_args(self, epoch: int) -> None:
        """Save training arguments and epoch.

        Parameters
        ----------
        epoch : int
            Epoch number to save.
        """

        torch.save([self.args, epoch], self.exp_path / "args.pth")

    def _save_model(self, filename: str, payload: dict[str, Any]) -> None:
        """Save a checkpoint payload.

        Parameters
        ----------
        filename : str
            Checkpoint filename.
        payload : dict[str, Any]
            State dictionary payload.
        """

        save_path = self.exp_path / filename
        torch.save(payload, save_path)
        self.logger.debug("Saved model to %s", save_path)

    def _clip(self, parameters) -> None:
        """Clip gradients when ``grad_clip`` is configured.

        Parameters
        ----------
        parameters
            Iterable of parameters to clip.
        """

        grad_clip = getattr(self.args, "grad_clip", None)
        if grad_clip is not None:
            clip_grad_value_(parameters, float(grad_clip))

    @abstractmethod
    def train(self) -> None:
        """Run the training loop."""
        ...


__all__ = [
    "BaseTrainer",
    "data_paths",
    "epoch_length",
    "epochs",
    "legacy_or_standard_encoder",
    "paired_loaders",
    "to_device",
]
