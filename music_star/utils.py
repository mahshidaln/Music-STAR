"""Shared utilities for preprocessing, training, and audio export."""

from __future__ import annotations

import errno
import logging
import random
import shutil
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np


def setup_logger(logger_name: str, filename: str | Path | None = None) -> logging.Logger:
    """Create an idempotent logger.

    Parameters
    ----------
    logger_name:
        Logger namespace.
    filename:
        Optional file path. When omitted, only the logger object is configured.
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if filename is not None:
        filepath = Path(filename)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == filepath.resolve()
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(filepath)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def train_logger(args, path: Path) -> logging.Logger:
    """Configure a file-backed root logger for training runs."""

    path.mkdir(parents=True, exist_ok=True)
    rank = getattr(args, "rank", 0)
    filepath = path / f"main_{rank}.log"

    if rank != 0:
        sys.stdout = open(path / f"stdout_{rank}.log", "w")
        sys.stderr = open(path / f"stderr_{rank}.log", "w")

    class TrainLogFormatter(logging.Formatter):
        """Formatter that prefixes records with wall-clock elapsed time."""

        def __init__(self):
            super().__init__()
            self.start_time = time.time()

        def format(self, record):
            """Format a log record.

            Parameters
            ----------
            record : logging.LogRecord
                Log record to format.

            Returns
            -------
            str
                Formatted log line.
            """

            elapsed_seconds = round(record.created - self.start_time)
            prefix = (
                f"{time.strftime('%x %X')} - {timedelta(seconds=elapsed_seconds)}"
                f" - {record.levelname}"
            )
            message = record.getMessage().replace("\n", "\n" + " " * (len(prefix) + 3))
            return f"{prefix} - {message}"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = TrainLogFormatter()
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(args)
    return logger


def mu_law(x, mu: int = 255):
    """Mu-law encode a signal in [-1, 1] to integer values in [0, mu]."""

    x = np.clip(x, -1, 1)
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((x_mu + 1) / 2 * mu).astype("int16")


def inv_mu_law(x, mu: float = 255.0):
    """Invert :func:`mu_law` back to approximately [-1, 1]."""

    x = np.array(x).astype(np.float32)
    y = 2.0 * (x - (mu + 1.0) / 2.0) / (mu + 1.0)
    return np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** np.abs(y) - 1.0)


def copy_files(files, from_path: Path, to_path: Path):
    """Copy files while preserving paths relative to ``from_path``."""

    from_path = Path(from_path)
    to_path = Path(to_path)
    for file_path in files:
        file_path = Path(file_path)
        out_file_path = to_path / file_path.relative_to(from_path)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, out_file_path)


def save_audio(x, path: Path, rate: int):
    """Write a WAV file, creating parent directories when needed."""

    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, x)


def save_wav_image(wav, path: Path):
    """Save a simple waveform plot."""

    from matplotlib import pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.plot(wav)
    plt.savefig(path)
    plt.close()


def free_port(host: str = "", low: int = 20000, high: int = 40000) -> int:
    """Return an available TCP port in the configured range."""

    while True:
        port = random.randint(low, high)
        with socket.socket() as sock:
            try:
                sock.bind((host, port))
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    continue
                raise
            return port


class LossManager:
    """Track scalar losses over an epoch."""

    def __init__(self, name: str):
        self.name = name
        self.losses: list[float] = []

    def reset(self):
        """Clear all stored loss values."""

        self.losses = []

    def add(self, val: float):
        """Add one scalar loss value.

        Parameters
        ----------
        val : float
            Loss value to record.
        """

        self.losses.append(float(val))

    def epoch_mean(self) -> float:
        """Return the mean loss for the current epoch.

        Returns
        -------
        float
            Mean loss, or ``0.0`` when no values are stored.
        """

        if not self.losses:
            return 0.0
        return float(np.mean(self.losses))

    summarize_epoch = epoch_mean

    def losses_sum(self) -> float:
        """Return the sum of stored losses.

        Returns
        -------
        float
            Sum of all recorded losses.
        """

        return float(sum(self.losses))
