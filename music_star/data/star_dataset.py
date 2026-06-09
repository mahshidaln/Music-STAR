"""Paired source/mixture HDF5 datasets used by the STAR encoder recipes."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from music_star.utils import mu_law, setup_logger

logger = setup_logger(__name__, "star_data.log")


class PairedH5Dataset(torch_data.Dataset):
    """Read aligned random slices from source and mixture HDF5 directories.

    Parameters
    ----------
    source_path : pathlib.Path
        Directory containing source HDF5 files.
    mix_path : pathlib.Path
        Directory containing aligned mixture or target HDF5 files.
    segment_length : int
        Number of samples per slice.
    dataset_name : str, optional
        HDF5 dataset name.
    epoch_length : int, optional
        Number of samples exposed per epoch.
    """

    def __init__(
        self,
        source_path: Path,
        mix_path: Path,
        segment_length: int,
        dataset_name: str = "wav",
        epoch_length: int = 10000,
    ):
        self.path = Path(source_path)
        self.mix_path = Path(mix_path)
        self.segment_length = segment_length
        self.epoch_length = epoch_length
        self.dataset_name = dataset_name

        self.file_paths = sorted(self.path.glob("**/*.h5"))
        if not self.file_paths:
            raise FileNotFoundError(f"No .h5 files found in {self.path}")

        logger.info("Paired dataset created. %s files. Path: %s", len(self.file_paths), self.path)

    def __getitem__(self, index):
        """Return one aligned source/mixture pair.

        Parameters
        ----------
        index : int
            Ignored dataset index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mu-law encoded source and paired mixture tensors.
        """

        del index
        source, mix = self.try_random_slice()
        if self.dataset_name == "wav":
            source = mu_law(source / 2**15)
            mix = mu_law(mix / 2**15)
        return torch.tensor(source), torch.tensor(mix)

    def __len__(self):
        """Return configured epoch length.

        Returns
        -------
        int
            Number of examples per epoch.
        """

        return self.epoch_length

    def try_random_slice(self):
        """Read an aligned random slice from matching HDF5 files.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Source and paired waveform slices.
        """

        source_file = random.choice(self.file_paths)
        mix_file = self._matching_mix_path(source_file)
        source = self._read_h5_file(source_file)
        mix = self._read_h5_file(mix_file)
        return self._read_aligned_slice(source, source_file, mix, mix_file)

    def _matching_mix_path(self, source_file: Path) -> Path:
        legacy_stem = source_file.stem[-3:]
        legacy_mix = self.mix_path / f"{legacy_stem}.0.h5"
        if legacy_mix.exists():
            return legacy_mix

        direct_mix = self.mix_path / source_file.name
        if direct_mix.exists():
            return direct_mix

        raise FileNotFoundError(f"No matching mixture file for {source_file} in {self.mix_path}")

    def _read_h5_file(self, h5file_path: Path):
        import h5py

        with h5py.File(h5file_path, "r") as h5file:
            if self.dataset_name not in h5file:
                raise KeyError(f"{h5file_path} does not contain dataset {self.dataset_name!r}")
            return np.asarray(h5file[self.dataset_name])

    def _read_aligned_slice(self, source, source_path: Path, mix, mix_path: Path):
        length = min(source.shape[0], mix.shape[0])
        if length < self.segment_length:
            raise ValueError(
                f"Aligned files {source_path} and {mix_path} are shorter than "
                f"segment_length={self.segment_length}"
            )

        start_time = random.randint(0, length - self.segment_length)
        source_data = source[start_time : start_time + self.segment_length]
        mix_data = mix[start_time : start_time + self.segment_length]
        return source_data.T, mix_data.T


class StarDataset:
    """Build train and validation loaders for a source domain and mixture path.

    Parameters
    ----------
    args
        Namespace with loader settings.
    source_path : pathlib.Path
        Source domain directory with ``train`` and ``val`` subdirectories.
    mix_path : pathlib.Path
        Paired mixture or target directory with matching splits.
    """

    def __init__(self, args, source_path: Path, mix_path: Path):
        dataset_name = getattr(args, "h5_dataset_name", "wav")
        segment_length = getattr(args, "segment_length", getattr(args, "seq_len", 16000))
        epoch_length = getattr(args, "epoch_length", getattr(args, "epoch_len", 10000))

        self.train_dataset = PairedH5Dataset(
            Path(source_path) / "train",
            Path(mix_path) / "train",
            segment_length,
            dataset_name=dataset_name,
            epoch_length=epoch_length,
        )
        self.train_loader = torch_data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.train_iter = iter(self.train_loader)

        self.valid_dataset = PairedH5Dataset(
            Path(source_path) / "val",
            Path(mix_path) / "val",
            segment_length,
            dataset_name=dataset_name,
            epoch_length=epoch_length,
        )
        self.valid_loader = torch_data.DataLoader(
            self.valid_dataset,
            batch_size=args.batch_size,
            num_workers=max(args.num_workers // 10, 1),
            pin_memory=torch.cuda.is_available(),
        )
        self.valid_iter = iter(self.valid_loader)


__all__ = ["PairedH5Dataset", "StarDataset"]
