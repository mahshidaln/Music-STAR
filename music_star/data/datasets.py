"""Dataset helpers for training from preprocessed HDF5 audio."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from music_star.data.audio import PitchAugmentation
from music_star.utils import mu_law, setup_logger

logger = setup_logger(__name__, "log_data.log")


class H5Dataset(torch_data.Dataset):
    """Read random fixed-length slices from a directory of HDF5 files.

    Parameters
    ----------
    args
        Namespace with ``segment_length``, ``epoch_length``, and optionally
        ``h5_dataset_name``.
    path : pathlib.Path
        Directory containing HDF5 files.
    augmentation : callable | None, optional
        Optional waveform augmentation function.
    """

    def __init__(self, args, path: Path, augmentation=None):
        self.path = Path(path)
        self.segment_length = args.segment_length
        self.epoch_length = args.epoch_length
        self.dataset_name = getattr(args, "h5_dataset_name", "wav")
        self.augmentation = augmentation
        self.file_paths = sorted(self.path.glob("**/*.h5"))

        if not self.file_paths:
            raise FileNotFoundError(f"No .h5 files found in {self.path}")

        logger.info(
            "Dataset created. %s files, augmentation: %s. Path: %s",
            len(self.file_paths),
            self.augmentation is not None,
            self.path,
        )

    def read_data(self, dataset, path: Path):
        """Read one fixed-length random slice from an H5 dataset.

        Parameters
        ----------
        dataset
            Open HDF5 dataset object.
        path : pathlib.Path
            File path used for error reporting.

        Returns
        -------
        numpy.ndarray
            Sliced waveform data.
        """

        length = dataset.shape[0]
        if length < self.segment_length:
            raise ValueError(
                f"{path} has {length} samples, shorter than segment_length={self.segment_length}"
            )

        start_time = random.randint(0, length - self.segment_length)
        wav = dataset[start_time : start_time + self.segment_length]
        assert wav.shape[0] == self.segment_length
        return np.asarray(wav).T

    def read_h5_file(self, h5file_path: Path):
        """Read one random slice from an HDF5 file.

        Parameters
        ----------
        h5file_path : pathlib.Path
            HDF5 file path.

        Returns
        -------
        numpy.ndarray
            Sliced waveform data.
        """

        import h5py

        with h5py.File(h5file_path, "r") as h5file:
            if self.dataset_name not in h5file:
                raise KeyError(f"{h5file_path} does not contain dataset {self.dataset_name!r}")
            return self.read_data(h5file[self.dataset_name], h5file_path)

    def _random_file(self):
        return random.choice(self.file_paths)

    def get_random_slice(self):
        """Read one random slice from a random file.

        Returns
        -------
        numpy.ndarray
            Sliced waveform data.
        """

        return self.read_h5_file(self._random_file())

    def __getitem__(self, index):
        """Return one original/augmented waveform pair.

        Parameters
        ----------
        index : int
            Ignored dataset index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mu-law encoded original and augmented waveform tensors.
        """

        del index
        wav = self.get_random_slice()
        if self.augmentation:
            augmented = self.augmentation(wav)
        else:
            augmented = wav
        return torch.tensor(mu_law(wav / 2**15)), torch.tensor(mu_law(augmented / 2**15))

    def __len__(self):
        """Return configured epoch length.

        Returns
        -------
        int
            Number of examples per epoch.
        """

        return self.epoch_length


class Dataset:
    """Build train and validation loaders for a single domain path.

    Parameters
    ----------
    args
        Namespace with data-loader and augmentation options.
    domain_path : pathlib.Path
        Directory containing ``train`` and ``val`` HDF5 subdirectories.
    """

    def __init__(self, args, domain_path: Path):
        augmentation = PitchAugmentation(args) if args.data_aug else None

        self.train_dataset = H5Dataset(args, Path(domain_path) / "train", augmentation=augmentation)
        self.train_loader = torch_data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.train_iter = iter(self.train_loader)

        self.valid_dataset = H5Dataset(args, Path(domain_path) / "val", augmentation=augmentation)
        self.valid_loader = torch_data.DataLoader(
            self.valid_dataset,
            batch_size=args.batch_size,
            num_workers=max(args.num_workers // 10, 0),
            pin_memory=torch.cuda.is_available(),
        )
        self.valid_iter = iter(self.valid_loader)
