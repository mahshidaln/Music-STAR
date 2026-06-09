import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if importlib.util.find_spec("h5py") is None:
    raise unittest.SkipTest("h5py is not installed")
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import h5py

from music_star.data.datasets import H5Dataset


class H5DatasetTest(unittest.TestCase):
    def test_h5_dataset_returns_mu_law_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            h5_path = train_dir / "example.h5"
            with h5py.File(h5_path, "w") as h5file:
                h5file.create_dataset("wav", data=np.arange(64, dtype=np.float32))

            args = SimpleNamespace(
                segment_length=16,
                epoch_length=3,
                h5_dataset_name="wav",
            )

            dataset = H5Dataset(args, train_dir)
            x, y = dataset[0]

            self.assertEqual(len(dataset), 3)
            self.assertEqual(tuple(x.shape), (16,))
            self.assertEqual(tuple(y.shape), (16,))
            self.assertGreaterEqual(x.min(), 0)
            self.assertLessEqual(x.max(), 255)
