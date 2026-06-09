import importlib.util
import unittest
from types import SimpleNamespace

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed")

import torch

from music_star.models.wavenet import WaveNet


class WaveNetTest(unittest.TestCase):
    def test_wavenet_forward_shape(self):
        args = SimpleNamespace(
            blocks=1,
            layers=2,
            kernel_size=2,
            skip_channels=8,
            residual_channels=4,
            latent_d=3,
        )
        model = WaveNet(args)
        x = torch.randint(0, 256, (2, 16))
        c = torch.randn(2, 3, 1)

        output = model(x, c)

        self.assertEqual(tuple(output.shape), (2, 256, 16))
