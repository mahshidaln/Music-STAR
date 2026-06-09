import unittest

import numpy as np

from music_star.utils import LossManager, inv_mu_law, mu_law


class UtilsTest(unittest.TestCase):
    def test_mu_law_round_trip_is_bounded(self):
        signal = np.linspace(-1, 1, 17)
        encoded = mu_law(signal)
        decoded = inv_mu_law(encoded)

        self.assertGreaterEqual(encoded.min(), 0)
        self.assertLessEqual(encoded.max(), 255)
        self.assertGreaterEqual(decoded.min(), -1)
        self.assertLessEqual(decoded.max(), 1)

    def test_loss_manager_tracks_epoch_mean(self):
        losses = LossManager("test")
        losses.add(1)
        losses.add(3)

        self.assertEqual(losses.epoch_mean(), 2.0)
        self.assertEqual(losses.losses_sum(), 4.0)

        losses.reset()
        self.assertEqual(losses.epoch_mean(), 0.0)
