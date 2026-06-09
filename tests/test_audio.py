import tempfile
import unittest
from pathlib import Path

from music_star.data.audio import Audio


class AudioTest(unittest.TestCase):
    def test_track_names_are_unique_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ["002.1.wav", "001.0.wav", "001.1.wav"]:
                (root / name).touch()

            self.assertEqual(Audio.track_names(root), ["001", "002"])
