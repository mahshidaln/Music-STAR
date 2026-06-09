import unittest
from pathlib import Path

from music_star.parser import get_parser


class ParserTest(unittest.TestCase):
    def test_parser_accepts_preprocess_defaults(self):
        args = get_parser().parse_args(
            [
                "--operation-mode",
                "preprocess",
                "--data-path",
                "data/raw",
                "--train-ratio",
                "80",
                "--val-ratio",
                "15",
            ]
        )

        self.assertEqual(args.operation_mode, ["preprocess"])
        self.assertEqual(args.data_path, Path("data/raw"))
        self.assertEqual(args.train_ratio, 0.8)
        self.assertEqual(args.val_ratio, 0.15)
        self.assertEqual(args.stems, [0, 1, 2])
