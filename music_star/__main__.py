"""Executable module for ``python -m music_star``."""

from __future__ import annotations

import random

from music_star.parser import get_parser


def _run_preprocess(args):
    """Run audio preprocessing from parsed CLI arguments.

    Parameters
    ----------
    args
        Parsed command-line namespace.

    Raises
    ------
    ValueError
        If ``--data-path`` is missing.
    """

    from music_star.data.audio import Audio

    if args.data_path is None:
        raise ValueError("--data-path is required for preprocessing")

    audio = Audio(
        data_path=args.data_path,
        preprocessed_path=args.preprocessed_path,
        h5_path=args.h5_path,
        split_path=args.split_path,
        stems=args.stems,
        in_sr=args.input_rate,
        out_sr=args.sample_rate,
        in_channel=args.in_channel,
        out_channel=args.out_channel,
        file_type=args.file_type or "wav",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    audio.preprocess_audio()


def _run_train(args):
    """Run the training workflow from parsed CLI arguments.

    Parameters
    ----------
    args
        Parsed command-line namespace.
    """

    import torch
    from torch import distributed

    from music_star.train import StarLatentTrainer

    if args.world_size > 1:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        distributed.init_process_group(
            backend=args.dist_backend,
            init_method="tcp://" + args.master,
            rank=args.rank,
            world_size=args.world_size,
        )

    trainer = StarLatentTrainer(args)
    trainer.train()


def main(argv=None):
    """Execute the ``python -m music_star`` command.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments. When ``None``, arguments are read from
        ``sys.argv`` by ``argparse``.
    """

    parser = get_parser()
    args = parser.parse_args(argv)
    random.seed(args.seed)

    if "preprocess" in args.operation_mode:
        _run_preprocess(args)
    if "train" in args.operation_mode:
        _run_train(args)


if __name__ == "__main__":
    main()
