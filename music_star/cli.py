"""Command-line entry points for config-driven training and generation."""

from __future__ import annotations

import argparse

from music_star.train import list_training_configs


def train_main(argv=None) -> None:
    """Run the training CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments forwarded to :func:`music_star.train.main`.
    """

    from music_star.train import main

    main(argv)


def generate_main(argv=None) -> None:
    """Run the generation CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments forwarded to :func:`music_star.translate.main`.
    """

    from music_star.translate import main

    main(argv)


def configs_main(argv=None) -> None:
    """List bundled training procedure configs.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments for the listing command.
    """

    parser = argparse.ArgumentParser(description="List bundled Music-STAR training configs.")
    parser.parse_args(argv)
    for name in list_training_configs():
        print(name)


__all__ = ["configs_main", "generate_main", "train_main"]
