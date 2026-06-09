"""Translation and generation entry points for Music-STAR checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from music_star.generation import GenerationConfig, generate_files


@dataclass
class TranslationResult:
    """Generated translation output paths.

    Parameters
    ----------
    outputs : list[pathlib.Path]
        WAV files produced by a translation variation.
    """

    outputs: list[Path]


class WaveNetTranslate:
    """Generate audio using an encoder and one WaveNet decoder state."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    def translate(self) -> TranslationResult:
        """Generate audio with one decoder.

        Returns
        -------
        TranslationResult
            Paths to generated WAV files.
        """

        return TranslationResult(outputs=generate_files(self.config))


class DoubleDecoderTranslate:
    """Generate audio from both decoder states stored in a double-decoder checkpoint."""

    decoder_state_keys = ("decoder_state", "decoder2_state")

    def __init__(self, config: GenerationConfig):
        self.config = config

    def translate(self) -> TranslationResult:
        """Generate audio with both decoder states.

        Returns
        -------
        TranslationResult
            Paths to generated WAV files from both decoders.
        """

        outputs: list[Path] = []
        for state_key in self.decoder_state_keys:
            config = GenerationConfig(
                checkpoint_dir=self.config.checkpoint_dir,
                files=self.config.files,
                output=self.config.output,
                model_file=self.config.model_file,
                decoder_state_key=state_key,
                rate=self.config.rate,
                batch_size=self.config.batch_size,
                split_size=self.config.split_size,
                sample_len=self.config.sample_len,
                method=self.config.method,
                device=self.config.device,
            )
            outputs.extend(generate_files(config))
        return TranslationResult(outputs=outputs)


class SingleInstrumentTranslate(WaveNetTranslate):
    """Single-instrument baseline for already separated stems."""


class EmbeddingSupervisedTranslate(WaveNetTranslate):
    """Embedding-supervised encoder plus target WaveNet decoder."""


class MixtureSupervisedTranslate(WaveNetTranslate):
    """Music-STAR mixture-supervised generation."""


class StemSupervisedTranslate(DoubleDecoderTranslate):
    """Music-STAR stem-supervised generation with two target-stem decoders."""


class UniverseTranslate(WaveNetTranslate):
    """Backward-compatible translation name for universal checkpoints."""


class MusicStarTranslate(WaveNetTranslate):
    """Backward-compatible translation name for Music-STAR checkpoints."""


def translate_checkpoint(config: GenerationConfig, variation: str = "wavenet") -> TranslationResult:
    """Run one generation variation for a checkpoint."""

    if variation in {
        "wavenet",
        "single_instrument",
        "embedding_supervised",
        "music_star_mixture_supervised",
    }:
        return WaveNetTranslate(config).translate()
    if variation in {"double_decoder", "music_star_stem_supervised"}:
        return DoubleDecoderTranslate(config).translate()
    raise ValueError(f"Unknown translation variation: {variation!r}")


def main(argv=None) -> None:
    """Run the checkpoint generation command.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments. When ``None``, arguments are read from
        ``sys.argv`` by ``argparse``.
    """

    parser = argparse.ArgumentParser(description="Generate audio from Music-STAR checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory.")
    parser.add_argument("--model-file", default="bestmodel_0.pth", help="Checkpoint file name.")
    parser.add_argument(
        "--variation",
        choices=[
            "wavenet",
            "single_instrument",
            "embedding_supervised",
            "music_star_mixture_supervised",
            "music_star_stem_supervised",
            "double_decoder",
        ],
        default="wavenet",
        help="Generation variation to run.",
    )
    parser.add_argument(
        "--decoder-state-key",
        default="decoder_state",
        help="Decoder state key for the wavenet variation.",
    )
    parser.add_argument(
        "--files", type=Path, nargs="+", required=True, help="Input WAV/H5 files or directory."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--rate", type=int, default=16000, help="Output sample rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument("--split-size", type=int, default=20, help="Latent split size.")
    parser.add_argument("--sample-len", type=int, help="Optional input sample length.")
    parser.add_argument("--method", choices=["sample", "max"], default="sample")
    parser.add_argument("--device", help="Override device, for example cpu or cuda.")
    parsed = parser.parse_args(argv)

    result = translate_checkpoint(
        GenerationConfig(
            checkpoint_dir=parsed.checkpoint_dir,
            files=parsed.files,
            output=parsed.output,
            model_file=parsed.model_file,
            decoder_state_key=parsed.decoder_state_key,
            rate=parsed.rate,
            batch_size=parsed.batch_size,
            split_size=parsed.split_size,
            sample_len=parsed.sample_len,
            method=parsed.method,
            device=parsed.device,
        ),
        variation=parsed.variation,
    )
    for output in result.outputs:
        print(output)


__all__ = [
    "DoubleDecoderTranslate",
    "EmbeddingSupervisedTranslate",
    "MixtureSupervisedTranslate",
    "MusicStarTranslate",
    "SingleInstrumentTranslate",
    "StemSupervisedTranslate",
    "TranslationResult",
    "UniverseTranslate",
    "WaveNetTranslate",
    "main",
    "translate_checkpoint",
]
