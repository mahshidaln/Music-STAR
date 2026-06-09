"""Smoke tests for loading and exercising Music-STAR checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from music_star.checkpoints import DEFAULT_MODEL_FILE, load_decoder, load_encoder
from music_star.models.universal import cross_entropy_loss


@dataclass
class DecoderSmokeResult:
    """Smoke result for one decoder state.

    Parameters
    ----------
    state_key : str
        Decoder state key inside the checkpoint.
    inference_shape : tuple[int, ...]
        Shape returned by the decoder inference forward pass.
    train_loss : float
        Synthetic training loss after one forward pass.
    notes : list[str]
        Compatibility notes gathered while loading or routing latents.
    """

    state_key: str
    inference_shape: tuple[int, ...]
    train_loss: float
    notes: list[str] = field(default_factory=list)


@dataclass
class CheckpointSmokeResult:
    """Smoke result for one checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Checkpoint directory that was tested.
    encoder_type : str
        Encoder class name.
    encoder_shape : tuple[int, ...]
        Shape returned by the encoder forward pass.
    decoder_results : list[DecoderSmokeResult]
        Per-decoder smoke results.
    """

    checkpoint_dir: Path
    encoder_type: str
    encoder_shape: tuple[int, ...]
    decoder_results: list[DecoderSmokeResult]


def decoder_state_keys(checkpoint: dict) -> list[str]:
    """Return decoder state keys from a checkpoint dictionary.

    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint payload.

    Returns
    -------
    list[str]
        Decoder state keys sorted by their checkpoint order.
    """

    keys = ["decoder_state"]
    if "decoder2_state" in checkpoint:
        keys.append("decoder2_state")
    return [key for key in keys if key in checkpoint]


def route_condition(
    encoded: torch.Tensor,
    decoder_condition_channels: int,
    state_key: str,
    available_decoder_keys: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Route encoder latents to a decoder with matching condition channels.

    Parameters
    ----------
    encoded : torch.Tensor
        Encoder output with shape ``(batch, channels, time)``.
    decoder_condition_channels : int
        Number of channels expected by the decoder condition projection.
    state_key : str
        Decoder state key being routed.
    available_decoder_keys : list[str]
        Decoder keys available in the checkpoint.

    Returns
    -------
    tuple[torch.Tensor, list[str]]
        Routed latent tensor and compatibility notes.

    Raises
    ------
    ValueError
        If the latent channels cannot be routed safely.
    """

    encoder_channels = int(encoded.shape[1])
    notes: list[str] = []
    if encoder_channels == decoder_condition_channels:
        return encoded, notes

    if encoder_channels == decoder_condition_channels * len(available_decoder_keys):
        decoder_index = available_decoder_keys.index(state_key)
        start = decoder_index * decoder_condition_channels
        end = start + decoder_condition_channels
        notes.append(
            f"routed encoder channels {start}:{end} to {state_key} "
            f"from {encoder_channels}-channel latent"
        )
        return encoded[:, start:end, :], notes

    raise ValueError(
        f"Encoder produced {encoder_channels} channels but {state_key} expects "
        f"{decoder_condition_channels}; cannot infer routing."
    )


def smoke_test_checkpoint(
    checkpoint_dir: str | Path,
    model_file: str = DEFAULT_MODEL_FILE,
    sample_length: int | None = None,
) -> CheckpointSmokeResult:
    """Load a checkpoint and run tiny inference/training-style passes.

    Parameters
    ----------
    checkpoint_dir : str | pathlib.Path
        Directory containing ``args.pth`` and checkpoint files.
    model_file : str, optional
        Checkpoint filename.
    sample_length : int | None, optional
        Synthetic waveform length. Defaults to ``encoder_pool`` from args.

    Returns
    -------
    CheckpointSmokeResult
        Structured smoke-test result.
    """

    checkpoint_path = Path(checkpoint_dir)
    loaded_encoder = load_encoder(checkpoint_path, model_file=model_file)
    checkpoint = loaded_encoder.checkpoint
    args = loaded_encoder.args
    length = int(sample_length or getattr(args, "encoder_pool", 800))
    x = torch.full((1, length), 128.0)

    encoder = loaded_encoder.model
    encoder.train()
    with torch.no_grad():
        encoded = encoder(x)
    encoder_shape = tuple(int(dim) for dim in encoded.shape)

    available_decoder_keys = decoder_state_keys(checkpoint)
    decoder_results: list[DecoderSmokeResult] = []
    for state_key in available_decoder_keys:
        loaded_decoder = load_decoder(checkpoint_path, model_file=model_file, state_key=state_key)
        decoder = loaded_decoder.model
        decoder.train()
        condition_channels = int(
            loaded_decoder.checkpoint[state_key]["layers.0.condition.weight"].shape[1]
        )
        train_encoded = encoder(x)
        condition, route_notes = route_condition(
            train_encoded,
            condition_channels,
            state_key,
            available_decoder_keys,
        )
        logits = decoder(x, condition)
        loss = cross_entropy_loss(logits, x).mean()
        decoder.zero_grad(set_to_none=True)
        encoder.zero_grad(set_to_none=True)
        loss.backward()
        decoder_results.append(
            DecoderSmokeResult(
                state_key=state_key,
                inference_shape=tuple(int(dim) for dim in logits.shape),
                train_loss=float(loss.detach().cpu()),
                notes=[*loaded_encoder.notes, *loaded_decoder.notes, *route_notes],
            )
        )

    return CheckpointSmokeResult(
        checkpoint_dir=checkpoint_path,
        encoder_type=type(encoder).__name__,
        encoder_shape=encoder_shape,
        decoder_results=decoder_results,
    )


def smoke_test_checkpoint_root(
    checkpoint_root: str | Path,
    model_file: str = DEFAULT_MODEL_FILE,
    sample_length: int | None = None,
) -> list[CheckpointSmokeResult]:
    """Run checkpoint smoke tests for all experiment directories in a root.

    Parameters
    ----------
    checkpoint_root : str | pathlib.Path
        Directory containing experiment checkpoint directories.
    model_file : str, optional
        Checkpoint filename.
    sample_length : int | None, optional
        Synthetic waveform length passed to each checkpoint smoke test.

    Returns
    -------
    list[CheckpointSmokeResult]
        Smoke results sorted by checkpoint directory name.
    """

    root = Path(checkpoint_root)
    return [
        smoke_test_checkpoint(path, model_file=model_file, sample_length=sample_length)
        for path in sorted(child for child in root.iterdir() if child.is_dir())
    ]


def format_smoke_results(results: list[CheckpointSmokeResult]) -> str:
    """Format smoke results for command-line output.

    Parameters
    ----------
    results : list[CheckpointSmokeResult]
        Smoke results to format.

    Returns
    -------
    str
        Human-readable report.
    """

    lines: list[str] = []
    for result in results:
        lines.append(
            f"{result.checkpoint_dir.name}: encoder={result.encoder_type} "
            f"shape={result.encoder_shape}"
        )
        for decoder in result.decoder_results:
            note = f" notes={decoder.notes}" if decoder.notes else ""
            lines.append(
                f"  {decoder.state_key}: logits={decoder.inference_shape} "
                f"loss={decoder.train_loss:.4f}{note}"
            )
    return "\n".join(lines)


def main(argv=None) -> None:
    """Run checkpoint smoke tests from the command line.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments. When ``None``, ``argparse`` reads from
        ``sys.argv``.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test Music-STAR checkpoints.")
    parser.add_argument("checkpoint_root", type=Path, help="Directory of checkpoint folders.")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE, help="Checkpoint filename.")
    parser.add_argument(
        "--sample-length",
        type=int,
        default=None,
        help="Synthetic waveform length. Defaults to the checkpoint encoder_pool.",
    )
    parsed = parser.parse_args(argv)

    print(
        format_smoke_results(
            smoke_test_checkpoint_root(
                parsed.checkpoint_root,
                parsed.model_file,
                sample_length=parsed.sample_length,
            )
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CheckpointSmokeResult",
    "DecoderSmokeResult",
    "decoder_state_keys",
    "format_smoke_results",
    "main",
    "route_condition",
    "smoke_test_checkpoint",
    "smoke_test_checkpoint_root",
]
