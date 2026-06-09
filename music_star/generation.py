"""Audio generation utilities for trained Music-STAR checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from music_star.checkpoints import DEFAULT_MODEL_FILE, load_decoder, load_encoder
from music_star.models import WaveNet, WavenetGenerator
from music_star.smoke import decoder_state_keys, route_condition
from music_star.utils import inv_mu_law, mu_law, save_audio


@dataclass
class GenerationConfig:
    """Configuration for checkpoint-based audio generation.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Directory containing ``args.pth`` and model checkpoint files.
    files : list[pathlib.Path]
        Input WAV/HDF5 files or a single directory to scan.
    output : pathlib.Path
        Directory where generated WAV files are written.
    model_file : str, optional
        Checkpoint file name.
    decoder_state_key : str, optional
        Decoder state key inside the checkpoint.
    rate : int, optional
        Audio sample rate.
    batch_size : int, optional
        Generation batch size.
    split_size : int, optional
        Latent split size for autoregressive generation.
    sample_len : int | None, optional
        Optional input length limit in samples.
    method : {"sample", "max"}, optional
        WaveNet sampling strategy.
    device : str | None, optional
        Torch device override.
    """

    checkpoint_dir: Path
    files: list[Path]
    output: Path
    model_file: str = DEFAULT_MODEL_FILE
    decoder_state_key: str = "decoder_state"
    rate: int = 16000
    batch_size: int = 1
    split_size: int = 20
    sample_len: int | None = None
    method: str = "sample"
    device: str | None = None


def _device(name: str | None = None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_audio(path: Path, rate: int) -> torch.Tensor:
    if path.suffix == ".wav":
        import librosa

        data, _ = librosa.load(path, sr=rate, mono=True)
        encoded = mu_law(data)
    elif path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as h5file:
            encoded = mu_law(h5file["wav"][:] / 2**15)
    else:
        raise ValueError(f"Unsupported input file type: {path}")
    return torch.tensor(encoded).unsqueeze(0).float()


def _collect_files(files: list[Path]) -> list[Path]:
    if len(files) == 1 and files[0].is_dir():
        return sorted([*files[0].glob("**/*.wav"), *files[0].glob("**/*.h5")])
    return files


def generate_files(config: GenerationConfig) -> list[Path]:
    """Generate WAV files from a checkpoint and input WAV/H5 files.

    This uses the pure PyTorch autoregressive generator. The historical
    ``nv-wavenet`` path is intentionally left out because it required a custom
    CUDA extension and machine-specific build.
    """

    device = _device(config.device)
    loaded_encoder = load_encoder(config.checkpoint_dir, model_file=config.model_file)
    loaded_decoder = load_decoder(
        config.checkpoint_dir,
        model_file=config.model_file,
        state_key=config.decoder_state_key,
    )
    encoder = loaded_encoder.model.to(device).eval()
    decoder = cast(WaveNet, loaded_decoder.model.to(device).eval())
    generator = WavenetGenerator(decoder, batch_size=config.batch_size, wav_freq=config.rate)

    input_files = _collect_files(config.files)
    if not input_files:
        raise FileNotFoundError("No input WAV/H5 files found for generation")

    xs = []
    for file_path in input_files:
        data = _read_audio(file_path, config.rate)
        if config.sample_len is not None:
            data = data[:, : config.sample_len]
        xs.append(data.to(device))
    stacked = torch.stack(xs).contiguous()

    outputs: list[Path] = []
    config.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        encodings = []
        for batch in torch.split(stacked, config.batch_size):
            encodings.append(encoder(batch))
        z = torch.cat(encodings, dim=0)
        condition_channels = int(
            loaded_decoder.checkpoint[config.decoder_state_key]["layers.0.condition.weight"].shape[
                1
            ]
        )
        z, _ = route_condition(
            z,
            condition_channels,
            config.decoder_state_key,
            decoder_state_keys(loaded_decoder.checkpoint),
        )

        generated = []
        for z_batch in torch.split(z, config.batch_size):
            pieces = []
            generator.reset()
            for cond in torch.split(z_batch, config.split_size, dim=-1):
                pieces.append(generator.generate(cond, method=config.method).cpu())
            generated.append(torch.cat(pieces, dim=-1))
        generated_audio = torch.cat(generated, dim=0)

    suffix = config.decoder_state_key.removesuffix("_state").replace("decoder", "d")
    for sample, source_path in zip(generated_audio, input_files):
        wav = inv_mu_law(sample.cpu().numpy()).squeeze()
        output_path = config.output / f"{source_path.stem}_{suffix}.wav"
        save_audio(wav, output_path, rate=config.rate)
        outputs.append(output_path)

    return outputs


__all__ = ["GenerationConfig", "generate_files"]
