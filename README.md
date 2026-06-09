# Music-STAR: a Style Translation system for Audio-based Re-instrumentation

This is the code repository for the ISMIR 2022 paper *[Music-STAR: a Style Translation system for Audio-based Re-instrumentation](https://archives.ismir.net/ismir2022/paper/000050.pdf)* by Mahshid Alinoori and Vassilios Tzerpos.

- The audio samples and supplementary material can be found [here](https://mahshidaln.github.io/Music-STAR).
- The conference materials can be found [here](https://ismir2022program.ismir.net/poster_139.html).
- The model checkpoints can be found [here](https://drive.google.com/drive/folders/14UvNwR6I-NRJ6d_kG2men8L45KSdPgnv?usp=sharing).
- The StarNet dataset used in the paper can be found [here](https://zenodo.org/record/6917099).
- The fork of Demucs project used for musical source separation in the paper can be found [here](https://github.com/mahshidaln/demucs).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Audio preprocessing also requires `ffmpeg` and `ffprobe` on `PATH`.

## Run preprocessing

The expected raw audio naming pattern is `TRACK.STEM.wav`, for example
`001.0.wav`, `001.1.wav`, and `001.2.wav`.

```bash
music-star \
  --operation-mode preprocess \
  --data-path data/raw \
  --preprocessed-path data/preprocessed \
  --h5-path data/h5 \
  --split-path data/splits \
  --stems 0 1 2
```

This copies raw files, trims stems to equal length, removes silence, converts
WAV files to HDF5, and creates train/validation/test splits.

## Package usage

```python
from music_star.config import load_builtin_config
from music_star.data import Audio, H5Dataset
from music_star.checkpoints import load_decoder, load_encoder
from music_star.models import Encoder, LegacyEncoder, WaveNet
```

Legacy public checkpoints should be loaded with `music_star.checkpoints`. The
loader detects the legacy GLU encoder architecture by the
`dilated_convs.*.extra.*` weights and instantiates `LegacyEncoder`; standard
encoder states still use `Encoder`.

## Project Tree

```text
.
├── README.md                       # Project overview, usage, and package map.
├── pyproject.toml                  # Package metadata, dependencies, scripts, Ruff, pytest, and mypy.
├── requirements.txt                # Compatibility dependency list for pip users.
├── uv.lock                         # Reproducible uv dependency lock file.
├── music_star/                     # Single import package for all reusable code.
│   ├── __main__.py                 # Main preprocess/train command entry point.
│   ├── checkpoints.py              # Legacy checkpoint loading, state-key detection, and arg repair.
│   ├── cli.py                      # Small helper CLIs such as built-in config listing.
│   ├── config.py                   # JSON training-procedure config dataclasses and loaders.
│   ├── generation.py               # Portable PyTorch generation/inference helpers.
│   ├── launch.py                   # Distributed training launcher.
│   ├── parser.py                   # Historical command-line parser for preprocessing/training flags.
│   ├── smoke.py                    # Checkpoint load, inference, and synthetic backward smoke tests.
│   ├── train.py                    # Config-driven training entry point.
│   ├── translate.py                # Generation CLI for the supported Music-STAR variations.
│   ├── utils.py                    # Audio/file utilities shared by preprocessing and training.
│   ├── configs/                    # Clean JSON configs, one per training procedure.
│   ├── data/                       # Audio preprocessing, HDF5 datasets, and StarNet dataset wrappers.
│   ├── evaluation/                 # Pitch, SDR, timbre, and triplet-network evaluation tools.
│   ├── models/                     # Encoder, recovered legacy encoder, WaveNet, and generator modules.
│   └── trainers/                   # Modular trainers, split by training solution.
└── tests/                          # Unit tests for parser, data, config, checkpoint, and model behavior.
```

The trainer modules are intentionally split by procedure:

- `music_star/trainers/base.py` defines shared trainer dataclasses and lifecycle hooks.
- `music_star/trainers/adversarial_universal.py` implements the universal adversarial baseline.
- `music_star/trainers/decoder_finetuner.py` implements frozen-encoder decoder finetuning.
- `music_star/trainers/embedding_supervised.py` implements latent embedding supervision.
- `music_star/trainers/music_star_mixture.py` implements mixture-supervised Music-STAR training.
- `music_star/trainers/music_star_stem.py` implements two-decoder stem-supervised Music-STAR training.
- `music_star/trainers/factory.py` maps config names to concrete trainer classes.

## Training Procedure Configs

Training procedures are represented as JSON configs in `music_star/configs/`.
They describe reusable model/loss solutions from the paper, not individual
checkpoint folders:

- `recipe_universal_adversarial.json`: reconstruction cross entropy plus latent
  discriminator cross entropy weighted by `d_lambda`, corresponding to the
  universal translation baseline.
- `recipe_decoder_finetune.json`: frozen encoder plus WaveNet reconstruction
  cross entropy for the single-instrument decoder finetuning baseline.
- `recipe_embedding_supervised.json`: frozen universal encoder plus L1 latent
  matching from mixture audio to one instrument stem embedding.
- `recipe_music_star_mixture_supervised.json`: source mixture encoder plus
  target mixture teacher-forced WaveNet decoder with cross entropy.
- `recipe_music_star_stem_supervised.json`: source mixture encoder plus two
  target-stem WaveNet decoders with cross entropy.

Useful commands:

```bash
music-star-configs
music-star-train-config --config-name recipe_embedding_supervised
music-star-generate \
  --variation music_star_mixture_supervised \
  --checkpoint-dir /Users/mahshid/Desktop/checkpoints/star-wave-ps2v \
  --files path/to/input.wav \
  --output outputs/
music-star-generate \
  --variation music_star_stem_supervised \
  --checkpoint-dir /Users/mahshid/Desktop/checkpoints/star-double \
  --files path/to/input.wav \
  --output outputs/
```

## Tests

```bash
python -m pytest
```

The tests cover parser behavior, utility functions, HDF5 slicing, config loading,
checkpoint compatibility, and WaveNet tensor shapes without requiring training
data, ffmpeg invocation, or GPU.
