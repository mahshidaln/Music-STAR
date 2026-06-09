from types import SimpleNamespace

import torch

from music_star.checkpoints import (
    decoder_args_for_state,
    is_legacy_encoder_state,
    load_encoder,
    strip_legacy_encoder_extra_keys,
)
from music_star.models import Encoder, LegacyEncoder, WaveNet
from music_star.smoke import route_condition


def _args(**overrides):
    values = {
        "encoder_blocks": 1,
        "encoder_layers": 1,
        "encoder_channels": 4,
        "latent_d": 3,
        "encoder_func": "relu",
        "encoder_pool": 2,
        "blocks": 1,
        "layers": 1,
        "kernel_size": 2,
        "skip_channels": 5,
        "residual_channels": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_strip_legacy_encoder_extra_keys_allows_strict_load():
    args = _args()
    model = Encoder(args)
    state = model.state_dict()
    state["dilated_convs.0.extra.weight"] = torch.zeros(8, 4, 1)
    state["dilated_convs.0.extra.bias"] = torch.zeros(8)

    stripped, notes = strip_legacy_encoder_extra_keys(state)

    model.load_state_dict(stripped, strict=True)
    assert len(notes) == 1
    assert "ignored 2 legacy encoder extra parameters" in notes[0]


def test_legacy_encoder_state_loads_strictly():
    args = _args()
    state = LegacyEncoder(args).state_dict()

    model = LegacyEncoder(args)
    model.load_state_dict(state, strict=True)

    assert is_legacy_encoder_state(state)
    output = model(torch.zeros(2, 16))
    assert output.shape == (2, 3, 8)


def test_load_encoder_uses_legacy_architecture(tmp_path):
    args = _args()
    encoder = LegacyEncoder(args)

    torch.save([args, 0], tmp_path / "args.pth")
    torch.save({"encoder_state": encoder.state_dict()}, tmp_path / "bestmodel_0.pth")

    loaded = load_encoder(tmp_path)

    assert isinstance(loaded.model, LegacyEncoder)
    assert "loaded recovered legacy GLU encoder architecture" in loaded.notes


def test_decoder_args_follow_checkpoint_condition_channels():
    args = _args(latent_d=8)
    decoder_state = WaveNet(_args(latent_d=3)).state_dict()

    patched_args, notes = decoder_args_for_state(args, decoder_state)

    assert patched_args.latent_d == 3
    assert "patched latent_d" in notes[0]


def test_route_condition_splits_double_decoder_latents():
    encoded = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)

    first, first_notes = route_condition(
        encoded, 4, "decoder_state", ["decoder_state", "decoder2_state"]
    )
    second, second_notes = route_condition(
        encoded,
        4,
        "decoder2_state",
        ["decoder_state", "decoder2_state"],
    )

    assert torch.equal(first, encoded[:, :4, :])
    assert torch.equal(second, encoded[:, 4:, :])
    assert "0:4" in first_notes[0]
    assert "4:8" in second_notes[0]
