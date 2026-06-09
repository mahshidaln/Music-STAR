from music_star.config import list_builtin_configs, load_builtin_config
from music_star.train import list_training_configs, load_training_config


def test_builtin_configs_load():
    names = list_builtin_configs()

    assert "recipe_embedding_supervised.json" in names
    assert all(name.startswith("recipe_") for name in names)
    for name in names:
        config = load_builtin_config(name)
        namespace = config.to_namespace()
        assert config.name
        assert config.recipe in {
            "decoder_finetune",
            "embedding_supervised",
            "music_star_mixture_supervised",
            "music_star_stem_supervised",
            "universal_adversarial",
        }
        assert namespace.recipe == config.recipe


def test_training_config_helpers_list_procedures_only():
    names = list_training_configs()

    assert names == [
        "recipe_decoder_finetune.json",
        "recipe_embedding_supervised.json",
        "recipe_music_star_mixture_supervised.json",
        "recipe_music_star_stem_supervised.json",
        "recipe_universal_adversarial.json",
    ]
    assert load_training_config("recipe_embedding_supervised").recipe == "embedding_supervised"
