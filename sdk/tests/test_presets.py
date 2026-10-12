from omniedu.presets import is_alias, is_preset, known_presets, resolve_model, served_name


def test_short_names_resolve() -> None:
    assert resolve_model("4B") == "OpenDCAI/Omni-Edu-4B"
    assert resolve_model("9b") == "OpenDCAI/Omni-Edu-9B"
    assert resolve_model("OmniEdu-27B") == "OpenDCAI/Omni-Edu-27B"
    assert resolve_model("omni-edu-27b") == "OpenDCAI/Omni-Edu-27B"


def test_full_ids_and_paths_pass_through() -> None:
    assert resolve_model("OpenDCAI/Omni-Edu-9B") == "OpenDCAI/Omni-Edu-9B"
    assert resolve_model("/models/Omni-Edu-4B") == "/models/Omni-Edu-4B"


def test_presets_are_complete() -> None:
    assert set(known_presets()) == {"4B", "9B", "27B"}


def test_served_name_matches_what_the_serve_command_prints() -> None:
    """``omniedu serve`` passes ``--served-model-name omniedu-<size>``, and the
    client derives the same name from the same alias."""
    assert served_name("4B") == "omniedu-4b"
    assert served_name("9b") == "omniedu-9b"
    assert served_name("OmniEdu-27B") == "omniedu-27b"
    assert served_name("OpenDCAI/Omni-Edu-27B") == "omniedu-27b"


def test_is_preset_distinguishes_aliases_from_other_names() -> None:
    assert is_preset("4B") and is_preset("OpenDCAI/Omni-Edu-9B") and is_preset("27b")
    assert not is_preset("my-server-name")
    assert not is_preset("/models/Omni-Edu-4B")


def test_only_the_short_alias_is_rewritten_to_the_served_name() -> None:
    """A full Hub id means the user may have let vLLM keep its default served
    name, so it must be sent as-is; only the alias form is rewritten."""
    assert is_alias("4B") and is_alias("omniedu-27b") and is_alias("OmniEdu-9B")
    assert not is_alias("OpenDCAI/Omni-Edu-9B")
    assert not is_alias("/models/Omni-Edu-4B")
    assert not is_alias("qwen3-8b")


def test_empty_name_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        resolve_model("  ")
