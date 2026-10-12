"""Checkpoint names and default generation settings."""

from __future__ import annotations

from .types import GenerationParams

#: Short names accepted by ``OmniEdu.from_pretrained``.
PRESETS: dict[str, str] = {
    "4B": "OpenDCAI/Omni-Edu-4B",
    "9B": "OpenDCAI/Omni-Edu-9B",
    "27B": "OpenDCAI/Omni-Edu-27B",
}

#: Paper evaluation protocol: greedy, no thinking.
DEFAULT_GENERATION = GenerationParams(temperature=0.0, max_new_tokens=2048, enable_thinking=False)


def resolve_model(name: str) -> str:
    """Map ``"4B"`` to a Hub id, leaving anything else untouched.

    Case is ignored and an optional ``omniedu-`` / ``omni-edu-`` prefix is
    accepted, so ``"4b"``, ``"OmniEdu-4B"`` and ``"4B"`` all resolve. Full Hub
    ids and local checkpoint paths pass straight through.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model must be a non-empty string")
    key = name.strip()
    if key in PRESETS:
        return PRESETS[key]
    lowered = key.lower()
    for short, repo in PRESETS.items():
        if lowered in (short.lower(), f"omniedu-{short.lower()}", f"omni-edu-{short.lower()}"):
            return repo
    return key


def known_presets() -> dict[str, str]:
    return dict(PRESETS)


def served_name(name: str) -> str:
    """The name a served preset answers to, i.e. what ``omniedu serve`` passes to
    vLLM's ``--served-model-name``.

    ``"27B"`` and ``"OpenDCAI/Omni-Edu-27B"`` both give ``"omniedu-27b"``.  The
    client and ``omniedu serve`` share this so the printed vLLM command and the
    requests the toolkit sends agree; pass ``served_model=`` (or start vLLM under
    the checkpoint's own id) to talk to a server named differently.
    """
    return "omniedu-" + resolve_model(name).rsplit("-", 1)[-1].lower()


def is_preset(name: str) -> bool:
    """True when ``name`` resolves to one of the shipped checkpoints."""
    try:
        return resolve_model(name) in PRESETS.values()
    except ValueError:
        return False


def is_alias(name: str) -> bool:
    """True when the caller wrote a short alias (``4B``, ``omniedu-27b``).

    Distinct from :func:`is_preset`, which is also true for a full Hub id.  Only
    the alias form is rewritten to the served name, because only then is it clear
    the user followed ``omniedu serve``; a full id is sent unchanged so a server
    started under vLLM's default name still works.
    """
    key = str(name).strip().lower()
    return any(
        key in (short.lower(), f"omniedu-{short.lower()}", f"omni-edu-{short.lower()}")
        for short in PRESETS
    )
