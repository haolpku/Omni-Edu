"""Backend selection."""

from __future__ import annotations

from typing import Any

from .base import Backend, Generation
from .openai_backend import OpenAIBackend, api_base


def build_backend(model: str, kind: str = "auto", **kwargs: Any) -> Backend:
    """Create a backend.

    ``kind`` is ``"openai"``, ``"transformers"`` or ``"auto"``.  ``auto`` picks
    the OpenAI-compatible backend when a ``base_url`` was supplied and the local
    ``transformers`` backend otherwise.
    """
    kind = (kind or "auto").lower()
    if kind == "auto":
        kind = "openai" if kwargs.get("base_url") else "transformers"

    if kind == "openai":
        base_url = kwargs.pop("base_url", None)
        if not base_url:
            raise ValueError("the openai backend needs base_url=...")
        return OpenAIBackend(model, base_url=base_url, **kwargs)

    if kind == "transformers":
        from .transformers_backend import TransformersBackend

        kwargs.pop("base_url", None)
        kwargs.pop("api_key", None)
        return TransformersBackend(model, **kwargs)

    raise ValueError(f"unknown backend {kind!r}; expected 'auto', 'openai' or 'transformers'")


__all__ = ["Backend", "Generation", "OpenAIBackend", "api_base", "build_backend"]
