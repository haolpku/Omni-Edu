"""Text normalisation used by the deterministic cleaning stage."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[，。．,\.；;：:、！!？?（）\(\)\[\]{}]")


def normalize_text(value: Any, *, optional: bool = False, form: str | None = None) -> str | None:
    """Normalise line endings and strip surrounding whitespace.

    ``optional=True`` returns ``None`` instead of ``""`` for empty input, and
    ``form`` applies a Unicode normalisation form (``"NFC"`` or ``"NFKC"``).
    """
    if value is None:
        return None if optional else ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    if form:
        text = unicodedata.normalize(form, text)
    text = text.strip()
    if optional and not text:
        return None
    return text


def norm_key(value: Any) -> str:
    """Aggressive comparison key: drop whitespace and punctuation, fold case."""
    text = "" if value is None else str(value)
    return _PUNCTUATION.sub("", _WHITESPACE.sub("", text)).lower()


def is_empty(value: Any, *extra: str) -> bool:
    """True for missing values and for the usual textual null spellings."""
    tokens = {"", "nan", "none", "null", "[]"} | {token.lower() for token in extra}
    return str(value).strip().lower() in tokens
