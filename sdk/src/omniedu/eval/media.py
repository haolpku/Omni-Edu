"""Image bytes -> ``data:`` URL, with one MIME sniffer for the whole tree.

``mmmu_pro``, ``exams_v``, ``tutorbench`` and ``mdk12_bench`` had each grown
their own copy of this; they disagreed on how to label anything that was not a
PNG.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff_mime(data: bytes) -> str:
    for signature, mime in _MAGIC:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # Unrecognised payloads are passed through as JPEG, as the runners did.
    return "image/jpeg"


def data_url(value: Any) -> str | None:
    """Build a ``data:`` URL from raw bytes, a path, or a row's image field.

    Accepts ``bytes``, a local path, or the ``{"bytes": ...}`` / ``{"path": ...}``
    mapping that the benchmark loaders produce.  Returns ``None`` when there is
    nothing to encode, matching the runners' "no image in this row" path.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    elif isinstance(value, dict):
        raw = value.get("bytes")
        if not raw and value.get("path"):
            raw = Path(value["path"]).read_bytes()
        elif raw:
            raw = bytes(raw)
    elif isinstance(value, (str, Path)):
        text = str(value)
        if not text:
            return None
        raw = Path(text).read_bytes()
    else:
        raise TypeError(f"unsupported image value: {type(value).__name__}")
    if not raw:
        return None
    return f"data:{sniff_mime(raw)};base64,{base64.b64encode(raw).decode()}"
