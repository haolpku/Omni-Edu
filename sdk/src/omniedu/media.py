"""Image handling: local files, URLs, bytes and PIL objects.

Every benchmark runner in the evaluation tree grew its own copy of this code
(``data_url`` / ``image_url`` helpers with subtly different MIME handling). The
toolkit keeps one implementation.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Iterable

IMAGE_MARKER = "<image>"

#: Signature -> MIME type, checked in order.
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
    # Unrecognised payloads go through as JPEG, matching the evaluation runners.
    return "image/jpeg"


def to_data_url(data: bytes) -> str:
    return f"data:{sniff_mime(data)};base64,{base64.b64encode(data).decode()}"


def _pil_to_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    if getattr(image, "mode", None) not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalise_image(value: Any, *, base_dir: str | Path | None = None) -> str:
    """Return a URL the OpenAI-compatible backend can send.

    Accepts a local path (``str`` / ``Path``), an ``http(s)`` or ``data:`` URL,
    raw ``bytes``, a PIL image, or a mapping with ``url`` / ``bytes`` / ``path``.
    """
    if value is None:
        raise ValueError("image is None")

    if isinstance(value, dict):
        if value.get("url"):
            return str(value["url"])
        if value.get("bytes"):
            return to_data_url(bytes(value["bytes"]))
        if value.get("path"):
            value = value["path"]
        else:
            raise ValueError(f"unsupported image mapping: {sorted(value)!r}")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return to_data_url(bytes(value))

    if isinstance(value, Path):
        return to_data_url(value.expanduser().read_bytes())

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("image is an empty string")
        if text.startswith(("http://", "https://", "data:")):
            return text
        path = Path(text).expanduser()
        if not path.is_absolute() and base_dir is not None:
            candidate = Path(base_dir) / path
            if candidate.exists():
                path = candidate
        return to_data_url(path.read_bytes())

    if hasattr(value, "save") and hasattr(value, "mode"):  # PIL.Image
        return to_data_url(_pil_to_bytes(value))

    raise TypeError(f"unsupported image type: {type(value).__name__}")


def normalise_images(values: Any, *, base_dir: str | Path | None = None) -> list[str]:
    """Normalise a single image or an iterable of images into a list of URLs."""
    if values is None:
        return []
    if isinstance(values, dict) or isinstance(values, (str, bytes, bytearray, memoryview, Path)):
        return [normalise_image(values, base_dir=base_dir)]
    if hasattr(values, "save") and hasattr(values, "mode"):
        return [normalise_image(values, base_dir=base_dir)]
    return [normalise_image(item, base_dir=base_dir) for item in values]


def to_pil(url: str) -> Any:
    """Turn anything :func:`normalise_image` returns into a PIL image.

    The local ``transformers`` backend wants real images rather than URLs.
    """
    from PIL import Image

    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    if url.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(url).convert("RGB")


def build_user_content(text: str, image_urls: Iterable[str] | None = None) -> Any:
    """Build an OpenAI-style user content value.

    ``<image>`` markers inside ``text`` are replaced by the supplied images in
    order, which is how the training mixture records image placement. Images
    left over (or text with no markers) are appended after the text.
    """
    urls = list(image_urls or [])
    if not urls:
        return text

    chunks = text.split(IMAGE_MARKER) if IMAGE_MARKER in text else [text]
    parts: list[dict[str, Any]] = []
    consumed = 0
    for index, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if index < len(chunks) - 1 and consumed < len(urls):
            parts.append({"type": "image_url", "image_url": {"url": urls[consumed]}})
            consumed += 1
    for url in urls[consumed:]:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    if not parts:
        parts.append({"type": "text", "text": ""})
    return parts


def to_transformers_content(text: str, image_urls: Iterable[str] | None = None) -> Any:
    """Same layout as :func:`build_user_content`, with PIL images attached.

    The local processor wants ``{"type": "image", "image": <PIL.Image>}``
    instead of a URL reference.
    """
    urls = list(image_urls or [])
    if not urls:
        return text
    content = build_user_content(text, urls)
    converted: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            converted.append({"type": "image", "image": to_pil(part["image_url"]["url"])})
        else:
            converted.append(part)
    return converted
