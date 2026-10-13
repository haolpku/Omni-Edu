import base64

import pytest

from omniedu.media import (
    build_user_content,
    normalise_image,
    normalise_images,
    sniff_mime,
    to_data_url,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def test_sniff_mime() -> None:
    assert sniff_mime(PNG_BYTES) == "image/png"
    assert sniff_mime(JPEG_BYTES) == "image/jpeg"
    assert sniff_mime(b"GIF89a" + b"\x00" * 8) == "image/gif"
    assert sniff_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"


def test_bytes_become_a_data_url() -> None:
    url = normalise_image(PNG_BYTES)
    assert url.startswith("data:image/png;base64,")
    assert url == to_data_url(PNG_BYTES)


def test_remote_urls_pass_through() -> None:
    assert normalise_image("https://example.com/a.png") == "https://example.com/a.png"
    existing = "data:image/png;base64,AAAA"
    assert normalise_image(existing) == existing


def test_local_path_is_read(tmp_path) -> None:
    path = tmp_path / "x.png"
    path.write_bytes(PNG_BYTES)
    assert normalise_image(str(path)).startswith("data:image/png;base64,")


def test_mapping_and_relative_paths(tmp_path) -> None:
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(PNG_BYTES)
    url = normalise_image("img/a.png", base_dir=tmp_path)
    assert url.startswith("data:image/png;base64,")
    assert normalise_image({"bytes": PNG_BYTES}).startswith("data:image/png;base64,")
    assert normalise_image({"url": "https://example.com/b.png"}).endswith("b.png")


def test_single_and_multiple_images_normalise() -> None:
    assert len(normalise_images(PNG_BYTES)) == 1
    assert len(normalise_images([PNG_BYTES, JPEG_BYTES])) == 2
    assert normalise_images(None) == []


def test_text_only_stays_a_string() -> None:
    assert build_user_content("just text") == "just text"


def test_image_markers_are_replaced_in_order() -> None:
    parts = build_user_content("before <image> middle <image> after", ["u1", "u2"])
    assert [part["type"] for part in parts] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert parts[1]["image_url"]["url"] == "u1"
    assert parts[3]["image_url"]["url"] == "u2"


def test_leftover_images_are_appended() -> None:
    parts = build_user_content("no marker here", ["u1", "u2"])
    assert parts[0] == {"type": "text", "text": "no marker here"}
    assert [part["image_url"]["url"] for part in parts[1:]] == ["u1", "u2"]


def test_unsupported_type_raises() -> None:
    with pytest.raises(TypeError):
        normalise_image(42)
