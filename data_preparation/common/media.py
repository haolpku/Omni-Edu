"""Image spec resolution shared by stage 4 scoring and stage 6 assembly.

A record references images in two ways:

* ``payload["images"]``: paths to local files;
* ``payload["image_ref"] = {"origin_index": i, ...}``: the image lives at that row of a stage 1
  snapshot, listed in ``SOURCE_MEDIA``.

Specs are plain dicts, e.g. ``{"type": "file", "path": ...}`` or
``{"type": "parquet", "path": ..., "column": ..., "row_index": ...}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SOURCES = Path(__file__).resolve().parent.parent / "stage1_capability_taxonomy_and_source_collection"
SUBJECT = SOURCES / "subject_competence"

#: multimodal source → (stage 1 snapshot path, image column)
SOURCE_MEDIA: dict[str, tuple[str, str]] = {
    "CMM-Math (image)": ("huggingface/cmm-math/train_data.parquet", "image"),
    "ScienceQA (image)": (
        "huggingface/ScienceQA/data/train-00000-of-00001-1028f23e353fbe3e.parquet",
        "image",
    ),
    "Geometry3K": ("huggingface/geometry3k/data/train-00000-of-00001.parquet", "images"),
    "TQA (diagram)": ("huggingface/tqa/data/train-00000-of-00001.parquet", "image"),
    "AI2D": ("huggingface/ai2d/data/train-00000-of-00001.parquet", "image"),
}


def parquet_column(path: Path, column: str) -> list[Any]:
    """Read one parquet column."""
    import pyarrow.parquet as pq

    return pq.read_table(Path(path), columns=[column])[column].to_pylist()


def media_specs(record: dict[str, Any], *, snapshot_root: Path | None = None) -> list[dict[str, Any]]:
    """Image specs referenced by one record."""
    payload = record.get("payload") or record
    specs: list[dict[str, Any]] = []

    for image in payload.get("images") or []:
        if isinstance(image, str):
            specs.append({"type": "file", "path": image})
        elif isinstance(image, dict) and image.get("path"):
            specs.append({"type": "file", "path": image["path"]})

    for ref in payload.get("image_ref") or []:
        if isinstance(ref, dict) and ref.get("archive"):
            specs.append({"type": "archive", "archive": ref["archive"], "member": ref["member"]})

    image_ref = payload.get("image_ref") if isinstance(payload.get("image_ref"), dict) else None
    source = str(record.get("source") or "")
    if image_ref and image_ref.get("has_image") and source in SOURCE_MEDIA:
        relative, column = SOURCE_MEDIA[source]
        root = snapshot_root or SUBJECT
        specs.append(
            {
                "type": "parquet",
                "path": str(root / relative),
                "column": column,
                "row_index": int(image_ref.get("origin_index", 0)),
                "list_index": 0 if column == "images" else None,
            }
        )
    return specs
