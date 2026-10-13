#!/usr/bin/env python3
"""Apply Stage-1 exclusions to the existing structurally cleaned outputs.

This is a no-delete projection:

* old ``cleaned/kept_*`` files remain untouched;
* records excluded by official-eval or cross-dataset Stage-1 dedup are routed;
* same-source canonical UID changes do not discard an already-cleaned record;
* DA-20K/XES splits are replaced with the Stage-1 content-key split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from dedup import PIPE, TRAIN_INPUTS, content_key, read_jsonl, source_slug

DATASETS = list(TRAIN_INPUTS)
OUTPUT_LAYER = "cleaned"
GLOBAL_OUTPUT = PIPE / "stage2_clean_after_stage1"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_old_cleaned(slug: str) -> list[dict[str, Any]]:
    root = PIPE / slug / "cleaned"
    result = []
    for filename in ["kept_text.jsonl", "kept_multimodal.jsonl"]:
        path = root / filename
        if path.exists():
            result.extend(read_jsonl(path))
    return result


def load_stage1_canonical() -> dict[str, dict[str, Any]]:
    result = {}
    for slug in DATASETS:
        path = PIPE / slug / "cleaned/kept.jsonl"
        for row in read_jsonl(path):
            key = content_key(row)
            if key in result:
                raise RuntimeError(f"duplicate Stage-1 final key: {key}")
            result[key] = row
    return result


def load_eval_exclusion_keys() -> set[str]:
    result = set()
    for slug in DATASETS:
        path = PIPE / slug / "cleaned/removed_official_eval_overlap.jsonl"
        result.update(content_key(row) for row in read_jsonl(path))
    return result


def apply_stage1(
    slug: str,
    rows: list[dict[str, Any]],
    stage1_by_key: dict[str, dict[str, Any]],
    eval_exclusion_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    kept = []
    removed = []
    reasons: Counter[str] = Counter()
    for row in rows:
        key = content_key(row)
        canonical = stage1_by_key.get(key)
        if canonical is None:
            reason = (
                "official_eval_overlap"
                if key in eval_exclusion_keys
                else "absent_from_stage1_final"
            )
            routed = dict(row)
            routed.update(
                _reason=reason,
                _dedup_stage="stage1",
                _dedup_application="filtered_from_existing_cleaned_output",
                _content_key=((row.get("_stage1") or {}).get("content_key_sha256")),
            )
            removed.append(routed)
            reasons[reason] += 1
            continue

        canonical_slug = source_slug(canonical)
        if canonical_slug != slug:
            routed = dict(row)
            routed.update(
                _reason="cross_dataset_duplicate",
                _kept_uid=canonical.get("uid"),
                _kept_in=canonical_slug,
                _dedup_stage="stage1",
                _dedup_application="filtered_from_existing_cleaned_output",
            )
            removed.append(routed)
            reasons["cross_dataset_duplicate"] += 1
            continue

        projected = dict(row)
        projected["meta"] = dict(row.get("meta") or {})
        projected["meta"]["split"] = (canonical.get("meta") or {}).get(
            "split", projected["meta"].get("split", "train")
        )
        if (canonical.get("meta") or {}).get("split_basis"):
            projected["meta"]["split_basis"] = canonical["meta"]["split_basis"]
        projected["_stage1"] = canonical.get("_stage1") or {}
        projected["_stage1_application"] = {
            "stage1_canonical_uid": canonical.get("uid"),
            "cleaned_representative_uid": row.get("uid"),
            "same_uid": row.get("uid") == canonical.get("uid"),
            "action": "kept_after_stage1",
        }
        kept.append(projected)
        reasons[
            (
                "kept_same_uid"
                if row.get("uid") == canonical.get("uid")
                else "kept_same_source_representative"
            )
        ] += 1
    return kept, removed, reasons


def split_outputs(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        modality = "multimodal" if row.get("image_ref") else "text"
        split = str((row.get("meta") or {}).get("split") or "train")
        result[(modality, split)].append(row)
    return result


def write_dataset_outputs(
    slug: str,
    kept: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    reasons: Counter[str],
    origin_count: int,
) -> dict[str, Any]:
    out = PIPE / slug / OUTPUT_LAYER
    text = [row for row in kept if not row.get("image_ref")]
    multimodal = [row for row in kept if row.get("image_ref")]
    write_jsonl(out / "kept.jsonl", kept)
    write_jsonl(out / "kept_text.jsonl", text)
    write_jsonl(out / "kept_multimodal.jsonl", multimodal)
    write_jsonl(out / "removed_by_stage1.jsonl", removed)
    split_rows = split_outputs(kept)
    split_counts = {}
    for modality in ["text", "multimodal"]:
        for split in ["train", "validation", "test"]:
            rows = split_rows.get((modality, split), [])
            write_jsonl(out / f"{modality}_{split}.jsonl", rows)
            split_counts[f"{modality}_{split}"] = len(rows)
    report = {
        "input_cleaned": origin_count,
        "kept": len(kept),
        "removed_by_stage1": len(removed),
        "text": len(text),
        "multimodal": len(multimodal),
        "actions": dict(reasons),
        "splits": split_counts,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    stage1_by_key = load_stage1_canonical()
    eval_exclusion_keys = load_eval_exclusion_keys()
    old_cleaned_by_slug = {slug: load_old_cleaned(slug) for slug in DATASETS}
    missing_inputs = [slug for slug, rows in old_cleaned_by_slug.items() if not rows]
    if missing_inputs:
        raise FileNotFoundError(
            "Old layer2 kept inputs were pruned for: "
            + ", ".join(missing_inputs)
            + ". Run build_structure_sft.py to regenerate them before reapplying."
        )
    all_kept = []
    all_removed = []
    dataset_reports = {}
    for slug in DATASETS:
        old_cleaned = old_cleaned_by_slug[slug]
        kept, removed, reasons = apply_stage1(slug, old_cleaned, stage1_by_key, eval_exclusion_keys)
        dataset_reports[slug] = write_dataset_outputs(
            slug, kept, removed, reasons, len(old_cleaned)
        )
        all_kept.extend(kept)
        all_removed.extend(removed)

    write_jsonl(GLOBAL_OUTPUT / "kept.jsonl", all_kept)
    write_jsonl(GLOBAL_OUTPUT / "removed_by_stage1.jsonl", all_removed)
    for (modality, split), rows in split_outputs(all_kept).items():
        write_jsonl(GLOBAL_OUTPUT / f"{modality}_{split}.jsonl", rows)

    report = {
        "input_cleaned": sum(values["input_cleaned"] for values in dataset_reports.values()),
        "kept": len(all_kept),
        "removed_by_stage1": len(all_removed),
        "datasets": dataset_reports,
        "note": (
            "Existing cleaned outputs are unchanged. This layer applies only "
            "Stage-1 exact dedup/evaluation exclusions and content-key splits."
        ),
    }
    (GLOBAL_OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
