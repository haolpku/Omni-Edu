#!/usr/bin/env python3
"""QC for the cleaned-output projection after Stage-1 dedup."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from apply_dedup_to_cleaned import (
    DATASETS,
    GLOBAL_OUTPUT,
    OUTPUT_LAYER,
    load_old_cleaned,
    load_stage1_canonical,
)
from dedup import PIPE, content_key, read_jsonl, source_slug


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def main() -> None:
    failures: list[dict[str, Any]] = []
    stage1_by_key = load_stage1_canonical()
    eval_keys = {
        content_key(row) for row in read_jsonl(PIPE / "stage1_eval_index/reserved_records.jsonl")
    }
    all_kept = []
    dataset_checks = {}

    for slug in DATASETS:
        old_rows = load_old_cleaned(slug)
        old_by_uid = {str(row.get("uid")): row for row in old_rows}
        out = PIPE / slug / OUTPUT_LAYER
        projection_report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        old_source_available = bool(old_rows)
        input_count = (
            len(old_rows) if old_source_available else int(projection_report["input_cleaned"])
        )
        kept = load_rows(out / "kept.jsonl")
        removed = load_rows(out / "removed_by_stage1.jsonl")
        if input_count != len(kept) + len(removed):
            failures.append(
                {
                    "check": "dataset_accounting",
                    "dataset": slug,
                    "input": input_count,
                    "accounted": len(kept) + len(removed),
                }
            )

        for row in kept:
            uid = str(row.get("uid"))
            old = old_by_uid.get(uid)
            key = content_key(row)
            canonical = stage1_by_key.get(key)
            if old_source_available and old is None:
                failures.append({"check": "kept_uid_from_old_cleaned", "dataset": slug, "uid": uid})
                continue
            if old_source_available:
                for field in ["messages", "grounding", "image_ref"]:
                    if row.get(field) != old.get(field):
                        failures.append(
                            {
                                "check": "cleaned_content_unchanged",
                                "dataset": slug,
                                "uid": uid,
                                "field": field,
                            }
                        )
            if canonical is None or source_slug(canonical) != slug:
                failures.append(
                    {
                        "check": "stage1_canonical_source",
                        "dataset": slug,
                        "uid": uid,
                    }
                )
            elif (row.get("meta") or {}).get("split") != (canonical.get("meta") or {}).get("split"):
                failures.append(
                    {
                        "check": "stage1_split_applied",
                        "dataset": slug,
                        "uid": uid,
                    }
                )
            if key in eval_keys:
                failures.append(
                    {
                        "check": "official_eval_overlap_absent",
                        "dataset": slug,
                        "uid": uid,
                    }
                )

        invalid_reasons = Counter(
            str(row.get("_reason"))
            for row in removed
            if row.get("_reason") not in {"official_eval_overlap", "cross_dataset_duplicate"}
        )
        if invalid_reasons:
            failures.append(
                {
                    "check": "only_stage1_dedup_removed",
                    "dataset": slug,
                    "reasons": dict(invalid_reasons),
                }
            )

        expected_splits = Counter(
            (
                "multimodal" if row.get("image_ref") else "text",
                str((row.get("meta") or {}).get("split") or "train"),
            )
            for row in kept
        )
        for modality in ["text", "multimodal"]:
            for split in ["train", "validation", "test"]:
                actual = len(load_rows(out / f"{modality}_{split}.jsonl"))
                expected = expected_splits[(modality, split)]
                if actual != expected:
                    failures.append(
                        {
                            "check": "split_file_accounting",
                            "dataset": slug,
                            "modality": modality,
                            "split": split,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

        dataset_checks[slug] = {
            "input_cleaned": input_count,
            "kept": len(kept),
            "removed_by_stage1": len(removed),
            "old_source_available": old_source_available,
            "removed_reasons": dict(Counter(str(row.get("_reason")) for row in removed)),
        }
        all_kept.extend(kept)

    keys = [content_key(row) for row in all_kept]
    if len(keys) != len(set(keys)):
        failures.append(
            {
                "check": "global_unique_content_key",
                "records": len(keys),
                "unique_keys": len(set(keys)),
            }
        )

    global_rows = load_rows(GLOBAL_OUTPUT / "kept.jsonl")
    if [str(row.get("uid")) for row in global_rows] != [str(row.get("uid")) for row in all_kept]:
        failures.append({"check": "global_kept_matches_dataset_outputs"})

    report = {
        "datasets": dataset_checks,
        "input_cleaned": sum(row["input_cleaned"] for row in dataset_checks.values()),
        "kept": len(all_kept),
        "unique_content_keys": len(set(keys)),
        "removed_by_stage1": sum(row["removed_by_stage1"] for row in dataset_checks.values()),
        "failure_count": len(failures),
        "failures": failures,
    }
    (GLOBAL_OUTPUT / "QC_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
