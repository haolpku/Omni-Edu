#!/usr/bin/env python3
"""Deterministic QC for curriculum-grounding Stage-1 outputs."""

from __future__ import annotations

import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dedup import PIPE, TRAIN_INPUTS, content_key, read_jsonl


DATASETS = list(TRAIN_INPUTS)


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def main() -> None:
    failures: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    eval_path = PIPE / "stage1_eval_index/reserved_records.jsonl"
    eval_rows = load_rows(eval_path)
    eval_keys = {content_key(row) for row in eval_rows}

    final_rows: list[dict[str, Any]] = []
    all_routed_uids: set[str] = set()
    routed_pointer_rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, Any] = {}

    for slug, origin_path in TRAIN_INPUTS.items():
        out = PIPE / slug / "cleaned"
        kept = load_rows(out / "kept.jsonl")
        intra = load_rows(out / "removed_intra_duplicates.jsonl")
        eval_removed = load_rows(out / "removed_official_eval_overlap.jsonl")
        cross = load_rows(out / "removed_cross_duplicates.jsonl")
        origin_count = count_jsonl(origin_path)
        accounted = len(kept) + len(intra) + len(eval_removed) + len(cross)
        dataset_counts[slug] = {
            "origin": origin_count,
            "kept": len(kept),
            "removed_intra": len(intra),
            "removed_eval_overlap": len(eval_removed),
            "removed_cross": len(cross),
            "accounted": accounted,
        }
        if accounted != origin_count:
            failures.append(
                {
                    "check": "dataset_accounting",
                    "dataset": slug,
                    "origin": origin_count,
                    "accounted": accounted,
                }
            )
        for row in kept:
            if content_key(row) in eval_keys:
                failures.append(
                    {
                        "check": "official_eval_leakage",
                        "dataset": slug,
                        "uid": row.get("uid"),
                    }
                )
        for row in eval_removed:
            if content_key(row) not in eval_keys:
                failures.append(
                    {
                        "check": "invalid_eval_overlap_route",
                        "dataset": slug,
                        "uid": row.get("uid"),
                    }
                )
        final_rows.extend(kept)
        routed_pointer_rows.extend(intra)
        routed_pointer_rows.extend(cross)
        all_routed_uids.update(
            str(row.get("uid"))
            for row in [*intra, *eval_removed, *cross]
            if row.get("uid")
        )

    final_uid_counts = Counter(
        str(row.get("uid")) for row in final_rows if row.get("uid")
    )
    duplicate_uids = [uid for uid, count in final_uid_counts.items() if count > 1]
    if duplicate_uids:
        failures.append(
            {
                "check": "unique_final_uid",
                "count": len(duplicate_uids),
                "examples": duplicate_uids[:20],
            }
        )

    global_key_members: dict[str, list[str]] = defaultdict(list)
    for row in final_rows:
        global_key_members[content_key(row)].append(str(row.get("uid")))
    duplicate_keys = {
        key: uids for key, uids in global_key_members.items() if len(uids) > 1
    }
    if duplicate_keys:
        failures.append(
            {
                "check": "unique_global_content_key",
                "count": len(duplicate_keys),
                "examples": list(duplicate_keys.values())[:20],
            }
        )

    known_uids = set(final_uid_counts) | all_routed_uids
    dangling = [
        {
            "uid": row.get("uid"),
            "kept_uid": row.get("_kept_uid"),
            "reason": row.get("_reason"),
        }
        for row in routed_pointer_rows
        if row.get("_kept_uid") and str(row.get("_kept_uid")) not in known_uids
    ]
    if dangling:
        failures.append(
            {
                "check": "dedup_pointer_exists",
                "count": len(dangling),
                "examples": dangling[:20],
            }
        )

    tar_members: dict[str, set[str]] = {}
    image_refs_checked = 0
    for row in [*final_rows, *eval_rows]:
        for ref in row.get("image_ref") or []:
            image_refs_checked += 1
            archive = Path(str(ref.get("archive") or ""))
            member = str(ref.get("member") or "")
            if not archive.is_file():
                failures.append(
                    {
                        "check": "image_archive_exists",
                        "uid": row.get("uid"),
                        "archive": str(archive),
                    }
                )
                continue
            archive_key = str(archive)
            if archive_key not in tar_members:
                try:
                    with tarfile.open(archive, "r:*") as handle:
                        tar_members[archive_key] = {
                            item.name for item in handle.getmembers() if item.isfile()
                        }
                except tarfile.TarError as exc:
                    failures.append(
                        {
                            "check": "image_archive_readable",
                            "archive": archive_key,
                            "error": str(exc),
                        }
                    )
                    tar_members[archive_key] = set()
            if member not in tar_members[archive_key]:
                failures.append(
                    {
                        "check": "image_member_exists",
                        "uid": row.get("uid"),
                        "archive": archive_key,
                        "member": member,
                    }
                )

    standards = load_rows(
        PIPE / "achieve-the-core/cleaned/kept.jsonl"
    )
    standard_ids = [str(row.get("id")) for row in standards]
    if len(standard_ids) != len(set(standard_ids)):
        failures.append({"check": "unique_achieve_standard_id"})

    edge_checks = {}
    for filename in ["kept_prerequisite_edges.jsonl", "kept_similar_to_edges.jsonl"]:
        rows = load_rows(PIPE / "junyi-graph/cleaned" / filename)
        signatures = [
            json.dumps(
                [
                    row.get("source"),
                    row.get("target"),
                    row.get("relation"),
                    row.get("attributes"),
                ],
                sort_keys=True,
            )
            for row in rows
        ]
        edge_checks[filename] = len(rows)
        if len(signatures) != len(set(signatures)):
            failures.append(
                {"check": "unique_junyi_edge", "file": filename}
            )

    edukg = load_rows(PIPE / "edukg/cleaned/kept_entities.jsonl")
    edukg_ids = [str(row.get("entity_id")) for row in edukg]
    if len(edukg_ids) != len(set(edukg_ids)):
        failures.append({"check": "unique_edukg_entity_id"})

    openscied = load_rows(PIPE / "openscied/cleaned/kept_pages.jsonl")
    page_keys = [str(row.get("visible_text")) for row in openscied]
    if len(page_keys) != len(set(page_keys)):
        failures.append({"check": "unique_openscied_visible_text"})

    raw_inventory_path = PIPE / "stage1_raw_files/file_inventory.jsonl"
    raw_file_checks: dict[str, Any] = {"available": raw_inventory_path.exists()}
    if raw_inventory_path.exists():
        inventory = load_rows(raw_inventory_path)
        duplicate_groups = load_rows(
            PIPE / "stage1_raw_files/duplicate_groups.jsonl"
        )
        routed_files = load_rows(
            PIPE / "stage1_raw_files/routed_exact_file_duplicates.jsonl"
        )
        paths = [str(row.get("relative_path")) for row in inventory]
        if len(paths) != len(set(paths)):
            failures.append({"check": "unique_raw_inventory_path"})
        by_hash: dict[str, list[str]] = defaultdict(list)
        for row in inventory:
            by_hash[str(row.get("sha256"))].append(str(row.get("relative_path")))
        expected_groups = {
            digest: sorted(group_paths)
            for digest, group_paths in by_hash.items()
            if len(group_paths) > 1
        }
        reported_groups = {
            str(row.get("sha256")): sorted(map(str, row.get("paths") or []))
            for row in duplicate_groups
        }
        if expected_groups != reported_groups:
            failures.append(
                {
                    "check": "raw_duplicate_group_completeness",
                    "expected": len(expected_groups),
                    "reported": len(reported_groups),
                }
            )
        expected_routed = sum(len(paths_) - 1 for paths_ in expected_groups.values())
        if len(routed_files) != expected_routed:
            failures.append(
                {
                    "check": "raw_duplicate_routing_accounting",
                    "expected": expected_routed,
                    "reported": len(routed_files),
                }
            )
        raw_file_checks.update(
            files=len(inventory),
            bytes=sum(int(row.get("size_bytes") or 0) for row in inventory),
            duplicate_groups=len(duplicate_groups),
            routed_duplicates=len(routed_files),
        )

    checks.update(
        dataset_counts=dataset_counts,
        final_curriculum_items=len(final_rows),
        final_unique_content_keys=len(global_key_members),
        reserved_eval_records=len(eval_rows),
        reserved_eval_unique_content_keys=len(eval_keys),
        image_refs_checked=image_refs_checked,
        image_archives_checked=len(tar_members),
        achieve_standards=len(standards),
        junyi_edges=edge_checks,
        edukg_entities=len(edukg),
        openscied_pages=len(openscied),
        raw_files=raw_file_checks,
    )
    report = {
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
    }
    path = PIPE / "STAGE1_QC_REPORT.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
