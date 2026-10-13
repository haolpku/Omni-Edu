#!/usr/bin/env python3
"""Validate final bucketed K-Center outputs and write a compact report."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent  # stage5_token_budgeted_diversity_selection
SFT = HERE.parent  # data_preparation
STAGE4 = SFT / "stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering"
RESULTS = HERE / "results"
FILTERED = STAGE4 / "quality_scoring/results/quality_filtered_pools"
FINAL = RESULTS / "bucketed_kcenter_final"
CATEGORIES = (
    "subject_competence",
    "diagnostic_reasoning",
    "curriculum_grounding",
    "pedagogical_action",
)
#: final sizes reported in the paper, used to check the selection
EDUCATION_ROWS = 60_951
EDUCATION_TOKENS = 11_997_566
GENERAL_ROWS = 9_048
GENERAL_TOKENS = 3_958_203
COMBINED_ROWS = 69_999
COMBINED_TOKENS = 15_955_769
def rows(path: Path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def row_id(row: dict) -> str:
    return f"{row['category']}::{row['source']}::{row['uid']}"


def main():
    all_ids = set()
    content_hashes = set()
    category_summary = {}
    bucket_rows = []
    errors = []
    for category in CATEGORIES:
        candidate_ids = {row_id(row) for row in rows(FILTERED / f"{category}.jsonl")}
        selected = list(rows(FINAL / f"{category}.jsonl"))
        summary = json.loads((FINAL / f"{category}.summary.json").read_text())
        selected_ids = [row_id(row) for row in selected]
        if len(selected_ids) != len(set(selected_ids)):
            errors.append(f"{category}: duplicate selected IDs")
        missing = set(selected_ids) - candidate_ids
        if missing:
            errors.append(f"{category}: {len(missing)} rows not in filtered pool")
        if all_ids.intersection(selected_ids):
            errors.append(f"{category}: cross-category duplicate IDs")
        all_ids.update(selected_ids)
        selected_tokens = sum(row["kcenter"]["supervised_tokens"] for row in selected)
        expected_count = sum(item["selected_count"] for item in summary.values())
        expected_tokens = sum(item["selected_tokens"] for item in summary.values())
        if len(selected) != expected_count or selected_tokens != expected_tokens:
            errors.append(f"{category}: output and summary mismatch")
        for row in selected:
            digest = hashlib.sha256(
                json.dumps(
                    row["payload"],
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if digest in content_hashes:
                errors.append(f"{category}: duplicate payload {row_id(row)}")
            content_hashes.add(digest)
        category_summary[category] = {
            "candidates": len(candidate_ids),
            "selected": len(selected),
            "selected_tokens": selected_tokens,
        }
        for bucket, item in sorted(summary.items()):
            budget = item["token_budget"]
            if (
                budget is not None
                and item["selected_count"] < item["candidate_count"]
                and item["selected_tokens"] < budget
            ):
                errors.append(f"{bucket}: token budget not reached")
            bucket_rows.append((category, bucket, item))

    selected_count = sum(item["selected"] for item in category_summary.values())
    selected_tokens = sum(item["selected_tokens"] for item in category_summary.values())
    if selected_count != EDUCATION_ROWS:
        errors.append(
            f"education-specific rows {selected_count} != paper {EDUCATION_ROWS}"
        )
    if selected_tokens != EDUCATION_TOKENS:
        errors.append(
            f"education-specific tokens {selected_tokens} != paper {EDUCATION_TOKENS}"
        )
    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "categories": category_summary,
        "selected_count": selected_count,
        "selected_tokens": selected_tokens,
        "paper_education_rows": EDUCATION_ROWS,
        "paper_education_tokens": EDUCATION_TOKENS,
        "general_rows": GENERAL_ROWS,
        "general_tokens": GENERAL_TOKENS,
        "combined_rows": selected_count + GENERAL_ROWS,
        "combined_tokens": selected_tokens + GENERAL_TOKENS,
        "paper_combined_rows": COMBINED_ROWS,
        "paper_combined_tokens": COMBINED_TOKENS,
    }
    (FINAL / "validation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Bucket-level K-Center final statistics",
        "",
        f"- status: **{result['status'].upper()}**",
        f"- education-specific: {selected_count:,} examples / {selected_tokens:,} supervised tokens (paper: {EDUCATION_ROWS:,} / {EDUCATION_TOKENS:,})",
        f"- after merging the frozen general-purpose examples: {result['combined_rows']:,} / {result['combined_tokens']:,} supervised tokens (paper: {COMBINED_ROWS:,} / {COMBINED_TOKENS:,})",
        f"- duplicate ids, cross-category payload duplicates, candidate-subset and token-budget errors: {len(errors)}",
        "",
        "## Categories",
        "",
        "| category | filtered candidates | k-center selected | supervised tokens |",
        "|---|---:|---:|---:|",
    ]
    for category, item in category_summary.items():
        lines.append(
            f"| {category} | {item['candidates']:,} | {item['selected']:,} | "
            f"{item['selected_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "## Buckets",
            "",
            "| category | bucket | candidates | selected | token budget | actual tokens |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for category, bucket, item in bucket_rows:
        budget = "all" if item["token_budget"] is None else f"{item['token_budget']:,}"
        lines.append(
            f"| {category} | {bucket} | {item['candidate_count']:,} | "
            f"{item['selected_count']:,} | {budget} | {item['selected_tokens']:,} |"
        )
    if errors:
        lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])
    (FINAL / "KCenter_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
