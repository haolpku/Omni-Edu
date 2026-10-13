#!/usr/bin/env python3
"""Stage-1 deduplication for locally available curriculum-grounding data.

The pass is deterministic and no-delete:

* curriculum items: normalized exact content + image identity
* official evaluation data: indexed before training data to prevent leakage
* same-content annotations: retained on the canonical record as provenance
* standards, graph entities/edges, and course pages: family-specific exact keys
* all local raw assets: byte-level SHA-256 inventory and duplicate routing

The resulting curriculum-item records are still pre-SFT artifacts. Their
``_stage1.merged_groundings`` field is authoritative; messages are re-rendered
only in the later rewrite stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

from build_structure_sft import (
    format_tal_options,
    load_standard_catalog,
    make_kp_record,
    make_mathfish_record,
    normalize_paths,
    read_jsonl,
)


ROOT = Path("data_preparation/stage1_capability_taxonomy_and_source_collection/curriculum_grounding")
PIPE = Path("data_preparation/stage2_deterministic_cleaning_and_evaluation_decontamination/curriculum_grounding")
DATE = "2026-08-06"
METHOD = "normalized_exact_content+image_identity"
KEEP_SYMBOLS = set("<>=+-*/^\\{}[]|")

TRAIN_INPUTS = OrderedDict(
    [
        ("DA-20K", PIPE / "DA-20K/origin/origin.jsonl"),
        ("tal-scq5k-cn", PIPE / "tal-scq5k-cn/origin/origin.jsonl"),
        ("tal-scq5k-en", PIPE / "tal-scq5k-en/origin/origin.jsonl"),
        ("XES3G5M", PIPE / "XES3G5M/origin/origin.jsonl"),
        ("mathfish", PIPE / "mathfish/origin/origin.jsonl"),
    ]
)

SOURCE_PRIORITY = {
    "XES3G5M": 50,
    "DA-20K": 40,
    "mathfish": 30,
    "TAL-SCQ5K-CN": 20,
    "TAL-SCQ5K-EN": 20,
}
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}
GENERATED_SPLIT_DATASETS = {"DA-20K", "XES3G5M"}

TAL_ROOT = ROOT / "huggingface/TAL-SCQ5K"
TAL_SAQ_ROOT = ROOT / "github/TAL-SAQ"
MATHFISH_ROOT = ROOT / "huggingface/mathfish"
ACHIEVE_PATH = ROOT / "huggingface/achieve-the-core/standards.jsonl"
JUNYI_ROOT = ROOT / "direct/Junyi_preprocessed"
EDUKG_ROOT = ROOT / "direct/EDUKG"
OPENSCIED_ROOT = ROOT / "direct/OpenSciEd"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("$", "")
    return "".join(
        char for char in text if char.isalnum() or char in KEEP_SYMBOLS
    )


def prompt_body(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    content = str(messages[0].get("content") or "") if messages else ""
    return content.split("\n\n", 1)[1] if "\n\n" in content else content


def image_identity(record: dict[str, Any]) -> str:
    refs = record.get("image_ref") or []
    if not refs:
        return ""
    identities = []
    for ref in refs:
        archive = Path(str(ref.get("archive") or "")).name
        member = str(ref.get("member") or ref.get("filename") or "")
        identities.append(f"{archive}::{member}")
    return "|".join(sorted(identities))


def content_key(record: dict[str, Any]) -> str:
    text_key = normalize_text(prompt_body(record))
    # Very short generic strings are unsafe to deduplicate without answer
    # confirmation. Salt them by source id so they survive for Stage-2 review.
    if len(text_key) < 16:
        source_id = str((record.get("meta") or {}).get("source_id") or record.get("uid"))
        text_key = f"{text_key}::short::{source_id}"
    images = image_identity(record)
    return f"{text_key}::images::{images}" if images else text_key


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def content_stable_split(key: str) -> str:
    value = int(key_digest(key)[:8], 16) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "validation"
    return "test"


def source_name(record: dict[str, Any]) -> str:
    return str((record.get("meta") or {}).get("source_dataset") or "unknown")


def source_split(record: dict[str, Any]) -> str:
    return str((record.get("meta") or {}).get("split") or "train")


def source_slug(record: dict[str, Any]) -> str:
    name = source_name(record)
    mapping = {
        "DA-20K": "DA-20K",
        "TAL-SCQ5K-CN": "tal-scq5k-cn",
        "TAL-SCQ5K-EN": "tal-scq5k-en",
        "XES3G5M": "XES3G5M",
        "mathfish": "mathfish",
    }
    return mapping.get(name, re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-"))


def grounding_envelope(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": record.get("uid"),
        "source_dataset": source_name(record),
        "source_id": (record.get("meta") or {}).get("source_id"),
        "split": source_split(record),
        "grounding": record.get("grounding") or {},
    }


def grounding_signature(record: dict[str, Any]) -> str:
    return stable_json((record.get("grounding") or {}))


def canonical_score(record: dict[str, Any]) -> tuple[int, int, int, int, str]:
    grounding = record.get("grounding") or {}
    labels = grounding.get("knowledge_paths") or grounding.get("standard_alignments") or []
    split = source_split(record)
    return (
        SPLIT_PRIORITY.get(split, 0),
        SOURCE_PRIORITY.get(source_name(record), 0),
        len(labels),
        len(prompt_body(record)),
        str(record.get("uid") or ""),
    )


def attach_stage1(
    canonical: dict[str, Any],
    group: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    result = dict(canonical)
    envelopes: list[dict[str, Any]] = []
    seen_groundings: set[str] = set()
    for record in sorted(group, key=lambda row: str(row.get("uid") or "")):
        envelope = grounding_envelope(record)
        signature = stable_json(envelope)
        if signature not in seen_groundings:
            seen_groundings.add(signature)
            envelopes.append(envelope)
    result["_stage1"] = {
        "dedup_stage": "stage1",
        "dedup_date": DATE,
        "dedup_method": METHOD,
        "content_key_sha256": key_digest(key),
        "canonical_uid": canonical.get("uid"),
        "source_uids": [record.get("uid") for record in group],
        "merged_groundings": envelopes,
        "requires_sft_rerender": len(envelopes) > 1,
    }
    return result


def route_duplicate(
    record: dict[str, Any],
    canonical: dict[str, Any],
    reason: str,
    key: str,
) -> dict[str, Any]:
    routed = dict(record)
    routed.update(
        _reason=reason,
        _kept_uid=canonical.get("uid"),
        _kept_in=source_name(canonical),
        _dedup_stage="stage1",
        _dedup_date=DATE,
        _dedup_method=METHOD,
        _content_key_sha256=key_digest(key),
    )
    return routed


def group_records(
    rows: Iterable[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        groups.setdefault(content_key(row), []).append(row)
    return groups


def load_train_rows() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for slug, path in TRAIN_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run build_structure_sft.py once to create layer0"
            )
        result[slug] = list(read_jsonl(path))
    return result


def load_tal_eval(language: str) -> list[dict[str, Any]]:
    suffix = "CN" if language == "zh" else "EN"
    dataset = f"TAL-SCQ5K-{suffix}"
    path = TAL_ROOT / dataset / "test.jsonl"
    result = []
    for raw in read_jsonl(path):
        source_id = str(raw.get("queId") or raw.get("qid") or "")
        record = make_kp_record(
            uid=f"tal-scq5k-{suffix.casefold()}-test::{source_id}",
            source_dataset=dataset,
            source_id=source_id,
            question=str(raw.get("problem") or ""),
            options=format_tal_options(raw.get("answer_option_list")),
            paths=normalize_paths(raw.get("knowledge_point_routes")),
            language=language,
            split="official_test",
            license_tag="MIT",
            source_meta={
                "difficulty": raw.get("difficulty"),
                "question_type": raw.get("qtype"),
                "reserved_for_evaluation": True,
            },
        )
        result.append(record)
    return result


def load_tal_saq_eval(language: str) -> list[dict[str, Any]]:
    suffix = "CN" if language == "zh" else "EN"
    filename = "TAL-SAQ7K-CN.jsonl" if language == "zh" else "TAL-SAQ6K-EN.jsonl"
    dataset = f"TAL-SAQ-{'7K-CN' if language == 'zh' else '6K-EN'}"
    result = []
    for raw in read_jsonl(TAL_SAQ_ROOT / filename):
        source_id = str(raw.get("queId") or "")
        record = make_kp_record(
            uid=f"{dataset.casefold()}::{source_id}",
            source_dataset=dataset,
            source_id=source_id,
            question=str(raw.get("problem") or ""),
            options=[],
            paths=normalize_paths(raw.get("knowledge_point_routes")),
            language=language,
            split="official_test",
            license_tag="MIT",
            source_meta={
                "difficulty": raw.get("difficulty"),
                "question_type": raw.get("qtype"),
                "reserved_for_evaluation": True,
            },
        )
        result.append(record)
    return result


def load_mathfish_eval(split: str) -> list[dict[str, Any]]:
    catalog = load_standard_catalog()
    result = []
    for raw in read_jsonl(MATHFISH_ROOT / f"{split}.jsonl"):
        record = make_mathfish_record(raw, catalog)
        source_id = str((record.get("meta") or {}).get("source_id") or "")
        record["uid"] = f"mathfish-{split}::{source_id}"
        record["meta"] = dict(record.get("meta") or {})
        record["meta"]["split"] = (
            "official_validation" if split == "dev" else "official_test"
        )
        record["meta"]["reserved_for_evaluation"] = True
        result.append(record)
    return result


def load_eval_rows() -> dict[str, list[dict[str, Any]]]:
    return OrderedDict(
        [
            ("tal-scq5k-cn-test", load_tal_eval("zh")),
            ("tal-scq5k-en-test", load_tal_eval("en")),
            ("tal-saq7k-cn-test", load_tal_saq_eval("zh")),
            ("tal-saq6k-en-test", load_tal_saq_eval("en")),
            ("mathfish-dev", load_mathfish_eval("dev")),
            ("mathfish-test", load_mathfish_eval("test")),
        ]
    )


def dedup_curriculum_items() -> dict[str, Any]:
    train_by_slug = load_train_rows()
    eval_by_name = load_eval_rows()
    output_root = PIPE / "stage1_eval_index"

    eval_rows = [row for rows in eval_by_name.values() for row in rows]
    eval_groups = group_records(eval_rows)
    eval_index = {
        key: [
            {
                "uid": row.get("uid"),
                "source_dataset": source_name(row),
                "split": source_split(row),
            }
            for row in group
        ]
        for key, group in eval_groups.items()
    }
    eval_collision_groups = [
        {
            "content_key_sha256": key_digest(key),
            "member_count": len(group),
            "members": eval_index[key],
        }
        for key, group in eval_groups.items()
        if len(group) > 1
    ]
    write_jsonl(output_root / "reserved_records.jsonl", eval_rows)
    write_jsonl(output_root / "duplicate_groups.jsonl", eval_collision_groups)

    intra_kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intra_removed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_variants: Counter[str] = Counter()

    for slug, rows in train_by_slug.items():
        for key, group in group_records(rows).items():
            canonical = max(group, key=canonical_score)
            if source_name(canonical) in GENERATED_SPLIT_DATASETS:
                canonical["meta"] = dict(canonical.get("meta") or {})
                canonical["meta"]["split"] = content_stable_split(key)
                canonical["meta"]["split_basis"] = "stage1_content_key"
            intra_kept[slug].append(attach_stage1(canonical, group, key))
            signatures = {grounding_signature(record) for record in group}
            if len(signatures) > 1:
                annotation_variants[slug] += len(group) - 1
            for record in group:
                if record is canonical:
                    continue
                reason = (
                    "same_content_additional_grounding"
                    if grounding_signature(record) != grounding_signature(canonical)
                    else "intra_dataset_duplicate"
                )
                intra_removed[slug].append(
                    route_duplicate(record, canonical, reason, key)
                )

    after_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eval_removed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slug, rows in intra_kept.items():
        for row in rows:
            key = content_key(row)
            matches = eval_index.get(key)
            if not matches:
                after_eval[slug].append(row)
                continue
            routed = dict(row)
            routed.update(
                _reason="official_eval_overlap",
                _eval_matches=matches,
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method=METHOD,
                _content_key_sha256=key_digest(key),
            )
            eval_removed[slug].append(routed)

    global_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for rows in after_eval.values():
        for row in rows:
            global_groups.setdefault(content_key(row), []).append(row)

    final_kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_removed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cross_groups = []
    for key, group in global_groups.items():
        canonical = max(group, key=canonical_score)
        # All members of a content group must stay in the most protected split.
        protected_split = max(
            (source_split(row) for row in group),
            key=lambda split: SPLIT_PRIORITY.get(split, 0),
        )
        canonical["meta"] = dict(canonical.get("meta") or {})
        canonical["meta"]["split"] = protected_split
        canonical["meta"]["split_basis"] = "stage1_cross_dataset_content_group"
        canonical = attach_stage1(canonical, group, key)
        canonical_slug = source_slug(canonical)
        final_kept[canonical_slug].append(canonical)
        owners = {source_name(row) for row in group}
        if len(owners) > 1:
            cross_groups.append(
                {
                    "content_key_sha256": key_digest(key),
                    "canonical_uid": canonical.get("uid"),
                    "canonical_dataset": source_name(canonical),
                    "split": protected_split,
                    "members": [
                        {
                            "uid": row.get("uid"),
                            "source_dataset": source_name(row),
                            "split": source_split(row),
                        }
                        for row in group
                    ],
                }
            )
        for row in group:
            if row.get("uid") == canonical.get("uid"):
                continue
            cross_removed[source_slug(row)].append(
                route_duplicate(
                    row,
                    canonical,
                    "cross_dataset_duplicate_merged_grounding",
                    key,
                )
            )

    write_jsonl(PIPE / "stage1_cross_dataset_groups.jsonl", cross_groups)
    datasets_report: dict[str, Any] = {}
    for slug in TRAIN_INPUTS:
        out = PIPE / slug / "cleaned"
        write_jsonl(out / "kept.jsonl", final_kept.get(slug, []))
        write_jsonl(
            out / "removed_intra_duplicates.jsonl", intra_removed.get(slug, [])
        )
        write_jsonl(
            out / "removed_official_eval_overlap.jsonl", eval_removed.get(slug, [])
        )
        write_jsonl(
            out / "removed_cross_duplicates.jsonl", cross_removed.get(slug, [])
        )
        datasets_report[slug] = {
            "origin": len(train_by_slug[slug]),
            "after_intra": len(intra_kept.get(slug, [])),
            "removed_intra": len(intra_removed.get(slug, [])),
            "additional_grounding_variants_merged": annotation_variants[slug],
            "removed_official_eval_overlap": len(eval_removed.get(slug, [])),
            "removed_cross_dataset": len(cross_removed.get(slug, [])),
            "final_kept": len(final_kept.get(slug, [])),
        }

    return {
        "datasets": datasets_report,
        "official_eval": {
            "records": len(eval_rows),
            "unique_content_keys": len(eval_groups),
            "duplicate_content_groups": len(eval_collision_groups),
            "sources": {name: len(rows) for name, rows in eval_by_name.items()},
        },
        "cross_dataset_duplicate_groups": len(cross_groups),
    }


def dedup_achieve() -> dict[str, Any]:
    rows = list(read_jsonl(ACHIEVE_PATH))
    by_id: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    by_description: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id.setdefault(normalize_text(row.get("id")), []).append(row)
        description_key = normalize_text(row.get("description"))
        if description_key:
            by_description[description_key].append(row)

    kept = []
    removed = []
    for key, group in by_id.items():
        canonical = group[0]
        kept.append(canonical)
        for row in group[1:]:
            routed = dict(row)
            routed.update(
                _reason="duplicate_standard_id",
                _kept_standard_id=canonical.get("id"),
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="normalized_exact_standard_id",
            )
            removed.append(routed)
    same_description = [
        {
            "description_key_sha256": key_digest(key),
            "standard_ids": sorted({str(row.get("id")) for row in group}),
            "description": group[0].get("description"),
        }
        for key, group in by_description.items()
        if len({str(row.get("id")) for row in group}) > 1
    ]
    out = PIPE / "achieve-the-core/cleaned"
    write_jsonl(out / "kept.jsonl", kept)
    write_jsonl(out / "removed_duplicate_ids.jsonl", removed)
    write_jsonl(out / "same_description_distinct_ids.jsonl", same_description)
    return {
        "origin": len(rows),
        "kept": len(kept),
        "removed_duplicate_ids": len(removed),
        "same_description_distinct_id_groups_kept": len(same_description),
    }


def dedup_junyi() -> dict[str, Any]:
    node_rows = []
    id_to_slug: dict[int, str] = {}
    with (JUNYI_ROOT / "vertex_id2idx").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            slug, raw_index = line.rstrip("\n").rsplit(",", 1)
            index = int(raw_index)
            id_to_slug[index] = slug
            node_rows.append({"node_id": index, "slug": slug})

    out = PIPE / "junyi-graph/cleaned"
    write_jsonl(out / "kept_nodes.jsonl", node_rows)
    report: dict[str, Any] = {"nodes": len(node_rows)}
    for relation, filename in [
        ("prerequisite_for", "prerequisite.json"),
        ("similar_to", "similarity.json"),
    ]:
        raw_edges = json.loads((JUNYI_ROOT / filename).read_text(encoding="utf-8"))
        seen: set[tuple[Any, ...]] = set()
        kept = []
        removed = []
        self_loops = []
        for index, raw in enumerate(raw_edges):
            if not isinstance(raw, list) or len(raw) < 2:
                edge = {
                    "source_index": index,
                    "raw": raw,
                    "_reason": "malformed_graph_edge",
                }
                removed.append(edge)
                continue
            source, target = int(raw[0]), int(raw[1])
            extras = raw[2:]
            key = (source, target, *extras)
            edge = {
                "source": source,
                "source_slug": id_to_slug.get(source),
                "target": target,
                "target_slug": id_to_slug.get(target),
                "relation": relation,
                "attributes": extras,
            }
            if source == target:
                self_loops.append(edge)
            if key in seen:
                edge.update(
                    _reason="duplicate_graph_edge",
                    _dedup_stage="stage1",
                    _dedup_date=DATE,
                    _dedup_method="exact_ordered_edge",
                )
                removed.append(edge)
            else:
                seen.add(key)
                kept.append(edge)
        stem = relation.replace("_for", "")
        write_jsonl(out / f"kept_{stem}_edges.jsonl", kept)
        write_jsonl(out / f"removed_{stem}_edges.jsonl", removed)
        write_jsonl(out / f"routed_{stem}_self_loops.jsonl", self_loops)
        report[relation] = {
            "origin": len(raw_edges),
            "kept": len(kept),
            "removed_exact": len(removed),
            "self_loops_kept_for_review": len(self_loops),
        }
    return report


TURTLE_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')


def decode_turtle_string(value: str) -> str:
    try:
        return str(json.loads(value))
    except (json.JSONDecodeError, TypeError):
        return value[1:-1]


def iter_turtle_blocks(path: Path) -> Iterator[str]:
    current: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                if current:
                    yield "".join(current)
                    current = []
                continue
            current.append(line)
    if current:
        yield "".join(current)


def parse_edukg_entity(block: str, source_file: str) -> dict[str, Any] | None:
    identifier = re.match(r"\s*<([^>]+)>", block)
    if not identifier:
        return None
    entity_id = identifier.group(1)
    entity_type_match = re.search(r"\s+a\s+<([^>]+)>", block)
    label_match = re.search(r"rdfs:label\s+(\"(?:\\.|[^\"\\])*\")", block)
    label = decode_turtle_string(label_match.group(1)) if label_match else ""
    locations = []
    temp_match = re.search(r"ns1:temp\s+(.+?)\s+\.\s*$", block, re.S)
    if temp_match:
        for encoded in TURTLE_STRING_RE.findall(temp_match.group(1)):
            decoded = decode_turtle_string(encoded)
            try:
                payload = json.loads(decoded)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                locations.append(payload)
    return {
        "entity_id": entity_id,
        "entity_type": entity_type_match.group(1) if entity_type_match else "",
        "label": label,
        "locations": locations,
        "source_file": source_file,
    }


def dedup_edukg() -> dict[str, Any]:
    rows = []
    for filename in ["main.ttl", "material.ttl"]:
        path = EDUKG_ROOT / filename
        rows.extend(
            row
            for block in iter_turtle_blocks(path)
            if (row := parse_edukg_entity(block, filename)) is not None
        )
    by_id: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id.setdefault(row["entity_id"], []).append(row)
        label_key = normalize_text(row.get("label"))
        if label_key:
            labels[label_key].append(row)
    kept = []
    removed = []
    for entity_id, group in by_id.items():
        canonical = group[0]
        kept.append(canonical)
        for row in group[1:]:
            routed = dict(row)
            routed.update(
                _reason="duplicate_entity_id",
                _kept_entity_id=entity_id,
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="exact_entity_uri",
            )
            removed.append(routed)
    label_collisions = [
        {
            "label": group[0].get("label"),
            "entity_ids": sorted({row["entity_id"] for row in group}),
            "source_files": sorted({row["source_file"] for row in group}),
        }
        for group in labels.values()
        if len({row["entity_id"] for row in group}) > 1
    ]
    out = PIPE / "edukg/cleaned"
    write_jsonl(out / "kept_entities.jsonl", kept)
    write_jsonl(out / "removed_duplicate_entity_ids.jsonl", removed)
    write_jsonl(out / "same_label_distinct_entities.jsonl", label_collisions)
    return {
        "origin_entities": len(rows),
        "kept_entities": len(kept),
        "removed_duplicate_entity_ids": len(removed),
        "same_label_distinct_entity_groups_kept": len(label_collisions),
    }


class VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in {"script", "style", "svg", "nav", "header", "footer"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "nav", "header", "footer"}:
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


def dedup_openscied() -> dict[str, Any]:
    rows = []
    for path in sorted(OPENSCIED_ROOT.rglob("*.html")):
        parser = VisibleTextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(parser.parts)
        rows.append(
            {
                "uid": f"openscied::{path.relative_to(OPENSCIED_ROOT)}",
                "relative_path": str(path.relative_to(OPENSCIED_ROOT)),
                "title": parser.parts[0] if parser.parts else "",
                "visible_text": text,
            }
        )
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = normalize_text(row["visible_text"])
        if not key:
            key = f"empty::{row['uid']}"
        groups.setdefault(key, []).append(row)
    kept = []
    removed = []
    for key, group in groups.items():
        canonical = group[0]
        kept.append(canonical)
        for row in group[1:]:
            routed = dict(row)
            routed.update(
                _reason="duplicate_course_page",
                _kept_uid=canonical["uid"],
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="normalized_exact_visible_html_text",
                _content_key_sha256=key_digest(key),
            )
            removed.append(routed)
    out = PIPE / "openscied/cleaned"
    write_jsonl(out / "kept_pages.jsonl", kept)
    write_jsonl(out / "removed_duplicate_pages.jsonl", removed)
    return {
        "origin_pages": len(rows),
        "kept_pages": len(kept),
        "removed_duplicate_pages": len(removed),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_raw_files() -> Iterator[Path]:
    excluded_dirs = {".git", ".cache", "__pycache__", "_pipeline"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        if path.suffix in {".partial", ".part", ".tmp"}:
            continue
        yield path


def audit_raw_files() -> dict[str, Any]:
    inventory = []
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(iter_raw_files()):
        relative = path.relative_to(ROOT)
        digest = file_sha256(path)
        row = {
            "relative_path": str(relative),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "source_area": relative.parts[0] if relative.parts else "",
            "source_dataset": relative.parts[1] if len(relative.parts) > 1 else "",
        }
        inventory.append(row)
        by_hash[digest].append(row)
    duplicate_groups = []
    routed = []
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        canonical = min(group, key=lambda row: row["relative_path"])
        duplicate_groups.append(
            {
                "sha256": digest,
                "size_bytes": canonical["size_bytes"],
                "canonical_path": canonical["relative_path"],
                "paths": [row["relative_path"] for row in group],
            }
        )
        for row in group:
            if row is canonical:
                continue
            routed.append(
                {
                    **row,
                    "_reason": "exact_file_duplicate",
                    "_kept_path": canonical["relative_path"],
                    "_dedup_stage": "stage1",
                    "_dedup_date": DATE,
                    "_dedup_method": "sha256_bytes",
                }
            )
    out = PIPE / "stage1_raw_files"
    write_jsonl(out / "file_inventory.jsonl", inventory)
    write_jsonl(out / "duplicate_groups.jsonl", duplicate_groups)
    write_jsonl(out / "routed_exact_file_duplicates.jsonl", routed)
    return {
        "files_hashed": len(inventory),
        "bytes_hashed": sum(row["size_bytes"] for row in inventory),
        "exact_duplicate_groups": len(duplicate_groups),
        "duplicate_files_routed": len(routed),
    }


def write_scope() -> None:
    scope = {
        "date": DATE,
        "record_level_included": {
            "curriculum_items": [
                "DA-20K",
                "TAL-SCQ5K-CN/EN train",
                "XES3G5M question metadata",
                "MathFish train",
            ],
            "official_eval_leakage_index": [
                "TAL-SCQ5K-CN/EN test",
                "TAL-SAQ7K-CN",
                "TAL-SAQ6K-EN",
                "MathFish dev/test",
            ],
            "reference_and_graph": [
                "Achieve-the-Core standards",
                "EDUKG main/material entities",
                "Junyi prerequisite/similarity graph",
                "OpenSciEd downloaded unit pages",
            ],
        },
        "file_level_only": [
            "downloaded curriculum-standard PDFs and HTML landing pages",
            "TIMSS/PIRLS/Eedi/ASSISTments archives",
            "SLP tables",
            "all other local non-generated assets",
        ],
        "excluded_from_record_level": {
            "K12-KGraph": "user-owned and already processed; do not re-clean",
            "FoundationalASSIST": "gated/unavailable locally",
            "ASSISTments_old_releases": "no problem stems; diagnostic/KT work postponed",
            "Eedi_and_SLP_interactions": "student-event data; not direct curriculum SFT",
            "TIMSS_2023": "large assessment microdata archives; no normalized item adapter yet",
            "PIRLS_2021": "partial download",
            "Smarter_Balanced": "only landing-page HTML is local; linked specifications absent",
            "MathFish_tasks": "prompt-expanded derivative of MathFish dev; evaluation-only",
        },
    }
    path = PIPE / "STAGE1_SCOPE.json"
    path.write_text(
        json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_markdown_report(report: dict[str, Any]) -> None:
    items = report["curriculum_items"]
    lines = [
        "# Curriculum Stage-1 Dedup Report",
        "",
        f"- Date: {DATE}",
        f"- Method: `{METHOD}`; deterministic and no-delete.",
        "- Official evaluation content is indexed before training content.",
        "- Multimodal identity includes archive/member references; text-only matching is not used to collapse different diagrams.",
        "- Same-content labels are preserved in `_stage1.merged_groundings`; later SFT messages must be re-rendered.",
        "",
        "## Curriculum items",
    ]
    for slug, values in items["datasets"].items():
        lines.append(
            "- "
            f"`{slug}`: origin {values['origin']}; "
            f"intra removed {values['removed_intra']}; "
            f"eval overlap routed {values['removed_official_eval_overlap']}; "
            f"cross-dataset routed {values['removed_cross_dataset']}; "
            f"final kept {values['final_kept']}."
        )
    eval_report = items["official_eval"]
    lines.extend(
        [
            "",
            "## Evaluation isolation",
            (
                f"- Reserved records: {eval_report['records']}; unique content keys: "
                f"{eval_report['unique_content_keys']}; duplicate groups inside/across "
                f"evaluation sources: {eval_report['duplicate_content_groups']}."
            ),
            "",
            "## Other structural resources",
            f"- Achieve-the-Core: {stable_json(report['achieve_the_core'])}",
            f"- EDUKG: {stable_json(report['edukg'])}",
            f"- Junyi graph: {stable_json(report['junyi'])}",
            f"- OpenSciEd pages: {stable_json(report['openscied'])}",
        ]
    )
    if "raw_files" in report:
        raw = report["raw_files"]
        lines.extend(
            [
                "",
                "## Raw-file audit",
                (
                    f"- Hashed {raw['files_hashed']} files / {raw['bytes_hashed']} bytes; "
                    f"found {raw['exact_duplicate_groups']} exact duplicate groups and "
                    f"routed {raw['duplicate_files_routed']} non-canonical file entries."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "- Record-level scope and exclusions are machine-readable in `STAGE1_SCOPE.json`.",
            "- No Stage-2 quality/K12 relevance filtering or Stage-3 SFT rewriting is performed here.",
            "",
        ]
    )
    (PIPE / "STAGE1_DEDUP_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-file-audit",
        action="store_true",
        help="Skip byte-level hashing of every local raw asset.",
    )
    args = parser.parse_args()

    write_scope()
    report = {
        "date": DATE,
        "method": METHOD,
        "curriculum_items": dedup_curriculum_items(),
        "achieve_the_core": dedup_achieve(),
        "junyi": dedup_junyi(),
        "edukg": dedup_edukg(),
        "openscied": dedup_openscied(),
    }
    if not args.skip_file_audit:
        report["raw_files"] = audit_raw_files()
    report_path = PIPE / "STAGE1_DEDUP_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
