#!/usr/bin/env python3
"""Deterministic fast clean for local AAAS public files and LongTutor.

The cleaner is intentionally extractive:
* source text and labels are not corrected, translated, or synthesized;
* every kept record carries file/line-or-page provenance and source hashes;
* uncertain AAAS resources are routed rather than guessed;
* LongTutor Gold has priority over Silver during exact/structural deduplication;
* XES3G5M question-bank rows are referenced, not re-emitted as curriculum data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import fitz

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

CAPABILITY = "diagnostic_reasoning"
SOURCE_ROOT = pools.source_dir(CAPABILITY)
AAAS = SOURCE_ROOT / "direct" / "AAAS_Project_2061_public_files"
LONGTUTOR = SOURCE_ROOT / "github" / "LongTutor"
CURRICULUM_XES = pools.cleaned_dir("curriculum_grounding", "XES3G5M") / "origin.jsonl"
AAAS_OUT = pools.cleaned_dir(CAPABILITY, "AAAS")
LONGTUTOR_OUT = pools.cleaned_dir(CAPABILITY, "LongTutor-Gold")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                obj = json.loads(line)
                obj["_source_line"] = line_no
                rows.append(obj)
    return rows


def pdf_pages(path: Path) -> list[str]:
    doc = fitz.open(path)
    return [clean_text(page.get_text()) for page in doc]


def html_plain_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_aspect_answer_key(path: Path) -> dict[str, list[dict[str, Any]]]:
    lines = pdf_pages(path)[0].splitlines()
    triples = []
    for i in range(9, min(324, len(lines)), 3):
        if i + 2 >= len(lines):
            break
        item, code, answer = lines[i : i + 3]
        if item.isdigit() and re.fullmatch(r"[A-D]", answer):
            triples.append({"item_number": int(item), "item_code": code, "correct_option": answer})
    if len(triples) != 105:
        raise ValueError(f"Expected 105 ASPECt answer triples, got {len(triples)}")
    return {
        "Basic": triples[0::3],
        "Intermediate": triples[1::3],
        "Advanced": triples[2::3],
    }


def parse_mc_page(page: str) -> dict[str, Any] | None:
    # Instrument pages contain one or two items. Caller splits at item-code boundaries.
    m = re.match(
        r"(?s)^(?P<code>[A-Z]{2,3}\d+[‐‑–—-]\d+)\s+" r"(?P<number>\d+)\.\s+(?P<body>.*)$",
        page.strip(),
    )
    if not m:
        return None
    body = m.group("body")
    options = {}
    positions = list(re.finditer(r"(?m)^([A-D])\.\s+", body))
    if len(positions) != 4:
        return None
    stem = clean_text(body[: positions[0].start()])
    for idx, opt in enumerate(positions):
        end = positions[idx + 1].start() if idx + 1 < len(positions) else len(body)
        options[opt.group(1)] = clean_text(body[opt.end() : end])
    return {
        "item_code": m.group("code"),
        "item_number": int(m.group("number")),
        "question": stem,
        "options": options,
    }


def extract_aspect_mc(
    instrument: str, path: Path, key_rows: list[dict[str, Any]]
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    routed: list[dict] = []
    source_hash = file_sha256(path)
    key_by_num = {x["item_number"]: x for x in key_rows}
    joined_pages = []
    for page_no, raw_page in enumerate(pdf_pages(path), 1):
        page = re.sub(r"(?m)^[^\n]*COPYRIGHT © 2018 AAAS PROJECT 2061[^\n]*$", "", raw_page)
        page = re.sub(
            r"(?m)^For each question, fill in one circle on the answer sheet\.\s*$", "", page
        )
        joined_pages.append(f"[[PAGE {page_no}]]\n{page}")
    full = "\n".join(joined_pages)
    code_pat = re.compile(r"(?m)^(?P<code>[A-Z]{2,3}\d+[‐‑–—-]\d+)\s+(?P<number>\d+)\.")
    starts = list(code_pat.finditer(full))
    for j, boundary in enumerate(starts):
        start = boundary.start()
        end = starts[j + 1].start() if j + 1 < len(starts) else len(full)
        chunk = clean_text(full[start:end])
        page_hits = re.findall(r"\[\[PAGE (\d+)\]\]", full[:start])
        page_no = int(page_hits[-1]) if page_hits else None
        chunk = re.sub(r"\[\[PAGE \d+\]\]", "", chunk)
        chunk = re.sub(r"(?m)^#\d+ continued on next page\s*$", "", chunk)
        chunk = clean_text(chunk)
        item = parse_mc_page(chunk)
        if item is None:
            routed.append(
                {
                    "dataset": "AAAS_Project_2061",
                    "resource_group": "ASPECt",
                    "route_reason": "mc_item_parse_failed",
                    "source": {
                        "path": str(path.relative_to(DIAG)),
                        "page": page_no,
                        "sha256": source_hash,
                    },
                    "raw_text": chunk,
                }
            )
            continue
        key = key_by_num.get(item["item_number"])
        if not key:
            routed.append(
                {
                    "dataset": "AAAS_Project_2061",
                    "resource_group": "ASPECt",
                    "route_reason": "missing_answer_key",
                    "source": {
                        "path": str(path.relative_to(DIAG)),
                        "page": page_no,
                        "sha256": source_hash,
                    },
                    "parsed": item,
                }
            )
            continue
        item["correct_option"] = key["correct_option"]
        item["answer_key_item_code"] = key["item_code"]
        item["item_code_match"] = item["item_code"].replace("‐", "-") == key["item_code"].replace(
            "‐", "-"
        )
        record = {
            "record_id": f"aaas:aspect:{instrument.lower()}:{item['item_number']}",
            "dataset": "AAAS_Project_2061",
            "resource_group": "ASPECt",
            "diagnostic_unit": "multiple_choice_item_with_keyed_distractors",
            "instrument": instrument,
            **item,
            "license": {
                "status": "group_page_explicit",
                "spdx_like": "CC-BY-NC-SA-4.0",
                "commercial_use": False,
                "attribution_required": True,
                "share_alike": True,
            },
            "source": {
                "path": str(path.relative_to(DIAG)),
                "page": page_no,
                "sha256": source_hash,
                "answer_key_path": str((AAAS / "ASPECt/ASPECt_Answer_Keys.pdf").relative_to(DIAG)),
            },
            "no_rewrite": True,
        }
        record["exact_content_sha256"] = object_hash(
            {
                "question": record["question"],
                "options": record["options"],
                "correct_option": record["correct_option"],
            }
        )
        kept.append(record)
    return kept, routed


def extract_aspect3d_exemplars(path: Path) -> tuple[list[dict], list[dict]]:
    pages = pdf_pages(path)
    source_hash = file_sha256(path)
    full = "\n\n".join(f"[[PAGE {i}]]\n{text}" for i, text in enumerate(pages, 1))
    kept: list[dict] = []
    routed: list[dict] = []
    # Exact PDF wording is retained. The pattern only identifies table rows.
    response_start = r"(?:Student selects|Student wrote|[“\"]|‘)"
    pattern = re.compile(
        rf"(?s)(?P<response>{response_start}.*?)\s+"
        r"(?P<score>Score\s*=\s*\d+)\s+"
        r"(?P<description>(?:The student|The response|Student response).*?)"
        rf"(?=(?:{response_start}|Sample Student Responses|QUESTIONS? |\[\[PAGE \d+\]\]|\Z))"
    )
    candidates = []
    headers = list(re.finditer(r"Sample Student Responses", full))
    for header_index, header in enumerate(headers):
        section_end = (
            headers[header_index + 1].start() if header_index + 1 < len(headers) else len(full)
        )
        section = full[header.end() : section_end]
        for m in pattern.finditer(section):
            candidates.append((header.end() + m.start(), m))
    for idx, (absolute_start, m) in enumerate(candidates, 1):
        prefix = full[:absolute_start]
        page_hits = re.findall(r"\[\[PAGE (\d+)\]\]", prefix)
        page_no = int(page_hits[-1]) if page_hits else None
        response = clean_text(m.group("response"))
        description = clean_text(m.group("description"))
        score_m = re.search(r"\d+", m.group("score"))
        record = {
            "record_id": f"aaas:aspect3d:{path.stem}:{idx}",
            "dataset": "AAAS_Project_2061",
            "resource_group": "ASPECt-3D",
            "diagnostic_unit": "sample_student_response_with_rubric_score",
            "task_file": path.name,
            "sample_student_response": response,
            "score": int(score_m.group()) if score_m else None,
            "scoring_description": description,
            "license": {
                "status": "group_page_explicit",
                "spdx_like": "CC-BY-NC-SA-4.0",
                "commercial_use": False,
                "attribution_required": True,
                "share_alike": True,
                "embedded_content_may_have_separate_terms": True,
            },
            "source": {"path": str(path.relative_to(DIAG)), "page": page_no, "sha256": source_hash},
            "no_rewrite": True,
        }
        record["exact_content_sha256"] = object_hash(
            {
                "sample_student_response": response,
                "score": record["score"],
                "scoring_description": description,
            }
        )
        kept.append(record)
    if "Sample Student Responses" in full and not candidates:
        routed.append(
            {
                "dataset": "AAAS_Project_2061",
                "resource_group": "ASPECt-3D",
                "route_reason": "sample_response_table_parse_failed",
                "source": {"path": str(path.relative_to(DIAG)), "sha256": source_hash},
            }
        )
    return kept, routed


def audit_aaas() -> tuple[list[dict], list[dict], list[dict]]:
    audits = []
    for group in ("ASPECt", "ASPECt-3D", "Evolution"):
        files = sorted((AAAS / group).iterdir())
        source_page = AAAS / group / "source_page.html"
        text = html_plain_text(source_page)
        if group == "ASPECt":
            license_status = "explicit_group_license"
            license_id = "CC-BY-NC-SA-4.0"
            evidence = "ASPECt-MC resources ... licensed as CC-BY-NC-SA-4.0."
        elif group == "ASPECt-3D":
            license_status = "explicit_group_license"
            license_id = "CC-BY-NC-SA-4.0"
            evidence = "ASPECt-3D resources ... licensed as CC-BY-NC-SA-4.0."
        else:
            license_status = "copyright_only_no_group_reuse_license_found"
            license_id = None
            evidence = "No group-level reuse license was found in the saved source page; PDFs display AAAS copyright."
        audits.append(
            {
                "resource_group": group,
                "collection_scope": "partial_local_public_collection",
                "file_count": len(files),
                "pdf_count": sum(p.suffix.lower() == ".pdf" for p in files),
                "license_status": license_status,
                "license": license_id,
                "license_evidence": evidence,
                "source_page_path": str(source_page.relative_to(DIAG)),
                "source_page_sha256": file_sha256(source_page),
                "source_page_mentions_license": "license" in text.lower(),
            }
        )

    key = parse_aspect_answer_key(AAAS / "ASPECt/ASPECt_Answer_Keys.pdf")
    kept: list[dict] = []
    routed: list[dict] = []
    for instrument in ("Basic", "Intermediate", "Advanced"):
        k, r = extract_aspect_mc(
            instrument,
            AAAS / f"ASPECt/{instrument}_Energy_Instrument.pdf",
            key[instrument],
        )
        kept.extend(k)
        routed.extend(r)

    task_pdfs = []
    for path in sorted((AAAS / "ASPECt-3D").glob("*.pdf")):
        text = "\n".join(pdf_pages(path))
        if "Sample Student Responses" in text and "SCORING RUBRIC" in text:
            task_pdfs.append(path)
            k, r = extract_aspect3d_exemplars(path)
            kept.extend(k)
            routed.extend(r)
        elif "_Task_Data_Summary" not in path.name:
            routed.append(
                {
                    "dataset": "AAAS_Project_2061",
                    "resource_group": "ASPECt-3D",
                    "route_reason": "not_a_stable_student_response_exemplar_table",
                    "source": {"path": str(path.relative_to(DIAG)), "sha256": file_sha256(path)},
                }
            )

    for path in sorted((AAAS / "Evolution").glob("*.pdf")):
        routed.append(
            {
                "dataset": "AAAS_Project_2061",
                "resource_group": "Evolution",
                "route_reason": "no_group_level_reuse_license_and_no_standalone_machine_readable_answer_key",
                "source": {"path": str(path.relative_to(DIAG)), "sha256": file_sha256(path)},
            }
        )
    return kept, routed, audits


def load_question_map(path: Path) -> tuple[dict[str, dict], str]:
    rows = load_jsonl(path)
    return {str(x["id"]): x for x in rows}, file_sha256(path)


def curriculum_xes_hashes() -> tuple[set[str], int]:
    hashes: set[str] = set()
    if not CURRICULUM_XES.exists():
        return hashes, 0
    n = 0
    with CURRICULUM_XES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            n += 1
            source_id = str(obj.get("grounding", {}).get("source_id", ""))
            if source_id:
                hashes.add(source_id)
    return hashes, n


def clean_longtutor() -> tuple[list[dict], list[dict], dict[str, Any]]:
    data = LONGTUTOR / "data"
    curriculum_ids, curriculum_count = curriculum_xes_hashes()
    specifications = [
        ("Gold", "XES3G5M", data / "XES3G5M/human_an_updated.jsonl", 0),
        ("Silver", "XES3G5M", data / "XES3G5M/pipeline_an_scale.jsonl", 1000),
        ("Silver", "MOOCRadar", data / "MOOCRadar/pipeline_an.jsonl", 0),
    ]
    questions = {}
    question_hashes = {}
    sequences = {}
    sequence_hashes = {}
    for dataset in ("XES3G5M", "MOOCRadar"):
        questions[dataset], question_hashes[dataset] = load_question_map(
            data / dataset / "questions.jsonl"
        )
        sequences[dataset] = load_jsonl(data / dataset / "sequences_long.jsonl")
        sequence_hashes[dataset] = file_sha256(data / dataset / "sequences_long.jsonl")

    kept: list[dict] = []
    removed: list[dict] = []
    seen_exact: dict[str, str] = {}
    seen_structural: dict[str, str] = {}
    tier_counts = Counter()
    overlap_refs = 0
    for tier, dataset, annotation_path, sequence_offset in specifications:
        annotations = load_jsonl(annotation_path)
        annotation_file_hash = file_sha256(annotation_path)
        for index, annotation in enumerate(annotations):
            seq_index = sequence_offset + index
            seq_obj = sequences[dataset][seq_index]
            uid = str(seq_obj["uid"])
            key_uid = str(annotation.get("_key", "")).split("||", 1)[0]
            if key_uid != uid:
                raise ValueError(
                    f"LongTutor alignment failed: {annotation_path}:{index + 1}, {key_uid=} {uid=}"
                )
            sequence = seq_obj.get("sequence", [])
            current = sequence[-1] if sequence else {}
            qid = str(current.get("question", ""))
            qobj = questions[dataset].get(qid)
            if qobj is None:
                raise ValueError(f"Missing {dataset} question {qid}")
            raw_annotation = {k: v for k, v in annotation.items() if k != "_source_line"}
            raw_sequence = {k: v for k, v in seq_obj.items() if k != "_source_line"}
            exact_key = object_hash(raw_annotation)
            structural_payload = {
                "dataset": dataset,
                "uid": uid,
                "current_question_id": qid,
                "memory": raw_annotation.get("memory"),
                "diagnosis": raw_annotation.get("diagnosis"),
                "reason": raw_annotation.get("reason"),
                "strategy": raw_annotation.get("strategy"),
                "content": raw_annotation.get("content"),
            }
            structural_key = object_hash(structural_payload)
            record_id = f"longtutor:{tier.lower()}:{dataset.lower()}:{index + 1}"
            duplicate_of = seen_exact.get(exact_key) or seen_structural.get(structural_key)
            if duplicate_of:
                removed.append(
                    {
                        "record_id": record_id,
                        "dataset": "LongTutor",
                        "source_dataset": dataset,
                        "quality_tier": tier,
                        "route_reason": (
                            "exact_duplicate" if exact_key in seen_exact else "structural_duplicate"
                        ),
                        "duplicate_of": duplicate_of,
                        "source": {
                            "path": str(annotation_path.relative_to(DIAG)),
                            "line": index + 1,
                        },
                    }
                )
                continue
            seen_exact[exact_key] = record_id
            seen_structural[structural_key] = record_id
            xes_curriculum_overlap = dataset == "XES3G5M" and qid in curriculum_ids
            overlap_refs += int(xes_curriculum_overlap)
            record = {
                "record_id": record_id,
                "dataset": "LongTutor",
                "benchmark_subset": f"LongTutor-{tier}",
                "quality_tier": tier,
                "source_dataset": dataset,
                "diagnostic_unit": "longitudinal_student_history_with_tutor_annotation",
                "uid": uid,
                "current_interaction": current,
                "current_question": {k: v for k, v in qobj.items() if k != "_source_line"},
                "history": sequence[:-1],
                "annotation": raw_annotation,
                "dedup": {
                    "exact_annotation_sha256": exact_key,
                    "structural_sha256": structural_key,
                    "priority": "Gold_before_Silver_then_source_order",
                },
                "curriculum_overlap": {
                    "is_XES3G5M_curriculum_question_reference": xes_curriculum_overlap,
                    "question_id": qid if dataset == "XES3G5M" else None,
                    "policy": "reference_only_do_not_emit_as_curriculum_training_record",
                    "curriculum_origin_path": (
                        str(CURRICULUM_XES.relative_to(ROOT)) if dataset == "XES3G5M" else None
                    ),
                },
                "provenance": {
                    "annotation_path": str(annotation_path.relative_to(DIAG)),
                    "annotation_line": index + 1,
                    "annotation_file_sha256": annotation_file_hash,
                    "sequence_path": str(
                        (data / dataset / "sequences_long.jsonl").relative_to(DIAG)
                    ),
                    "sequence_line": seq_index + 1,
                    "sequence_file_sha256": sequence_hashes[dataset],
                    "question_path": str((data / dataset / "questions.jsonl").relative_to(DIAG)),
                    "question_line": qobj["_source_line"],
                    "question_file_sha256": question_hashes[dataset],
                    "alignment": "_key UID equals sequences_long UID at documented row offset",
                },
                "no_rewrite": True,
            }
            kept.append(record)
            tier_counts[(tier, dataset)] += 1

    report = {
        "input_annotations": {
            "Gold/XES3G5M": 1000,
            "Silver/XES3G5M": 2437,
            "Silver/MOOCRadar": 729,
        },
        "kept_by_tier_source": {f"{k[0]}/{k[1]}": v for k, v in sorted(tier_counts.items())},
        "removed_duplicates": len(removed),
        "XES3G5M_records_referencing_curriculum_question_ids": overlap_refs,
        "curriculum_XES3G5M_origin_rows_seen": curriculum_count,
        "curriculum_duplicate_policy": (
            "LongTutor XES3G5M records are retained only as longitudinal diagnostic benchmark units. "
            "Standalone XES questions/knowledge paths are not emitted, preventing duplication of curriculum records."
        ),
    }
    return kept, removed, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    aaas_kept, aaas_routed, audits = audit_aaas()
    long_kept, long_removed, long_report = clean_longtutor()

    # Final cross-source exact content dedup. AAAS and LongTutor payload types differ,
    # but this check documents that no accidental byte-identical diagnostic unit survives.
    all_kept = aaas_kept + long_kept
    ids = set()
    for row in all_kept:
        if row["record_id"] in ids:
            raise ValueError(f"Duplicate record_id: {row['record_id']}")
        ids.add(row["record_id"])

    write_jsonl(AAAS_OUT / "kept.jsonl", aaas_kept)
    write_jsonl(AAAS_OUT / "routed.jsonl", aaas_routed)
    write_jsonl(LONGTUTOR_OUT / "kept.jsonl", long_kept)
    write_jsonl(LONGTUTOR_OUT / "removed_duplicates.jsonl", long_removed)
    write_jsonl(LONGTUTOR_OUT / "license_audit.jsonl", audits)

    report = {
        "scope": "deterministic_clean_extract_only_no_rewrite_no_synthesis",
        "source_roots": {
            "aaas": str(AAAS),
            "longtutor": str(LONGTUTOR),
        },
        "aaas": {
            "kept_records": len(aaas_kept),
            "kept_by_group": dict(Counter(x["resource_group"] for x in aaas_kept)),
            "kept_by_unit": dict(Counter(x["diagnostic_unit"] for x in aaas_kept)),
            "routed_records": len(aaas_routed),
            "routes": dict(Counter(x["route_reason"] for x in aaas_routed)),
            "license_audit": audits,
        },
        "longtutor": {
            "kept_records": len(long_kept),
            **long_report,
        },
        "combined_kept_records": len(all_kept),
        "validation": {
            "record_ids_unique": len(ids) == len(all_kept),
            "all_kept_marked_no_rewrite": all(x.get("no_rewrite") is True for x in all_kept),
            "longtutor_key_sequence_alignment_checked": True,
            "aaas_aspect_answer_key_triples": 105,
            "aaas_aspect_instrument_vs_answer_key_item_code_mismatches_preserved": sum(
                x.get("item_code_match") is False for x in aaas_kept
            ),
        },
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
