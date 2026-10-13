#!/usr/bin/env python3
"""Deterministic fast-clean for Bridge, SemEval 2013 Task 7, and CLC-FCE.

Only official/original data files are read.  In particular, Bridge ``outputs``,
``results``, and prompts are outside this pipeline's input scope.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import pyarrow.parquet as pq

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

CAPABILITY = "diagnostic_reasoning"
SOURCE_ROOT = pools.source_dir(CAPABILITY)
BRIDGE_OUT = pools.cleaned_dir(CAPABILITY, "Bridge")
SEMEVAL_OUT = pools.cleaned_dir(CAPABILITY, "SemEval")
CLC_OUT = pools.cleaned_dir(CAPABILITY, "CLC-FCE")
BRIDGE_DIR = SOURCE_ROOT / "github" / "Bridge" / "dataset"
SEMEVAL_ZIP = SOURCE_ROOT / "direct" / "SemEval_2013_Task7_author_archive" / "semeval-5way.zip"
CLC_TAR = SOURCE_ROOT / "direct" / "CLC_FCE" / "fce_v2.1.bea19.tar.gz"
HF_ROOT = SOURCE_ROOT / "huggingface"

LABELS = {
    0: "correct",
    1: "contradictory",
    2: "partially_correct_incomplete",
    3: "irrelevant",
    4: "non_domain",
}
SEMEVAL_SPLITS = {
    "train": "train",
    "test-unseen-answers": "test_ua",
    "test-unseen-questions": "test_uq",
    "test-unseen-domains": "test_ud",
}


def text(value: Any, *, optional: bool = False) -> str | None:
    if value is None:
        return None if optional else ""
    value = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return value or (None if optional else "")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return f"{prefix}_{digest[:20]}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def semantic_text(value: Any) -> str:
    """Conservative exact-semantic normalization; no spelling/paraphrase edits."""
    value = unicodedata.normalize("NFKC", text(value))
    return " ".join(value.split()).casefold()


def dedup_by_key(
    rows: list[dict[str, Any]],
    key_fn,
    *,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for row in rows:
        key = canonical_json(key_fn(row))
        if key in seen:
            rejected.append(
                {
                    "reason": "exact_normalized_semantic_duplicate_within_split",
                    "dataset": source,
                    "split": row["split"],
                    "source_record_id": row.get("source_record_id"),
                    "duplicate_of": seen[key],
                }
            )
        else:
            seen[key] = row.get("source_record_id") or row["record_id"]
            kept.append(row)
    return kept, rejected


def remove_train_eval_collisions(
    rows: list[dict[str, Any]], key_fn, *, source: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_keys = {canonical_json(key_fn(row)) for row in rows if row["split"] != "train"}
    kept, rejected = [], []
    for row in rows:
        if row["split"] == "train" and canonical_json(key_fn(row)) in eval_keys:
            rejected.append(
                {
                    "reason": "exact_train_evaluation_semantic_overlap",
                    "dataset": source,
                    "split": "train",
                    "source_record_id": row.get("source_record_id"),
                }
            )
        else:
            kept.append(row)
    return kept, rejected


def bridge_key(row: dict[str, Any]) -> Any:
    return {
        k: row[k]
        for k in (
            "lesson_topic",
            "conversation_history",
            "original_tutor_response",
            "expert_revised_response",
            "expert_revision_dialogue",
            "student_error",
            "strategy",
            "intention",
        )
    }


def normalize_turn(raw: dict[str, Any], *, source_ids: bool) -> dict[str, Any]:
    out = {"user": text(raw.get("user")), "text": text(raw.get("text"))}
    if source_ids:
        out["source_turn_id"] = raw.get("id")
        out["time"] = text(raw.get("time"), optional=True)
    else:
        out["is_revised"] = bool(raw.get("is_revised", False))
    return out


def clean_bridge() -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    expected = {
        "c_id",
        "lesson_topic",
        "c_h",
        "c_r",
        "c_r_",
        "c_revision",
        "e",
        "z_what",
        "z_why",
    }
    source_files = []
    input_rows = 0
    for split, filename in (
        ("train", "train.json"),
        ("validation", "validation.json"),
        ("test", "test.json"),
    ):
        path = BRIDGE_DIR / filename
        source_files.append(str(path.relative_to(ROOT)))
        raw_rows = json.loads(path.read_text(encoding="utf-8"))
        input_rows += len(raw_rows)
        normalized = []
        for raw in raw_rows:
            if set(raw) != expected:
                raise ValueError(f"Bridge schema mismatch in {filename}: {set(raw)}")
            row = {
                "dataset": "Bridge",
                "split": split,
                "source_record_id": text(raw["c_id"]),
                "lesson_topic": text(raw["lesson_topic"]),
                "conversation_history": [normalize_turn(x, source_ids=True) for x in raw["c_h"]],
                "original_tutor_response": [normalize_turn(x, source_ids=True) for x in raw["c_r"]],
                "expert_revised_response": [
                    normalize_turn(x, source_ids=False) for x in raw["c_r_"]
                ],
                "expert_revision_dialogue": [
                    normalize_turn(x, source_ids=False) for x in raw["c_revision"]
                ],
                "student_error": text(raw["e"]),
                "strategy": text(raw["z_what"]),
                "intention": text(raw["z_why"]),
            }
            if (
                not row["source_record_id"]
                or not row["conversation_history"]
                or not row["original_tutor_response"]
                or not row["expert_revised_response"]
            ):
                rejected.append(
                    {
                        "reason": "missing_required_bridge_structure",
                        "split": split,
                        "source_record_id": row["source_record_id"],
                    }
                )
            else:
                row["record_id"] = stable_id("bridge", bridge_key(row))
                normalized.append(row)
        kept, dup = dedup_by_key(normalized, bridge_key, source="Bridge")
        rejected.extend(dup)
        rows_by_split[split] = kept

    before_cross = {
        f"{a}__{b}": len(
            {canonical_json(bridge_key(x)) for x in rows_by_split[a]}
            & {canonical_json(bridge_key(x)) for x in rows_by_split[b]}
        )
        for a, b in combinations(rows_by_split, 2)
    }
    all_rows = sum(rows_by_split.values(), [])
    all_rows, cross_rejected = remove_train_eval_collisions(all_rows, bridge_key, source="Bridge")
    rejected.extend(cross_rejected)
    all_rows.sort(
        key=lambda r: (("train", "validation", "test").index(r["split"]), r["source_record_id"])
    )
    write_jsonl(BRIDGE_OUT / "kept.jsonl", all_rows)
    write_jsonl(BRIDGE_OUT / "rejected.jsonl", rejected)
    report = {
        "dataset": "Bridge",
        "scope": "official_repository_dataset_only_no_outputs_no_model_results_no_rewrite",
        "source_files": source_files,
        "explicitly_excluded_paths": [
            "github/Bridge/outputs",
            "github/Bridge/results",
            "github/Bridge/prompts",
        ],
        "input_rows": input_rows,
        "kept_rows": len(all_rows),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "split_counts": dict(sorted(Counter(x["split"] for x in all_rows).items())),
        "cross_split_exact_semantic_overlap_before_train_filter": before_cross,
        "dedup_policy": (
            "Exact equality after line-ending/outer-whitespace normalization on "
            "all task-semantic fields; source conversation/turn IDs are excluded."
        ),
        "split_policy": (
            "Official train/validation/test names retained. Split-local duplicate "
            "removal first; exact train-to-evaluation semantic copies are removed "
            "from train while validation/test remain in their official split."
        ),
    }
    write_json(BRIDGE_OUT / "report.json", report)
    return report


def semeval_key(row: dict[str, Any]) -> Any:
    return {
        "question": semantic_text(row["question"]),
        "reference_answer": semantic_text(row["reference_answer"]),
        "student_answer": semantic_text(row["student_answer"]),
        "label_5way": row["label_5way"],
    }


def parse_semeval_core() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inventory = Counter()
    reliability = Counter()
    with ZipFile(SEMEVAL_ZIP) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            parts = name.split("/")
            inventory["xml_total"] += 1
            if len(parts) >= 3 and parts[1] == "reliability":
                reliability["xml_files"] += 1
                root = ET.fromstring(archive.read(name))
                reliability["student_annotations"] += len(
                    root.findall("./studentAnswers/studentAnswer")
                )
                continue
            if len(parts) != 4 or parts[2] != "Core":
                continue
            corpus, source_split = parts[:2]
            if corpus not in {"beetle", "sciEntsBank"}:
                continue
            split = SEMEVAL_SPLITS[source_split]
            root = ET.fromstring(archive.read(name))
            question = text(root.findtext("questionText"))
            refs = []
            for ref in root.findall("./referenceAnswers/referenceAnswer"):
                refs.append(
                    {
                        "source_reference_id": text(ref.attrib.get("id")),
                        "category": text(ref.attrib.get("category"), optional=True),
                        "answer": text(ref.text),
                    }
                )
            if not refs:
                raise ValueError(f"No reference answer in {name}")
            best = [x for x in refs if x["category"] == "BEST"]
            selected = (best or refs)[0]["answer"]
            for answer in root.findall("./studentAnswers/studentAnswer"):
                label = text(answer.attrib.get("accuracy")).replace("-", "_")
                row = {
                    "dataset": "SemEval_2013_Task7",
                    "source_corpus": corpus,
                    "split": split,
                    "source_record_id": text(answer.attrib.get("id")),
                    "source_question_id": text(root.attrib.get("id")),
                    "source_xml": name,
                    "module": text(root.attrib.get("module"), optional=True),
                    "question_type": text(root.attrib.get("qtype"), optional=True),
                    "student_answer_type": text(root.attrib.get("stype"), optional=True),
                    "question": question,
                    "reference_answer": selected,
                    "reference_answers": refs,
                    "student_answer": text(answer.text),
                    "label_5way": label,
                    "answer_match": text(answer.attrib.get("answerMatch"), optional=True),
                    "source_count": int(answer.attrib.get("count", "1")),
                }
                row["semantic_id"] = stable_id("semeval_sem", semeval_key(row))
                row["record_id"] = stable_id(
                    "semeval",
                    {
                        "source_corpus": corpus,
                        "split": split,
                        "source_record_id": row["source_record_id"],
                        "source_xml": name,
                    },
                )
                rows.append(row)
                inventory[f"{corpus}:{split}:core_xml"] += 0
            inventory[f"{corpus}:{split}:core_xml"] += 1
    return rows, {
        "archive_inventory": dict(sorted(inventory.items())),
        "reliability_material_not_in_task_records": dict(reliability),
    }


def dedup_semeval(rows: list[dict[str, Any]]):
    rejected: list[dict[str, Any]] = []
    stage = []
    for corpus in ("beetle", "sciEntsBank"):
        corpus_rows = [r for r in rows if r["source_corpus"] == corpus]
        for split in sorted({r["split"] for r in corpus_rows}):
            kept, dup = dedup_by_key(
                [r for r in corpus_rows if r["split"] == split],
                semeval_key,
                source=f"SemEval:{corpus}",
            )
            stage.extend(kept)
            rejected.extend(dup)
    final = []
    for corpus in ("beetle", "sciEntsBank"):
        kept, cross = remove_train_eval_collisions(
            [r for r in stage if r["source_corpus"] == corpus],
            semeval_key,
            source=f"SemEval:{corpus}",
        )
        final.extend(kept)
        rejected.extend(cross)
    return final, rejected


def load_hf_rows(dataset: str) -> list[dict[str, Any]]:
    rows = []
    paths = {
        "Beetle": {
            "train": "train-00001.parquet",
            "test_ua": "test-ua-00001.parquet",
            "test_uq": "test-uq-00001.parquet",
        },
        "SciEntsBank": {
            "train": "train-00001.parquet",
            "test_ua": "test-ua-00001.parquet",
            "test_uq": "test-uq-00001.parquet",
            "test_ud": "test-ud-00001.parquet",
        },
    }[dataset]
    for split, filename in paths.items():
        for raw in pq.read_table(HF_ROOT / dataset / "data" / filename).to_pylist():
            row = {
                "dataset": dataset,
                "split": split,
                "source_record_id": text(raw["id"]),
                "question": text(raw["question"]),
                "reference_answer": text(raw["reference_answer"]),
                "student_answer": text(raw["student_answer"]),
                "label_5way": LABELS[int(raw["label"])],
            }
            rows.append(row)
    return rows


def collision_audit(
    semeval_raw: list[dict[str, Any]], semeval_clean: list[dict[str, Any]]
) -> dict[str, Any]:
    source_rows = {
        "Beetle": load_hf_rows("Beetle"),
        "SciEntsBank": load_hf_rows("SciEntsBank"),
        "SemEval_beetle_raw_core": [x for x in semeval_raw if x["source_corpus"] == "beetle"],
        "SemEval_scientsbank_raw_core": [
            x for x in semeval_raw if x["source_corpus"] == "sciEntsBank"
        ],
        "SemEval_beetle_clean": [x for x in semeval_clean if x["source_corpus"] == "beetle"],
        "SemEval_scientsbank_clean": [
            x for x in semeval_clean if x["source_corpus"] == "sciEntsBank"
        ],
    }
    sets = {
        name: {canonical_json(semeval_key(x)) for x in rows} for name, rows in source_rows.items()
    }
    pairs = {}
    requested = (
        ("Beetle", "SciEntsBank"),
        ("Beetle", "SemEval_beetle_raw_core"),
        ("Beetle", "SemEval_scientsbank_raw_core"),
        ("SciEntsBank", "SemEval_beetle_raw_core"),
        ("SciEntsBank", "SemEval_scientsbank_raw_core"),
        ("SemEval_beetle_raw_core", "SemEval_scientsbank_raw_core"),
        ("Beetle", "SemEval_beetle_clean"),
        ("SciEntsBank", "SemEval_scientsbank_clean"),
    )
    for a, b in requested:
        intersection = sets[a] & sets[b]
        pairs[f"{a}__{b}"] = {
            "source_a_rows": len(source_rows[a]),
            "source_b_rows": len(source_rows[b]),
            "source_a_unique_semantic_keys": len(sets[a]),
            "source_b_unique_semantic_keys": len(sets[b]),
            "exact_semantic_collision_keys": len(intersection),
            "a_key_coverage_by_b": (len(intersection) / len(sets[a]) if sets[a] else None),
            "b_key_coverage_by_a": (len(intersection) / len(sets[b]) if sets[b] else None),
        }
    return {
        "key_definition": (
            "NFKC + whitespace collapse + casefold for question, selected reference "
            "answer, and student answer; exact normalized 5-way label equality. "
            "No fuzzy matching, stemming, spelling repair, or paraphrase matching."
        ),
        "pairwise": pairs,
        "canonical_source_recommendation": {
            "recommendation": (
                "Use the SemEval 2013 Task 7 author archive semeval-5way.zip, Core "
                "XML, as the canonical source for both Beetle and SciEntsBank task "
                "examples. Treat the local Hugging Face Beetle/SciEntsBank parquet "
                "copies as convenient derivatives, not additional independent data."
            ),
            "representation_note": (
                "Core and Extra XML contain duplicate task examples; Core is selected "
                "once. Extra and Dependency artifacts are not separately ingested. "
                "The reliability directory is annotation-agreement material and is "
                "audited but not mixed into official task split records."
            ),
        },
    }


def clean_semeval() -> dict[str, Any]:
    raw, inventory = parse_semeval_core()
    clean, rejected = dedup_semeval(raw)
    split_order = {"train": 0, "test_ua": 1, "test_uq": 2, "test_ud": 3}
    clean.sort(
        key=lambda r: (
            r["source_corpus"],
            split_order[r["split"]],
            r["source_record_id"],
        )
    )
    write_jsonl(SEMEVAL_OUT / "kept.jsonl", clean)
    write_jsonl(SEMEVAL_OUT / "rejected.jsonl", rejected)
    collisions = collision_audit(raw, clean)
    write_json(SEMEVAL_OUT / "source_overlap_audit.json", collisions)
    report = {
        "dataset": "SemEval_2013_Task7_5way",
        "scope": "official_author_archive_core_xml_no_rewrite",
        "source_files": [str(SEMEVAL_ZIP.relative_to(ROOT))],
        "input_core_student_answer_rows": len(raw),
        "kept_rows": len(clean),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "split_counts": {
            f"{corpus}:{split}": count
            for (corpus, split), count in sorted(
                Counter((x["source_corpus"], x["split"]) for x in clean).items()
            )
        },
        "label_counts": dict(sorted(Counter(x["label_5way"] for x in clean).items())),
        "dedup_policy": (
            "Exact semantic key after NFKC, whitespace collapse, and casefold on "
            "question/selected-reference/student-answer plus exact label. First "
            "BEST reference is selected for Beetle compatibility; all official "
            "reference answers remain in reference_answers."
        ),
        "split_policy": (
            "Official train, unseen-answers, unseen-questions, and unseen-domains "
            "splits retained as train/test_ua/test_uq/test_ud. Exact split-local "
            "duplicates are removed; exact train/evaluation collisions are removed "
            "from train only."
        ),
        **inventory,
    }
    write_json(SEMEVAL_OUT / "report.json", report)
    return report


def clc_key(row: dict[str, Any]) -> Any:
    return {
        "text": row["learner_text"],
        "annotations": row["annotations"],
        "l1": row["l1"],
        "age": row["age"],
        "question": row["question"],
    }


def clean_clc() -> dict[str, Any]:
    essay_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edit_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_files = []
    with tarfile.open(CLC_TAR, "r:gz") as archive:
        for split in ("train", "dev", "test"):
            member = f"fce/json/fce.{split}.json"
            source_files.append(f"{CLC_TAR.relative_to(ROOT)}::{member}")
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(member)
            for line_number, raw_line in enumerate(io.TextIOWrapper(handle, encoding="utf-8"), 1):
                raw = json.loads(raw_line)
                # Character offsets are defined against the byte-decoded JSON string.
                # Do not strip or normalize it: trailing whitespace is offset-bearing.
                learner_text = raw["text"]
                if not isinstance(learner_text, str):
                    raise TypeError(f"CLC-FCE text is not a string: {raw['id']}")
                source_script_id = text(raw["id"])
                source_record_id = f"{source_script_id}::q={raw.get('q')}"
                annotations = []
                for annotator_id, edits in raw["edits"]:
                    normalized_edits = []
                    for edit_index, edit in enumerate(edits):
                        if len(edit) == 4:
                            start, end, correction, error_type = edit
                        elif len(edit) == 3:
                            start, end, correction = edit
                            error_type = None
                        else:
                            raise ValueError(f"Unexpected CLC-FCE edit shape: {edit!r}")
                        start, end = int(start), int(end)
                        correction = "" if correction is None else str(correction)
                        valid = 0 <= start <= end <= len(learner_text)
                        error_span = learner_text[start:end] if valid else None
                        item = {
                            "annotator_id": str(annotator_id),
                            "edit_index": edit_index,
                            "char_start": start,
                            "char_end": end,
                            "learner_error": error_span,
                            "correction": correction,
                            "error_type": text(error_type, optional=True),
                            "operation": (
                                "insertion"
                                if start == end
                                else ("deletion" if not correction else "replacement")
                            ),
                        }
                        if not valid:
                            rejected.append(
                                {
                                    "reason": "invalid_character_offsets",
                                    "split": split,
                                    "source_record_id": source_record_id,
                                    "annotation": item,
                                }
                            )
                        else:
                            normalized_edits.append(item)
                            edit_rows.append(
                                {
                                    "dataset": "CLC_FCE",
                                    "split": split,
                                    "source_record_id": source_record_id,
                                    "source_script_id": source_script_id,
                                    **item,
                                }
                            )
                    annotations.append(
                        {
                            "annotator_id": str(annotator_id),
                            "edits": normalized_edits,
                        }
                    )
                row = {
                    "dataset": "CLC_FCE",
                    "split": split,
                    "source_record_id": source_record_id,
                    "source_script_id": source_script_id,
                    "l1": text(raw.get("l1"), optional=True),
                    "age": text(raw.get("age"), optional=True),
                    "question": raw.get("q"),
                    "answer_score": raw.get("answer-s"),
                    "script_score": raw.get("script-s"),
                    "learner_text": learner_text,
                    "annotations": annotations,
                    "source_line_number": line_number,
                }
                row["record_id"] = stable_id("clcfce", clc_key(row))
                essay_rows[split].append(row)

    clean_by_split = {}
    for split, rows in essay_rows.items():
        clean_by_split[split], dup = dedup_by_key(rows, clc_key, source="CLC_FCE")
        rejected.extend(dup)
    before = {
        f"{a}__{b}": len(
            {canonical_json(clc_key(x)) for x in clean_by_split[a]}
            & {canonical_json(clc_key(x)) for x in clean_by_split[b]}
        )
        for a, b in combinations(("train", "dev", "test"), 2)
    }
    essays = sum((clean_by_split[x] for x in ("train", "dev", "test")), [])
    essays, cross = remove_train_eval_collisions(essays, clc_key, source="CLC_FCE")
    rejected.extend(cross)
    kept_ids = {(x["split"], x["source_record_id"]) for x in essays}
    edit_rows = [x for x in edit_rows if (x["split"], x["source_record_id"]) in kept_ids]
    edit_rows.sort(
        key=lambda x: (
            ("train", "dev", "test").index(x["split"]),
            x["source_record_id"],
            x["annotator_id"],
            x["edit_index"],
        )
    )
    write_jsonl(CLC_OUT / "kept.jsonl", essays)
    write_jsonl(CLC_OUT / "edits.jsonl", edit_rows)
    write_jsonl(CLC_OUT / "rejected.jsonl", rejected)
    report = {
        "dataset": "CLC_FCE.1_BEA2019",
        "scope": "official_release_json_character_edits_no_rewrite",
        "source_files": source_files,
        "input_essays": sum(len(x) for x in essay_rows.values()),
        "kept_essays": len(essays),
        "kept_edits": len(edit_rows),
        "rejected_rows_or_annotations": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "split_counts": dict(sorted(Counter(x["split"] for x in essays).items())),
        "edit_operation_counts": dict(sorted(Counter(x["operation"] for x in edit_rows).items())),
        "annotator_count_distribution": dict(
            sorted(
                Counter(len(x["annotations"]) for x in essays).items(),
                key=lambda item: item[0],
            )
        ),
        "cross_split_exact_semantic_overlap_before_train_filter": before,
        "structure_policy": (
            "Preserve original learner essay and official per-annotator character "
            "offset edits. Each edit exposes learner_error, correction, offsets, "
            "annotator, and insertion/deletion/replacement type. No corrected essay "
            "or synthetic error explanation is generated."
        ),
        "split_policy": (
            "Official BEA2019 train/dev/test split retained. Exact split-local "
            "duplicates are removed; exact train-to-dev/test semantic copies would "
            "be removed from train."
        ),
    }
    write_json(CLC_OUT / "report.json", report)
    return report


def main() -> None:
    bridge = clean_bridge()
    semeval = clean_semeval()
    clc = clean_clc()
    summary = {
        "pipeline": "bridge_semeval_clc_fce",
        "no_rewrite": True,
        "bridge": bridge,
        "semeval": semeval,
        "clc_fce": clc,
    }
    print(
        canonical_json(
            {
                "bridge_kept": bridge["kept_rows"],
                "semeval_kept": semeval["kept_rows"],
                "clc_fce_essays": clc["kept_essays"],
                "clc_fce_edits": clc["kept_edits"],
            }
        )
    )


if __name__ == "__main__":
    main()
