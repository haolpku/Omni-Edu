#!/usr/bin/env python3
"""Deterministic fast clean for WikiHint.

The script does not rewrite hints and does not call a model. It:
1. assigns a conservative K12/subject relevance bucket using existing fields
   plus fixed lexical rules;
2. removes structurally invalid rows, exact duplicate questions, and hints
   that literally contain the gold answer;
3. preserves the official train/test boundary;
4. emits kept, deferred, removed, and QC/audit artifacts.
"""

from __future__ import annotations

import argparse
import collections
import copy
import difflib
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path

import sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import cleaning, pools
from typing import Any

HERE = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_SOURCE = HERE.parents[1] / "github" / "WikiHint" / "WikiHint"

SPLIT_FILES = {"train": "training.json", "test": "test.json"}
REQUIRED_ROW_FIELDS = ("id", "question", "answer", "hints")
REQUIRED_QUESTION_FIELDS = ("question", "question_type", "difficulty", "candidate_answers")
REQUIRED_ANSWER_FIELDS = ("answer",)
REQUIRED_HINT_FIELDS = ("hint", "rank", "metrics")
REQUIRED_HINT_METRICS = (
    "relevance",
    "readability",
    "convergence",
    "familiarity",
    "answer_leakage",
)

# Clearly out-of-scope entertainment, celebrity, sport, brand/company, and
# internet trivia. Matching is against the normalized gold answer.
ENTERTAINMENT_ANSWERS = {
    "angry birds",
    "barbie",
    "batman",
    "breaking bad",
    "captain america",
    "chewbacca",
    "cthulhu",
    "dc comics",
    "dc extended universe",
    "disney",
    "donkey kong",
    "eric van der woodsen",
    "game boy",
    "gangnam style",
    "hannibal",
    "hermione granger",
    "homer simpson",
    "how i met your mother",
    "inception",
    "iron man",
    "it",
    "jack sparrow",
    "jesse pinkman",
    "malcolm in the middle",
    "master chief",
    "meme",
    "monty python",
    "naughty dog",
    "nintendo 64",
    "nintendo switch",
    "obi wan kenobi",
    "one piece",
    "patrick star",
    "pepe the frog",
    "peter griffin",
    "pikachu",
    "samurai",
    "sasuke uchiha",
    "scrubs",
    "she s the man",
    "sing",
    "sonic the hedgehog",
    "star wars",
    "the last of us",
    "the legend of zelda",
    "the lord of the rings",
    "the matrix",
    "the simpsons",
    "the wolf of wall street",
    "toad",
    "willy wonka",
    "yoda",
}
MUSIC_CELEBRITY_ANSWERS = {
    "anthony hopkins",
    "beyonce",
    "bob dylan",
    "bryan cranston",
    "charlie chaplin",
    "christina aguilera",
    "david hasselhoff",
    "elvis presley",
    "eminem",
    "guns n roses",
    "h p lovecraft",
    "iron maiden",
    "john cena",
    "kanye west",
    "leslie nielsen",
    "madonna",
    "matt lanter",
    "michael jackson",
    "nick cannon",
    "neil patrick harris",
    "paris hilton",
    "paul mccartney",
    "pewdiepie",
    "queen",
    "scarlet johansson",
    "selena gomez",
    "usher",
    "we are the champions",
}
SPORT_ANSWERS = {
    "bobby fischer",
    "cristiano ronaldo",
    "detroit tigers",
    "diego maradona",
    "fc barcelona",
    "hermann maier",
    "kobe bryant",
    "lebron james",
    "lionel messi",
    "novak djokovic",
    "olympique de marseille",
    "roger federer",
    "seattle seahawks",
    "serena williams",
    "super g",
    "tom brady",
    "wayne rooney",
    "wladimir klitschko",
    "zlatan ibrahimovi",
}
BRAND_PRODUCT_ANSWERS = {
    "android",
    "apple",
    "bmw",
    "firefox",
    "hbo",
    "imdb",
    "instagram",
    "sap",
    "toyota",
    "twitter",
    "walmart",
    "youtube",
}
GENERIC_TRIVIA_ANSWERS = {
    "baccarat",
    "beer",
    "kazoo",
    "mp3",
    "photograph",
    "red",
}

# Strong school-subject signals. These are deliberately conservative.
SCIENCE_ANSWERS = {
    "geiger counter",
    "hydrogen",
    "jupiter",
    "mount st helens",
    "pacific ocean",
    "seismometer",
}
MATH_ANSWERS = {"pythagorean theorem", "python"}
HISTORY_CIVICS_ANSWERS = {
    "apollo",
    "archimedes",
    "book of genesis",
    "boris johnson",
    "buddhism",
    "colosseum",
    "egypt",
    "elon musk",
    "ezra pound",
    "francis drake",
    "hinduism",
    "israel",
    "julius caesar",
    "mahatma gandhi",
    "mike pence",
    "mona lisa",
    "moses",
    "nelson mandela",
    "ottoman empire",
    "pablo picasso",
    "pain",
    "ra",
    "robert oppenheimer",
    "ronald reagan",
    "sanskrit",
    "scientology",
    "shigeru miyamoto",
    "sigmund freud",
    "socialism",
    "tolkien",
    "trireme",
    "trojan horse",
    "virgil",
    "yom kippur war",
}

SUBJECT_KEYWORDS = re.compile(
    r"\b("
    r"algebra|angle|atom|biology|chemical|chemistry|climate|combustion|"
    r"constitution|democracy|earthquake|element|empire|energy|equation|"
    r"fluid mechanics|geograph|geometry|government|histor|language|literature|"
    r"mathematic|mountain|ocean|philosoph|physics|planet|politic|religion|"
    r"science|scientific|seismic|solar system|theorem|war"
    r")\b"
)
ENTERTAINMENT_KEYWORDS = re.compile(
    r"\b("
    r"actor|actress|album|anime|band|box office|celebrity|comic|console|"
    r"film|football|franchise|game|gaming|manga|movie|musician|rapper|"
    r"sitcom|singer|single|song|superhero|television|tv series|video game|"
    r"wrestler"
    r")\b"
)


def normalize(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = text.replace("’", "'")
    return re.sub(r"[^\w]+", " ", text).strip()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_fields(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in REQUIRED_ROW_FIELDS:
        if key not in row or row[key] in (None, "", []):
            out.append(key)
    q = row.get("question")
    if isinstance(q, dict):
        for key in REQUIRED_QUESTION_FIELDS:
            if key not in q or q[key] in (None, "", []):
                out.append(f"question.{key}")
    elif "question" not in out:
        out.append("question:not_object")
    a = row.get("answer")
    if isinstance(a, dict):
        for key in REQUIRED_ANSWER_FIELDS:
            if key not in a or a[key] in (None, ""):
                out.append(f"answer.{key}")
    elif "answer" not in out:
        out.append("answer:not_object")
    hints = row.get("hints")
    if isinstance(hints, list):
        for idx, hint in enumerate(hints):
            if not isinstance(hint, dict):
                out.append(f"hints[{idx}]:not_object")
                continue
            for key in REQUIRED_HINT_FIELDS:
                if key not in hint or hint[key] in (None, ""):
                    out.append(f"hints[{idx}].{key}")
            metrics = hint.get("metrics")
            if isinstance(metrics, dict):
                for key in REQUIRED_HINT_METRICS:
                    if key not in metrics or metrics[key] is None:
                        out.append(f"hints[{idx}].metrics.{key}")
            else:
                out.append(f"hints[{idx}].metrics:not_object")
    return out


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    q = row["question"]
    answer = normalize(row["answer"]["answer"])
    question = normalize(q["question"])
    major = q["question_type"].get("major", "")
    minor = q["question_type"].get("minor", "")
    reasons: list[str] = []

    if answer in MATH_ANSWERS:
        return "keep", ["subject_allowlist:math_computing"]
    if answer in SCIENCE_ANSWERS:
        return "keep", ["subject_allowlist:science"]
    if answer in HISTORY_CIVICS_ANSWERS:
        return "keep", ["subject_allowlist:history_civics_humanities"]

    if answer in ENTERTAINMENT_ANSWERS:
        return "defer", ["out_of_scope:entertainment_fiction_games"]
    if answer in MUSIC_CELEBRITY_ANSWERS:
        return "defer", ["out_of_scope:celebrity_music_media"]
    if answer in SPORT_ANSWERS:
        return "defer", ["out_of_scope:sports"]
    if answer in BRAND_PRODUCT_ANSWERS:
        return "defer", ["out_of_scope:brand_company_internet"]
    if answer in GENERIC_TRIVIA_ANSWERS:
        return "defer", ["out_of_scope:generic_trivia"]

    # TREC-style type fields are useful but too broad alone. Location is kept
    # as geography; entity types need explicit subject evidence.
    if major == "LOC:LOCATION":
        return "keep", ["field_signal:location_geography"]
    if SUBJECT_KEYWORDS.search(question):
        reasons.append("lexical_signal:school_subject")
        return "keep", reasons
    if major == "DESC:DESCRIPTION":
        return "keep", ["field_signal:description_concept"]
    if major == "ENTY:ENTITY" and minor in {
        "substance:Element and substance",
        "techmeth:Techniques and method",
        "lang:Language",
    }:
        return "keep", [f"field_signal:{minor}"]

    if ENTERTAINMENT_KEYWORDS.search(question):
        return "defer", ["lexical_signal:entertainment_or_sport"]
    if major == "HUM:HUMAN":
        return "defer", ["field_signal:person_or_group_trivia"]
    if major == "ENTY:ENTITY":
        return "defer", ["field_signal:generic_entity_trivia"]
    return "defer", ["no_strong_k12_subject_signal"]


def literal_answer_in_hint(answer: str, hint: str) -> bool:
    a = normalize(answer)
    h = normalize(hint)
    # Short/ambiguous answers such as "It", "Ra", and "Red" are excluded to
    # avoid false positives. Matching uses token boundaries after normalization.
    if len(a) < 4 or len(a.split()) == 1 and a in {"this", "that", "pain"}:
        return False
    return re.search(rf"(?<!\w){re.escape(a)}(?!\w)", h) is not None


def leakage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [
        h["metrics"]["answer_leakage"]
        for row in rows
        for h in row["hints"]
        if isinstance(h.get("metrics", {}).get("answer_leakage"), (int, float))
    ]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "threshold_counts": {
            "ge_0.5": sum(v >= 0.5 for v in vals),
            "ge_0.6": sum(v >= 0.6 for v in vals),
            "ge_0.7": sum(v >= 0.7 for v in vals),
            "ge_0.8": sum(v >= 0.8 for v in vals),
            "ge_0.9": sum(v >= 0.9 for v in vals),
            "eq_1.0": sum(v == 1.0 for v in vals),
        },
    }


def duplicate_stats(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, list[str]]] = {}
    for split, rows in rows_by_split.items():
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for row in rows:
            groups[normalize(row["question"]["question"])].append(row["id"])
        by_split[split] = groups
    cross = set(by_split["train"]) & set(by_split["test"])
    return {
        "within_split_question_duplicate_groups": {
            split: sum(len(ids) > 1 for ids in groups.values())
            for split, groups in by_split.items()
        },
        "within_split_question_duplicate_rows": {
            split: sum(len(ids) for ids in groups.values() if len(ids) > 1)
            for split, groups in by_split.items()
        },
        "cross_split_exact_question_groups": len(cross),
        "cross_split_exact_question_examples": [
            {
                "question_normalized": key,
                "train_ids": by_split["train"][key],
                "test_ids": by_split["test"][key],
            }
            for key in sorted(cross)[:20]
        ],
    }


def overlap_stats(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    answer_groups: dict[str, dict[str, list[str]]] = {
        "train": collections.defaultdict(list),
        "test": collections.defaultdict(list),
    }
    hint_groups: dict[str, list[tuple[str, str, int]]] = collections.defaultdict(list)
    for split, rows in rows_by_split.items():
        for row in rows:
            answer_groups[split][normalize(row["answer"]["answer"])].append(row["id"])
            for hint_index, hint in enumerate(row["hints"]):
                hint_groups[normalize(hint["hint"])].append((split, row["id"], hint_index))

    shared_answers = set(answer_groups["train"]) & set(answer_groups["test"])
    exact_hint_dups = {k: v for k, v in hint_groups.items() if len(v) > 1}
    cross_hint_dups = {
        k: v for k, v in exact_hint_dups.items() if {x[0] for x in v} == {"train", "test"}
    }

    # A lightweight deterministic near-duplicate probe: compare train/test
    # questions only when the normalized gold answer is identical.
    train_by_answer: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    test_by_answer: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows_by_split["train"]:
        train_by_answer[normalize(row["answer"]["answer"])].append(row)
    for row in rows_by_split["test"]:
        test_by_answer[normalize(row["answer"]["answer"])].append(row)
    near_pairs: list[dict[str, Any]] = []
    for answer in sorted(set(train_by_answer) & set(test_by_answer)):
        for train_row in train_by_answer[answer]:
            tq = normalize(train_row["question"]["question"])
            for test_row in test_by_answer[answer]:
                eq = normalize(test_row["question"]["question"])
                ratio = difflib.SequenceMatcher(None, tq, eq).ratio()
                if ratio >= 0.8:
                    near_pairs.append(
                        {
                            "answer_normalized": answer,
                            "train_id": train_row["id"],
                            "test_id": test_row["id"],
                            "similarity": ratio,
                            "train_question": train_row["question"]["question"],
                            "test_question": test_row["question"]["question"],
                        }
                    )

    return {
        "cross_split_shared_answer_groups": len(shared_answers),
        "cross_split_shared_answer_train_rows": sum(
            len(answer_groups["train"][a]) for a in shared_answers
        ),
        "cross_split_shared_answer_test_rows": sum(
            len(answer_groups["test"][a]) for a in shared_answers
        ),
        "exact_hint_duplicate_groups": len(exact_hint_dups),
        "exact_hint_duplicate_occurrences": sum(len(v) for v in exact_hint_dups.values()),
        "cross_split_exact_hint_duplicate_groups": len(cross_hint_dups),
        "cross_split_same_answer_question_pairs_ge_0.8": len(near_pairs),
        "cross_split_same_answer_question_pair_examples": sorted(
            near_pairs, key=lambda x: (-x["similarity"], x["train_id"], x["test_id"])
        )[:20],
    }


def build(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    raw: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, Any] = {}
    for split, filename in SPLIT_FILES.items():
        path = source_dir / filename
        with path.open(encoding="utf-8") as f:
            raw[split] = json.load(f)
        source_files[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(raw[split]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    deferred: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    kept: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    literal_leak_hints: list[dict[str, Any]] = []
    seen_questions: dict[str, set[str]] = {"train": set(), "test": set()}

    for split in ("train", "test"):
        for source_index, original in enumerate(raw[split]):
            row = copy.deepcopy(original)
            miss = missing_fields(row)
            if miss:
                reasons = ["structural_invalid:missing_required_fields"]
                removed_rows.append(
                    {"split": split, "source_index": source_index, "row": original, "missing": miss}
                )
                audit.append(
                    {
                        "id": original.get("id"),
                        "split": split,
                        "source_index": source_index,
                        "decision": "remove",
                        "reasons": reasons,
                        "missing_fields": miss,
                    }
                )
                continue

            question_key = normalize(row["question"]["question"])
            if question_key in seen_questions[split]:
                reasons = ["duplicate:exact_question_within_official_split"]
                removed_rows.append(
                    {
                        "split": split,
                        "source_index": source_index,
                        "row": original,
                        "reasons": reasons,
                    }
                )
                audit.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "source_index": source_index,
                        "decision": "remove",
                        "reasons": reasons,
                    }
                )
                continue
            seen_questions[split].add(question_key)

            clean_hints: list[dict[str, Any]] = []
            removed_hint_indices: list[int] = []
            answer = row["answer"]["answer"]
            for hint_index, hint in enumerate(row["hints"]):
                if literal_answer_in_hint(answer, hint["hint"]):
                    removed_hint_indices.append(hint_index)
                    literal_leak_hints.append(
                        {
                            "id": row["id"],
                            "split": split,
                            "hint_index": hint_index,
                            "rank": hint.get("rank"),
                            "answer": answer,
                            "hint": hint["hint"],
                            "answer_leakage_metric": hint["metrics"].get("answer_leakage"),
                            "reason": "literal_gold_answer_in_hint",
                        }
                    )
                else:
                    clean_hints.append(hint)
            row["hints"] = clean_hints

            if not row["hints"]:
                reasons = ["structural_invalid:no_hints_after_literal_leak_filter"]
                removed_rows.append(
                    {
                        "split": split,
                        "source_index": source_index,
                        "row": original,
                        "reasons": reasons,
                    }
                )
                audit.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "source_index": source_index,
                        "decision": "remove",
                        "reasons": reasons,
                        "removed_hint_indices": removed_hint_indices,
                    }
                )
                continue

            decision, reasons = classify(row)
            row["_clean"] = {
                "official_split": split,
                "source_index": source_index,
                "decision": decision,
                "reasons": reasons,
                "literal_answer_hints_removed": len(removed_hint_indices),
            }
            (kept if decision == "keep" else deferred)[split].append(row)
            audit.append(
                {
                    "id": row["id"],
                    "split": split,
                    "source_index": source_index,
                    "decision": decision,
                    "reasons": reasons,
                    "literal_answer_hints_removed": len(removed_hint_indices),
                    "remaining_hints": len(row["hints"]),
                }
            )

    # Exact question overlap must remain absent after cleaning.
    clean_dups = duplicate_stats(kept)
    source_dups = duplicate_stats(raw)

    for split in ("train", "test"):
        dump_json(out_dir / f"wikihint_k12_{split}.json", kept[split])
        dump_json(out_dir / f"wikihint_deferred_{split}.json", deferred[split])
    cleaning.write_pool(
        "pedagogical_action",
        "WikiHint",
        kept["train"] + kept["test"],
        removed={"removed": removed_rows},
    )
    dump_jsonl(out_dir / "decision_audit.jsonl", audit)
    dump_jsonl(out_dir / "removed_rows.jsonl", removed_rows)
    dump_jsonl(out_dir / "literal_answer_leakage_hints.jsonl", literal_leak_hints)

    all_raw = raw["train"] + raw["test"]
    all_kept = kept["train"] + kept["test"]
    all_deferred = deferred["train"] + deferred["test"]
    decisions = collections.Counter(x["decision"] for x in audit)
    reasons = collections.Counter(r for x in audit for r in x["reasons"])
    report = {
        "pipeline": "wikihint-fast-clean",
        "policy": {
            "generation_or_rewrite_used": False,
            "official_split_preserved": True,
            "scope": "conservative K12/subject relevance; obvious entertainment/person/generic trivia deferred",
            "literal_leak_rule": "remove hint only when normalized full gold answer occurs literally, except ambiguous short answers",
            "metric_rule": "answer_leakage is audited, not threshold-filtered; it is a continuous similarity metric",
        },
        "source_files": source_files,
        "counts": {
            "source_rows": {split: len(rows) for split, rows in raw.items()},
            "kept_rows": {split: len(rows) for split, rows in kept.items()},
            "deferred_rows": {split: len(rows) for split, rows in deferred.items()},
            "removed_rows": len(removed_rows),
            "source_hints": sum(len(x["hints"]) for x in all_raw),
            "kept_hints": sum(len(x["hints"]) for x in all_kept),
            "deferred_hints": sum(len(x["hints"]) for x in all_deferred),
            "literal_answer_hints_removed": len(literal_leak_hints),
            "decision_counts": dict(sorted(decisions.items())),
            "reason_counts": dict(sorted(reasons.items())),
        },
        "empty_field_qc": {
            "source_rows_with_missing_required_fields": sum(
                bool(missing_fields(x)) for x in all_raw
            ),
            "kept_rows_with_missing_required_fields": sum(
                bool(missing_fields(x)) for x in all_kept
            ),
        },
        "duplicate_qc_source": source_dups,
        "duplicate_qc_kept": clean_dups,
        "content_overlap_qc_source": overlap_stats(raw),
        "content_overlap_qc_kept": overlap_stats(kept),
        "answer_leakage_source": leakage_summary(all_raw),
        "answer_leakage_kept": leakage_summary(all_kept),
        "answer_leakage_deferred": leakage_summary(all_deferred),
        "assertions": {
            "official_source_counts_are_900_100": (
                len(raw["train"]) == 900 and len(raw["test"]) == 100
            ),
            "source_ids_match_split_prefix": all(x["id"].startswith("train_") for x in raw["train"])
            and all(x["id"].startswith("test_") for x in raw["test"]),
            "no_exact_question_overlap_kept_train_test": (
                clean_dups["cross_split_exact_question_groups"] == 0
            ),
            "no_exact_question_duplicates_within_kept": all(
                n == 0 for n in clean_dups["within_split_question_duplicate_groups"].values()
            ),
            "all_kept_rows_have_hints": all(x["hints"] for x in all_kept),
            "all_kept_rows_complete": all(not missing_fields(x) for x in all_kept),
            "no_literal_gold_answer_in_kept_hints": all(
                not literal_answer_in_hint(x["answer"]["answer"], h["hint"])
                for x in all_kept
                for h in x["hints"]
            ),
            "row_accounting_balances": (
                len(all_raw) == len(all_kept) + len(all_deferred) + len(removed_rows)
            ),
        },
    }
    dump_json(out_dir / "qc_report.json", report)
    write_markdown_report(out_dir / "REPORT.md", report, literal_leak_hints)
    return report


def write_markdown_report(
    path: Path, report: dict[str, Any], leak_hints: list[dict[str, Any]]
) -> None:
    c = report["counts"]
    rc = c["reason_counts"]
    lines = [
        "# WikiHint deterministic fast-clean report",
        "",
        "## Policy",
        "",
        "- No hint rewriting and no model generation.",
        "- Preserve the official `training.json` / `test.json` boundary.",
        "- Keep conservative K12/subject-related material; defer obvious entertainment,",
        "  celebrity/person, sports, brand/company, and generic trivia.",
        "- Remove exact duplicate questions within a split (first occurrence wins).",
        "- Remove only hints that literally contain the normalized full gold answer.",
        "- Audit the provided `answer_leakage` metric; do not threshold-filter on it.",
        "",
        "## Counts",
        "",
        f"- Source rows: train {c['source_rows']['train']}, test {c['source_rows']['test']}.",
        f"- Kept rows: train {c['kept_rows']['train']}, test {c['kept_rows']['test']}.",
        f"- Deferred rows: train {c['deferred_rows']['train']}, test {c['deferred_rows']['test']}.",
        f"- Removed rows: {c['removed_rows']}.",
        f"- Source hints: {c['source_hints']}; kept hints: {c['kept_hints']};",
        f"  deferred hints: {c['deferred_hints']}; literal-answer hints removed: {c['literal_answer_hints_removed']}.",
        "",
        "## Main filtering reasons",
        "",
    ]
    for reason, count in sorted(rc.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{reason}`: {count}")
    lines += [
        "",
        "## QC",
        "",
        f"- Source missing-required-field rows: {report['empty_field_qc']['source_rows_with_missing_required_fields']}.",
        f"- Kept missing-required-field rows: {report['empty_field_qc']['kept_rows_with_missing_required_fields']}.",
        "- Source exact-question duplicate groups: "
        f"train {report['duplicate_qc_source']['within_split_question_duplicate_groups']['train']}, "
        f"test {report['duplicate_qc_source']['within_split_question_duplicate_groups']['test']}; "
        f"cross-split {report['duplicate_qc_source']['cross_split_exact_question_groups']}.",
        "- Kept exact-question duplicate groups: "
        f"train {report['duplicate_qc_kept']['within_split_question_duplicate_groups']['train']}, "
        f"test {report['duplicate_qc_kept']['within_split_question_duplicate_groups']['test']}; "
        f"cross-split {report['duplicate_qc_kept']['cross_split_exact_question_groups']}.",
        "- Source cross-split overlap: "
        f"{report['content_overlap_qc_source']['cross_split_shared_answer_groups']} shared-answer groups; "
        f"{report['content_overlap_qc_source']['cross_split_exact_hint_duplicate_groups']} exact-hint duplicate groups; "
        f"{report['content_overlap_qc_source']['cross_split_same_answer_question_pairs_ge_0.8']} "
        "same-answer question pairs with character similarity >=0.8.",
        f"- Source answer-leakage metric mean/median: "
        f"{report['answer_leakage_source']['mean']:.6f} / "
        f"{report['answer_leakage_source']['median']:.6f}.",
        f"- Source hints with metric >=0.8: "
        f"{report['answer_leakage_source']['threshold_counts']['ge_0.8']}; "
        f">=0.9: {report['answer_leakage_source']['threshold_counts']['ge_0.9']}; "
        f"==1.0: {report['answer_leakage_source']['threshold_counts']['eq_1.0']}.",
        f"- Literal full-answer hint leaks removed: {len(leak_hints)}.",
        "",
        "All machine-checkable assertions are recorded in `qc_report.json`.",
        "The deferred set is retained for later human/domain review rather than silently discarded.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    report = build(args.source_dir.resolve(), args.out_dir.resolve())
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    failed = [k for k, ok in report["assertions"].items() if not ok]
    if failed:
        raise SystemExit(f"QC assertions failed: {failed}")


if __name__ == "__main__":
    main()
