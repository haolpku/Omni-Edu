#!/usr/bin/env python3
"""Near-deduplicate the 4K COIG subject-relevant subset.

Inspired by DataFlow's SimHash/MinHash filters, but implemented dependency-free:
- SimHash multi-index creates candidates efficiently.
- Character 5-gram Jaccard verifies candidates.
- For records with a separate `input`, that question/task field is the main
  comparison text, preventing long shared reading passages from collapsing
  different questions.
- Existing specialist corpora have priority over COIG.
- No-delete routing is preserved.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

PIPE = pools.STAGE2 / "subject_competence"
SRC = PIPE / "COIG-CQIA/relevance/kept_relevant.jsonl"
OUT = PIPE / "COIG-CQIA/cleaned"
DATE = "2026-08-05"
NGRAM = 5
INTRA_THRESHOLD = 0.94
CROSS_COIG_THRESHOLD = 0.95
EXISTING_THRESHOLD = 0.97
KEEP_SYMBOLS = set("<>=+-*/^\\{}")


def norm(value: Any) -> str:
    text = str(value or "").lower().replace("$", "")
    return "".join(ch for ch in text if ch.isalnum() or ch in KEEP_SYMBOLS)


OPTION_RE = re.compile(r"(?im)(^|[\s　])(?:[\(（]?\s*[a-h]\s*[\)）]|[a-h][\.:、])\s*")
HEADING_RE = re.compile(
    r"(?im)(?:^|[\n\r])\s*(?:question|options?|article|passage|"
    r"题目内容|题目|问题|选项)\s*[:：]\s*"
)


def canon(value: Any) -> str:
    text = HEADING_RE.sub("\n", str(value or ""))
    text = OPTION_RE.sub(" ", text)
    return norm(text)


def shingles(text: str) -> set[str]:
    if len(text) <= NGRAM:
        return {text} if text else set()
    return {text[i : i + NGRAM] for i in range(len(text) - NGRAM + 1)}


def hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(tokens: set[str]) -> int:
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        value = hash64(token)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def buckets(fingerprint: int) -> tuple[int, int, int, int]:
    return tuple((fingerprint >> (16 * i)) & 0xFFFF for i in range(4))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def question_text(row: dict[str, Any]) -> str:
    # COIG frequently stores passage/context in instruction and the actual task
    # in input. Using input as the main key avoids merging different questions
    # about the same long article.
    extra = str(row.get("input") or "").strip()
    return canon(extra if extra else row.get("instruction") or row.get("problem"))


def context_text(row: dict[str, Any]) -> str:
    return canon(row.get("instruction") or "")


def source_priority(source: str) -> int:
    order = [
        "exam/coig_exam_sampled_clean.jsonl",
        "chinese_traditional/trad-multi-choice",
        "chinese_traditional/translate_classical_chinese.jsonl",
        "chinese_traditional/chengyu.jsonl",
        "chinese_traditional/poem.jsonl",
        "logi_qa/logi-qa.jsonl",
        "wiki/10why_final.jsonl",
        "wiki/zgbk.jsonl",
        "wiki/digital_wiki.jsonl",
    ]
    for index, prefix in enumerate(order):
        if source.startswith(prefix):
            return index
    return len(order)


def quality_rank(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -source_priority(str(row.get("_source_file") or "")),
        int(row.get("human_verified") is True),
        # For reading comprehension, preserve the copy containing the complete
        # passage rather than a duplicate that contains only a wrapper.
        len(str(row.get("instruction") or "")),
        len(str(row.get("output") or "")),
    )


class NearIndex:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.grams: list[set[str]] = []
        self.context_grams: list[set[str]] = []
        self.bucket_map: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]

    def add(self, row: dict[str, Any]) -> int:
        qgrams = shingles(question_text(row))
        fingerprint = simhash(qgrams)
        index = len(self.rows)
        self.rows.append(row)
        self.grams.append(qgrams)
        self.context_grams.append(shingles(context_text(row)))
        for position, value in enumerate(buckets(fingerprint)):
            self.bucket_map[position][value].append(index)
        return index

    def candidates(self, row: dict[str, Any]) -> tuple[set[int], set[str], set[str]]:
        qgrams = shingles(question_text(row))
        cgrams = shingles(context_text(row))
        fingerprint = simhash(qgrams)
        result: set[int] = set()
        for position, value in enumerate(buckets(fingerprint)):
            result.update(self.bucket_map[position].get(value, []))
        return result, qgrams, cgrams


def compatible_context(
    row: dict[str, Any], candidate: dict[str, Any], cg: set[str], candidate_cg: set[str]
) -> bool:
    # If both have a separate input, shared task text is enough; the instruction
    # is often a long passage or a harmless wrapper. For an exact/near question,
    # requiring context similarity only when both contexts are substantive
    # prevents accidental collisions of short generic questions.
    if row.get("input") and candidate.get("input"):
        if len(question_text(row)) >= 25:
            return True
        return jaccard(cg, candidate_cg) >= 0.80
    return True


def best_match(row: dict[str, Any], index: NearIndex, threshold: float) -> tuple[int | None, float]:
    candidates, qgrams, cgrams = index.candidates(row)
    best_index = None
    best_score = 0.0
    for candidate_index in candidates:
        score = jaccard(qgrams, index.grams[candidate_index])
        if score < threshold or score <= best_score:
            continue
        candidate = index.rows[candidate_index]
        if not compatible_context(row, candidate, cgrams, index.context_grams[candidate_index]):
            continue
        best_index = candidate_index
        best_score = score
    return best_index, best_score


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def specialist_rows() -> Iterable[dict[str, Any]]:
    specs = [
        ("CJEval", PIPE / "CJEval/cleaned/kept.jsonl"),
        ("C3", PIPE / "C3/cleaned/kept.jsonl"),
        ("ARC", PIPE / "ARC/cleaned/kept.jsonl"),
        ("QASC", PIPE / "QASC/cleaned/kept.jsonl"),
        ("SciQ", PIPE / "SciQ/cleaned/kept.jsonl"),
        ("WorldTree", PIPE / "WorldTree/cleaned/kept.jsonl"),
        ("TQA (text)", PIPE / "TQA (text)/cleaned/kept.jsonl"),
        ("SciInstruct (CN math)", PIPE / "SciInstruct (CN math)/cleaned/kept.jsonl"),
    ]
    for owner, path in specs:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if owner == "C3":
                instruction = str(row.get("passage") or "")
                task = "\n".join([str(row.get("problem") or ""), str(row.get("options") or "")])
            else:
                instruction = ""
                task = "\n".join([str(row.get("problem") or ""), str(row.get("options") or "")])
            yield {
                "uid": row.get("uid"),
                "dataset": owner,
                "instruction": instruction,
                "input": task,
                "problem": task,
            }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = list(read_jsonl(SRC))
    rows.sort(key=quality_rank, reverse=True)

    # Pass 1: source-internal near dedup.
    by_source: dict[str, NearIndex] = defaultdict(NearIndex)
    intra_kept: list[dict[str, Any]] = []
    intra_removed: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("_source_file") or "")
        index = by_source[source]
        hit, score = best_match(row, index, INTRA_THRESHOLD)
        if hit is None:
            index.add(row)
            intra_kept.append(row)
        else:
            routed = dict(row)
            routed.update(
                _reason="near_duplicate_within_source",
                _kept_uid=index.rows[hit].get("uid"),
                _near_similarity=round(score, 6),
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="simhash_candidate+char5gram_jaccard",
            )
            intra_removed.append(routed)

    # Pass 2: between COIG source families.
    global_index = NearIndex()
    cross_coig_kept: list[dict[str, Any]] = []
    cross_coig_removed: list[dict[str, Any]] = []
    for row in intra_kept:
        hit, score = best_match(row, global_index, CROSS_COIG_THRESHOLD)
        if hit is None:
            global_index.add(row)
            cross_coig_kept.append(row)
        else:
            kept_row = global_index.rows[hit]
            if kept_row.get("_source_file") == row.get("_source_file"):
                # Already handled in the source-local pass.
                global_index.add(row)
                cross_coig_kept.append(row)
                continue
            routed = dict(row)
            routed.update(
                _reason="near_duplicate_across_coig_sources",
                _kept_uid=kept_row.get("uid"),
                _kept_in=kept_row.get("_source_file"),
                _near_similarity=round(score, 6),
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="simhash_candidate+char5gram_jaccard",
            )
            cross_coig_removed.append(routed)

    # Pass 3: existing specialist corpora have priority.
    specialist_index = NearIndex()
    for row in specialist_rows():
        specialist_index.add(row)

    final_kept: list[dict[str, Any]] = []
    existing_removed: list[dict[str, Any]] = []
    destinations: Counter[str] = Counter()
    for row in cross_coig_kept:
        hit, score = best_match(row, specialist_index, EXISTING_THRESHOLD)
        if hit is None:
            final_kept.append(row)
            continue
        kept_row = specialist_index.rows[hit]
        routed = dict(row)
        routed.update(
            _reason="near_duplicate_with_existing_specialist",
            _kept_uid=kept_row.get("uid"),
            _kept_in=kept_row.get("dataset"),
            _near_similarity=round(score, 6),
            _dedup_stage="stage1",
            _dedup_date=DATE,
            _dedup_method="simhash_candidate+char5gram_jaccard",
        )
        existing_removed.append(routed)
        destinations[str(kept_row.get("dataset"))] += 1

    write_jsonl(OUT / "kept.jsonl", final_kept)
    write_jsonl(OUT / "removed_near_intra.jsonl", intra_removed)
    write_jsonl(OUT / "removed_near_cross_coig.jsonl", cross_coig_removed)
    write_jsonl(OUT / "removed_near_existing.jsonl", existing_removed)

    report = {
        "date": DATE,
        "input": len(rows),
        "parameters": {
            "ngram": NGRAM,
            "intra_threshold": INTRA_THRESHOLD,
            "cross_coig_threshold": CROSS_COIG_THRESHOLD,
            "existing_threshold": EXISTING_THRESHOLD,
        },
        "removed_near_intra": len(intra_removed),
        "removed_near_cross_coig": len(cross_coig_removed),
        "removed_near_existing": len(existing_removed),
        "existing_destinations": dict(destinations),
        "final_kept": len(final_kept),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
