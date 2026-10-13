#!/usr/bin/env python3
"""Near-deduplicate merged Aquila nonmath + math-uncertain candidates.

Uses dependency-free SimHash candidate generation plus char-5gram Jaccard.
Reading-comprehension keys emphasize the final question/options, so multiple
questions sharing one passage are not collapsed. Existing specialist corpora
and filtered COIG have priority. All removed rows are retained in route files.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

PIPE = pools.STAGE2 / "subject_competence"
AQ = PIPE / "AquilaEdu/quality_filtered"
OUT = PIPE / "AquilaEdu/near_dedup"
DATE = "2026-08-05"
NGRAM = 5
INTRA_THRESHOLD = 0.94
EXISTING_THRESHOLD = 0.97
KEEP = set("<>=+-*/^\\{}")


def norm(value: Any) -> str:
    text = str(value or "").lower().replace("$", "")
    return "".join(c for c in text if c.isalnum() or c in KEEP)


WRAPPER_RE = re.compile(
    r"(?im)(?:^|[\n\r])\s*(?:question|options?|article|passage|"
    r"题目内容|题目|问题|选项|选择)\s*[:：]\s*"
)
OPTION_RE = re.compile(r"(?im)(^|[\s　])(?:[\(（]?\s*[a-h]\s*[\)）]|[a-h][\.:、．])\s*")
QUESTION_SPLIT_RE = re.compile(r"(?is)(?:question|问题)\s*[:：]")


def canon(value: Any) -> str:
    text = WRAPPER_RE.sub("\n", str(value or ""))
    text = OPTION_RE.sub(" ", text)
    return norm(text)


def dedup_text(row: dict[str, Any]) -> str:
    text = str(row.get("input") or row.get("problem") or row.get("instruction") or "")
    parts = QUESTION_SPLIT_RE.split(text)
    # For reading data, compare the final question/options rather than the long
    # shared passage. Require a meaningful suffix to avoid wrapper accidents.
    if len(parts) > 1 and len(parts[-1].strip()) >= 20:
        return canon(parts[-1])
    return canon(text)


def shingles(text: str) -> set[str]:
    if not text:
        return set()
    if len(text) <= NGRAM:
        return {text}
    return {text[i : i + NGRAM] for i in range(len(text) - NGRAM + 1)}


def h64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(tokens: set[str]) -> int:
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        value = h64(token)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def chunks(value: int) -> tuple[int, int, int, int]:
    return tuple((value >> (16 * i)) & 0xFFFF for i in range(4))


def jac(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class Index:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.grams: list[set[str]] = []
        self.maps = [defaultdict(list) for _ in range(4)]

    def add(self, row: dict[str, Any]) -> None:
        grams = shingles(dedup_text(row))
        fp = simhash(grams)
        idx = len(self.rows)
        self.rows.append(row)
        self.grams.append(grams)
        for pos, value in enumerate(chunks(fp)):
            self.maps[pos][value].append(idx)

    def match(self, row: dict[str, Any], threshold: float) -> tuple[int | None, float]:
        grams = shingles(dedup_text(row))
        fp = simhash(grams)
        candidates: set[int] = set()
        for pos, value in enumerate(chunks(fp)):
            candidates.update(self.maps[pos].get(value, []))
        best = None
        score = 0.0
        for idx in candidates:
            current = jac(grams, self.grams[idx])
            if current >= threshold and current > score:
                best, score = idx, current
        return best, score


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def quality_rank(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row.get("quality") or 0),
        float(row.get("c_q_score") or 0),
        len(str(row.get("output") or "")),
    )


def existing_rows() -> Iterable[dict[str, Any]]:
    specs = [
        ("CJEval", PIPE / "CJEval/cleaned/kept.jsonl"),
        ("ARC", PIPE / "ARC/cleaned/kept.jsonl"),
        ("QASC", PIPE / "QASC/cleaned/kept.jsonl"),
        ("SciQ", PIPE / "SciQ/cleaned/kept.jsonl"),
        ("WorldTree", PIPE / "WorldTree/cleaned/kept.jsonl"),
        ("TQA (text)", PIPE / "TQA (text)/cleaned/kept.jsonl"),
        ("C3", PIPE / "C3/cleaned/kept.jsonl"),
        ("COIG-CQIA", PIPE / "COIG-CQIA/cleaned/kept.jsonl"),
    ]
    for owner, path in specs:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if owner.startswith("reading-"):
                content = "\n".join(
                    [
                        str(row.get("passage") or ""),
                        f"Question: {row.get('problem') or ''}",
                        str(row.get("options") or ""),
                    ]
                )
            else:
                content = "\n".join([str(row.get("problem") or ""), str(row.get("options") or "")])
            yield {
                "uid": row.get("uid"),
                "dataset": owner,
                "input": content,
            }


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = list(read_jsonl(AQ / "math_uncertain.jsonl"))
    rows += list(read_jsonl(AQ / "kept_nonmath_candidate.jsonl"))
    rows.sort(key=quality_rank, reverse=True)

    index = Index()
    intra_kept: list[dict[str, Any]] = []
    intra_removed: list[dict[str, Any]] = []
    for row in rows:
        hit, score = index.match(row, INTRA_THRESHOLD)
        if hit is None:
            index.add(row)
            intra_kept.append(row)
        else:
            routed = dict(row)
            routed.update(
                _reason="near_duplicate_within_aquila",
                _kept_uid=index.rows[hit].get("uid"),
                _near_similarity=round(score, 6),
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="simhash_candidate+char5gram_jaccard",
            )
            intra_removed.append(routed)

    ext_index = Index()
    for row in existing_rows():
        ext_index.add(row)

    final: list[dict[str, Any]] = []
    cross_removed: list[dict[str, Any]] = []
    destinations: Counter[str] = Counter()
    for row in intra_kept:
        hit, score = ext_index.match(row, EXISTING_THRESHOLD)
        if hit is None:
            final.append(row)
        else:
            kept = ext_index.rows[hit]
            routed = dict(row)
            routed.update(
                _reason="near_duplicate_with_existing_specialist",
                _kept_uid=kept.get("uid"),
                _kept_in=kept.get("dataset"),
                _near_similarity=round(score, 6),
                _dedup_stage="stage1",
                _dedup_date=DATE,
                _dedup_method="simhash_candidate+char5gram_jaccard",
            )
            cross_removed.append(routed)
            destinations[str(kept.get("dataset"))] += 1

    write(OUT / "kept.jsonl", final)
    write(OUT / "removed_near_intra.jsonl", intra_removed)
    write(OUT / "removed_near_existing.jsonl", cross_removed)
    report = {
        "date": DATE,
        "input": len(rows),
        "intra_threshold": INTRA_THRESHOLD,
        "existing_threshold": EXISTING_THRESHOLD,
        "removed_near_intra": len(intra_removed),
        "removed_near_existing": len(cross_removed),
        "existing_destinations": dict(destinations),
        "kept": len(final),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
