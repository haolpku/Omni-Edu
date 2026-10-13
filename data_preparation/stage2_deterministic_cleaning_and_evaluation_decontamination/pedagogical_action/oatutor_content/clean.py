#!/usr/bin/env python3
"""Stream-clean OATutor Content from a .tar.gz without extracting members."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

OATUTOR_OUT = pools.cleaned_dir("pedagogical_action", "OATutor")
TEXT_KEYS = (
    "text",
    "content",
    "body",
    "value",
    "prompt",
    "question",
    "problem",
    "statement",
    "description",
    "message",
    "html",
    "markdown",
)
ID_KEYS = (
    "id",
    "_id",
    "uuid",
    "uid",
    "problem_id",
    "problemId",
    "question_id",
    "questionId",
    "item_id",
    "itemId",
    "content_id",
    "contentId",
)
TITLE_KEYS = ("title", "name", "label")
ANSWER_KEYS = (
    "answer",
    "answers",
    "correct_answer",
    "correctAnswer",
    "solution",
    "solutions",
    "response",
    "expected",
    "expected_answer",
    "expectedAnswer",
)
OPTION_KEYS = ("options", "choices", "answers", "selections")
HINT_KEYS = (
    "hint",
    "hints",
    "hint_text",
    "hintText",
    "feedback",
    "explanation",
    "explanations",
)
SCAFFOLD_KEYS = (
    "scaffold",
    "scaffolds",
    "scaffolding",
    "steps",
    "subgoals",
    "worked_example",
    "workedExample",
    "worked_solution",
    "workedSolution",
)
LICENSE_KEYS = (
    "license",
    "licence",
    "license_name",
    "licenseName",
    "license_url",
    "licenseUrl",
    "rights",
    "copyright",
    "attribution",
)
SOURCE_KEYS = (
    "source",
    "source_name",
    "sourceName",
    "source_url",
    "sourceUrl",
    "dataset",
    "repository",
    "repo",
    "origin",
    "url",
)
QUESTIONISH_KEYS = set(HINT_KEYS + SCAFFOLD_KEYS + ANSWER_KEYS + OPTION_KEYS) | {
    "question",
    "problem",
    "prompt",
    "statement",
    "stem",
}
QUESTION_TYPES = {
    "question",
    "problem",
    "exercise",
    "item",
    "assessment",
    "task",
    "multiple_choice",
    "multiple-choice",
    "mcq",
}
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean_text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value).replace("\x00", "")
    value = SPACE_RE.sub(" ", value).strip()
    return value or None


def canonical_text(value: str | None) -> str:
    if not value:
        return ""
    value = HTML_RE.sub(" ", value)
    value = unicodedata.normalize("NFKC", value).casefold()
    return SPACE_RE.sub(" ", value).strip()


def first_scalar(obj: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in obj:
            text = clean_text(obj[key])
            if text:
                return text
    return None


def collect_strings(value: Any, *, _depth: int = 0) -> list[str]:
    if _depth > 8 or value is None:
        return []
    text = clean_text(value)
    if text:
        return [text]
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            out.extend(collect_strings(item, _depth=_depth + 1))
    elif isinstance(value, dict):
        preferred = first_scalar(value, TEXT_KEYS)
        if preferred:
            out.append(preferred)
        else:
            for item in value.values():
                out.extend(collect_strings(item, _depth=_depth + 1))
    return unique_texts(out)


def unique_texts(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = canonical_text(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def extract_named_strings(obj: dict[str, Any], keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        if key in obj:
            out.extend(collect_strings(obj[key]))
    return unique_texts(out)


def extract_prompt(obj: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "prompt", "statement", "stem"):
        if key not in obj:
            continue
        value = obj[key]
        direct = clean_text(value)
        if direct:
            return direct
        if isinstance(value, dict):
            nested = first_scalar(value, TEXT_KEYS + TITLE_KEYS)
            if nested:
                return nested
    typ = canonical_text(first_scalar(obj, ("type", "kind", "object_type", "objectType")))
    if typ in QUESTION_TYPES or any(k in obj for k in QUESTIONISH_KEYS):
        return first_scalar(obj, TEXT_KEYS + TITLE_KEYS)
    return None


def extract_metadata(obj: dict[str, Any], keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    wanted = {k.casefold() for k in keys}

    def walk(value: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                folded = key.casefold()
                if folded in wanted or any(token in folded for token in wanted):
                    values.extend(collect_strings(item))
                elif folded in {"metadata", "meta", "provenance", "sourceinfo", "source_info"}:
                    walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(obj, 0)
    return unique_texts(values)


def likely_container(obj: dict[str, Any]) -> bool:
    if extract_prompt(obj):
        return False
    list_values = [value for value in obj.values() if isinstance(value, list)]
    return bool(list_values) and any(
        isinstance(item, dict) for value in list_values for item in value[:3]
    )


def iter_candidate_objects(root: Any) -> Iterable[tuple[Any, str]]:
    if isinstance(root, dict):
        yield root, "$"
        if likely_container(root):
            for key, value in root.items():
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, dict):
                            yield item, f"$.{key}[{index}]"
    elif isinstance(root, list):
        for index, item in enumerate(root):
            if isinstance(item, dict):
                yield item, f"$[{index}]"


def parse_json_payload(raw: bytes) -> tuple[list[Any], int]:
    text = raw.decode("utf-8-sig")
    if not text.strip():
        return [], 1
    try:
        return [json.loads(text)], 0
    except json.JSONDecodeError:
        pass
    values: list[Any] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return values, bad or (0 if values else 1)


def make_record(
    obj: dict[str, Any],
    member_name: str,
    json_path: str,
    archive: str,
) -> dict[str, Any] | None:
    prompt = extract_prompt(obj)
    hints = extract_named_strings(obj, HINT_KEYS)
    scaffolds = extract_named_strings(obj, SCAFFOLD_KEYS)
    if not prompt or not (hints or scaffolds):
        return None
    answers = extract_named_strings(obj, ANSWER_KEYS)
    options = extract_named_strings(obj, OPTION_KEYS)
    licenses = extract_metadata(obj, LICENSE_KEYS)
    sources = extract_metadata(obj, SOURCE_KEYS)
    source_id = first_scalar(obj, ID_KEYS)
    title = first_scalar(obj, TITLE_KEYS)
    prompt_norm = canonical_text(prompt)
    content_identity = {
        "prompt": prompt_norm,
        "hints": sorted(canonical_text(x) for x in hints),
        "scaffolds": sorted(canonical_text(x) for x in scaffolds),
        "answers": sorted(canonical_text(x) for x in answers),
        "options": sorted(canonical_text(x) for x in options),
    }
    return {
        "record_id": sha256_text(stable_json(content_identity)),
        "question_key": sha256_text(prompt_norm),
        "question": {
            "source_id": source_id,
            "title": title,
            "prompt": prompt,
            "options": options,
            "answers": answers,
        },
        "hints": hints,
        "scaffolds": scaffolds,
        "license": licenses,
        "source": sources,
        "provenance": [
            {
                "archive": archive,
                "member": member_name,
                "json_path": json_path,
                "source_id": source_id,
            }
        ],
    }


def init_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(
        """CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            question_key TEXT NOT NULL,
            record_json TEXT NOT NULL
        )"""
    )
    db.execute("CREATE INDEX records_question_key ON records(question_key)")
    return db


def merge_record(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    existing["provenance"].extend(new["provenance"])
    existing["license"] = unique_texts(existing["license"] + new["license"])
    existing["source"] = unique_texts(existing["source"] + new["source"])
    return existing


def clean_content(archive: Path, output: Path, max_member_bytes: int) -> int:
    """? content pool ????????????????? content.jsonl?"""
    output.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        raise SystemExit(f"input archive not found: {archive}")

    stats: Counter[str] = Counter()
    members_by_suffix: Counter[str] = Counter()
    fd, db_name = tempfile.mkstemp(prefix=".oatutor-clean-", suffix=".sqlite3", dir=output)
    os.close(fd)
    db_path = Path(db_name)
    db = init_db(db_path)
    archive_label = str(archive)

    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                stats["tar_members"] += 1
                if not member.isfile():
                    stats["non_file_members"] += 1
                    continue
                suffix = Path(member.name).suffix.casefold() or "<none>"
                members_by_suffix[suffix] += 1
                if suffix not in {".json", ".jsonl", ".ndjson"}:
                    stats["non_json_members"] += 1
                    continue
                stats["json_members"] += 1
                if member.size == 0:
                    stats["empty_json_members"] += 1
                    continue
                if member.size > max_member_bytes:
                    stats["oversize_json_members"] += 1
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    stats["unreadable_json_members"] += 1
                    continue
                try:
                    raw = stream.read(max_member_bytes + 1)
                    if len(raw) > max_member_bytes:
                        stats["oversize_json_members"] += 1
                        continue
                    roots, bad = parse_json_payload(raw)
                except (UnicodeError, OSError, ValueError):
                    roots, bad = [], 1
                if bad:
                    stats["invalid_json_values"] += bad
                if not roots:
                    stats["empty_or_invalid_json_members"] += 1
                    continue
                stats["parsed_json_values"] += len(roots)
                for root in roots:
                    for obj, json_path in iter_candidate_objects(root):
                        stats["candidate_objects"] += 1
                        record = make_record(obj, member.name, json_path, archive_label)
                        if record is None:
                            stats["filtered_no_question_or_pedagogy"] += 1
                            continue
                        stats["eligible_records"] += 1
                        row = db.execute(
                            "SELECT record_json FROM records WHERE record_id=?",
                            (record["record_id"],),
                        ).fetchone()
                        if row:
                            stats["exact_duplicates"] += 1
                            merged = merge_record(json.loads(row[0]), record)
                            db.execute(
                                "UPDATE records SET record_json=? WHERE record_id=?",
                                (stable_json(merged), record["record_id"]),
                            )
                        else:
                            db.execute(
                                "INSERT INTO records VALUES (?, ?, ?)",
                                (
                                    record["record_id"],
                                    record["question_key"],
                                    stable_json(record),
                                ),
                            )
                if stats["json_members"] % 10000 == 0:
                    db.commit()
        db.commit()

        duplicate_question_groups = db.execute(
            "SELECT COUNT(*) FROM (SELECT question_key FROM records GROUP BY question_key HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        same_question_duplicate_records = db.execute(
            "SELECT COALESCE(SUM(n - 1), 0) FROM (SELECT COUNT(*) n FROM records GROUP BY question_key HAVING n > 1)"
        ).fetchone()[0]
        stats["same_question_duplicate_groups"] = duplicate_question_groups
        stats["same_question_duplicate_records"] = same_question_duplicate_records

        records_path = output / "content.jsonl"
        duplicates_path = output / "same_question_duplicates.jsonl"
        tmp_records = records_path.with_suffix(".jsonl.tmp")
        tmp_duplicates = duplicates_path.with_suffix(".jsonl.tmp")
        kept = 0
        with tmp_records.open("w", encoding="utf-8") as out, tmp_duplicates.open(
            "w", encoding="utf-8"
        ) as dup:
            for question_key, count in db.execute(
                "SELECT question_key, COUNT(*) FROM records GROUP BY question_key ORDER BY question_key"
            ):
                rows = [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT record_json FROM records WHERE question_key=? ORDER BY record_id",
                        (question_key,),
                    )
                ]
                if count > 1:
                    dup.write(
                        stable_json(
                            {
                                "question_key": question_key,
                                "variants": rows,
                                "dedup_policy": "kept one representative; all variants retained in this audit file",
                            }
                        )
                        + "\n"
                    )
                winner = min(
                    rows,
                    key=lambda r: (
                        -(len(r["hints"]) + len(r["scaffolds"])),
                        -sum(len(x) for x in r["hints"] + r["scaffolds"]),
                        r["record_id"],
                    ),
                )
                all_provenance = [provenance for row in rows for provenance in row["provenance"]]
                winner["provenance"] = all_provenance
                winner["license"] = unique_texts(value for row in rows for value in row["license"])
                winner["source"] = unique_texts(value for row in rows for value in row["source"])
                winner["dedup"] = {
                    "same_question_variant_count": count,
                    "variant_record_ids": [row["record_id"] for row in rows],
                    "selection": "max pedagogy entries, then max pedagogy text length, then record_id",
                }
                out.write(stable_json(winner) + "\n")
                kept += 1
        os.replace(tmp_records, records_path)
        os.replace(tmp_duplicates, duplicates_path)
        stats["records_after_exact_dedup"] = db.execute("SELECT COUNT(*) FROM records").fetchone()[
            0
        ]
        stats["records_kept"] = kept
        stats["records_removed_total"] = stats["eligible_records"] - kept

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "archive": archive_label,
                "size_bytes": archive.stat().st_size,
                "streaming": True,
                "archive_extracted": False,
            },
            "outputs": {
                "records": str(records_path),
                "same_question_duplicate_audit": str(duplicates_path),
            },
            "counts": dict(sorted(stats.items())),
            "member_suffix_counts": dict(members_by_suffix.most_common()),
            "deduplication": {
                "exact": "SHA-256 of normalized prompt + normalized hints/scaffolds/answers/options; provenance merged",
                "same_question": "SHA-256 of normalized prompt; one richest pedagogical variant kept",
                "normalization": "Unicode NFKC, HTML tags removed for keys, casefold, whitespace collapse",
            },
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        db.close()
        for extra in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            try:
                extra.unlink()
            except FileNotFoundError:
                pass
    return 0


# ---------------------------------------------------------------- joined

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tarfile
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path


def norm(x):
    if x is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(x)).replace("\x00", "").split()).strip()


def stable(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def strings(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [norm(v)] if norm(v) else []
    if isinstance(v, (int, float)):
        return [str(v)]
    if isinstance(v, list):
        out = []
        for z in v:
            out += strings(z)
        return list(dict.fromkeys(out))
    if isinstance(v, dict):
        for k in ("text", "content", "body", "value", "title", "stepTitle"):
            if k in v and norm(v[k]):
                return [norm(v[k])]
        out = []
        for z in v.values():
            out += strings(z)
        return list(dict.fromkeys(out))
    return []


def join_guided_dialogues(archive: Path, output: Path) -> None:
    """?????????????????????? kept.jsonl?"""
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "joined.sqlite3")
    db.executescript(
        "DROP TABLE IF EXISTS problem;DROP TABLE IF EXISTS step;DROP TABLE IF EXISTS tutor;CREATE TABLE problem(id TEXT PRIMARY KEY,j TEXT,member TEXT);CREATE TABLE step(id TEXT PRIMARY KEY,pid TEXT,j TEXT,member TEXT);CREATE TABLE tutor(sid TEXT PRIMARY KEY,j TEXT,member TEXT);"
    )
    stats = Counter()
    with tarfile.open(archive, "r|gz") as t:
        for m in t:
            if (
                not m.isfile()
                or not m.name.endswith(".json")
                or not m.name.startswith("content-pool/")
            ):
                continue
            f = t.extractfile(m)
            try:
                x = json.loads(f.read().decode("utf-8-sig"))
            except Exception:
                stats["bad_json"] += 1
                continue
            parts = Path(m.name).parts
            if len(parts) == 3 and isinstance(x, dict):
                pid = norm(x.get("id"))
                if pid:
                    db.execute(
                        "INSERT OR REPLACE INTO problem VALUES(?,?,?)",
                        (pid, json.dumps(x, ensure_ascii=False), m.name),
                    )
                    stats["problems"] += 1
            elif len(parts) == 5 and parts[-2] != "tutoring" and isinstance(x, dict):
                sid = norm(x.get("id"))
                pid = parts[1]
                if sid:
                    db.execute(
                        "INSERT OR REPLACE INTO step VALUES(?,?,?,?)",
                        (sid, pid, json.dumps(x, ensure_ascii=False), m.name),
                    )
                    stats["steps"] += 1
            elif "tutoring" in parts and isinstance(x, list):
                sid = parts[-1].replace("DefaultPathway.json", "")
                db.execute(
                    "INSERT OR REPLACE INTO tutor VALUES(?,?,?)",
                    (sid, json.dumps(x, ensure_ascii=False), m.name),
                )
                stats["tutoring"] += 1
    db.commit()
    rows = []
    rejected = []
    q = "SELECT s.id,s.pid,s.j,s.member,p.j,p.member,t.j,t.member FROM step s LEFT JOIN problem p ON p.id=s.pid LEFT JOIN tutor t ON t.sid=s.id"
    for sid, pid, sj, sm, pj, pm, tj, tm in db.execute(q):
        s = json.loads(sj)
        p = json.loads(pj) if pj else {}
        t = json.loads(tj) if tj else []
        question = norm(s.get("stepTitle") or s.get("title") or s.get("body") or s.get("stepBody"))
        hints = []
        scaff = []
        for z in t:
            if not isinstance(z, dict):
                continue
            text = norm(z.get("text") or z.get("body") or z.get("title"))
            if not text:
                continue
            typ = norm(z.get("type")).casefold()
            (hints if typ == "hint" else scaff).append(text)
        hints = list(dict.fromkeys(hints))
        scaff = list(dict.fromkeys(scaff))
        if not question or not (hints or scaff):
            rejected.append(
                {
                    "step_id": sid,
                    "reason": "missing_question_or_pedagogy",
                    "step_member": sm,
                    "tutoring_member": tm,
                }
            )
            continue
        answers = strings(s.get("stepAnswer"))
        options = strings(s.get("choices"))
        identity = {
            "q": question.casefold(),
            "h": sorted(x.casefold() for x in hints),
            "s": sorted(x.casefold() for x in scaff),
            "a": sorted(x.casefold() for x in answers),
            "o": sorted(x.casefold() for x in options),
        }
        rows.append(
            {
                "record_id": "oatutor::" + sha(stable(identity))[:24],
                "dataset": "OATutor-Content",
                "problem_id": pid,
                "step_id": sid,
                "course_name": norm(p.get("courseName")) or None,
                "lesson": norm(p.get("lesson")) or None,
                "problem_title": norm(p.get("title")) or None,
                "question": question,
                "options": options,
                "answers": answers,
                "hints": hints,
                "scaffolds": scaff,
                "license": norm(p.get("license")) or None,
                "oer": norm(p.get("oer")) or None,
                "provenance": {
                    "archive": str(archive),
                    "problem_member": pm,
                    "step_member": sm,
                    "tutoring_member": tm,
                },
            }
        )
    # exact dedup, merge provenance
    keep = {}
    dups = []
    for r in rows:
        k = r["record_id"]
        if k in keep:
            dups.append(
                {
                    "reason": "exact_duplicate",
                    "record_id": k,
                    "duplicate_provenance": r["provenance"],
                }
            )
        else:
            keep[k] = r
    with (output / "kept.jsonl").open("w") as f:
        for r in keep.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (output / "rejected.jsonl").open("w") as f:
        for r in rejected + dups:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "dataset": "OATutor-Content",
        "counts": dict(stats)
        | {
            "joined_steps": len(rows),
            "kept": len(keep),
            "rejected_missing": len(rejected),
            "exact_duplicates": len(dups),
        },
        "input": str(archive),
        "streaming": True,
        "no_rewrite": True,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    db.close()
    os.remove(output / "joined.sqlite3")


def main() -> None:
    """?? OATutor??????????????????????"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", type=Path, required=True, help="content pool ??")
    parser.add_argument("--joined", type=Path, required=True, help="???????????")
    parser.add_argument("--max-member-bytes", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    clean_content(args.content, OATUTOR_OUT, args.max_member_bytes)
    join_guided_dialogues(args.joined, OATUTOR_OUT)


if __name__ == "__main__":
    main()
