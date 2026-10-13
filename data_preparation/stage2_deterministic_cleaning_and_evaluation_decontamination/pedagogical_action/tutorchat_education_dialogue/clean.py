#!/usr/bin/env python3
"""Pure-code cleaning for TutorChat and Education Dialogue Dataset.

The cleaner preserves official splits, normalizes speaker labels only, drops
empty/malformed conversations, performs exact dialogue deduplication, and
protects evaluation splits from train leakage. It never paraphrases messages.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

SPLIT_ORDER = ("eval", "validation", "train")
HF_PARQUET_API = (
    "https://datasets-server.huggingface.co/parquet" "?dataset=princeton-nlp%2FTutorChat"
)
EDD_API = (
    "https://api.github.com/repos/" "google-research-datasets/Education-Dialogue-Dataset/contents"
)
TAGGED_TURN_RE = re.compile(
    r"(?:^|</s>\s*)<s>\s*(assistant|user|system)\s*:\s*(.*?)(?=</s>|$)",
    re.IGNORECASE | re.DOTALL,
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
END_MARKER_RE = re.compile(r"^\s*\[?\s*end of conversation\s*\]?\s*$", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: Any) -> str | None:
    """Remove invalid controls and edge whitespace without rewriting wording."""
    if not isinstance(value, str):
        return None
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = CONTROL_RE.sub("", value).strip()
    return value or None


def dialogue_key(messages: list[dict[str, str]]) -> str:
    # Role-normalized, text-exact identity after the minimal sanitation above.
    payload = [[m["role"], m["content"]] for m in messages]
    return sha256_bytes(stable_json(payload).encode("utf-8"))


def download(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    h = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "dialogue-fast-clean/1"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            size += len(chunk)
    os.replace(tmp, destination)
    return {"url": url, "path": str(destination), "bytes": size, "sha256": h.hexdigest()}


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "dialogue-fast-clean/1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def acquire_sources(cache_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    parquet_manifest = fetch_json(HF_PARQUET_API)
    for item in parquet_manifest["parquet_files"]:
        split = item["split"]
        dest = cache_dir / "TutorChat" / split / item["filename"]
        meta = download(item["url"], dest)
        meta.update(dataset="TutorChat", split=split, format="parquet")
        sources.append(meta)
    manifests.append(
        {
            "dataset": "TutorChat",
            "source": "princeton-nlp/TutorChat",
            "manifest_url": HF_PARQUET_API,
            "official_splits": sorted({x["split"] for x in sources if x["dataset"] == "TutorChat"}),
            "license": {"status": "not_specified_in_dataset_card", "value": None},
            "synthetic_provenance": {
                "synthetic": True,
                "evidence": "per-row speak0/speak1 model fields",
            },
        }
    )

    contents = fetch_json(EDD_API)
    for item in contents:
        name = item["name"]
        if not name.startswith("conversations_") or not name.endswith(".json"):
            continue
        split = "eval" if name == "conversations_eval.json" else "train"
        dest = cache_dir / "Education-Dialogue-Dataset" / split / name
        meta = download(item["download_url"], dest)
        meta.update(dataset="Education-Dialogue-Dataset", split=split, format="json")
        sources.append(meta)
    manifests.append(
        {
            "dataset": "Education-Dialogue-Dataset",
            "source": "google-research-datasets/Education-Dialogue-Dataset",
            "manifest_url": EDD_API,
            "official_splits": ["train", "eval"],
            "license": {
                "status": "not_specified_in_repository",
                "value": None,
                "note": "No LICENSE file and no GitHub-detected license at acquisition.",
            },
            "synthetic_provenance": {
                "synthetic": True,
                "generator": "Gemini Ultra",
                "evidence": "official repository README",
            },
        }
    )
    return manifests, sources


def parse_tagged_conversation(value: Any) -> list[dict[str, str]]:
    text = normalize_text(value)
    if not text:
        return []
    messages: list[dict[str, str]] = []
    for match in TAGGED_TURN_RE.finditer(text):
        content = normalize_text(match.group(2))
        if content:
            messages.append({"role": match.group(1).lower(), "content": content})
    return messages


def tutorchat_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str | None]:
    conversation = row.get("conversation")
    # PyArrow commonly materializes this list column as numpy.ndarray. Use the
    # raw ordered turns plus the explicit ``name`` variant; source-provided
    # processed tags are inconsistent for teacher-start variants.
    if hasattr(conversation, "tolist"):
        conversation = conversation.tolist()
    if not isinstance(conversation, (list, tuple)) or len(conversation) < 3:
        return [], "conversation_not_valid_sequence"
    texts = [normalize_text(x) for x in conversation]
    if any(x is None for x in texts):
        return [], "non_string_or_empty_turn"

    name = str(row.get("name") or "").casefold()
    mode = str(row.get("mode") or "").casefold()
    # ``mode=openbook`` describes access to source material, not who speaks
    # first. Prefer the explicit generation variant in ``name``. The previous
    # ordering treated every open-book teacher-start dialogue as student-start
    # and inverted user/assistant roles.
    if "studentstart" in name:
        first_dialogue_role = "user"
    elif "teacherstart" in name:
        first_dialogue_role = "assistant"
    elif mode == "openbook":
        first_dialogue_role = "user"
    else:
        return [], "speaker_order_not_identifiable"

    messages = [{"role": "system", "content": texts[0]}]
    for index, text in enumerate(texts[1:]):
        role = (
            first_dialogue_role
            if index % 2 == 0
            else ("assistant" if first_dialogue_role == "user" else "user")
        )
        messages.append({"role": role, "content": text})
    return messages, None


def education_dialogue_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str | None]:
    conversation = row.get("conversation")
    if not isinstance(conversation, list):
        return [], "conversation_not_valid_sequence"
    messages: list[dict[str, str]] = []
    role_map = {"teacher": "assistant", "student": "user"}
    for turn in conversation:
        if not isinstance(turn, dict):
            return [], "turn_not_object"
        source_role = str(turn.get("role") or "").strip().casefold()
        role = role_map.get(source_role)
        text = normalize_text(turn.get("text"))
        if not role:
            return [], "unknown_role"
        if not text:
            return [], "empty_turn"
        if END_MARKER_RE.fullmatch(text):
            continue
        messages.append({"role": role, "content": text})
    return messages, None


def validate_messages(messages: list[dict[str, str]]) -> str | None:
    dialogue = [m for m in messages if m["role"] != "system"]
    if len(dialogue) < 2:
        return "fewer_than_two_dialogue_turns"
    roles = {m["role"] for m in dialogue}
    if not {"user", "assistant"}.issubset(roles):
        return "missing_user_or_assistant"
    return None


def classify_level(dataset: str, metadata: dict[str, Any]) -> tuple[str, str]:
    if dataset == "Education-Dialogue-Dataset":
        return "k12", "official_generation_prompt_says_teacher_in_school"
    path = str(metadata.get("textbook_folder") or "").casefold()
    higher_markers = (
        "engineering",
        "calculus",
        "linear_algebra",
        "differential_equations",
        "university",
        "college",
        "organic_chemistry",
        "statistics",
        "mechanics",
        "thermodynamics",
        "electromagnet",
    )
    k12_markers = (
        "arithmetic_and_basic_math",
        "prealgebra",
        "elementary",
        "middle_school",
        "high_school",
    )
    if any(x in path for x in higher_markers):
        return "higher_ed", "textbook_path_keyword"
    if any(x in path for x in k12_markers):
        return "k12", "textbook_path_keyword"
    return "unknown", "no_explicit_or_reliable_level_metadata"


def iter_tutorchat(source: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(source["path"])
    index = 0
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def iter_education_dialogue(source: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(source["path"], encoding="utf-8") as handle:
        rows = json.load(handle)
    for index, row in enumerate(rows):
        yield index, row


class Cleaner:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.audit_dir = output_dir / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(output_dir / "dedup.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS seen (
               dialogue_id TEXT PRIMARY KEY, dataset TEXT, split TEXT,
               source_file TEXT, source_row INTEGER)"""
        )
        self.handles: dict[tuple[str, str], Any] = {}
        self.duplicates = gzip.open(self.audit_dir / "duplicates.jsonl.gz", "wt", encoding="utf-8")
        self.stats: dict[str, Counter[str]] = defaultdict(Counter)
        self.levels: dict[str, Counter[str]] = defaultdict(Counter)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.duplicates.close()
        self.db.commit()
        self.db.close()

    def output_handle(self, dataset: str, split: str):
        key = (dataset, split)
        if key not in self.handles:
            directory = pools.cleaned_dir("pedagogical_action", dataset)
            directory.mkdir(parents=True, exist_ok=True)
            self.handles[key] = gzip.open(directory / f"{split}.jsonl.gz", "wt", encoding="utf-8")
        return self.handles[key]

    def write_standard_pools(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for dataset in sorted({key[0] for key in self.handles}):
            directory = pools.cleaned_dir("pedagogical_action", dataset)
            kept_path = directory / "kept.jsonl"
            written = 0
            with kept_path.open("w", encoding="utf-8") as out:
                for split in SPLIT_ORDER:
                    part = directory / f"{split}.jsonl.gz"
                    if not part.exists():
                        continue
                    with gzip.open(part, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            out.write(line)
                            written += 1
            counts[dataset] = written
        return counts

    def process_source(self, source: dict[str, Any]) -> None:
        dataset, split = source["dataset"], source["split"]
        iterator = (
            iter_tutorchat(source) if dataset == "TutorChat" else (iter_education_dialogue(source))
        )
        parser = tutorchat_messages if dataset == "TutorChat" else (education_dialogue_messages)
        source_name = Path(source["path"]).name

        for source_row, row in iterator:
            stats = self.stats[f"{dataset}/{split}"]
            stats["input"] += 1
            messages, reason = parser(row)
            reason = reason or validate_messages(messages)
            if reason:
                stats["dropped_bad"] += 1
                stats[f"dropped_bad:{reason}"] += 1
                continue

            metadata: dict[str, Any]
            if dataset == "TutorChat":
                metadata = {
                    key: row.get(key)
                    for key in (
                        "mode",
                        "name",
                        "stop_reason",
                        "textbook_folder",
                        "speak0",
                        "speak1",
                        "num_turns",
                    )
                    if row.get(key) is not None
                }
                synthetic = {
                    "synthetic": True,
                    "models": sorted({str(row.get(k)) for k in ("speak0", "speak1") if row.get(k)}),
                }
                license_info = {"status": "not_specified_in_dataset_card", "value": None}
            else:
                metadata = dict(row.get("background_info") or {})
                synthetic = {"synthetic": True, "generator": "Gemini Ultra"}
                license_info = {"status": "not_specified_in_repository", "value": None}

            level, level_basis = classify_level(dataset, metadata)
            did = dialogue_key(messages)
            existing = self.db.execute(
                "SELECT dataset,split,source_file,source_row FROM seen WHERE dialogue_id=?",
                (did,),
            ).fetchone()
            if existing:
                kind = (
                    "within_split"
                    if existing[:2] == (dataset, split)
                    else ("cross_split" if existing[1] != split else "cross_dataset")
                )
                stats[f"dropped_duplicate:{kind}"] += 1
                self.duplicates.write(
                    stable_json(
                        {
                            "dialogue_id": did,
                            "duplicate_kind": kind,
                            "dropped": {
                                "dataset": dataset,
                                "split": split,
                                "source_file": source_name,
                                "source_row": source_row,
                            },
                            "kept": {
                                "dataset": existing[0],
                                "split": existing[1],
                                "source_file": existing[2],
                                "source_row": existing[3],
                            },
                        }
                    )
                    + "\n"
                )
                continue

            self.db.execute(
                "INSERT INTO seen VALUES (?,?,?,?,?)",
                (did, dataset, split, source_name, source_row),
            )
            record = {
                "dialogue_id": did,
                "dataset": dataset,
                "split": split,
                "messages": messages,
                "metadata": metadata,
                "education_level": {"label": level, "basis": level_basis},
                "license": license_info,
                "synthetic_provenance": synthetic,
                "source_provenance": {
                    "source_file": source_name,
                    "source_row": source_row,
                    "source_sha256": source["sha256"],
                },
            }
            self.output_handle(dataset, split).write(stable_json(record) + "\n")
            stats["kept"] += 1
            self.levels[f"{dataset}/{split}"][level] += 1
            if stats["input"] % 5000 == 0:
                self.db.commit()
                print(f"{dataset}/{split}: {stats['input']} input, {stats['kept']} kept")


def main() -> None:
    """?? TutorChat ? Education-Dialogue-Dataset???????"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Raw download cache (default: temporary directory, deleted after run).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temp: tempfile.TemporaryDirectory[str] | None = None
    if args.cache_dir:
        cache_dir = args.cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp = tempfile.TemporaryDirectory(prefix="dialogue-clean-")
        cache_dir = Path(temp.name)

    started = utc_now()
    manifests, sources = acquire_sources(cache_dir)
    # Protect held-out data: register eval, then validation, before train.
    priority = {"eval": 0, "validation": 1, "train": 2}
    sources.sort(key=lambda x: (priority[x["split"]], x["dataset"], x["path"]))

    cleaner = Cleaner(output_dir)
    try:
        for source in sources:
            print(f"Processing {source['dataset']}/{source['split']}: {source['path']}")
            cleaner.process_source(source)
        counts = cleaner.stats
        levels = cleaner.levels
        kept_pools = cleaner.write_standard_pools()
    finally:
        cleaner.close()

    finished = utc_now()
    for dataset in kept_pools:
        report = {
            "dataset": dataset,
            "started_at": started,
            "finished_at": finished,
            "policy": {
                "content_rewriting": False,
                "text_changes": "line-ending normalization, invalid-control removal, edge trim only",
                "bad_dialogue_filter": "drop malformed/empty turns and conversations lacking both user and assistant",
                "dedup": "SHA-256 over exact minimally-sanitized role/content sequence",
                "cross_split_precedence": ["eval", "validation", "train"],
                "split_preservation": "records never move splits; lower-priority exact duplicates are dropped",
            },
            "dataset_manifest": manifests.get(dataset),
            "counts": {
                key: dict(value) for key, value in sorted(counts.items()) if key.startswith(f"{dataset}/")
            },
            "education_level_counts": {
                key: dict(value) for key, value in sorted(levels.items()) if key.startswith(f"{dataset}/")
            },
            "kept_rows": kept_pools[dataset],
        }
        directory = pools.cleaned_dir("pedagogical_action", dataset)
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if temp:
        temp.cleanup()
    print(json.dumps({"kept_pools": kept_pools}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
