"""LongTutor: official protocol, deterministic memory/diagnosis scoring.

Prompts are rebuilt with the benchmark's own ``scripts/eval_ai_tutor.py`` helpers
and the diagnosis labels and strategy map are imported from
``scripts/gpt_memory_diagnose.py``, so no local prompt copy can drift from the
official one.  The official LongTutor scripts are not redistributed; point
``--official-root LongTutor=PATH`` at the checkout (it must contain ``scripts``).

Teaching quality is a separate judge step (``omniedu.eval.judge``); this module
reports the deterministic memory, diagnosis and strategy metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..bundle import read_jsonl, write_workload
from ..engine import THINKING_OFF, Sink, chat, endpoints, fan_out
from ..io import resume
from ..metrics import with_judged

NAME = "LongTutor"
MEMORY_TYPES = ["information_extraction", "multi_session_reasoning", "hallucination_check"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--longtutor-repo", default=None, help="LongTutor checkout (holds scripts/)"
    )
    parser.add_argument(
        "--longtutor-data-dir", default=None, help="directory holding features/tests shards"
    )
    parser.add_argument(
        "--longtutor-shard",
        type=int,
        default=None,
        help="run a single shard (default: every features.shard*.jsonl present)",
    )


def _import_official(repo: Path) -> tuple[Any, Any, Any, Any]:
    scripts = repo / "scripts"
    if not scripts.is_dir():
        raise SystemExit(
            f"LongTutor official scripts not found under {repo}; set --official-root {NAME}=PATH"
        )
    sys.path.insert(0, str(scripts))
    from eval_ai_tutor import _build_prompts, _extract_mem_queries_from_test_obj
    from gpt_memory_diagnose import DIAGNOSES, strategy_MAP

    return _build_prompts, _extract_mem_queries_from_test_obj, DIAGNOSES, strategy_MAP


def extract_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def normalize(
    obj: Any, queries: list[Any], key: str, diagnoses: Any, strategy_map: Any
) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    diagnosis = obj.get("diagnosis")
    strategy = obj.get("strategy")
    if diagnosis not in diagnoses or strategy != strategy_map.get(diagnosis):
        return None
    memory = obj.get("memory")
    if not isinstance(memory, list):
        return None
    mapping: dict[str, str] = {}
    for index, item in enumerate(memory):
        if isinstance(item, dict):
            mapping[str(item.get("id", f"Q{index + 1}")).upper()] = str(item.get("answer", ""))
    obj["memory"] = [
        {"id": f"Q{index + 1}", "answer": mapping.get(f"Q{index + 1}", "")}
        for index in range(len(queries))
    ]
    if not isinstance(obj.get("reason"), str) or not isinstance(obj.get("content"), str):
        return None
    obj["_key"] = key
    return obj


def _payload(model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": THINKING_OFF,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    urls = endpoints(args.base_urls)
    repo = Path(
        args.longtutor_repo
        or getattr(args, "official", {}).get(NAME)
        or Path(args.data_root) / "longtutor"
    )
    data_dir = Path(args.longtutor_data_dir or Path(args.data_root) / "longtutor")
    build_prompts, extract_queries, diagnoses, strategy_map = _import_official(repo)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = args.max_tokens if args.max_tokens is not None else 3000

    if args.longtutor_shard is None:
        shard_files = sorted(data_dir.glob("features.shard*.jsonl"))
    else:
        shard_files = [data_dir / f"features.shard{args.longtutor_shard}.jsonl"]
    if not shard_files:
        raise SystemExit(f"no features.shard*.jsonl under {data_dir}")

    async with httpx.AsyncClient(timeout=900) as client:
        for shard in shard_files:
            tests_path = data_dir / shard.name.replace("features.", "tests.")
            await _run_shard(
                args,
                client,
                urls,
                output_dir,
                shard,
                tests_path,
                build_prompts,
                extract_queries,
                diagnoses,
                strategy_map,
                max_tokens,
            )

    merged = output_dir / "predictions.jsonl"
    with merged.open("wb") as target:
        for shard_path in sorted(output_dir.glob("predictions.shard*.jsonl")):
            target.write(shard_path.read_bytes())
    return score(
        merged,
        len([line for line in merged.read_text(encoding="utf-8").splitlines() if line.strip()]),
        args,
        data_dir,
    )


async def _run_shard(
    args: argparse.Namespace,
    client: Any,
    urls: list[str],
    output_dir: Path,
    features_path: Path,
    tests_path: Path,
    build_prompts: Any,
    extract_queries: Any,
    diagnoses: Any,
    strategy_map: Any,
    max_tokens: int,
) -> None:
    features = [
        json.loads(line)
        for line in features_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tests = [
        json.loads(line)
        for line in tests_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(features) == len(tests)

    output = output_dir / features_path.name.replace("features.", "predictions.")
    errors_path = output.with_name(output.stem + ".errors.jsonl")
    done = resume(
        output, key="_key", failed=lambda row: str(row.get("reason") or "").startswith("ERROR:")
    )
    sink = Sink(output)
    errors = Sink(errors_path)

    async def worker(index: int, pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
        sample, test = pair
        key = sample["_eval_key"]
        if key in done:
            return
        system, user = build_prompts(sample, test)
        queries = extract_queries(test)
        result = None
        last = ""
        for attempt in range(5):
            content, _ = await chat(
                client, urls[index % len(urls)], _payload(args.model, system, user, max_tokens)
            )
            last = content
            result = normalize(extract_obj(content), queries, key, diagnoses, strategy_map)
            if result is not None:
                break
        if result is None:
            await errors.write(
                {"_key": key, "model": args.model, "error": "invalid_output", "raw_output": last}
            )
            return
        result["_corpus"] = sample.get("_corpus")
        await sink.write(result)

    await fan_out(
        list(zip(features, tests)),
        worker,
        concurrency=args.concurrency,
        chunk=max(32, args.concurrency * 4),
    )


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def score(
    predictions: Path, count: int, args: argparse.Namespace, data_dir: Path
) -> dict[str, Any]:
    gold: dict[str, Any] = {}
    for path in sorted(data_dir.glob("*_clean.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                gold[record["_key"]] = record
    rows = read_jsonl(predictions)
    write_workload(
        predictions.parent, "longtutor_teaching.jsonl", _teaching_workload(rows, gold, args.model)
    )

    memory_correct = [0, 0, 0]
    memory_total = [0, 0, 0]
    diag_gold: Counter[str] = Counter()
    diag_pred: Counter[str] = Counter()
    diag_tp: Counter[str] = Counter()
    diagnosis_correct = strategy_correct = matched = invalid = 0
    by_corpus: dict[str, dict[str, Any]] = {}

    for row in rows:
        reference = gold.get(row.get("_key"))
        if not reference:
            continue
        matched += 1
        corpus = row.get("_corpus") or "unknown"
        bucket = by_corpus.setdefault(corpus, {"count": 0, "diagnosis_correct": 0})
        bucket["count"] += 1
        gold_memory = reference.get("memory") or []
        pred_memory = row.get("memory") or []
        for position in range(3):
            memory_total[position] += 1
            gold_answer = (
                gold_memory[position].get("answer")
                if position < len(gold_memory) and isinstance(gold_memory[position], dict)
                else ""
            )
            pred_answer = (
                pred_memory[position].get("answer")
                if position < len(pred_memory) and isinstance(pred_memory[position], dict)
                else ""
            )
            memory_correct[position] += _norm(gold_answer) == _norm(pred_answer)
        gold_diag = _norm(reference.get("diagnosis"))
        pred_diag = _norm(row.get("diagnosis"))
        diag_gold[gold_diag] += 1
        diag_pred[pred_diag] += 1
        if gold_diag == pred_diag and gold_diag:
            diagnosis_correct += 1
            diag_tp[gold_diag] += 1
            bucket["diagnosis_correct"] += 1
        strategy_correct += _norm(reference.get("strategy")) == _norm(row.get("strategy"))
        invalid += not all(_norm(row.get(key)) for key in ["diagnosis", "strategy", "content"])

    labels = set(diag_gold) | set(diag_pred)
    per_label: dict[str, Any] = {}
    f1s: list[float] = []
    for label in sorted(labels):
        if not label:
            continue
        tp = diag_tp[label]
        pred_count = diag_pred[label]
        gold_count = diag_gold[label]
        precision = tp / pred_count if pred_count else 0
        recall = tp / gold_count if gold_count else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        f1s.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "pred": pred_count,
            "gold": gold_count,
        }
    for value in by_corpus.values():
        value["diagnosis_accuracy"] = (
            value["diagnosis_correct"] / value["count"] if value["count"] else None
        )

    total_memory = sum(memory_total) or 1
    summary = {
        "benchmark": "LongTutor",
        "model": args.model,
        "predictions": count,
        "matched_gold": matched,
        "missing_gold": count - matched,
        "invalid_predictions": invalid,
        "memory": {
            "correct": sum(memory_correct),
            "total": sum(memory_total),
            "accuracy": sum(memory_correct) / total_memory,
            "by_type": {
                name: {
                    "correct": memory_correct[index],
                    "total": memory_total[index],
                    "accuracy": (
                        memory_correct[index] / memory_total[index] if memory_total[index] else None
                    ),
                }
                for index, name in enumerate(MEMORY_TYPES)
            },
        },
        "diagnosis": {
            "correct": diagnosis_correct,
            "total": matched,
            "accuracy": diagnosis_correct / matched if matched else None,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "per_label": per_label,
        },
        "strategy_exact_accuracy": strategy_correct / matched if matched else None,
        "by_corpus": by_corpus,
        "teaching_quality": "LLM judge pending",
        "protocol": "LongTutor clean gold; normalized exact memory answers; exact diagnosis/strategy",
    }
    values: dict[str, float | None] = {
        "Acc": (
            None
            if summary["diagnosis"]["accuracy"] is None
            else summary["diagnosis"]["accuracy"] * 100
        ),
        "F1": (
            None
            if summary["diagnosis"]["macro_f1"] is None
            else summary["diagnosis"]["macro_f1"] * 100
        ),
    }
    summary["metrics"] = with_judged("LongTutor", predictions.parent, values)
    return summary


def _teaching_workload(
    rows: list[dict[str, Any]], gold: dict[str, Any], model: str
) -> list[dict[str, Any]]:
    """The teaching-quality judge input: reference and prediction side by side."""
    workload = []
    for row in rows:
        reference = gold.get(row.get("_key"))
        if not reference:
            continue
        workload.append(
            {
                "judge_id": f"longtutor::{model}::{row.get('_key')}",
                "benchmark": "LongTutor",
                "model": model,
                "sample_id": row.get("_key"),
                "reference": {
                    key: reference.get(key)
                    for key in ["memory", "diagnosis", "reason", "strategy", "content"]
                },
                "prediction": {
                    key: row.get(key)
                    for key in ["memory", "diagnosis", "reason", "strategy", "content"]
                },
                "corpus": row.get("_corpus"),
            }
        )
    return workload
