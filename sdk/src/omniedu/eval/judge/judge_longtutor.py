"""LongTutor teaching-quality judge (memory, diagnosis and content rubric).

Reads rows with ``judge_id``, ``model``, ``reference`` and ``prediction`` (the
shape ``omniedu.eval.bundle`` writes) and produces one judgment per row.  Memory
and diagnosis matches are forced true when the strings are already identical, so
the judge only grades the genuinely ambiguous cases and the teaching-content
rubric.  ROUGE-L is computed locally (see :func:`rouge_l`) and rides along in the
same record, so the lexical `R-L` column needs no separate pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..client import normalize_base_url
from ..io import iter_jsonl, load_jsonl

MEM_TYPES = ["Information Extraction", "Multi-session Reasoning", "Hallucination Check"]
SCORE_KEYS = [
    "history_score",
    "strategy_score",
    "coherence_score",
    "comprehension_score",
    "appropriateness_score",
    "overall_score",
]
_local = threading.local()

SYSTEM = """You are a strict educational evaluator. Judge every requested field independently.

Memory and diagnosis match rules:
- true only when MODEL conveys the same core meaning as GOLD for the query.
- false for contradictions or factual errors. Ignore wording, length, and punctuation differences.

Teaching-content rubric (integer 1-5; 1 very poor, 2 poor, 3 acceptable, 4 good, 5 excellent):
1. history_score: uses the student's learning history appropriately to transition and connect new knowledge; not forced or irrelevant.
2. strategy_score: precisely targets the same misconception or goal rather than giving generic help.
3. coherence_score: logically consistent, fluent, clear, natural, and well structured.
4. comprehension_score: concise, easy to understand, minimal jargon, and appropriate difficulty for the learner.
5. appropriateness_score (Appropriateness/ZPD): keep language simple (K-6 level), do not use complex academic jargon unless explaining a definition; the response should be neither too hard nor too easy for the learner, and should match the level of the GOLD reference teaching content.
6. overall_score: holistic teaching quality.

Return ONLY one valid JSON object, without markdown:
{"memory_match": [true, false, true], "diagnosis_match": true, "history_score": 1, "strategy_score": 1, "coherence_score": 1, "comprehension_score": 1, "appropriateness_score": 1, "overall_score": 1, "reason": "concise justification covering all teaching dimensions"}
The memory_match array must have exactly three booleans in the supplied order. For an item marked exact_match=true, always return true."""


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def rouge_l(gold_text: str, pred_text: str) -> float:
    """The paper's `R-L` column: ``rouge_score`` with its default tokenizer.

    This is the upstream call in LongTutor's ``compute_ai_tutor_eval_metrics.py`` --
    ``RougeScorer(["rougeL"], use_stemmer=True)`` -- and it is what the reported
    numbers come from.  Checked against a real run: recomputing this way over the
    ``gpt-5.6-sol`` workload reproduces the stored value to six decimals
    (0.365068, the paper's 0.3651), where a character-level variant gives 0.2328.

    Note the tokenizer lowercases and keeps only ``[a-z0-9]`` runs, so on Chinese
    teaching content the score is driven by the few surviving tokens (mostly the
    ``[2412]``-style references).  That is the benchmark's definition of the
    column, so it is kept as-is rather than "improved" into a real lexical overlap.

    Empty on either side scores 0, and the caller averages over every row, which is
    what the official script does.
    """
    if not (gold_text or "").strip() or not (pred_text or "").strip():
        return 0.0
    from rouge_score import rouge_scorer

    # A scorer per call, as the official script does, to avoid any thread-safety
    # surprise between the judging workers.
    return float(
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        .score(gold_text, pred_text)["rougeL"]
        .fmeasure
    )


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    return obj


def validate(obj: dict[str, Any]) -> None:
    memory = obj.get("memory_match")
    if (
        not isinstance(memory, list)
        or len(memory) != 3
        or any(not isinstance(x, bool) for x in memory)
    ):
        raise ValueError("bad memory_match")
    if not isinstance(obj.get("diagnosis_match"), bool):
        raise ValueError("bad diagnosis_match")
    for key in SCORE_KEYS:
        value = obj.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or int(value) != value
            or not 1 <= int(value) <= 5
        ):
            raise ValueError(f"bad {key}")
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        raise ValueError("bad reason")


def _client(base_url: str, api_key: str) -> OpenAI:
    if not hasattr(_local, "client"):
        _local.client = OpenAI(
            base_url=normalize_base_url(base_url), api_key=api_key, timeout=180, max_retries=0
        )
    return _local.client


def judge(row: dict[str, Any], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    gold, prediction = row["reference"], row["prediction"]
    gold_memory = gold.get("memory") or []
    pred_memory = prediction.get("memory") or []
    memory = []
    exact = []
    for index, name in enumerate(MEM_TYPES):
        gold_item = (
            gold_memory[index]
            if index < len(gold_memory) and isinstance(gold_memory[index], dict)
            else {}
        )
        pred_item = (
            pred_memory[index]
            if index < len(pred_memory) and isinstance(pred_memory[index], dict)
            else {}
        )
        is_exact = norm(gold_item.get("answer")) == norm(pred_item.get("answer"))
        exact.append(is_exact)
        memory.append(
            {
                "type": name,
                "query": norm(gold_item.get("query")),
                "gold_answer": norm(gold_item.get("answer")),
                "model_answer": norm(pred_item.get("answer")),
                "exact_match": is_exact,
            }
        )
    gold_diag, pred_diag = norm(gold.get("diagnosis")), norm(prediction.get("diagnosis"))
    diag_exact = bool(pred_diag) and pred_diag == gold_diag
    payload = {
        "memory": memory,
        "diagnosis": {"gold": gold_diag, "model": pred_diag, "exact_match": diag_exact},
        "teaching_content": {
            "gold": norm(gold.get("content")),
            "model": norm(prediction.get("content")),
        },
    }
    # Deterministic, LLM-free lexical overlap; computed before the API call so it
    # survives retries/exhaustion exactly like the official script's own ROUGE-L pass.
    rl = rouge_l(norm(gold.get("content")), norm(prediction.get("content")))
    last_error = None
    for attempt in range(args.retries):
        try:
            response = _client(args.base_url, api_key).chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            obj = extract_json(response.choices[0].message.content or "")
            validate(obj)
            for index, value in enumerate(exact):
                if value:
                    obj["memory_match"][index] = True
            if diag_exact:
                obj["diagnosis_match"] = True
            return {
                "judge_id": row["judge_id"],
                "benchmark": "LongTutor",
                "evaluated_model": row["model"],
                "judge_model": args.model,
                "memory_correct": obj["memory_match"],
                "diagnosis_correct": obj["diagnosis_match"],
                "diagnosis_pred": pred_diag,
                "diagnosis_gold": gold_diag,
                "content_scores": {key: int(obj[key]) for key in SCORE_KEYS},
                "content_reason": obj["reason"],
                "rouge_l": rl,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < args.retries:
                time.sleep(min(20, (2**attempt) + random.random()))
    return {
        "judge_id": row["judge_id"],
        "benchmark": "LongTutor",
        "evaluated_model": row["model"],
        "judge_model": args.model,
        "rouge_l": rl,
        "error": last_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {
        record["judge_id"]
        for record in (iter_jsonl(output) if output.exists() else ())
        if record.get("judge_id") and not record.get("error")
    }
    rows = [row for row in load_jsonl(args.input) if row["judge_id"] not in done]
    if args.limit:
        rows = rows[: args.limit]
    print(json.dumps({"already_done": len(done), "queued": len(rows)}), flush=True)

    lock = threading.Lock()
    errors = completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge, row, args, args.api_key) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                errors += bool(result.get("error"))
                if completed % 50 == 0 or completed == len(rows):
                    print(f"[longtutor] {len(done) + completed} errors={errors}", flush=True)
    return 0 if (args.allow_errors or not errors) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
