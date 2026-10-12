"""GAOKAO-Bench: replay the official objective and subjective prompts.

The prompts come from the release's ``Bench/Obj_Prompt.json`` and
``Bench/Sub_Prompt.json``, including the official two-stage objective prompt for
the correction task.  The message layout matches the official client
(``Models/openai_gpt4.py``): the benchmark's ``prefix_prompt`` is the **system**
message and the question is the **user** message.

Objective answers are parsed locally; subjective answers are scored by the
judge over ``judge/prompts/gaokao_subject_prompts.json``.

Temperature is 0.0 per the paper's protocol (Appendix D); the upstream client
defaults to 0.3 for objective questions, so this is an intentional deviation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from ..bundle import write_workload
from ..engine import THINKING_OFF, chat, endpoints, fan_out
from ..metrics import GAOKAO as GAOKAO_COLUMNS
from ..metrics import build, judgments, workload

NAME = "GAOKAO-Bench"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gaokao-kind", choices=["objective", "subjective", "both"], default="both"
    )
    parser.add_argument(
        "--gaokao-repo", default=None, help="GAOKAO-Bench checkout (holds Bench/ and Data/)"
    )


def extract_choice_answer(text: str, question_type: str, count: int) -> list[str]:
    if question_type == "single_choice":
        blocks = re.findall(r"\u3010\u7b54\u6848\u3011\s*[:\uff1a]?\s*([A-D])", text)
        if blocks:
            return [blocks[-1]]
        reversed_found = re.findall(r"[A-D]", text[::-1])
        return [reversed_found[0]] if reversed_found else []
    if question_type == "multi_question_choice":
        found = re.findall(r"\u3010\u7b54\u6848\u3011\s*[:\uff1a]?\s*([A-Z])", text)
        if len(found) == count:
            return found
        return re.findall(r"[A-Z]", text)[:count]
    if question_type == "multi_choice":
        compact = re.sub(r"\s+", "", text)
        cut = compact.find("\u3010\u7b54\u6848\u3011")
        tail = compact[cut:] if cut >= 0 else compact[-10:]
        answer = "".join(re.findall(r"[A-D]", tail))
        return [answer] if answer else []
    if question_type == "five_out_of_seven":
        return re.findall(r"[A-G]", text)[:5]
    return []


def _payload(model: str, system: str, question: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": THINKING_OFF,
    }


def load_jobs(kind: str, repo: Path) -> tuple[list[dict[str, Any]], list[tuple[Any, Path, Any]]]:
    prompt_file = repo / "Bench" / ("Obj_Prompt.json" if kind == "objective" else "Sub_Prompt.json")
    data_dir = (
        repo / "Data" / ("Objective_Questions" if kind == "objective" else "Subjective_Questions")
    )
    specs = json.loads(prompt_file.read_text(encoding="utf-8"))["examples"]
    jobs: list[tuple[Any, Path, Any]] = []
    for spec in specs:
        path = data_dir / (spec["keyword"] + ".json")
        for row in json.loads(path.read_text(encoding="utf-8"))["example"]:
            jobs.append((spec, path, row))
    return specs, jobs


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    urls = endpoints(args.base_urls)
    kinds = ["objective", "subjective"] if args.gaokao_kind == "both" else [args.gaokao_kind]
    repo = Path(
        args.gaokao_repo
        or getattr(args, "official", {}).get(NAME)
        or Path(args.data_root) / "gaokao_bench"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"benchmark": "GAOKAO-Bench", "model": args.model, "kinds": {}}
    async with httpx.AsyncClient(timeout=600) as client:
        for kind in kinds:
            # One directory per question kind: the official objective scorer globs
            # every ``*.json`` it is pointed at, so it must not see the subjective
            # answers (and vice versa).
            summary["kinds"][kind] = await _run_kind(
                args, client, urls, repo, output_dir / kind, kind
            )
    summary["metrics"] = _metrics(args, repo, output_dir)
    return summary


def _objective(repo: Path, output_dir: Path, python: str) -> dict[str, Any] | None:
    """Run the benchmark's own ``OBJ_score_evaluation.py`` and read its result.

    Objective answers are scored by the official script -- which awards partial
    credit on multi-select questions -- and never re-implemented here.
    """
    script = repo / "Bench" / "OBJ_score_evaluation.py"
    if not script.is_file():
        return None
    # The scorer prints one warning per mismatched answer; keep that in a log
    # next to the run instead of on the caller's console.
    log = output_dir / "official_objective_score.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(
            [python, script.name, f"--obj_output_dir={output_dir / 'objective'}"],
            cwd=script.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = output_dir / "objective" / "result" / "correction_score.json"
    if not result.is_file():
        return None
    return json.loads(result.read_text(encoding="utf-8"))


def _metrics(args: argparse.Namespace, repo: Path, output_dir: Path) -> dict[str, Any]:
    values: dict[str, float | None] = {}
    objective = _objective(repo, output_dir, str(args.python))
    if objective:
        correct = float(objective.get("correct_score") or 0)
        total = float(objective.get("total_score") or 0)
        values["Obj Rate"] = correct / total * 100 if total else None
        for subject, block in (objective.get("subject") or {}).items():
            if subject in GAOKAO_COLUMNS:
                values[subject] = float(block.get("scoring_rate") or 0) * 100

    sources = workload(output_dir)
    got = maximum = 0.0
    for record in judgments(output_dir):
        score = (record.get("parsed") or {}).get("score")
        top = (sources.get(record["judge_id"]) or {}).get("score")
        if score is None or not top:
            continue
        got += float(score)
        maximum += float(top)
    if maximum:
        values["Subj Rate"] = got / maximum * 100
    if objective and maximum:
        values["Full Rate"] = (
            (float(objective.get("correct_score") or 0) + got)
            / (float(objective.get("total_score") or 0) + maximum)
            * 100
        )
    return build("GAOKAO-Bench", values)


async def _run_kind(
    args: argparse.Namespace,
    client: Any,
    urls: list[str],
    repo: Path,
    output_dir: Path,
    kind: str,
) -> dict[str, Any]:
    max_tokens = (
        args.max_tokens if args.max_tokens is not None else (2048 if kind == "objective" else 4096)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, jobs = load_jobs(kind, repo)
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    for spec in specs:
        path = output_dir / f"{args.model}_{spec['keyword']}.json"
        if path.exists():
            try:
                for row in json.loads(path.read_text(encoding="utf-8")).get("example", []):
                    if not row.get("error"):
                        existing[(spec["keyword"], str(row["index"]), str(row.get("year")))] = row
            except Exception:
                pass

    buckets: dict[str, list[dict[str, Any]]] = {spec["keyword"]: [] for spec in specs}
    done_count = 0

    async def worker(index: int, job: tuple[Any, Path, Any]) -> None:
        nonlocal done_count
        spec, _, row = job
        key = (spec["keyword"], str(row["index"]), str(row.get("year")))
        if key in existing:
            record = existing[key]
        else:
            url = urls[index % len(urls)]
            prompt = spec["prefix_prompt"]
            question = row["question"].strip() + "\n"
            error = None
            response = ""
            if isinstance(prompt, list):  # official correction is two-stage
                first, error = await chat(
                    client, url, _payload(args.model, prompt[0], question, max_tokens)
                )
                if not error:
                    start = first.find("\u3010\u7b54\u6848\u3011")
                    end = first.find("<eoa>")
                    article2 = (
                        first[start + 4 : end if end >= 0 else None].strip() if start >= 0 else ""
                    )
                    response, error = await chat(
                        client,
                        url,
                        _payload(
                            args.model,
                            prompt[1],
                            "Article 1:" + question + "\nArticle 2:" + article2,
                            max_tokens,
                        ),
                    )
            else:
                response, error = await chat(
                    client, url, _payload(args.model, prompt, question, max_tokens)
                )
            gold = row.get("standard_answer", row.get("answer"))
            if not isinstance(gold, list):
                gold = [gold]
            record = {
                "index": row["index"],
                "year": row.get("year"),
                "category": row.get("category"),
                "score": row.get("score"),
                "question": question,
                "standard_answer": gold,
                "analysis": row.get("analysis"),
                "model_output": response,
            }
            if kind == "objective":
                record["model_answer"] = extract_choice_answer(response, spec["type"], len(gold))
            if error:
                record["error"] = error
        buckets[spec["keyword"]].append(record)
        done_count += 1
        if done_count % 25 == 0:
            print(f"[gaokao:{kind}] {done_count}/{len(jobs)}", flush=True)

    await fan_out(jobs, lambda index, job: worker(index, job), concurrency=args.concurrency)
    for spec in specs:
        rows = sorted(
            buckets[spec["keyword"]], key=lambda item: (str(item.get("year")), int(item["index"]))
        )
        target = output_dir / f"{args.model}_{spec['keyword']}.json"
        target.write_text(
            json.dumps(
                {
                    "model_name": args.model,
                    "keyword": spec["keyword"],
                    "question_type": spec["type"],
                    "example": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if kind == "subjective":
        write_workload(
            output_dir,
            "gaokao_subjective.jsonl",
            _subjective_workload(buckets, specs, args.model),
        )
    return {
        "kind": kind,
        "count": done_count,
        "errors": sum("error" in item for rows in buckets.values() for item in rows),
    }


def _subjective_workload(
    buckets: dict[str, list[dict[str, Any]]], specs: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """The GAOKAO judge input: one row per subjective answer, keyed by subject."""
    matrix = {spec["keyword"]: spec for spec in specs}
    rows = []
    for keyword, items in buckets.items():
        for item in sorted(items, key=lambda entry: (str(entry.get("year")), int(entry["index"]))):
            rows.append(
                {
                    "judge_id": f"gaokao::{model}::{keyword}::{item.get('year')}::{item['index']}",
                    "benchmark": "GAOKAO-Bench",
                    "sample_id": f"{keyword}::{item.get('year')}::{item['index']}",
                    "question": item.get("question"),
                    "reference_answer": item.get("standard_answer"),
                    "reference_solution": item.get("analysis"),
                    "model_response": item.get("model_output"),
                    "question_type": matrix[keyword]["type"],
                    "year": item.get("year"),
                    "score": item.get("score"),
                    "metadata": {"keyword": keyword, "category": item.get("category")},
                }
            )
    return rows
