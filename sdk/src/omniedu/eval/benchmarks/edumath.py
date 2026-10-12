"""EDUMATH: standards-aligned generation plus the official MaC classifier.

The generation prompt reproduces ``bryanchrist/EDUMATH``
``generate_with_interests.py`` verbatim: the instruction paragraph, the "Here
are some examples:" block and the "Write a new question about ..." sentence are
copied character for character (the Qwen branch), each sample is conditioned on a
student interest from the official ``topics`` list plus a grade and math topic,
and up to eight reference examples come from the same subtopic (topped up from
other subtopics of the same grade when the subtopic is small).

Two deliberate differences from upstream, both plumbing: this talks to an
OpenAI-compatible endpoint instead of loading a local HF model, and sampling uses
a per-sample seed so a rerun reproduces the same prompts (upstream is unseeded).

``--official-protocol`` switches to the upstream keep-until-N semantics used to
match the paper's ``Classifier GOOD`` denominator; the default path generates a
fixed N items.  The classifier stage replicates the official ``annotate.py``
classifier path (ModernBERT) and needs ``torch`` + ``transformers``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from ..bundle import read_jsonl, write_workload
from ..engine import THINKING_OFF, Sink, chat, endpoints, fan_out
from ..io import resume
from ..metrics import with_judged

NAME = "EDUMATH"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_SEED = 20260811

# Upstream `topics` list, de-duplicated preserving order exactly as upstream does
# with `interests = list(dict.fromkeys(topics))`.
INTERESTS = [
    "Superman",
    "Batman",
    "Wonder Woman",
    "Barbie",
    "Power Rangers",
    "basketball",
    "soccer",
    "football",
    "volleyball",
    "field hockey",
    "Fortnite",
    "Spiderman",
    "Iron Man",
    "Captain America",
    "Captain Marvel",
    "Thor, the God of Thunder",
    "Black Panther",
    "Taylor Swift",
    "swimming",
    "Pok\u00e9mon",
    "Super Mario",
    "Naruto",
    "unicorns",
    "Hello Kitty",
    "Minecraft",
    "lacrosse",
    "cheer leading",
    "LeBron James",
    "Steph Curry",
    "Patrick Mahomes",
    "Serena Williams",
    "dogs",
    "cats",
    "dinosaurs",
    "Harry Potter",
    "cars",
    "planes",
    "trains",
    "pizza",
    "cookies",
    "ice cream",
    "candy",
    "Frozen (Elsa and Anna)",
    "Star Wars",
    "Paw Patrol",
    "My Little Pony",
    "Minions",
    "Jurassic Park",
    "SpongeBob SquarePants",
    "Disney Princesses",
    "Toy Story",
    "The Incredibles",
    "Scooby-Doo",
    "Peppa Pig",
    "Dora the Explorer",
    "Pikachu",
    "Thomas the Tank Engine",
    "Sonic the Hedgehog",
    "Transformers",
    "Cinderella",
    "Moana",
    "Shrek",
    "Winnie the Pooh",
    "Tom and Jerry",
    "Sesame Street",
    "The Lion King",
    "Alice in Wonderland",
    "The Little Mermaid",
    "Peter Pan",
    "Aladdin",
    "The Jungle Book",
    "Pocahontas",
    "Beauty and the Beast",
    "Frozen",
    "Ratatouille",
    "Finding Nemo",
    "Cars",
    "Up",
    "The Simpsons",
    "Looney Tunes",
    "Teenage Mutant Ninja Turtles",
    "Mythical Creatures (dragons, unicorns)",
    "Dinosaurs",
    "Space and Astronauts",
    "Robots",
    "Aliens",
    "Exploring the Ocean",
    "Underwater Creatures",
    "Pirates",
    "Fairies",
    "Wizards",
    "Magic Tricks",
    "Time Travel",
    "Detectives and Mystery",
    "Inventions",
    "The Avengers",
    "The Justice League",
    "Dance and Ballet",
    "Music Instruments",
    "Art and Drawing",
    "Science Experiments",
    "Cooking and Baking",
    "DIY Crafts",
    "Board Games",
    "Puzzles",
    "Riddles",
    "Pets (cats, dogs, hamsters)",
    "Farm Animals",
    "Zoo Animals",
    "Wildlife Conservation",
    "Plants and Gardening",
    "Hiking and Nature",
    "Weather and Meteorology",
    "The Solar System",
    "Camping",
    "National Parks",
    "Trains and Railroads",
    "Planes and Aviation",
    "Cars and Racing",
    "Construction Vehicles",
    "Firefighters",
    "Police Officers",
    "Doctors and Nurses",
    "Astronauts and Space Exploration",
    "Animals and Wildlife",
    "Space and Astronomy",
    "Robots and Technology",
    "Underwater Life",
    "Fairy Tales and Folklore",
    "Outer Space",
    "Music and Instruments",
    "Insects and Bugs",
    "Historical Figures",
    "Countries and Cultures",
    "Mythical Creatures",
    "Magic and Wizards",
    "Friendship and Relationships",
    "Ocean Life",
    "Cars and Vehicles",
    "Famous Inventors",
    "Famous Artists",
    "Ancient Civilizations",
    "Space Exploration",
    "Gardening",
    "Environmental Conservation",
    "Pirates and Treasure",
    "Famous Scientists",
    "Computer Programming",
    "Unexplained Mysteries",
    "Planets and the Solar System",
    "Cartoons and Animated Shows",
    "Photography",
    "Books and Reading",
    "Volcanoes",
    "Mythology",
    "Ancient Egypt",
    "Reptiles and Amphibians",
    "Recycling",
    "Fairy Gardens",
    "Indoor Games",
    "Marine Biology",
    "Virtual Reality",
    "Natural Disasters",
    "Construction and Building",
    "the Circus and Performing Arts",
    "Science Fiction",
    "Pottery and Ceramics",
    "Famous Explorers",
    "Birds and Bird Watching",
    "Famous Landmarks",
    "Health and Nutrition",
    "Myths and Legends",
    "Fashion and Clothing",
    "DIY Science Projects",
    "Cultural Festivals",
    "Forests and Trees",
    "Mummies",
    "Famous Composers",
    "Circus Animals",
    "Geology",
    "Farm Life",
    "Travel and Adventure",
    "Ballet and Dance",
    "Whales and Dolphins",
    "Mystery Stories",
    "Hiking and Camping",
    "Games and Puzzles",
    "Space Aliens and UFOs",
]

INSTRUCTION = (
    "You are an experienced teacher tasked with writing word problems and solutions for 3rd-5th grade students. "
    "The question you write will be based on a grade level and math topic(s). The question's content should exactly "
    "match and incorporate ALL of the mathematical topics and constraints listed in the math topic(s). The question "
    "and answer pair you write should be solvable with the information presented in the question, contain an accurate "
    "solution, and contain language and context appropriate for a 3rd-5th grade student in a school setting (i.e., no "
    "harmful language and topics should be appropriate for school settings)."
)
# Byte-for-byte the tail upstream renders: the odd whitespace (`\n    \n`, the
# trailing spaces, the line holding a single space) is the indentation of
# upstream's triple-quoted f-string, and the newline inside `Solution:\n` is what
# a `\n` escape produces there.
PROMPT_TAIL = (
    "\n    \nHere are some examples: \n{examples}\n \n"
    "Write a new question about {interest} based on the grade level and math topic(s) below. "
    "Make sure to incorporate all of the information in the math topic(s) into your question; for example, "
    "if the math topic(s) mention remainders, the problem you write should require a remainder. Also make sure "
    'to write "Question:" followed by your question and "Solution:\n" followed by your solution. \n{query}'
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stem-csv", default=None, help="<data-root>/edumath/stem.csv")
    parser.add_argument(
        "--official-protocol",
        action="store_true",
        help="use upstream keep-until-N sampling instead of a fixed N",
    )
    parser.add_argument(
        "--reuse-file", default=None, help="reuse valid rows from an earlier generations.jsonl"
    )
    parser.add_argument("--classifier-model", default=None, help="local EDUMATH_classifier path")
    parser.add_argument(
        "--classifier-tokenizer", default=None, help="local ModernBERT tokenizer path"
    )


def example_block(row: dict[str, Any]) -> str:
    return (
        f"Grade: {row['grade']}\n\nMath Topic(s):\n{row['math_topic']}\n\n"
        f"Question: {row['question']}\n\nSolution:\n{row['solution']}"
    )


def build_prompt(grade: Any, math_topic: Any, interest: str, examples: list[str]) -> str:
    query = f"Grade: {grade}\n\nMath Topic(s):\n{math_topic}\n\nQuestion: "
    return INSTRUCTION + PROMPT_TAIL.format(
        examples="\n\n".join(examples), interest=interest, query=query
    )


def parse(text: str) -> tuple[str, str]:
    question = solution = ""
    if "Question:" in text:
        question = text.split("Question:", 1)[1].split("Solution:", 1)[0].strip()
    if "Solution:" in text:
        solution = text.split("Solution:", 1)[1].strip()
    return question, solution


def build_candidate(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    groups: dict[str, list[dict[str, Any]]],
    subtopics: list[str],
    index: int,
) -> dict[str, Any]:
    substandard = subtopics[index % len(subtopics)]
    rows = groups[substandard]
    seed = args.seed + index
    rng = random.Random(seed)
    interest = rng.choice(INTERESTS)
    num_examples = min(8, len(rows))
    examples: list[str] = []
    if num_examples < 8:
        pool = frame[
            (frame.substandard != substandard) & (frame.grade == rows[0]["grade"])
        ].to_dict("records")
        if pool:
            for _ in range(8 - num_examples):
                block = example_block(pool[rng.randrange(len(pool))])
                if block not in examples:
                    examples.append(block)
    for _ in range(num_examples):
        block = example_block(rows[rng.randrange(len(rows))])
        if block not in examples:
            examples.append(block)
    return {
        "sample_index": index,
        "grade": int(rows[0]["grade"]),
        "standard": rows[0]["standard"],
        "substandard": substandard,
        "math_topic": rows[0]["math_topic"],
        "interest": interest,
        "seed": seed,
        "prompt": build_prompt(rows[0]["grade"], rows[0]["math_topic"], interest, examples),
    }


def kept_ok(record: dict[str, Any], seen: set[str], stem_questions: set[str]) -> bool:
    question = record.get("question") or ""
    solution = record.get("solution") or ""
    return (
        bool(question)
        and question not in seen
        and "The final answer is" in solution
        and question not in stem_questions
    )


def _payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "chat_template_kwargs": THINKING_OFF,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    if getattr(args, "seed", None) is None:
        args.seed = DEFAULT_SEED
    urls = endpoints(args.base_urls)
    stem = Path(args.stem_csv or Path(args.data_root) / "edumath" / "stem.csv")
    frame = pd.read_csv(stem)
    groups = {
        key: value.to_dict("records") for key, value in frame.groupby("substandard", sort=False)
    }
    subtopics = list(groups)
    output = Path(args.output_dir) / "generations.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=600) as client:
        if args.official_protocol:
            summary = await _run_official(args, client, urls, frame, groups, subtopics, output)
        else:
            summary = await _run_fixed(args, client, urls, frame, groups, subtopics, output)
    write_workload(
        output.parent, "edumath_quality.jsonl", _quality_workload(read_jsonl(output), args.model)
    )
    if getattr(args, "classifier_model", None) and getattr(args, "classifier_tokenizer", None):
        # The official MaC classifier is a local ModernBERT checkpoint, so this
        # stage only runs when the caller points at it (needs ``omniedu[eval-local]``).
        _score_with_classifier(Path(args.output_dir), output, args)
    summary["metrics"] = _metrics(output, Path(args.output_dir))
    return summary


def _score_with_classifier(output_dir: Path, generations: Path, args: argparse.Namespace) -> None:
    """Run the official classifier over the generated items and store its summary."""
    items = pd.DataFrame(
        [
            {
                "grade": row.get("grade"),
                "math_topic": row.get("math_topic"),
                "question": row.get("question"),
                "solution": row.get("solution"),
            }
            for row in read_jsonl(generations)
        ],
        columns=["grade", "math_topic", "question", "solution"],
    )
    target = output_dir / "classifier"
    target.mkdir(parents=True, exist_ok=True)
    source = target / "items.csv"
    items.to_csv(source, index=False)
    result = run_classifier(
        source,
        target / "classified_official.csv",
        classifier_model=args.classifier_model,
        classifier_tokenizer=args.classifier_tokenizer,
    )
    (target / "classified_official.summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _metrics(output: Path, output_dir: Path) -> dict[str, Any]:
    """``Format Pass`` is ours; the classifier and the judge are optional stages."""
    rows = read_jsonl(output)
    values: dict[str, float | None] = {
        "Format Pass": (
            (sum(bool(row.get("format_ok")) for row in rows) / len(rows) * 100) if rows else None
        )
    }
    summary = output_dir / "classifier" / "classified_official.summary.json"
    if summary.is_file():
        try:
            good = json.loads(summary.read_text(encoding="utf-8")).get("good_rate")
        except json.JSONDecodeError:
            good = None
        values["Classifier GOOD"] = None if good is None else float(good) * 100
    return with_judged("EDUMATH", output_dir, values)


def _split_raw(raw: str) -> tuple[str | None, str | None]:
    """Recover (question, solution) when the literal ``Question:`` marker is absent.

    Upstream's parser only accepts the two literal markers, so a model that writes
    the stem directly before ``Solution:`` leaves ``question`` empty and the judge
    would grade an empty stem.
    """
    for marker in ("Solution:", "Solution\uff1a"):
        if marker in raw:
            head, tail = raw.split(marker, 1)
            if head.strip() and tail.strip():
                return head.strip(), tail.strip()
    if "Question:" in raw:
        tail = raw.split("Question:", 1)[1]
        if "Solution:" in tail:
            question, solution = tail.split("Solution:", 1)
            if question.strip() and solution.strip():
                return question.strip(), solution.strip()
    return None, None


def _quality_workload(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Rows the EDUMATH judge can actually grade (a non-empty stem and solution)."""
    workload = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        solution = str(row.get("solution") or "").strip()
        judge_question, judge_solution = question, solution
        recovered_question = recovered_solution = False
        if not judge_question or not judge_solution:
            part_question, part_solution = _split_raw(str(row.get("raw_output") or ""))
            if part_question and not judge_question:
                judge_question, recovered_question = part_question, True
            if part_solution and not judge_solution:
                judge_solution, recovered_solution = part_solution, True
        if not judge_question or not judge_solution:
            continue
        workload.append(
            {
                "judge_id": f"edumath::{model}::{row.get('sample_index')}",
                "benchmark": "EDUMATH",
                "model": model,
                "sample_id": row.get("sample_index"),
                "grade": row.get("grade"),
                "standard": row.get("standard"),
                "substandard": row.get("substandard"),
                "math_topic": row.get("math_topic"),
                "question": judge_question,
                "solution": judge_solution,
                "generated_question": question,
                "generated_solution": solution,
                "question_recovered": recovered_question,
                "solution_recovered": recovered_solution,
                "raw_output": row.get("raw_output"),
                "format_ok": row.get("format_ok"),
            }
        )
    return workload


async def _run_fixed(
    args: argparse.Namespace,
    client: Any,
    urls: list[str],
    frame: pd.DataFrame,
    groups: dict[str, list[dict[str, Any]]],
    subtopics: list[str],
    output: Path,
) -> dict[str, Any]:
    jobs = [
        build_candidate(args, frame, groups, subtopics, index) for index in range(args.num_samples)
    ]
    done = resume(output, key="sample_index", failed=lambda row: bool(row.get("error")))
    sink = Sink(output)
    new = 0

    async def worker(index: int, job: dict[str, Any]) -> None:
        nonlocal new
        if str(job["sample_index"]) in done:
            return
        url = urls[index % len(urls)]
        response, _ = await chat(client, url, _payload(args.model, job["prompt"]))
        question, solution = parse(response)
        record = {key: value for key, value in job.items() if key != "prompt"}
        record.update(
            {
                "model": args.model,
                "question": question,
                "solution": solution,
                "raw_output": response,
                "format_ok": bool(question and solution),
            }
        )
        if response.startswith("__ERROR__"):
            record["error"] = response
        await sink.write(record)
        new += 1
        if (len(done) + new) % 50 == 0:
            print(f"[edumath] {len(done) + new}/{len(jobs)}", flush=True)

    await fan_out(jobs, worker, concurrency=args.concurrency, chunk=max(64, args.concurrency * 4))
    return _summarize(output, args.model, "fixed_n")


async def _run_official(
    args: argparse.Namespace,
    client: Any,
    urls: list[str],
    frame: pd.DataFrame,
    groups: dict[str, list[dict[str, Any]]],
    subtopics: list[str],
    output: Path,
) -> dict[str, Any]:
    """Upstream keep-until-N: keep generating until ``--num-samples`` valid items."""
    stem_questions = set(frame["question"].astype(str))
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    if args.reuse_file:
        reuse = Path(args.reuse_file)
        if reuse.exists():
            for line in reuse.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("error"):
                    continue
                if kept_ok(record, seen, stem_questions):
                    seen.add(record["question"])
                    kept.append(
                        {
                            **{k: v for k, v in record.items() if k != "model"},
                            "model": args.model,
                            "source": "reuse",
                        }
                    )
            print(f"[edumath] reused {len(kept)} rows", flush=True)

    attempt = 0
    batch = max(args.concurrency * 4, 32)
    cap = args.num_samples * 200
    while len(kept) < args.num_samples and attempt < cap:
        candidates = [
            build_candidate(args, frame, groups, subtopics, attempt + offset)
            for offset in range(batch)
        ]
        attempt += batch

        results: list[tuple[str, str | None]] = [("", None)] * len(candidates)

        async def worker(index: int, candidate: dict[str, Any]) -> None:
            results[index] = await chat(
                client, urls[index % len(urls)], _payload(args.model, candidate["prompt"])
            )

        await fan_out(candidates, worker, concurrency=args.concurrency)
        for candidate, (response, _) in zip(candidates, results):
            question, solution = parse(response)
            record = {key: value for key, value in candidate.items() if key != "prompt"}
            record.update(
                {
                    "model": args.model,
                    "question": question,
                    "solution": solution,
                    "raw_output": response,
                    "format_ok": bool(question and solution),
                    "source": "new",
                }
            )
            if response.startswith("__ERROR__"):
                record["error"] = response
            elif kept_ok(record, seen, stem_questions):
                seen.add(question)
                kept.append(record)
        print(f"[edumath] attempts {attempt}, kept {len(kept)}/{args.num_samples}", flush=True)

    kept = kept[: args.num_samples]
    lines = []
    for index, record in enumerate(kept):
        record["sample_index"] = index
        lines.append(json.dumps(record, ensure_ascii=False))
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "benchmark": "EDUMATH-standards-aligned-generation",
        "protocol": "official_keep_until_ok",
        "model": args.model,
        "count": len(kept),
        "attempts": attempt,
        "reused": sum(1 for record in kept if record.get("source") == "reuse"),
        "temperature": 1.0,
        "seed": args.seed,
        "judge_status": "pending",
    }


def _summarize(output: Path, model: str, protocol: str) -> dict[str, Any]:
    values = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            values[record["sample_index"]] = record
        except json.JSONDecodeError:
            continue
    items = list(values.values())
    return {
        "benchmark": "EDUMATH-standards-aligned-generation",
        "model": model,
        "protocol": protocol,
        "count": len(items),
        "format_ok": sum(item["format_ok"] for item in items),
        "errors": sum(bool(item.get("error")) for item in items),
        "temperature": 1.0,
        "judge_status": "pending",
    }


def run_classifier(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    classifier_model: str | Path,
    classifier_tokenizer: str | Path,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict[str, Any]:
    """Portable entry point for EDUMATH's official ``annotate.py`` classifier path."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    frame = pd.read_csv(input_csv)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    required = ["grade", "math_topic", "question", "solution"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    texts = [
        "Grade Level: "
        + str(row["grade"])
        + "\n\nMath Topic(s):\n"
        + str(row["math_topic"])
        + "\n\nQuestion: "
        + str(row["question"]).rstrip().lstrip()
        + "\n\nSolution:\n"
        + str(row["solution"]).rstrip()
        for _, row in frame.iterrows()
    ]
    tokenizer = AutoTokenizer.from_pretrained(classifier_tokenizer, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_model, device_map="auto", local_files_only=True
    )
    model.eval()
    predictions: list[int] = []
    logits: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=8192
        ).to(model.device)
        with torch.no_grad():
            output = model(**inputs)
        predictions.extend(torch.argmax(output.logits, dim=1).cpu().tolist())
        logits.extend(output.logits.float().cpu().tolist())
    frame["classifier_labels"] = predictions
    frame["classifier_label_name"] = ["GOOD" if value == 0 else "NOT GOOD" for value in predictions]
    frame["classifier_logits"] = [json.dumps(value) for value in logits]
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return {
        "count": len(frame),
        "good": sum(value == 0 for value in predictions),
        "not_good": sum(value == 1 for value in predictions),
        "good_rate": (
            sum(value == 0 for value in predictions) / len(predictions) if predictions else None
        ),
    }
