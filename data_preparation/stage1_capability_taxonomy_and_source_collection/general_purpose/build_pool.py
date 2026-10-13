#!/usr/bin/env python3
"""?????????????? frozen ??????

????????????education-specific ??????????????????????? ??
??????? system prompt???? stage2~5 ??????????????? stage6 ????????

???????????????DataFlow-Instruct-10K 7,431?Tulu-3-SFT-mixture 1,495?MathV360K 122?
?? 9,048 ???????**????**??????????????

* DataFlow-Instruct-10K ? 10,000 ??? math / code / general instruction ?????????
  **?????????????/????**?K?12 ???????????????????
  ?? ``--filter-raw``???????????? 7,336 ?????? 2,044????? 620??
  ???? 7,431 ?? 95 ??? ``--tolerance`` ???
* Tulu-3-SFT-mixture ? MathV360K ?**????????**?Tulu-PersonaHub-IF-V3 572?
  Tulu-Aya-General-V3 348?Tulu-TableGPT-V3 208?Tulu-CoCoNot-V3 189?Tulu-SciRIFF-V3 178?
  GeoQA+ 88?TabMWP 19?GEOS 15???????????

?????????????? frozen ??????

????????????education-specific ??????????????????????? ??
??????? system prompt???? stage2~5 ??????????????? stage6 ????????

???????????????DataFlow-Instruct-10K 7,431?Tulu-3-SFT-mixture 1,495?MathV360K 122?
?? 9,048 ???????**????**??????????????

* DataFlow-Instruct-10K ? 10,000 ??? math / code / general instruction ?????? dataset card??
  ??????? **code ?**????????????? 2,569 ??? 2,000 ??????? ``` ???????
  ????**?????**????????? 81% ???????? 570 ????????????
  ???/?????????????????
* Tulu-3-SFT-mixture ? MathV360K ?**????**???Tulu-PersonaHub-IF-V3 572?Tulu-Aya-General-V3 348?
  Tulu-TableGPT-V3 208?Tulu-CoCoNot-V3 189?Tulu-SciRIFF-V3 178?GeoQA+ 88?TabMWP 19?GEOS 15?

???????????

1. ??????????????? ``{"messages": [...], "extra_info": {...}}``????? system?
2. ????????????????????``--no-check`` ?????

???

    # DataFlow???? 10k ?"??? + ?????"?????????????? 7,431 ??
    python .../build_pool.py --filter-raw <?? DataFlow train.jsonl> --output <???? dataflow ??>

    # ?????????????????DataFlow ?? --tolerance ???????
    python data_preparation/stage1_capability_taxonomy_and_source_collection/general_purpose/build_pool.py \
        --source dataflow-instruct-10k=<?????????> \
        --source tulu-3-sft-mixture=<????????> \
        --source mathv360k=<????????> --tolerance 200

????????????????? train.jsonl?????? ``--extract-from`` ?????????

    python .../build_pool.py --extract-from <???? train.jsonl> --extract-dir <????>
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "general_instruction_pool.jsonl"

#: ??? ? ?????????
SOURCES: dict[str, int] = {
    "dataflow-instruct-10k": 7_431,
    "tulu-3-sft-mixture": 1_495,
    "mathv360k": 122,
}
#: ????????/????? ? ???????
def row_text(row: dict) -> str:
    """??????????????/????????"""
    messages = row.get("messages") or row.get("conversations") or []
    return "\n".join(str(m.get("content") or m.get("value") or "") for m in messages)


def retain_dataflow(row: dict) -> bool:
    """DataFlow ??????????????????/?????K-12 ??????"""
    text = row_text(row)
    if CODE_RE.search(text):
        return False
    if ADVANCED_MATH_RE.search(text):
        return False
    return True


def filter_raw(source: Path, output: Path, expected: int) -> dict:
    """???????? DataFlow ????????????????????"""
    kept = 0
    dropped_code = dropped_math = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_rows(source):
            text = row_text(row)
            if CODE_RE.search(text):
                dropped_code += 1
                continue
            if ADVANCED_MATH_RE.search(text):
                dropped_math += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return {
        "output": str(output),
        "kept": kept,
        "dropped_code": dropped_code,
        "dropped_advanced_math": dropped_math,
        "expected": expected,
        "delta_vs_paper": kept - expected,
    }


def canonical_name(raw: str) -> str:
    """??????? dataset ??????????????????"""
    lowered = raw.lower()
    if "dataflow" in lowered:
        return "dataflow-instruct-10k"
    if "tulu" in lowered:
        return "tulu-3-sft-mixture"
    if "mathv360" in lowered:
        return "mathv360k"
    return raw

PROMPT_KEYS = ("prompt", "question", "problem", "instruction", "input")
RESPONSE_KEYS = ("response", "answer", "solution", "output", "completion")

#: DataFlow-Instruct-10K ????????? + ????????????? 7,431 ??
CODE_RE = re.compile(
    r"```|\bdef \w+\(|\bclass \w+[:(]|public static|console\.log|#include|"
    r"\bSELECT\b[\s\S]{0,120}\bFROM\b|pip install|import (?:os|sys|json|re|numpy|pandas)\b|"
    r"System\.out\.print|\bnpm \b|\bgit \b|\bprint\(|\bAPI\b",
    re.I,
)
ADVANCED_MATH_RE = re.compile(
    r"\\int\b|\\iint|\\oint|\\lim\b|\\sum_|\\prod_|\\binom|"
    r"\\begin\{(?:matrix|pmatrix|bmatrix|align)|\\mathbb|\\mathcal|\\gcd|\\pmod|"
    r"eigen|determinant|\bmatrix\b|integral|derivative|differential equation|complex number|"
    r"\bmodulo\b|group theory|topology|probability (?:density|distribution)|\\phi|\\theta|\\zeta|\\infty",
    re.I,
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open(encoding="utf-8-sig")


def iter_rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        yield from (data if isinstance(data, list) else data.get("data") or [])
        return
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_messages(value) -> list[dict]:
    """?????????system ?????"""
    messages = []
    for message in value or []:
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"human", "student"}:
            role = "user"
        elif role in {"gpt", "teacher", "tutor"}:
            role = "assistant"
        content = message.get("content") or message.get("value") or message.get("text")
        if role in {"system", "user", "assistant"} and str(content or "").strip():
            messages.append({"role": role, "content": content})
    return messages


def to_messages(row: dict) -> list[dict]:
    messages = normalize_messages(row.get("messages") or row.get("conversations") or row.get("conversation"))
    if messages:
        return messages
    prompt = next((row[key] for key in PROMPT_KEYS if row.get(key)), None)
    response = next((row[key] for key in RESPONSE_KEYS if row.get(key)), None)
    if prompt is None or response is None:
        raise ValueError("row has neither messages nor a prompt/response pair")
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def build_source(name: str, path: Path) -> list[dict]:
    """??????????????????????

    ``path`` ????????????????????? jsonl/json/parquet ??????????
    ?? Tulu / MathV360K ???????????????
    """
    files = (
        [p for p in sorted(path.iterdir()) if p.suffix in {".jsonl", ".json", ".parquet"}]
        if path.is_dir()
        else [path]
    )
    rows = []
    for file in files:
        for row in iter_rows(file):
            uid = row.get("uid") or row.get("id") or f"{name}-{len(rows):05d}"
            built = {
                "messages": to_messages(row),
                "extra_info": json.dumps(
                    {"category": "general_instruct", "dataset": name, "uid": str(uid)},
                    ensure_ascii=False,
                ),
            }
            if row.get("images"):
                built["images"] = list(row["images"])
            rows.append(built)
    return rows


def extract_from_release(path: Path, out_dir: Path) -> dict[str, int]:
    """????????????``<out_dir>/<???>/<?????>.jsonl``?

    ???????? ``--source`` ??????????????????????????????
    """
    handles = {}
    counts: dict[str, int] = {}
    for row in iter_rows(path):
        info = row.get("extra_info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        if info.get("category") not in {"general_instruct", "general_retention"}:
            continue
        raw_dataset = str(info.get("dataset") or "")
        name = canonical_name(raw_dataset)
        handle = handles.get(name)
        if handle is None:
            directory = out_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / f"{raw_dataset or 'unknown'}.jsonl").open("w", encoding="utf-8")
            handles[name] = handle
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = counts.get(name, 0) + 1
    for handle in handles.values():
        handle.close()
    return counts


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected <name>=<path>")
    name, raw_path = value.split("=", 1)
    if name not in SOURCES:
        raise argparse.ArgumentTypeError(f"unknown source {name!r}; expected one of {sorted(SOURCES)}")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", action="append", type=parse_source, default=[], metavar="NAME=PATH",
                        help="?????????????????????")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-check", action="store_true", help="??????????")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="?????????? DataFlow ????????+??????????")
    parser.add_argument("--filter-raw", type=Path, default=None,
                        help="??? DataFlow-Instruct-10K ?????????????")
    parser.add_argument("--extract-from", type=Path, default=None,
                        help="??????????????????????")
    parser.add_argument("--extract-dir", type=Path, default=HERE / "sources")
    args = parser.parse_args()

    if args.filter_raw is not None:
        output = args.output if args.output != OUTPUT else HERE / "sources/dataflow-instruct-10k.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(filter_raw(args.filter_raw, output, SOURCES["dataflow-instruct-10k"]), ensure_ascii=False, indent=2))
        return

    if args.extract_from is not None:
        counts = extract_from_release(args.extract_from, args.extract_dir)
        print(json.dumps({"extracted": counts, "dir": str(args.extract_dir)}, ensure_ascii=False, indent=2))
        return

    given = dict(args.source)
    missing = [name for name in SOURCES if name not in given]
    if missing:
        parser.error(f"missing sources: {', '.join(missing)}")

    pooled = []
    report = {}
    for name in SOURCES:
        rows = build_source(name, given[name])
        expected = SOURCES[name]
        tolerance = args.tolerance if name == "dataflow-instruct-10k" else 0
        if not args.no_check and abs(len(rows) - expected) > tolerance:
            parser.error(f"{name}: expected {expected} curated rows (+-{tolerance}), got {len(rows)}")
        pooled.extend(rows)
        report[name] = {"input": str(given[name]), "expected": expected, "kept": len(rows)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in pooled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "total": len(pooled), "sources": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
