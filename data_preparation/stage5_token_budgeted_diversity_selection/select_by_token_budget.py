#!/usr/bin/env python3
"""Run exact K-Center independently in each quality-filtered leaf bucket."""

import argparse
import hashlib
import json
import sys as _sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common.buckets import bucket_id, target_text

HERE = Path(__file__).resolve().parent  # stage5_token_budgeted_diversity_selection
SFT = HERE.parent  # data_preparation
STAGE4 = SFT / "stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering"
DEFAULT_INPUT_DIR = STAGE4 / "quality_scoring/results/quality_filtered_pools"
DEFAULT_OUTPUT_DIR = HERE / "results/bucketed_kcenter_final"
DEFAULT_EMBED_MODEL = Path("models/bge-m3")
DEFAULT_TOKENIZER = Path("models/Qwen3.5-9B-Base")
# Budgets are supervised-output tokens. They implement Inventory section 6 rather
# than forcing sample counts. A missing budget is an error, not an implicit drop.
TOKEN_BUDGETS = {
    # Subject competence
    "math/text/answer_only": 538,
    "math/text/cot": 2_305_000,
    "math/multimodal/answer_only": 24,
    "math/multimodal/cot": 600_000,
    "reading/reading-c3": 396_000,
    "reading/reading-race": 396_000,
    "science/text/cot": 1_742_000,
    "science/multimodal/cot": 765_000,
    "writing_feedback/en": 561_939,
    "writing_feedback/cn": 325_334,
    # Curriculum grounding
    "general_mapping/mixed_or_text": 82_000,
    "general_mapping/multimodal": 58_000,
    "knowledge_graph/text": 80_000,
    "knowledge_graph/multimodal": 181_000,
    "exercise_standard_mapping/multirelation": 225_000,
    "exercise_standard_mapping/strict": 160_000,
    # Diagnostic reasoning
    "process/text": 48_207,
    "process/multimodal": 119_000,
    "dialogue/no_conversation_history": 152_468,
    "dialogue/short_conversation_history": 8_645,
    "dialogue/long_conversation_history/official": 1_324_000,
    "dialogue/long_conversation_history/synthetic": 1_203_000,
    # Pedagogical action and scaffolding
    "scaffolding/minimal_scaffold": 265_000,
    "scaffolding/hint_framed_as_a_question": 79_500,
    "scaffolding/progressive_hints_no_question": 37_175,
    "questioning/one_question_per_turn": 174_000,
    "questioning/ordered_question_chain": 74_000,
    "multi_turn_dialogue": 931_000,
    "assessment/feedback_on_a_solution": 40_000,
    "assessment/feedback_on_writing": 15_000,
    "direct_explanation": 114_000,
    "dialogue/long_conversation_history": 1_270,
}


def read_rows(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def selection_bucket(row: dict) -> str:
    bucket = bucket_id(row["category"], row["source"], row["kind"])
    if bucket == "subject/math/text/cot":
        suffix = "openr1" if row["source"] == "openr1-default" else "other"
        return f"{bucket}/{suffix}"
    return bucket


def flatten_text(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_text(item))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            if key in {"provenance", "_stage1", "_stage1_application"}:
                continue
            result.extend(flatten_text(item))
        return result
    return []


def embedding_text(row: dict) -> str:
    parts = flatten_text(row.get("payload", {}))
    text = "\n".join(part.strip() for part in parts if part.strip())
    return f"task={selection_bucket(row)}\nsource={row['source']}\n{text}"


def fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["category"].encode())
        digest.update(b"\0")
        digest.update(row["source"].encode())
        digest.update(b"\0")
        digest.update(row["uid"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def embed_rows(
    rows: list[dict],
    model_path: Path,
    output_path: Path,
    batch_size: int,
) -> np.ndarray:
    metadata_path = output_path.with_suffix(".meta.json")
    expected = {"count": len(rows), "fingerprint": fingerprint(rows)}
    if output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        vectors = np.load(output_path)
        if metadata == expected and len(vectors) == len(rows):
            print(f"reuse embeddings: {output_path} {vectors.shape}", flush=True)
            return vectors
    model = SentenceTransformer(str(model_path), device="cuda")
    model.max_seq_length = 8192
    vectors = model.encode(
        [embedding_text(row) for row in rows],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float16)
    np.save(output_path, vectors)
    metadata_path.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"saved embeddings: {output_path} {vectors.shape}", flush=True)
    return vectors


def supervised_lengths(rows: list[dict], tokenizer_path: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    lengths = []
    batch = []
    for row in rows:
        batch.append(target_text(row))
        if len(batch) == 512:
            lengths.extend(
                len(ids)
                for ids in tokenizer(
                    batch,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            batch.clear()
    if batch:
        lengths.extend(
            len(ids)
            for ids in tokenizer(
                batch,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
    return np.asarray(lengths, dtype=np.int32)


def exact_kcenter(
    vectors: np.ndarray,
    lengths: np.ndarray,
    indices: list[int],
    token_budget: int | None,
    state_path: Path,
) -> list[int]:
    if token_budget is None:
        return indices
    local_vectors = torch.from_numpy(vectors[indices].astype(np.float32)).cuda().half()
    local_vectors = torch.nn.functional.normalize(local_vectors, dim=1)
    selected = []
    selected_mask = torch.zeros(
        len(indices),
        dtype=torch.bool,
        device="cuda",
    )
    min_dist = torch.full(
        (len(indices),),
        2.0,
        dtype=torch.float16,
        device="cuda",
    )
    if state_path.exists():
        state = np.load(state_path)
        selected = state["selected"].astype(np.int64).tolist()
        min_dist = torch.from_numpy(state["min_dist"]).cuda().half()
        selected_mask[selected] = True
        print(f"resume {state_path.stem}: {len(selected)}", flush=True)
    if not selected:
        centroid = torch.nn.functional.normalize(
            local_vectors.float().mean(dim=0),
            dim=0,
        ).half()
        first = int(torch.argmax(local_vectors @ centroid).item())
        selected.append(first)
        selected_mask[first] = True
    selected_tokens = int(lengths[[indices[item] for item in selected]].sum())
    if selected_tokens >= token_budget:
        cumulative = 0
        cutoff = 0
        for cutoff, item in enumerate(selected, start=1):
            cumulative += int(lengths[indices[item]])
            if cumulative >= token_budget:
                break
        return [indices[item] for item in selected[:cutoff]]
    while selected_tokens < token_budget and len(selected) < len(indices):
        latest = selected[-1]
        distance = 1.0 - (local_vectors @ local_vectors[latest])
        min_dist = torch.minimum(min_dist, distance)
        min_dist[selected_mask] = -1
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        selected_mask[nxt] = True
        selected_tokens += int(lengths[indices[nxt]])
        if len(selected) % 250 == 0:
            print(
                f"{state_path.stem}: selected={len(selected)}/{len(indices)} "
                f"tokens={selected_tokens:,}/{token_budget:,}",
                flush=True,
            )
        if len(selected) % 1000 == 0:
            np.savez(
                state_path,
                selected=np.asarray(selected, dtype=np.int64),
                min_dist=min_dist.float().cpu().numpy().astype(np.float16),
            )
    np.savez(
        state_path,
        selected=np.asarray(selected, dtype=np.int64),
        min_dist=min_dist.float().cpu().numpy().astype(np.float16),
    )
    return [indices[item] for item in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.output_dir / "states"
    state_dir.mkdir(exist_ok=True)
    input_path = args.input_dir / f"{args.category}.jsonl"
    rows = read_rows(input_path)
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        bucket = selection_bucket(row)
        if bucket not in TOKEN_BUDGETS:
            raise KeyError(f"missing token budget for {bucket}")
        buckets[bucket].append(index)
    vectors = embed_rows(
        rows,
        args.model,
        args.output_dir / f"{args.category}.embeddings.npy",
        args.batch_size,
    )
    lengths = supervised_lengths(rows, args.tokenizer)
    selected_rows = []
    summary = {}
    for bucket, indices in sorted(buckets.items()):
        state_name = hashlib.sha1(bucket.encode()).hexdigest()[:12]
        selected = exact_kcenter(
            vectors,
            lengths,
            indices,
            TOKEN_BUDGETS[bucket],
            state_dir / f"{args.category}.{state_name}.npz",
        )
        selected_tokens = int(lengths[selected].sum())
        summary[bucket] = {
            "candidate_count": len(indices),
            "candidate_tokens": int(lengths[indices].sum()),
            "token_budget": TOKEN_BUDGETS[bucket],
            "selected_count": len(selected),
            "selected_tokens": selected_tokens,
        }
        for rank, index in enumerate(selected):
            row = rows[index]
            row["kcenter"] = {
                "bucket": bucket,
                "rank": rank,
                "supervised_tokens": int(lengths[index]),
                "token_budget": TOKEN_BUDGETS[bucket],
            }
            selected_rows.append(row)
        print(json.dumps({bucket: summary[bucket]}), flush=True)
    output_path = args.output_dir / f"{args.category}.jsonl"
    with output_path.open("w") as output:
        for row in selected_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / f"{args.category}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "category": args.category,
                "candidates": len(rows),
                "selected": len(selected_rows),
                "tokens": sum(item["selected_tokens"] for item in summary.values()),
                "output": str(output_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
