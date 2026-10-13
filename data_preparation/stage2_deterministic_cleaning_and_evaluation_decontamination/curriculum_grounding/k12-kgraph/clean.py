#!/usr/bin/env python3
import collections
import hashlib
import json
import re
import tarfile
from pathlib import Path

SRC = (
    Path("stage1_capability_taxonomy_and_source_collection/curriculum_grounding/huggingface/K12-KGraph/discussion7_multimodal")
)
OUT = (
    Path("stage2_deterministic_cleaning_and_evaluation_decontamination/curriculum_grounding/K12-KGraph")
)
QPRE = re.compile(r"为什么在学习.+之前[，,]?需要先掌握", re.S)


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip())


def text_type(q):
    if q.startswith("题目：") or q.startswith("题目:"):
        return "exercise_solution_qa"
    if "这道题考察了什么概念" in q or "这道题考查了什么概念" in q:
        return "exercise_to_concept_grounding_qa"
    if QPRE.search(q):
        return "prerequisite_relation_qa"
    return "curriculum_concept_relation_qa"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kept = []
    rej = []
    seen = set()
    counts = collections.Counter()
    for i, l in enumerate((SRC / "train_textonly_qa.jsonl").open()):
        try:
            r = json.loads(l)
        except Exception as e:
            rej.append({"source": "text", "index": i, "reason": "bad_json"})
            continue
        q, a = norm(r.get("question", "")), norm(r.get("answer", ""))
        typ = text_type(q)
        if not q or not a:
            rej.append({"source": "text", "index": i, "reason": "empty"})
            continue
        key = hashlib.sha256((q + "\n" + a).encode()).hexdigest()
        if key in seen:
            rej.append({"source": "text", "index": i, "reason": "exact_duplicate"})
            continue
        seen.add(key)
        counts[typ] += 1
        kept.append(
            {
                "record_id": f"k12train:text:{i}",
                "dataset": "K12-Train",
                "task_type": typ,
                "question": r["question"],
                "answer": r["answer"],
                "images": [],
                "modality": "text",
                "language": "zh",
                "provenance": {
                    "source_release": "K12-KGraph discussion #7",
                    "license": "CC-BY-NC-SA-4.0",
                    "source_file": "K12-Train/train_textonly_qa.jsonl",
                    "source_index": i,
                },
            }
        )
    archives = {}
    for n in ("figures", "ve_boxed"):
        archives[n] = tarfile.open(SRC / f"images/{n}.tar.gz", "r:gz")
    for i, l in enumerate((SRC / "train_vqa.jsonl").open()):
        try:
            r = json.loads(l)
        except Exception:
            rej.append({"source": "vqa", "index": i, "reason": "bad_json"})
            continue
        inst, out = norm(r.get("instruction", "")), norm(r.get("output", ""))
        imgs = r.get("images") or []
        if not inst or not out or not imgs:
            rej.append({"source": "vqa", "index": i, "reason": "empty_required"})
            continue
        ok = True
        for im in imgs:
            parts = Path(im).parts
            root = parts[1]
            member = "/".join(parts[1:])
            try:
                archives[root].getmember(member)
            except Exception:
                ok = False
        if not ok:
            rej.append({"source": "vqa", "index": i, "reason": "missing_image"})
            continue
        key = hashlib.sha256((inst + "\n" + out + "\n" + json.dumps(imgs)).encode()).hexdigest()
        if key in seen:
            rej.append({"source": "vqa", "index": i, "reason": "exact_duplicate"})
            continue
        seen.add(key)
        typ = f"visual_{r['task_type']}"
        counts[typ] += 1
        kept.append(
            {
                "record_id": f"k12train:vqa:{r['sample_id']}",
                "dataset": "K12-Train-MM",
                "task_type": typ,
                "question": r["instruction"],
                "answer": r["output"],
                "images": imgs,
                "modality": "multimodal",
                "language": "zh",
                "graph_metadata": {
                    k: r.get(k)
                    for k in (
                        "task_type",
                        "sample_id",
                        "book",
                        "section",
                        "edge_key",
                        "source_id",
                        "target_id",
                    )
                },
                "provenance": {
                    "source_release": "K12-KGraph discussion #7",
                    "license": "CC-BY-NC-SA-4.0",
                    "source_file": "K12-Train/train_vqa.jsonl",
                    "source_index": i,
                    "image_archives": {
                        "figures": str(SRC / "images/figures.tar.gz"),
                        "ve_boxed": str(SRC / "images/ve_boxed.tar.gz"),
                    },
                },
            }
        )
    for a in archives.values():
        a.close()
    with (OUT / "kept.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "rejected.jsonl").open("w") as f:
        for r in rej:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {
        "input_text": 2267,
        "input_vqa": 5068,
        "kept": len(kept),
        "rejected": len(rej),
        "task_counts": dict(counts),
        "qc": {
            "all_required_nonempty": all(r["question"] and r["answer"] for r in kept),
            "vqa_all_have_images": all(r["images"] for r in kept if r["modality"] == "multimodal"),
            "exact_unique": len(seen) == len(kept),
        },
        "policy": "Preserve all usable K12-Train text/VQA. Sampling layer may exclude ordinary exercise_solution_qa from curriculum quotas because problem-solving is covered by subject_competence.",
    }
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
