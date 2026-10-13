#!/usr/bin/env python3
"""Build direct curriculum-structure SFT from DA-20K, TAL, XES3G5M, and MathFish.

The generated conversations only teach question/activity-to-curriculum
alignment. Source answers, analyses, and solution traces are intentionally not
read into the normalized records.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

PIPE = Path("data_preparation/stage2_deterministic_cleaning_and_evaluation_decontamination/curriculum_grounding")
DA20K_DIR = Path("github/DA-20K")
DA20K_IMAGE_MANIFEST = Path("direct/DA-20K/image_manifest.json")
DA20K_IMAGE_ARCHIVE = Path("direct/DA-20K/images.tar.gz")
XES_ARCHIVE = Path("direct/XES3G5M/XES3G5M.tar.gz")
MATHFISH_DIR = Path("huggingface/mathfish")
MATHFISH_IMAGE_DIR = MATHFISH_DIR / "images"
STANDARDS_PATH = Path("huggingface/achieve-the-core/standards.jsonl")
TAL_DIR = Path("huggingface/TAL-SCQ5K")

KEEP_IN_DEDUP_KEY = set("<>=+-*/^\\{}")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}
MATHFISH_ARCHIVES = {
    "fl_problem": "fl_problem.tar.gz",
    "im_lesson": "im_lesson.tar.gz",
    "im_modelingprompt": "im_modelingprompt.tar.gz",
    "im_practice": "im_practice.tar.gz",
    "im_task": "im_task.tar.gz",
}
RELATION_GROUPS = {
    "Addressing": "direct",
    "Alignment": "direct",
    "Building On": "building_on",
    "Building Towards": "building_towards",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", html.unescape(str(value or "")))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def mathml_to_text(raw_mathml: str) -> str:
    """Render common presentation MathML as compact readable plain text."""
    try:
        root = ElementTree.fromstring(html.unescape(raw_mathml))
    except ElementTree.ParseError:
        return clean_text(re.sub(r"<[^>]+>", "", html.unescape(raw_mathml)))

    def render_children(node: ElementTree.Element, separator: str = "") -> str:
        parts: list[str] = []
        if node.text:
            parts.append(node.text)
        for child in node:
            parts.append(render(child))
            if child.tail:
                parts.append(child.tail)
        return separator.join(part for part in parts if part)

    def render(node: ElementTree.Element) -> str:
        tag = local_xml_name(node.tag)
        children = list(node)
        rendered = [render(child) for child in children]
        if tag == "annotation":
            return ""
        if tag == "semantics":
            return rendered[0] if rendered else clean_text(node.text)
        if tag == "mfrac" and len(rendered) >= 2:
            return f"({rendered[0]})/({rendered[1]})"
        if tag == "msup" and len(rendered) >= 2:
            return f"{rendered[0]}^{{{rendered[1]}}}"
        if tag == "msub" and len(rendered) >= 2:
            return f"{rendered[0]}_{{{rendered[1]}}}"
        if tag == "msubsup" and len(rendered) >= 3:
            return f"{rendered[0]}_{{{rendered[1]}}}^{{{rendered[2]}}}"
        if tag == "msqrt":
            return f"sqrt({render_children(node)})"
        if tag == "mroot" and len(rendered) >= 2:
            return f"root({rendered[1]}, {rendered[0]})"
        if tag == "mfenced":
            opening = node.attrib.get("open", "(")
            closing = node.attrib.get("close", ")")
            separators = node.attrib.get("separators", ",")
            separator = separators[0] if separators else ","
            return f"{opening}{separator.join(rendered)}{closing}"
        if tag == "mover" and len(rendered) >= 2:
            accent = rendered[1].strip()
            if accent in {"¯", "‾", "―"}:
                return f"\\bar{{{rendered[0]}}}"
            if accent in {"^", "ˆ", "̂"}:
                return f"\\hat{{{rendered[0]}}}"
            if accent in {"→", "⃗"}:
                return f"\\vec{{{rendered[0]}}}"
            return f"\\overset{{{accent}}}{{{rendered[0]}}}"
        if tag == "munder" and len(rendered) >= 2:
            return f"{rendered[0]}_{{{rendered[1]}}}"
        if tag == "munderover" and len(rendered) >= 3:
            return f"{rendered[0]}_{{{rendered[1]}}}^{{{rendered[2]}}}"
        if tag in {"mtable", "mtr"}:
            separator = "\n" if tag == "mtable" else " | "
            return separator.join(rendered)
        return render_children(node)

    text = render(root)
    text = text.replace("\u2061", "").replace("\u2062", "·")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return clean_text(text)


class HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "br",
        "div",
        "p",
        "tbody",
        "thead",
        "tfoot",
        "li",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag == "table":
            self.table_depth += 1
            self.parts.append("\n")
        elif tag == "tr":
            self.parts.append("\n| ")
        elif tag in self.BLOCK_TAGS and not self.table_depth:
            self.parts.append("\n")
        elif tag == "sup":
            self.parts.append("^{")
        elif tag == "sub":
            self.parts.append("_{")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            self.table_depth = max(0, self.table_depth - 1)
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag == "tr":
            self.parts.append("\n")
        elif tag in self.BLOCK_TAGS and not self.table_depth:
            self.parts.append("\n")
        elif tag in {"sup", "sub"}:
            self.parts.append("}")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def clean_da20k_html(
    raw_html: Any, image_manifest: dict[str, dict[str, Any]] | None = None
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Remove site chrome, render MathML, and resolve downloaded inline images."""
    source = clean_text(raw_html)
    source = re.sub(
        r'<div\s+class=["\']exam-foot["\'][\s\S]*$',
        "",
        source,
        flags=re.IGNORECASE,
    )

    formulas: dict[str, str] = {}

    def replace_math(match: re.Match[str]) -> str:
        token = f"[[DA20K_MATH_{len(formulas)}]]"
        formulas[token] = mathml_to_text(match.group(0))
        return token

    source = re.sub(
        r"<math\b[\s\S]*?</math>",
        replace_math,
        source,
        flags=re.IGNORECASE,
    )

    image_ref: list[dict[str, str]] = []
    unresolved_images: list[dict[str, str]] = []

    def replace_image(match: re.Match[str]) -> str:
        image_tag = match.group(0)
        source_match = re.search(
            r'\bsrc\s*=\s*["\']([^"\']+)["\']',
            image_tag,
            flags=re.IGNORECASE,
        )
        placeholder = f"###IMAGE{len(image_ref) + len(unresolved_images)}###"
        url = clean_text(source_match.group(1)) if source_match else ""
        downloaded = (image_manifest or {}).get(url) or {}
        member = clean_text(downloaded.get("member"))
        if downloaded.get("status") == "ok" and member and DA20K_IMAGE_ARCHIVE.exists():
            image_ref.append(
                {
                    "placeholder": placeholder,
                    "archive": str(DA20K_IMAGE_ARCHIVE),
                    "member": member,
                    "filename": Path(member).name,
                }
            )
        else:
            unresolved_images.append({"placeholder": placeholder, "url": url})
        return placeholder

    source = re.sub(r"<img\b[^>]*>", replace_image, source, flags=re.IGNORECASE)
    parser = HTMLTextExtractor()
    parser.feed(source)
    parser.close()
    question = parser.text()
    for token, formula in formulas.items():
        question = question.replace(token, formula)
    question = re.sub(r"[ \t\f\v]+", " ", question)
    question = re.sub(r" *\n *", "\n", question)
    question = re.sub(r"\n{3,}", "\n\n", question)
    question = re.sub(r"^\s*\|\s*", "", question)
    question = re.sub(r"\s*\|\s*$", "", question)
    return clean_text(question), image_ref, unresolved_images


def normalize_for_dedup(value: str) -> str:
    value = clean_text(value).lower().replace("$", "")
    return "".join(char for char in value if char.isalnum() or char in KEEP_IN_DEDUP_KEY)


def stable_split(source_id: str) -> str:
    bucket = int(hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def parse_path(raw_path: Any) -> list[str]:
    path = clean_text(raw_path)
    if "->" in path:
        parts = path.split("->")
    elif "----" in path:
        parts = path.split("----")
    else:
        parts = [path]
    return [clean_text(part) for part in parts if clean_text(part)]


def normalize_paths(raw_paths: Any) -> list[list[str]]:
    normalized: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_path in raw_paths or []:
        path = parse_path(raw_path)
        signature = tuple(path)
        if path and signature not in seen:
            seen.add(signature)
            normalized.append(path)
    return normalized


def format_tal_options(raw_options: Any) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for group in raw_options or []:
        entries = group if isinstance(group, list) else [group]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = clean_text(entry.get("aoVal"))
            content = clean_text(entry.get("content"))
            if label or content:
                options.append({"label": label, "content": content})
    return options


def format_xes_options(raw_options: Any) -> list[dict[str, str]]:
    if not isinstance(raw_options, dict):
        return []
    return [
        {"label": clean_text(label), "content": clean_text(content)}
        for label, content in raw_options.items()
        if clean_text(label) or clean_text(content)
    ]


def render_question(question: str, options: list[dict[str, str]]) -> str:
    if not options:
        return question
    option_text = "\n".join(f"{option['label']}. {option['content']}".strip() for option in options)
    return f"{question}\n\n选项：\n{option_text}"


def render_question_en(question: str, options: list[dict[str, str]]) -> str:
    if not options:
        return question
    option_text = "\n".join(f"{option['label']}. {option['content']}".strip() for option in options)
    return f"{question}\n\nOptions:\n{option_text}"


def render_paths_zh(paths: list[list[str]]) -> str:
    return "知识点路径：\n" + "\n".join(f"- {' → '.join(path)}" for path in paths)


def render_paths_en(paths: list[list[str]]) -> str:
    return "Knowledge-point paths:\n" + "\n".join(f"- {' → '.join(path)}" for path in paths)


def make_kp_record(
    *,
    uid: str,
    source_dataset: str,
    source_id: str,
    question: str,
    options: list[dict[str, str]],
    paths: list[list[str]],
    language: str,
    split: str,
    license_tag: str,
    source_meta: dict[str, Any],
    image_ref: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    is_english = language == "en"
    rendered_question = (
        render_question_en(question, options) if is_english else render_question(question, options)
    )
    instruction = (
        "Identify the knowledge-point hierarchy for the following problem. "
        "Only label the knowledge-point paths; do not solve the problem."
        if is_english
        else "请判断下面题目主要对应的知识点。只标注知识点层级路径，不要求解题。"
    )
    response = render_paths_en(paths) if is_english else render_paths_zh(paths)
    return {
        "uid": uid,
        "task_type": "question_to_knowledge_path",
        "messages": [
            {"role": "user", "content": f"{instruction}\n\n{rendered_question}".strip()},
            {"role": "assistant", "content": response},
        ],
        "grounding": {
            "framework": source_dataset,
            "alignment_type": "native_annotation",
            "knowledge_paths": paths,
            "ancestor_paths": [path[:-1] for path in paths],
            "leaf_knowledge_points": [path[-1] for path in paths if path],
            "source_id": source_id,
        },
        "image_ref": image_ref or [],
        "meta": {
            "source_dataset": source_dataset,
            "source_id": source_id,
            "language": language,
            "subject": "mathematics",
            "split": split,
            "modality": "multimodal" if image_ref else "text",
            "license": license_tag,
            **source_meta,
        },
    }


def load_standard_catalog() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(STANDARDS_PATH)}


def standard_alignment(
    relation: str,
    code: str,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    standard = catalog.get(code)
    if not standard:
        return {
            "relation": relation,
            "relation_group": RELATION_GROUPS.get(relation, "unknown"),
            "standard_code": code,
            "standard_description": "",
            "unresolved": True,
        }
    return {
        "relation": relation,
        "relation_group": RELATION_GROUPS.get(relation, "unknown"),
        "standard_code": code,
        "standard_description": clean_text(standard.get("description")),
        "level": standard.get("level"),
        "parent": standard.get("parent"),
        "children": standard.get("children") or [],
        "connections": standard.get("connections") or {},
        "cluster_type": standard.get("cluster_type"),
        "modeling": bool(standard.get("modeling")),
    }


def render_standard_group(title: str, alignments: list[dict[str, Any]]) -> list[str]:
    if not alignments:
        return []
    lines = [f"{title}:"]
    for alignment in alignments:
        lines.append(
            "- "
            f"{alignment['standard_code']} — "
            f"{alignment['standard_description']} "
            f"(Source relation: {alignment['relation']})"
        )
    return lines


def render_standard_response(alignments: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for alignment in alignments:
        grouped[alignment["relation_group"]].append(alignment)
    lines: list[str] = []
    lines += render_standard_group("Directly aligned standards", grouped["direct"])
    lines += render_standard_group(
        "Previously learned standards used as a bridge", grouped["building_on"]
    )
    lines += render_standard_group(
        "Target standards this activity is building towards",
        grouped["building_towards"],
    )
    if grouped["unknown"]:
        lines += render_standard_group(
            "Other source-provided standard relations", grouped["unknown"]
        )
    return "\n".join(lines)


def cut_mathfish_answer_sections(text: str) -> str:
    stop_heading = re.compile(
        r"(?im)^\s*(?:Student Response|Activity Synthesis|Lesson Synthesis|"
        r"Solution|Solutions|Sample Response|Sample Solution|Answer|Answers)\s*$"
    )
    match = stop_heading.search(text)
    return text[: match.start()] if match else text


def cut_mathfish_student_segment(text: str) -> str:
    """Stop before the page returns from student material to teacher chrome."""
    stop_heading = re.compile(
        r"(?im)^\s*(?:Print|Launch|Narrative|Required Materials|"
        r"Required Preparation|Student Response|Activity Synthesis|"
        r"Lesson Synthesis|Solution|Solutions|Sample Response|"
        r"Sample Solution|Answer|Answers)\s*$"
    )
    match = stop_heading.search(text)
    return text[: match.start()] if match else text


def clean_mathfish_boilerplate(text: str, metadata: dict[str, Any]) -> str:
    pattern = re.compile(
        r"\n?Student Response\s+"
        r"Teachers with a valid work email address can\s+"
        r"click here to register or sign in\s+"
        r"for free access to Student Response\.\s*",
        re.IGNORECASE,
    )
    text = clean_text(pattern.sub("\n", clean_text(text)))

    # IM lesson pages mix teacher guidance, student-facing material, and an
    # answer-like synthesis in one field. Prefer the longest Student Facing
    # segment and stop before any response/solution/synthesis section.
    student_markers = list(re.finditer(r"(?im)^\s*Student Facing\s*$", text))
    if student_markers:
        candidates = [
            clean_text(cut_mathfish_student_segment(text[marker.end() :]))
            for marker in student_markers
        ]
        candidates = [candidate for candidate in candidates if candidate]
        if candidates:
            text = max(candidates, key=len)
    else:
        text = clean_text(cut_mathfish_answer_sections(text))

    title = clean_text(metadata.get("title"))
    if title and title.lower() not in text[: max(200, len(title) + 20)].lower():
        text = f"{title}\n{text}".strip()
    return clean_text(text)


def mathfish_archive_for(filename: str) -> Path | None:
    for prefix, archive_name in MATHFISH_ARCHIVES.items():
        if filename.startswith(prefix + "_"):
            return MATHFISH_IMAGE_DIR / archive_name
    return None


def mathfish_member_for(filename: str, metadata: dict[str, Any]) -> str:
    if filename.startswith("im_practice_"):
        grade = clean_text(metadata.get("grade / subject"))
        return f"images/{grade}/{filename}" if grade else f"images/{filename}"
    return f"images/{filename}"


def replace_mathfish_elements(
    text: str,
    elements: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[str, list[dict[str, str]], list[str]]:
    image_ref: list[dict[str, str]] = []
    unresolved: list[str] = []
    for placeholder, raw_value in (elements or {}).items():
        value = clean_text(raw_value)
        if "<table" in value.lower():
            text = text.replace(placeholder, value)
            continue
        filename = Path(value).name
        archive = mathfish_archive_for(filename)
        if (
            archive is None
            or not archive.exists()
            or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS
        ):
            unresolved.append(placeholder)
            continue
        image_ref.append(
            {
                "placeholder": placeholder,
                "archive": str(archive),
                "member": mathfish_member_for(filename, metadata),
                "filename": filename,
            }
        )
    text = clean_mathfish_boilerplate(text, metadata)
    image_ref = [ref for ref in image_ref if ref["placeholder"] in text]
    unresolved = [placeholder for placeholder in unresolved if placeholder in text]
    return text, image_ref, unresolved


def make_mathfish_record(
    raw: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_id = clean_text(raw.get("id"))
    metadata = raw.get("metadata") or {}
    text, image_ref, unresolved_images = replace_mathfish_elements(
        clean_text(raw.get("text")), raw.get("elements") or {}, metadata
    )
    alignments: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw.get("standards") or []:
        if not isinstance(item, list) or len(item) != 2:
            continue
        relation, code = clean_text(item[0]), clean_text(item[1])
        signature = (relation, code)
        if relation and code and signature not in seen:
            seen.add(signature)
            alignments.append(standard_alignment(relation, code, catalog))
    source = clean_text(raw.get("source"))
    license_tag = (
        "ODC-By-1.0 + CC-BY-NC-SA-4.0"
        if source == "Fishtank Learning"
        else "ODC-By-1.0 + CC-BY-4.0"
    )
    return {
        "uid": f"mathfish::{source_id}",
        "task_type": "activity_to_ccss_alignment",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Directly label the CCSS mathematics standards associated with "
                    "the following problem or activity. For every label, provide the "
                    "standard code, its full description, and whether the source marks "
                    "it as directly aligned, building on prior work, or building "
                    "towards a target. Do not solve the problem.\n\n"
                    f"{text}"
                ).strip(),
            },
            {"role": "assistant", "content": render_standard_response(alignments)},
        ],
        "grounding": {
            "framework": "Common Core State Standards for Mathematics",
            "catalog": "Achieve the Core",
            "alignment_type": "native_publisher_annotation",
            "standard_alignments": alignments,
            "source_id": source_id,
        },
        "image_ref": image_ref,
        "meta": {
            "source_dataset": "mathfish",
            "source_id": source_id,
            "source_publisher": source,
            "language": "en",
            "subject": "mathematics",
            "split": "train",
            "modality": "multimodal" if image_ref else "text",
            "license": license_tag,
            "problem_activity_type": metadata.get("problem_activity_type"),
            "grade_or_subject": metadata.get("grade / subject"),
            "unit": metadata.get("unit"),
            "lesson_number": metadata.get("lesson_number"),
            "source_url": metadata.get("url"),
            "unresolved_images": unresolved_images,
        },
    }


def load_tal(language: str) -> list[dict[str, Any]]:
    suffix = "CN" if language == "zh" else "EN"
    path = TAL_DIR / f"TAL-SCQ5K-{suffix}/train.jsonl"
    rows: list[dict[str, Any]] = []
    dataset_name = f"TAL-SCQ5K-{suffix}"
    for raw in read_jsonl(path):
        source_id = clean_text(raw.get("queId") or raw.get("qid"))
        rows.append(
            make_kp_record(
                uid=f"tal-scq5k-{suffix.lower()}::{source_id}",
                source_dataset=dataset_name,
                source_id=source_id,
                question=clean_text(raw.get("problem")),
                options=format_tal_options(raw.get("answer_option_list")),
                paths=normalize_paths(raw.get("knowledge_point_routes")),
                language=language,
                split="train",
                license_tag="MIT",
                source_meta={
                    "difficulty": raw.get("difficulty"),
                    "question_type": raw.get("qtype"),
                },
            )
        )
    return rows


def load_da20k() -> list[dict[str, Any]]:
    questions = json.loads((DA20K_DIR / "math_questions_content.json").read_text(encoding="utf-8"))
    question_tags = json.loads(
        (DA20K_DIR / "math_questions_knowledgetag.json").read_text(encoding="utf-8")
    )
    knowledge = json.loads(
        (DA20K_DIR / "math_questions_knowledge.json").read_text(encoding="utf-8")
    )
    image_manifest: dict[str, dict[str, Any]] = {}
    if DA20K_IMAGE_MANIFEST.exists():
        manifest_payload = json.loads(DA20K_IMAGE_MANIFEST.read_text(encoding="utf-8"))
        image_manifest = manifest_payload.get("images") or {}
    knowledge_by_id = {item["id"]: item for item in knowledge}
    knowledge_by_uuid = {item["uuid"]: item for item in knowledge}
    tag_ids_by_question: dict[int, list[int]] = defaultdict(list)
    for edge in question_tags:
        label_id = edge.get("label_id")
        question_id = edge.get("qid")
        if (
            isinstance(question_id, int)
            and isinstance(label_id, int)
            and label_id not in tag_ids_by_question[question_id]
        ):
            tag_ids_by_question[question_id].append(label_id)

    def knowledge_path(label_id: int) -> list[str]:
        node = knowledge_by_id.get(label_id)
        path: list[str] = []
        seen: set[str] = set()
        while node:
            uuid = clean_text(node.get("uuid"))
            if not uuid or uuid in seen:
                return []
            seen.add(uuid)
            name = clean_text(node.get("name"))
            if name:
                path.append(name)
            parent_uuid = clean_text(node.get("parent_uuid"))
            if not parent_uuid:
                break
            node = knowledge_by_uuid.get(parent_uuid)
            if node is None:
                return []
        return list(reversed(path))

    rows: list[dict[str, Any]] = []
    for raw in questions:
        source_id = clean_text(raw.get("id"))
        label_ids = tag_ids_by_question.get(raw.get("id"), [])
        paths = [knowledge_path(label_id) for label_id in label_ids]
        question, image_ref, unresolved_images = clean_da20k_html(raw.get("text"), image_manifest)
        rows.append(
            make_kp_record(
                uid=f"DA-20K::{source_id}",
                source_dataset="DA-20K",
                source_id=source_id,
                question=question,
                options=[],
                paths=[path for path in paths if path],
                language="zh",
                split=stable_split(source_id),
                license_tag="license-unverified; non-commercial use",
                source_meta={
                    "knowledge_label_ids": label_ids,
                    "unresolved_images": unresolved_images,
                },
                image_ref=image_ref,
            )
        )
    return rows


def xes_image_index(archive: tarfile.TarFile) -> dict[str, str]:
    result: dict[str, str] = {}
    for member in archive.getmembers():
        if member.isfile() and "/metadata/images/question_" in member.name:
            result[Path(member.name).stem] = member.name
    return result


def replace_xes_images(
    question: str, image_members: dict[str, str]
) -> tuple[str, list[dict[str, str]], list[str]]:
    matches = list(dict.fromkeys(re.findall(r"question_\d+-image_\d+", question)))
    image_ref: list[dict[str, str]] = []
    unresolved: list[str] = []
    for index, source_placeholder in enumerate(matches):
        canonical = f"###IMAGE{index}###"
        question = question.replace(source_placeholder, canonical)
        member = image_members.get(source_placeholder)
        if member:
            image_ref.append(
                {
                    "placeholder": canonical,
                    "archive": str(XES_ARCHIVE),
                    "member": member,
                    "filename": Path(member).name,
                }
            )
        else:
            unresolved.append(source_placeholder)
    return clean_text(question), image_ref, unresolved


def load_xes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tarfile.open(XES_ARCHIVE, "r:gz") as archive:
        questions_file = archive.extractfile("XES3G5M/metadata/questions.json")
        if questions_file is None:
            raise RuntimeError("XES3G5M metadata/questions.json is missing")
        questions = json.load(questions_file)
        members = xes_image_index(archive)
    for source_id, raw in questions.items():
        question, image_ref, unresolved = replace_xes_images(
            clean_text(raw.get("content")), members
        )
        record = make_kp_record(
            uid=f"XES3G5M::{source_id}",
            source_dataset="XES3G5M",
            source_id=str(source_id),
            question=question,
            options=format_xes_options(raw.get("options")),
            paths=normalize_paths(raw.get("kc_routes")),
            language="zh",
            split=stable_split(str(source_id)),
            license_tag="MIT",
            source_meta={
                "question_type": raw.get("type"),
                "unresolved_images": unresolved,
            },
            image_ref=image_ref,
        )
        rows.append(record)
    return rows


def label_signature(record: dict[str, Any]) -> str:
    grounding = record.get("grounding") or {}
    if record.get("task_type") == "activity_to_ccss_alignment":
        labels = [
            (
                item.get("relation"),
                item.get("standard_code"),
                item.get("standard_description"),
            )
            for item in grounding.get("standard_alignments") or []
        ]
    else:
        labels = grounding.get("knowledge_paths") or []
    return json.dumps(labels, ensure_ascii=False, sort_keys=True)


def record_dedup_key(record: dict[str, Any]) -> str:
    user_message = (record.get("messages") or [{}, {}])[0].get("content", "")
    key = normalize_for_dedup(user_message)
    image_refs = record.get("image_ref") or []
    if image_refs:
        # The same text placeholder can accompany different diagrams. Treating
        # those as duplicates would be a false positive, so image identity is
        # part of the key. We intentionally prefer false negatives here.
        image_identity = "|".join(
            f"{ref.get('archive', '')}::{ref.get('member', '')}" for ref in image_refs
        )
        key = f"{key}::images::{image_identity}"
    return key


def structural_error(record: dict[str, Any]) -> str | None:
    messages = record.get("messages") or []
    if len(messages) != 2 or not clean_text(messages[0].get("content")):
        return "missing_question"
    grounding = record.get("grounding") or {}
    if record.get("task_type") == "activity_to_ccss_alignment":
        alignments = grounding.get("standard_alignments") or []
        if not alignments:
            return "missing_alignment"
        if any(item.get("unresolved") for item in alignments):
            return "unresolved_standard"
    elif not grounding.get("knowledge_paths"):
        return "missing_alignment"
    if (record.get("meta") or {}).get("unresolved_images"):
        return "missing_image"
    if not clean_text(messages[1].get("content")):
        return "missing_response"
    return None


def structural_stage(
    rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    kept: list[dict[str, Any]] = []
    routed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        error = structural_error(row)
        if error:
            rejected = dict(row)
            rejected["_reason"] = error
            routed[error].append(rejected)
        else:
            kept.append(row)
    return kept, routed


def dedup_stage(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[record_dedup_key(row)].append(row)

    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for key, group in groups.items():
        signatures = {label_signature(row) for row in group}
        if len(signatures) > 1:
            conflict_uids = [row["uid"] for row in group]
            for row in group:
                conflict = dict(row)
                conflict.update(
                    _reason="same_question_conflicting_labels",
                    _conflict_uids=conflict_uids,
                    _dedup_key=key,
                )
                conflicts.append(conflict)
            continue
        kept.append(group[0])
        for row in group[1:]:
            duplicate = dict(row)
            duplicate.update(
                _reason="same_question_same_labels",
                _kept_uid=group[0]["uid"],
                _dedup_key=key,
            )
            duplicates.append(duplicate)
    return kept, duplicates, conflicts


def output_split_files(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for modality in ("text", "multimodal"):
        modality_rows = [row for row in rows if (row.get("meta") or {}).get("modality") == modality]
        write_jsonl(output_dir / f"kept_{modality}.jsonl", modality_rows)
        for split in ("train", "validation", "test"):
            split_rows = [
                row for row in modality_rows if (row.get("meta") or {}).get("split") == split
            ]
            if split_rows:
                write_jsonl(output_dir / f"{modality}_{split}.jsonl", split_rows)


def process_dataset(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_dir = PIPE / name
    write_jsonl(dataset_dir / "origin/origin.jsonl", rows)

    structurally_kept, routed = structural_stage(rows)
    write_jsonl(dataset_dir / "structural/kept.jsonl", structurally_kept)
    for reason, rejected in sorted(routed.items()):
        prefix = "routed" if reason == "missing_image" else "removed"
        write_jsonl(
            dataset_dir / f"structural/{prefix}_{reason}.jsonl",
            rejected,
        )

    kept, duplicates, conflicts = dedup_stage(structurally_kept)
    layer2 = dataset_dir / "cleaned"
    output_split_files(layer2, kept)
    write_jsonl(layer2 / "removed_duplicates.jsonl", duplicates)
    write_jsonl(layer2 / "routed_label_conflicts.jsonl", conflicts)

    split_counts = Counter((row.get("meta") or {}).get("split") for row in kept)
    modality_counts = Counter((row.get("meta") or {}).get("modality") for row in kept)
    relation_counts: Counter[str] = Counter()
    license_counts = Counter((row.get("meta") or {}).get("license") for row in kept)
    for row in kept:
        for alignment in (row.get("grounding") or {}).get("standard_alignments", []):
            relation_counts[alignment["relation"]] += 1
    report = {
        "dataset": name,
        "origin": len(rows),
        "structurally_kept": len(structurally_kept),
        "structural_routes": {reason: len(rejected) for reason, rejected in sorted(routed.items())},
        "dedup_kept": len(kept),
        "removed_duplicates": len(duplicates),
        "routed_label_conflicts": len(conflicts),
        "split_counts": dict(sorted(split_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "relation_occurrences": dict(sorted(relation_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
    }
    (dataset_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_final_outputs(reports: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    image_members: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        dataset_dir = PIPE / report["dataset"] / "cleaned"
        for path in sorted(dataset_dir.glob("kept_*.jsonl")):
            for row in read_jsonl(path):
                checked += 1
                messages = row.get("messages") or []
                if len(messages) != 2:
                    failures.append(f"{row.get('uid')}: messages != 2")
                if any(
                    forbidden in json.dumps(messages, ensure_ascii=False).lower()
                    for forbidden in (
                        '"answer_analysis"',
                        '"answer_value"',
                        '"analysis"',
                    )
                ):
                    failures.append(f"{row.get('uid')}: leaked source answer field")
                if row.get("task_type") == "activity_to_ccss_alignment":
                    for alignment in (row.get("grounding") or {}).get("standard_alignments", []):
                        if not alignment.get("standard_description"):
                            failures.append(f"{row.get('uid')}: unresolved standard description")
                placeholders = set(re.findall(r"###IMAGE\d+###", messages[0].get("content", "")))
                refs = {ref.get("placeholder") for ref in row.get("image_ref") or []}
                if placeholders != refs:
                    failures.append(f"{row.get('uid')}: image placeholders do not match refs")
                for ref in row.get("image_ref") or []:
                    archive = ref.get("archive")
                    member = ref.get("member")
                    if archive and member:
                        image_members[archive].add(member)
    image_refs_checked = sum(len(members) for members in image_members.values())
    missing_image_members: list[str] = []
    for archive_path, wanted_members in sorted(image_members.items()):
        archive = Path(archive_path)
        if not archive.exists():
            missing_image_members.extend(
                f"{archive_path}::{member}" for member in sorted(wanted_members)
            )
            continue
        with tarfile.open(archive, "r:gz") as handle:
            found = {
                member.name
                for member in handle
                if member.isfile() and member.name in wanted_members
            }
        missing_image_members.extend(
            f"{archive_path}::{member}" for member in sorted(wanted_members - found)
        )
    failures.extend(f"missing image member: {item}" for item in missing_image_members)
    result = {
        "records_checked": checked,
        "image_refs_checked": image_refs_checked,
        "missing_image_member_count": len(missing_image_members),
        "failure_count": len(failures),
        "failures": failures[:200],
    }
    (PIPE / "QC_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["DA-20K", "TAL-SCQ5K-CN", "TAL-SCQ5K-EN", "xes", "mathfish"],
        default=["DA-20K", "TAL-SCQ5K-CN", "TAL-SCQ5K-EN", "xes", "mathfish"],
    )
    args = parser.parse_args()

    catalog = load_standard_catalog()
    loaders = {
        "DA-20K": ("DA-20K", load_da20k),
        "TAL-SCQ5K-CN": ("tal-scq5k-cn", lambda: load_tal("zh")),
        "TAL-SCQ5K-EN": ("tal-scq5k-en", lambda: load_tal("en")),
        "xes": ("XES3G5M", load_xes),
        "mathfish": (
            "mathfish",
            lambda: [
                make_mathfish_record(raw, catalog)
                for raw in read_jsonl(MATHFISH_DIR / "train.jsonl")
            ],
        ),
    }
    reports: list[dict[str, Any]] = []
    for requested in args.datasets:
        name, loader = loaders[requested]
        print(f"Processing {name}...")
        report = process_dataset(name, loader())
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
    qc = validate_final_outputs(reports)
    print(json.dumps(qc, ensure_ascii=False))
    if qc["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
