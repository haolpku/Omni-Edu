#!/usr/bin/env python3
"""Clean two structured curriculum resources that are ready for extraction.

The script is deterministic and follows the pipeline's no-delete convention:

* Australian Curriculum v9: parse the official workbook without third-party
  dependencies, strip presentational HTML, preserve hierarchy fields, and
  route duplicate codes instead of silently dropping them.
* NCETM curriculum maps: extract Year 1--6 unit sequences from all locally
  saved HTML aliases, canonicalize the duplicate ``cp-``/non-``cp-`` pages,
  and retain provenance for every source page.

These are curriculum structure records rather than question-grounding SFT
records. They are intended as clean inputs for a later framework-aware
rendering pass.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

PIPE = Path("data_preparation/stage2_deterministic_cleaning_and_evaluation_decontamination/curriculum_grounding")
DATE = "2026-08-06"
AU_PATH = Path("direct/Australian_Curriculum/curriculum-workbook.xlsx")
NCETM_PAGES = Path("direct/NCETM/pages")
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "br",
        "p",
        "div",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_text(value: Any) -> str:
    text = str(value or "")
    if "<" in text and ">" in text:
        parser = TextExtractor()
        parser.feed(text)
        text = "".join(parser.parts)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\u202f", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalized_key(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value).casefold()).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def cell_column(reference: str) -> str:
    return "".join(char for char in reference if char.isalpha())


def load_xlsx_rows(path: Path) -> list[tuple[str, list[dict[str, str]]]]:
    """Read all worksheets using only the Python standard library."""
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(f"{{{XLSX_NS['m']}}}t"))
            for item in shared_root.findall("m:si", XLSX_NS)
        ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_nodes = workbook.find("m:sheets", XLSX_NS)
        names = [] if sheet_nodes is None else [sheet.attrib["name"] for sheet in sheet_nodes]
        result: list[tuple[str, list[dict[str, str]]]] = []
        for index, name in enumerate(names, start=1):
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))
            matrix: list[dict[str, str]] = []
            for row in root.findall(".//m:sheetData/m:row", XLSX_NS):
                values: dict[str, str] = {}
                for cell in row.findall("m:c", XLSX_NS):
                    ref = cell.attrib.get("r", "")
                    column = cell_column(ref)
                    value_node = cell.find("m:v", XLSX_NS)
                    value = "" if value_node is None else value_node.text or ""
                    if cell.attrib.get("t") == "s" and value:
                        value = shared[int(value)]
                    values[column] = value
                matrix.append(values)
            if not matrix:
                result.append((name, []))
                continue
            header = {column: clean_text(value) for column, value in matrix[0].items()}
            rows = [
                {
                    header[column]: value
                    for column, value in raw.items()
                    if column in header and header[column]
                }
                for raw in matrix[1:]
            ]
            result.append((name, rows))
        return result


def australian_record_type(sheet: str, row: dict[str, str]) -> str:
    if sheet == "Learning areas":
        if row.get("Elaboration"):
            return "elaboration"
        if row.get("Content Description"):
            return "content_description"
        if row.get("Sub-Strand"):
            return "sub_strand"
        if row.get("Strand"):
            return "strand"
        if row.get("Level Description"):
            return "level"
        if row.get("Subject"):
            return "subject"
        return "learning_area"
    if sheet == "Achievement standards":
        return "achievement_standard"
    if sheet == "Cross-curriculum priorities":
        if row.get("Organising idea indicator"):
            return "organising_idea_indicator"
        if row.get("Organising ideas title"):
            return "organising_idea"
        return "cross_curriculum_priority"
    if sheet == "General capabilities":
        if row.get("Indicator"):
            return "capability_indicator"
        if row.get("Level"):
            return "capability_level"
        if row.get("Sub-Element"):
            return "capability_sub_element"
        if row.get("Element"):
            return "capability_element"
        return "general_capability"
    return "unknown"


def clean_australian() -> dict[str, Any]:
    out = PIPE / "australian-curriculum-v9"
    sheets = load_xlsx_rows(AU_PATH)
    origin: list[dict[str, Any]] = []
    structural: list[dict[str, Any]] = []
    missing_code: list[dict[str, Any]] = []

    for sheet, rows in sheets:
        for row_number, row in enumerate(rows, start=2):
            raw = {key: value for key, value in row.items() if str(value or "").strip()}
            origin.append(
                {
                    "source_sheet": sheet,
                    "source_row": row_number,
                    "raw": raw,
                }
            )
            cleaned = {key: clean_text(value) for key, value in raw.items()}
            code = cleaned.get("Code", "")
            generated_code = False
            if sheet == "Achievement standards" and not code:
                # The official achievement-standards worksheet has no Code
                # column. Use a deterministic composite identifier rather than
                # discarding all 260 valid standards as "missing code".
                identity = "\x1f".join(
                    cleaned.get(field, "")
                    for field in (
                        "Learning Area",
                        "Subject",
                        "Pathway",
                        "Sequence",
                        "Level",
                        "Achievement Standard",
                    )
                )
                code = "AS-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                generated_code = True
            if not code:
                routed = {
                    "source_sheet": sheet,
                    "source_row": row_number,
                    "raw": raw,
                    "_reason": "missing_code",
                }
                missing_code.append(routed)
                continue
            record_type = australian_record_type(sheet, cleaned)
            record = {
                "uid": f"australian-curriculum-v9::{code}",
                "framework": "Australian Curriculum v9",
                "record_type": record_type,
                "code": code,
                "code_origin": "generated_composite" if generated_code else "source",
                "learning_area": cleaned.get("Learning Area")
                or cleaned.get("Cross-Curriculum Priority")
                or cleaned.get("General Capability"),
                "subject": cleaned.get("Subject") or None,
                "level": cleaned.get("Level") or None,
                "pathway": cleaned.get("Pathway") or None,
                "sequence": cleaned.get("Sequence") or None,
                "strand": cleaned.get("Strand") or None,
                "sub_strand": cleaned.get("Sub-Strand") or None,
                "element": cleaned.get("Element") or None,
                "sub_element": cleaned.get("Sub-Element") or None,
                "title": cleaned.get("Organising ideas title") or None,
                "description": (
                    cleaned.get("Content Description")
                    or cleaned.get("Elaboration")
                    or cleaned.get("Achievement Standard")
                    or cleaned.get("Organising idea indicator")
                    or cleaned.get("Description")
                    or cleaned.get("Indicator")
                    or cleaned.get("Level Description")
                ),
                "topics": cleaned.get("Topics") or None,
                "source": {
                    "file": str(AU_PATH),
                    "sheet": sheet,
                    "row": row_number,
                },
                "meta": {
                    "source_dataset": "Australian Curriculum v9",
                    "language": "en",
                    "jurisdiction": "Australia",
                    "extraction_date": DATE,
                    "license": "review_required",
                },
            }
            structural.append({key: value for key, value in record.items() if value is not None})

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in structural:
        groups[record["code"]].append(record)
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for code, group in groups.items():
        canonical = group[0]
        kept.append(canonical)
        for duplicate in group[1:]:
            routed = dict(duplicate)
            routed.update(
                _reason="duplicate_code",
                _kept_uid=canonical["uid"],
                _dedup_stage="exact_code",
            )
            duplicates.append(routed)

    kept.sort(key=lambda row: (row["source"]["sheet"], row["source"]["row"]))
    write_jsonl(out / "origin/origin.jsonl", origin)
    write_jsonl(out / "structural/kept.jsonl", structural)
    write_jsonl(out / "structural/removed_missing_code.jsonl", missing_code)
    write_jsonl(out / "cleaned/kept.jsonl", kept)
    write_jsonl(out / "cleaned/removed_duplicates.jsonl", duplicates)

    report = {
        "dataset": "Australian Curriculum v9",
        "source_file": str(AU_PATH),
        "source_sha256": sha256_file(AU_PATH),
        "extraction_date": DATE,
        "counts": {
            "origin": len(origin),
            "structural_kept": len(structural),
            "removed_missing_code": len(missing_code),
            "dedup_kept": len(kept),
            "removed_duplicate_code": len(duplicates),
        },
        "generated_code_count": sum(
            row.get("code_origin") == "generated_composite" for row in kept
        ),
        "by_sheet": dict(Counter(row["source"]["sheet"] for row in kept)),
        "by_record_type": dict(Counter(row["record_type"] for row in kept)),
    }
    write_json(out / "report.json", report)
    return report


class NCETMUnitParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_item = 0
        self.capture: str | None = None
        self.title_parts: list[str] = []
        self.meta_parts: list[str] = []
        self.href: str | None = None
        self.units: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        classes = set((attr.get("class") or "").split())
        if tag == "li" and "list-equal-height-item" in classes:
            self.in_item += 1
        if not self.in_item:
            return
        if tag == "a":
            self.href = attr.get("href")
        elif tag == "h4":
            self.capture = "title"
        elif tag == "p":
            self.capture = "meta"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h4", "p"}:
            self.capture = None
        if tag == "li" and self.in_item:
            title = clean_text("".join(self.title_parts))
            meta = clean_text("".join(self.meta_parts))
            match = re.fullmatch(r"Unit\s+(\d+)\s*[–-]\s*(\d+)\s+weeks?", meta)
            if title and match:
                self.units.append(
                    {
                        "title": title,
                        "unit_number": match.group(1),
                        "weeks": match.group(2),
                        "href": self.href or "",
                    }
                )
            self.in_item -= 1
            self.capture = None
            self.title_parts = []
            self.meta_parts = []
            self.href = None

    def handle_data(self, data: str) -> None:
        if self.capture == "title":
            self.title_parts.append(data)
        elif self.capture == "meta":
            self.meta_parts.append(data)


def ncetm_year(path: Path) -> int:
    match = re.search(r"year-(\d+)-curriculum-map", path.name)
    if not match:
        raise ValueError(f"Cannot determine year from {path}")
    return int(match.group(1))


def clean_ncetm() -> dict[str, Any]:
    out = PIPE / "ncetm-curriculum-maps"
    paths = sorted(NCETM_PAGES.glob("*year-*-curriculum-map.html"))
    origin: list[dict[str, Any]] = []
    parsed: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    for path in paths:
        parser = NCETMUnitParser()
        parser.feed(path.read_text(encoding="utf-8"))
        year = ncetm_year(path)
        origin.append(
            {
                "source_file": str(path),
                "source_sha256": sha256_file(path),
                "year": year,
                "parsed_unit_count": len(parser.units),
            }
        )
        if not parser.units:
            parse_failures.append(
                {
                    "source_file": str(path),
                    "year": year,
                    "_reason": "no_units_parsed",
                }
            )
        for unit in parser.units:
            unit_number = int(unit["unit_number"])
            parsed.append(
                {
                    "uid": f"ncetm-curriculum-map::year-{year}::unit-{unit_number}",
                    "framework": "NCETM Curriculum Prioritisation",
                    "record_type": "curriculum_unit",
                    "subject": "Mathematics",
                    "phase": "Primary",
                    "key_stage": "KS1" if year <= 2 else "KS2",
                    "year": year,
                    "unit_number": unit_number,
                    "title": unit["title"],
                    "duration_weeks": int(unit["weeks"]),
                    "source_url_path": unit["href"],
                    "source": {"file": str(path)},
                    "meta": {
                        "source_dataset": "NCETM curriculum maps",
                        "language": "en",
                        "jurisdiction": "England",
                        "extraction_date": DATE,
                        "license": "review_required",
                    },
                }
            )

    groups: dict[tuple[int, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in parsed:
        key = (
            record["year"],
            record["unit_number"],
            normalized_key(record["title"]),
            record["duration_weeks"],
        )
        groups[key].append(record)
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    slot_signatures: dict[tuple[int, int], set[tuple[str, int]]] = defaultdict(set)
    for record in parsed:
        slot_signatures[(record["year"], record["unit_number"])].add(
            (normalized_key(record["title"]), record["duration_weeks"])
        )
    conflict_slots = {slot for slot, signatures in slot_signatures.items() if len(signatures) > 1}

    for group in groups.values():
        # Prefer the explicit cp-* page as the canonical provenance.
        canonical = sorted(
            group,
            key=lambda row: (
                not Path(row["source"]["file"]).name.startswith("cp-"),
                row["source"]["file"],
            ),
        )[0]
        kept.append(canonical)
        for duplicate in group:
            if duplicate is canonical:
                continue
            routed = dict(duplicate)
            routed.update(
                _reason="duplicate_page_alias",
                _kept_uid=canonical["uid"],
                _kept_source_file=canonical["source"]["file"],
            )
            duplicates.append(routed)
    for record in parsed:
        if (record["year"], record["unit_number"]) in conflict_slots:
            routed = dict(record)
            routed["_reason"] = "same_year_unit_conflicting_content"
            conflicts.append(routed)

    kept.sort(key=lambda row: (row["year"], row["unit_number"]))
    write_jsonl(out / "origin/pages.jsonl", origin)
    write_jsonl(out / "structural/kept.jsonl", parsed)
    write_jsonl(out / "structural/removed_parse_failures.jsonl", parse_failures)
    write_jsonl(out / "cleaned/kept.jsonl", kept)
    write_jsonl(out / "cleaned/removed_duplicates.jsonl", duplicates)
    write_jsonl(out / "cleaned/routed_conflicts.jsonl", conflicts)

    report = {
        "dataset": "NCETM curriculum maps",
        "source_directory": str(NCETM_PAGES),
        "extraction_date": DATE,
        "counts": {
            "source_pages": len(paths),
            "parsed_units": len(parsed),
            "parse_failures": len(parse_failures),
            "dedup_kept": len(kept),
            "removed_duplicate_page_alias": len(duplicates),
            "routed_conflicts": len(conflicts),
        },
        "by_year": dict(Counter(str(row["year"]) for row in kept)),
        "weeks_by_year": {
            str(year): sum(row["duration_weeks"] for row in kept if row["year"] == year)
            for year in sorted({row["year"] for row in kept})
        },
    }
    write_json(out / "report.json", report)
    return report


def validate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    australian, ncetm = reports
    failures: list[str] = []
    au_counts = australian["counts"]
    if au_counts["origin"] != (au_counts["structural_kept"] + au_counts["removed_missing_code"]):
        failures.append("Australian origin accounting mismatch")
    if au_counts["structural_kept"] != (
        au_counts["dedup_kept"] + au_counts["removed_duplicate_code"]
    ):
        failures.append("Australian dedup accounting mismatch")
    nc_counts = ncetm["counts"]
    if nc_counts["parsed_units"] != (
        nc_counts["dedup_kept"] + nc_counts["removed_duplicate_page_alias"]
    ):
        failures.append("NCETM dedup accounting mismatch")
    if nc_counts["parse_failures"]:
        failures.append("NCETM has source pages with no parsed units")
    if nc_counts["routed_conflicts"]:
        failures.append("NCETM has conflicting year/unit definitions")
    report = {
        "date": DATE,
        "datasets": [item["dataset"] for item in reports],
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(PIPE / "EASY_CURRICULUM_QC_REPORT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", "australian", "ncetm"),
        default="all",
    )
    args = parser.parse_args()
    reports: list[dict[str, Any]] = []
    if args.dataset in {"all", "australian"}:
        reports.append(clean_australian())
    if args.dataset in {"all", "ncetm"}:
        reports.append(clean_ncetm())
    if args.dataset == "all":
        qc = validate_reports(reports)
        print(json.dumps({"reports": reports, "qc": qc}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(reports[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
