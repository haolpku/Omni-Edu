# Stage 2 — Deterministic cleaning and evaluation decontamination

Rule- and threshold-based cleaning, answer verification and deduplication; no model calls.

## dedup/ — two-layer deduplication (applies to every education dataset)

| File | Role |
| --- | --- |
| `dedup/within_source.py` | Within-dataset deduplication: one row per exact duplicate (keeps the longest CoT), the rest are routed no-delete |
| `dedup/across_sources.py` | Cross-dataset deduplication: normalized problem text plus an answer-consistency check before a row counts as a duplicate |

Both pick the deduplication key by task family (`math` = problem + options; `short_answer` = problem + student answer + label; `reading` = passage + problem + options; `dialogue` = canonical message JSON; `generic` = problem + options + answer). Datasets are discovered from the directory layout, so no inventory is maintained. Options: `--list-only`, `--dry-run`, and `--pool-root` for another root that follows the same layout.

### Standard output paths

Each dataset exposes its cleaning output at one fixed path, and the dedup scripts discover datasets by it (see `common/pools.py`):

```
<dataset>/cleaned/kept.jsonl               # cleaning output, input of both dedup layers
<dataset>/cleaned/rejected.jsonl           # dropped rows (optional)
<dataset>/cleaned/report.json              # per-dataset cleaning report (optional)
<dataset>/dedup/within/kept.jsonl          # within-dataset deduplication output
<dataset>/dedup/within/removed_intra_duplicates.jsonl
dedup/across_sources/<family>/kept.jsonl   # cross-dataset deduplication output
dedup/across_sources/<family>/removed_cross_duplicates.jsonl
```

## subject_competence/

| File | Datasets | Role |
| --- | --- | --- |
| `clean_math_family.py` | GSM8K, MATH, OpenR1, MathQA, SciInstruct (CN math) | structural cleaning |
| `clean_nonmath_family.py` | ARC, QASC, SciQ, SciInstruct (EN), CJEval, ScienceQA (text), ScienceQA (image), Geometry3K | structural cleaning, split into text and image pools |
| `clean_cmm_math.py` | CMM-Math (text cot), (text answer), (image) | structural cleaning, split into three pools |
| `mcq/clean.py` | RACE, C3, TQA (text), TQA (diagram), WorldTree, AI2D | shared multiple-choice normalization (passage, options, answer letter); image rows are referenced by origin index |
| `essays/clean.py` | ASAP 2.0, CSEE, ELLIPSE | essay, prompt, scores and feedback; scores are copied verbatim |
| `mixed_domain/aquila/` | AquilaEdu | quality-score filter → near-duplicate dedup → K-12 domain routing |
| `mixed_domain/coig/` | COIG-CQIA | domain routing → near-duplicate dedup |

## curriculum_grounding/

| File | Role |
| --- | --- |
| `dedup.py` | deduplication for curriculum sources (evaluation-decontamination index, byte-level asset inventory) |
| `qc.py` | structural and count checks on the deduplication output |
| `apply_dedup_to_cleaned.py` | projects deduplication exclusions onto the cleaned output (no-delete) |
| `check_cleaned_projection.py` | checks that projection |
| `build_structure_sft.py` | curriculum-structure alignment records (DA-20K, TAL-SCQ5K, XES3G5M, MathFish) |
| `curriculum_frameworks/clean.py` | Australian Curriculum v9 and NCETM framework documents only |
| `k12-kgraph/clean.py` | K12-KGraph only |

DA-20K image download belongs to source collection and lives in stage 1: `stage1_capability_taxonomy_and_source_collection/curriculum_grounding/da-20k/download_images.py`.

## diagnostic_reasoning/

| File | Role |
| --- | --- |
| `clean.py` | ErrorRadar, MAP (license-quarantined), StepVerify, algebra_misconceptions, Beetle, Beetle-Atomi-5way, SciEntsBank, MathDial (`--batch 1\|2` reruns a single batch) |
| `aaas_longtutor/clean.py` | AAAS public files and LongTutor |
| `bridge_semeval_clc_fce/clean.py` | Bridge, SemEval 2013 Task 7 and CLC-FCE |
| `drawedumath/clean.py` | DrawEduMath |
| `scratchmath/clean.py` | ScratchMath |
| `ErrorRadar/localize_images.py` | downloads the images referenced by the ErrorRadar pool and writes an image manifest |

## pedagogical_action/

Each source folder keeps one `clean.py`; a single run performs all processing for that source family (structural cleaning, deduplication, required joins and checks) and writes into each dataset's `cleaned/` directory.

| File | Datasets | Role |
| --- | --- | --- |
| `clean.py` | CIMA, TalkMoves, MultiHint, SocraticMATH, SocraticMATH-sol, EduAdapt | structural cleaning, exact deduplication, checks (`--batch 1\|2` reruns a single batch) |
| `feat_foxglove_sefora/clean.py` | FEAT, FOXGLOVE, SEFORA | essay and feedback cleaning, then feedback joined with its essay/task context |
| `oatutor_content/clean.py` | OATutor | content-pool extraction, then guided-dialogue joining |
| `remaining_sources/clean.py` | Eedi, MRBench | cleaning followed by structural validation |
| `socrateach_convolearn/clean.py` | SocraTeach, ConvoLearn | dialogue cleaning |
| `tmath_learningq/clean.py` | TMATH, LearningQ | cleaning followed by QC |
| `tutorchat_education_dialogue/clean.py` | TutorChat, Education-Dialogue-Dataset | dialogue cleaning and exact deduplication |
| `wiki_hint/clean.py` | WikiHint | cleaning |

## Shared helpers

`../common/`: JSONL/text/hash utilities, dataset path conventions (`pools.py`), task buckets (`buckets.py`), record builders and pool writers (`cleaning.py`).
