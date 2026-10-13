# Stage 1 — Capability taxonomy and source collection

Defines the capability categories and keeps the source snapshots for every stage. The dataset inventory is in the paper and is not repeated here.

```
stage1_capability_taxonomy_and_source_collection/
├── download_from_huggingface.py   # generic Hugging Face downloader (no source list baked in)
└── general_purpose/
    ├── build_pool.py              # builds the general-purpose pool
    └── general_instruction_pool.jsonl   # its output, merged verbatim at stage 6
```

Sources are stored per capability: `subject_competence/`, `curriculum_grounding/`, `diagnostic_reasoning/`, `pedagogical_action/`, each with `huggingface/`, `github/` and `direct/` subdirectories.

- Hugging Face sources: download with `download_from_huggingface.py` (see the script).
- GitHub sources: `git clone` into the matching `github/` subdirectory.
- Other sources: place them under `direct/`.

## general_purpose/ — general-purpose pool

`build_pool.py` standardizes the three curated sources into `general_instruction_pool.jsonl`.

```bash
python data_preparation/stage1_capability_taxonomy_and_source_collection/general_purpose/build_pool.py \
  --source dataflow-instruct-10k=<file or directory> --source tulu-3-sft-mixture=<directory of subsets> \
  --source mathv360k=<directory of subsets>
```

This component is only standardized here: original system prompts are preserved, it never enters stages 2–5, and it is merged verbatim at stage 6.
