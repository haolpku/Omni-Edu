# Stage 6 — Pedagogical instruction assignment and final assembly

Assigns one of 19 task-specific system instructions to each education-specific example by task bucket, then merges the 9,048 general-purpose examples and runs the format and media checks.

```
stage6_pedagogical_instruction_assignment_and_final_assembly/
├── build_release.py          # education-specific: unified messages, image materialization, checks
└── assign_system_prompts.py  # the 19 system instructions → merge general data → final release
```

```bash
python data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly/build_release.py
python data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly/assign_system_prompts.py
```

```
releases/assembled/train.jsonl      # education-specific, unified and with media, no system instruction yet
releases/omniedu/train.jsonl        # final corpus: education-specific + general-purpose examples
releases/omniedu/dataset_info.json  # LLaMA-Factory data config
releases/omniedu/system_prompt_registry.json   # the 19 task-specific templates
releases/omniedu/assembly_report.json          # counts, prompt distribution, sequence length report
```
