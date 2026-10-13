# Training

Full-parameter SFT of the OmniEdu models (4B, 9B, 27B) with LLaMA-Factory on the final stage-6 release.

```
training/
├── run_training.sh         # main entry: 4b | 9b | 27b
├── launch_27b_node.sh      # per-node entry for the two-node 27B run
└── automation/
    ├── train.py            # checks the release, renders the LLaMA-Factory config, starts torchrun
    ├── launch.sh           # wrapper around `train.py launch`
    ├── worker.sh           # wrapper around `train.py worker` (one node)
    └── h20_bond1.env       # example NCCL/GLOO environment file for the training cluster
```

Run from the repository root:

```bash
# 4B / 9B: one node, eight GPUs
bash training/run_training.sh 4b
bash training/run_training.sh 9b

# 27B: one copy per node, start the second node first
NODE_RANK=1 MASTER_ADDR=<master node address> bash training/launch_27b_node.sh
NODE_RANK=0 MASTER_ADDR=<master node address> bash training/launch_27b_node.sh

# print the rendered config without starting anything
bash training/run_training.sh 9b --dry-run
```

- Hyperparameters follow the paper; `--dry-run` prints the full rendered configuration.
- The release defaults to `data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly/releases/omniedu` and is validated before launch (JSONL structure, referenced images, 69,999 examples); `--allow-nonpaper-counts` relaxes the count.
- Base models and the framework are resolved as `models/` and `LLaMA-Factory/` under the repository root; both, as well as the output directory and NCCL environment file, can be overridden from the command line. Runs write checkpoints to `training/outputs/` and logs to `training/logs/`.
