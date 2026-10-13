#!/usr/bin/env python3
"""Parameterized LLaMA-Factory SFT launcher for the OmniEdu release.

The settings rendered by ``build_config`` mirror the training configuration
reported in the paper (SFT on the final stage-6 release, Table
"Training configuration for OmniEdu").  ``worker`` runs one node, ``launch``
starts the required number of nodes locally or over SSH.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, NoReturn

TRAINING_DIR = Path(__file__).resolve().parent.parent  # training/
DEFAULT_LLAMAFACTORY_DIR = Path("LLaMA-Factory")
DEFAULT_LOG_DIR = TRAINING_DIR / "logs"
DEFAULT_OUTPUT_ROOT = TRAINING_DIR / "outputs"
DEFAULT_EXTRA_SITE = Path("envs/llamafactory/lib/python3.13/site-packages")
DEFAULT_RELEASE_DIR = (
    Path("data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly")
    / "releases/omniedu"
)

#: final corpus size reported in the paper; the release used for training must match it
PAPER_TRAIN_ROWS = 69_999

#: one node holds eight accelerators, as reported in the paper
GPUS_PER_NODE = 8

#: chat template reported in the paper for the Qwen3.5/Qwen3.8 backbones
DEFAULT_TEMPLATE = "qwen3_5_nothink"

PROFILES = {
    "4b": {
        "nodes": 1,
        "model": Path("models/Qwen3.5-4B-Base"),
        "save_steps": 500,
    },
    "9b": {
        "nodes": 1,
        "model": Path("models/Qwen3.5-9B-Base"),
        "save_steps": 500,
    },
    "27b": {
        "nodes": 2,
        "model": Path("models/Qwen3.8-27B"),
        "save_steps": 200,
    },
}


def fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def fill_path_defaults(args: argparse.Namespace) -> None:
    profile = PROFILES[args.model_size]
    if not args.model_path:
        args.model_path = str(profile["model"])
    if not args.output_dir:
        release_name = resolve_path(args.release_dir).name
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / f"{args.model_size}-{release_name}")
    if not args.config_dir:
        args.config_dir = str(resolve_path(args.llamafactory_dir) / "configs" / "omniedu")


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        fail(f"environment file does not exist: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            fail(f"{path} line {line_number}: expected NAME=VALUE")
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            fail(f"{path} line {line_number}: invalid environment name {name!r}")
        result[name] = value.strip().strip("\"'")
    return result


def infer_dataset_key(dataset_info: dict[str, Any], requested: str | None) -> str:
    if requested:
        if requested not in dataset_info:
            fail(f"dataset key {requested!r} is absent from dataset_info.json")
        return requested

    candidates = [
        key
        for key, spec in dataset_info.items()
        if isinstance(spec, dict) and Path(str(spec.get("file_name", ""))).name == "train.jsonl"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and len(dataset_info) == 1:
        return next(iter(dataset_info))
    fail(
        "cannot infer dataset key unambiguously; pass --dataset-key "
        f"(available: {', '.join(dataset_info) or '<none>'})"
    )


def iter_image_paths(images: Any, line_number: int) -> Iterable[str]:
    if images is None:
        return
    if isinstance(images, str):
        yield images
        return
    if isinstance(images, list) and all(isinstance(item, str) for item in images):
        yield from images
        return
    fail(f"train.jsonl line {line_number}: images must be a string or list of strings")


def validate_release(
    release_dir: Path, requested_key: str | None, enforce_paper_rows: bool
) -> tuple[str, int, int, int]:
    train_file = release_dir / "train.jsonl"
    info_file = release_dir / "dataset_info.json"
    if not train_file.is_file() or train_file.stat().st_size == 0:
        fail(f"missing or empty {train_file}")
    if not info_file.is_file() or info_file.stat().st_size == 0:
        fail(f"missing or empty {info_file}")

    try:
        dataset_info = json.loads(info_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid dataset_info.json: {exc}")
    if not isinstance(dataset_info, dict) or not dataset_info:
        fail("dataset_info.json must be a non-empty JSON object")

    dataset_key = infer_dataset_key(dataset_info, requested_key)
    spec = dataset_info[dataset_key]
    if not isinstance(spec, dict):
        fail(f"dataset_info.json entry {dataset_key!r} must be an object")
    file_name = spec.get("file_name")
    if Path(str(file_name)).name != "train.jsonl":
        fail(f"dataset {dataset_key!r} must point to train.jsonl, got {file_name!r}")

    release_real = release_dir.resolve()
    checked_images: set[Path] = set()
    external_images: set[Path] = set()
    records = 0
    try:
        with train_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    fail(f"train.jsonl line {line_number} is blank")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    fail(f"train.jsonl line {line_number} is invalid JSON: {exc}")
                if not isinstance(record, dict):
                    fail(f"train.jsonl line {line_number} must be a JSON object")
                records += 1
                for image_value in iter_image_paths(record.get("images"), line_number):
                    raw_path = Path(image_value).expanduser()
                    image_path = raw_path if raw_path.is_absolute() else release_dir / raw_path
                    image_path = image_path.resolve()
                    if not raw_path.is_absolute():
                        try:
                            image_path.relative_to(release_real)
                        except ValueError:
                            fail(
                                f"train.jsonl line {line_number}: relative image path "
                                f"escapes release directory: {image_value}"
                            )
                    else:
                        try:
                            image_path.relative_to(release_real)
                        except ValueError:
                            external_images.add(image_path)
                    if image_path not in checked_images:
                        if not image_path.is_file():
                            fail(f"train.jsonl line {line_number}: missing image " f"{image_path}")
                        checked_images.add(image_path)
    except OSError as exc:
        fail(f"cannot read train.jsonl: {exc}")

    if records == 0:
        fail("train.jsonl contains no records")
    if enforce_paper_rows and records != PAPER_TRAIN_ROWS:
        fail(
            f"release holds {records} examples but the paper reports {PAPER_TRAIN_ROWS}; "
            "pass --allow-nonpaper-counts to train on a different corpus"
        )
    return dataset_key, records, len(checked_images), len(external_images)


def yaml_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def build_config(args: argparse.Namespace, dataset_key: str) -> dict[str, Any]:
    profile = PROFILES[args.model_size]
    resume: str | None = None
    if args.resume == "auto":
        detected = latest_checkpoint(resolve_path(args.output_dir))
        resume = str(detected) if detected else None
        print(
            f"resume=auto resolved to {resume or '<no complete checkpoint; start fresh>'}",
            flush=True,
        )
    elif args.resume:
        checkpoint = resolve_path(args.resume)
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            fail(f"resume checkpoint is incomplete or missing: {checkpoint}")
        resume = str(checkpoint)

    return {
        "model_name_or_path": str(resolve_path(args.model_path)),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "deepspeed": str(
            resolve_path(args.llamafactory_dir) / "examples/deepspeed/ds_z3_config.json"
        ),
        "dataset": dataset_key,
        "dataset_dir": str(resolve_path(args.release_dir)),
        "template": profile.get("template", DEFAULT_TEMPLATE),
        "cutoff_len": 32768,
        "packing": False,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 4,
        "output_dir": str(resolve_path(args.output_dir)),
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",
        "logging_steps": 10,
        "save_steps": profile["save_steps"],
        "save_total_limit": 1,
        "plot_loss": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-6,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch_fused",
        "bf16": True,
        "gradient_checkpointing": True,
        "seed": 42,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": resume,
    }


def render_config(config: dict[str, Any]) -> str:
    return "".join(f"{key}: {yaml_scalar(value)}\n" for key, value in config.items())


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if path.is_dir() and suffix.isdigit() and (path / "trainer_state.json").is_file():
                candidates.append((int(suffix), path))
    return max(candidates, default=(0, None))[1]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-size", required=True, choices=sorted(PROFILES))
    parser.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="final stage-6 release directory holding train.jsonl and dataset_info.json",
    )
    parser.add_argument(
        "--model-path", help="defaults to the workspace base model for --model-size"
    )
    parser.add_argument(
        "--output-dir", help="defaults to training/outputs/<model-size>-<release name>"
    )
    parser.add_argument("--dataset-key")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help="resume latest checkpoint, or resume from the supplied checkpoint path",
    )
    parser.add_argument(
        "--allow-nonpaper-counts",
        action="store_true",
        help="accept a release whose example count differs from the paper",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for the worker and LLaMA-Factory",
    )
    parser.add_argument(
        "--llamafactory-dir",
        default=str(DEFAULT_LLAMAFACTORY_DIR),
    )
    parser.add_argument(
        "--extra-site-packages",
        default=str(DEFAULT_EXTRA_SITE),
        help="dependency site-packages prepended to PYTHONPATH; pass '' to disable",
    )
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument(
        "--config-dir",
        help=(
            "directory that keeps the rendered LLaMA-Factory config for each run; "
            "defaults to <llamafactory-dir>/configs/omniedu"
        ),
    )
    parser.add_argument(
        "--env-file",
        help="shared NAME=VALUE file loaded on every worker (for NCCL/GLOO settings)",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=GPUS_PER_NODE,
        help="accelerators per node, eight in the paper's configuration",
    )
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")


def validate_common_args(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    fill_path_defaults(args)
    if not (1 <= args.master_port <= 65535):
        fail("--master-port must be between 1 and 65535")
    if args.nproc_per_node < 1:
        fail("--nproc-per-node must be positive")
    python_path = resolve_path(args.python)
    if not python_path.is_file():
        fail(f"Python executable does not exist: {python_path}")
    llamafactory_dir = resolve_path(args.llamafactory_dir)
    if not (llamafactory_dir / "src/llamafactory").is_dir():
        fail(f"invalid LLaMA-Factory directory: {llamafactory_dir}")
    if args.extra_site_packages:
        extra_site = resolve_path(args.extra_site_packages)
        if not extra_site.is_dir():
            fail(f"extra site-packages directory does not exist: {extra_site}")
    if args.env_file:
        load_env_file(resolve_path(args.env_file))
    model_path = resolve_path(args.model_path)
    if not model_path.is_dir():
        fail(f"model path is not a directory: {model_path}")
    release_dir = resolve_path(args.release_dir)
    dataset_key, records, image_count, external_image_count = validate_release(
        release_dir, args.dataset_key, not args.allow_nonpaper_counts
    )
    config = build_config(args, dataset_key)
    deepspeed = Path(config["deepspeed"])
    if not deepspeed.is_file():
        fail(f"missing LLaMA-Factory ZeRO-3 config: {deepspeed}")
    print(
        f"validated release={release_dir} dataset={dataset_key} "
        f"records={records} unique_images={image_count} "
        f"external_absolute_images={external_image_count}",
        flush=True,
    )
    return dataset_key, config


def worker(args: argparse.Namespace) -> int:
    expected_nodes = PROFILES[args.model_size]["nodes"]
    if args.nnodes != expected_nodes:
        fail(f"{args.model_size} requires {expected_nodes} node(s), " f"but --nnodes={args.nnodes}")
    if not 0 <= args.node_rank < args.nnodes:
        fail("--node-rank must be in [0, nnodes)")

    _, config = validate_common_args(args)
    config_text = render_config(config)
    train_command = [
        args.python,
        "-m",
        "llamafactory.cli",
        "train",
        "<generated-config>",
    ]
    print("--- generated LLaMA-Factory config ---")
    print(config_text, end="")
    print("--- command ---")
    print(shlex.join(train_command))
    if args.dry_run:
        print("dry-run: no process started")
        return 0

    llamafactory_dir = resolve_path(args.llamafactory_dir)

    log_dir = resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir = resolve_path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    # One stable config file per model size and rank: the rendered settings are
    # reproducible from train.py, so re-running should overwrite rather than
    # pile up timestamped copies.  Per-run history stays in the log files.
    config_path = config_dir / f"{args.model_size}.node{args.node_rank}.yaml"
    log_path = log_dir / f"{run_id}.node{args.node_rank}.log"
    config_path.write_text(config_text, encoding="utf-8")
    train_command[-1] = str(config_path)

    env = os.environ.copy()
    if args.env_file:
        env.update(load_env_file(resolve_path(args.env_file)))
    # LLaMA-Factory shells out to `torchrun`, which must come from the same
    # interpreter that owns the training dependencies.
    python_path = resolve_path(args.python)
    env["PATH"] = os.pathsep.join([str(python_path.parent), env.get("PATH", "")]).strip(
        os.pathsep
    )
    env.update(
        {
            "FORCE_TORCHRUN": "1",
            "NNODES": str(args.nnodes),
            "NODE_RANK": str(args.node_rank),
            "NPROC_PER_NODE": str(args.nproc_per_node),
            "MASTER_ADDR": args.master_addr,
            "MASTER_PORT": str(args.master_port),
            "CUDA_VISIBLE_DEVICES": env.get(
                "CUDA_VISIBLE_DEVICES", ",".join(str(gpu) for gpu in range(args.nproc_per_node))
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )
    python_paths = []
    if args.extra_site_packages:
        python_paths.append(str(resolve_path(args.extra_site_packages)))
    python_paths.append(str(llamafactory_dir / "src"))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    print(f"config={config_path}", flush=True)
    print(f"log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{time.strftime('%F %T')}] command={shlex.join(train_command)}\n")
        log_handle.flush()
        completed = subprocess.run(
            train_command,
            cwd=llamafactory_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def common_cli_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--model-size",
        args.model_size,
        "--release-dir",
        args.release_dir,
        "--model-path",
        args.model_path,
        "--output-dir",
        args.output_dir,
        "--python",
        args.python,
        "--llamafactory-dir",
        args.llamafactory_dir,
        "--extra-site-packages",
        args.extra_site_packages,
        "--log-dir",
        args.log_dir,
        "--config-dir",
        args.config_dir,
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--master-port",
        str(args.master_port),
    ]
    if args.dataset_key:
        result += ["--dataset-key", args.dataset_key]
    if args.env_file:
        result += ["--env-file", args.env_file]
    if args.resume:
        result += ["--resume", args.resume]
    if args.allow_nonpaper_counts:
        result += ["--allow-nonpaper-counts"]
    return result


def parse_hosts(value: str | None) -> list[str]:
    if not value:
        return []
    hosts = [host.strip() for host in value.split(",") if host.strip()]
    if len(hosts) != len(set(hosts)):
        fail("--hosts contains duplicate entries")
    return hosts


def launcher(args: argparse.Namespace) -> int:
    expected_nodes = PROFILES[args.model_size]["nodes"]
    hosts = parse_hosts(args.hosts)
    if expected_nodes == 2 and len(hosts) != 2:
        fail("27b requires exactly two comma-separated --hosts, or two manual workers")
    if expected_nodes == 1 and len(hosts) > 1:
        fail("4b/9b accept at most one --hosts entry")

    _, config = validate_common_args(args)
    master_addr = args.master_addr or (hosts[0] if hosts else "127.0.0.1")
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}"
    base_worker = [
        args.python,
        str(Path(__file__).resolve()),
        "worker",
        *common_cli_args(args),
        "--nnodes",
        str(expected_nodes),
        "--master-addr",
        master_addr,
        "--run-id",
        run_id,
    ]

    commands: list[list[str]] = []
    for rank in range(expected_nodes):
        worker_command = [*base_worker, "--node-rank", str(rank)]
        if hosts:
            remote_command = "bash -lc " + shlex.quote(shlex.join(worker_command))
            commands.append(["ssh", *args.ssh_option, hosts[rank], remote_command])
        else:
            commands.append(worker_command)

    print("--- generated LLaMA-Factory config ---")
    print(render_config(config), end="")
    print("--- launcher commands ---")
    for command in commands:
        print(shlex.join(command))
    if args.dry_run:
        print("dry-run: no local, SSH, or training process started")
        return 0

    processes = [subprocess.Popen(command) for command in commands]
    remaining = set(range(len(processes)))
    failed_code = 0

    def stop_children(signum: int, _frame: Any) -> None:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    while remaining:
        for index in tuple(remaining):
            code = processes[index].poll()
            if code is None:
                continue
            remaining.remove(index)
            if code != 0 and failed_code == 0:
                failed_code = code
                for peer_index in remaining:
                    processes[peer_index].terminate()
        if remaining:
            time.sleep(1)
    return failed_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified education SFT worker and SSH launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker", help="run one local node")
    add_common_arguments(worker_parser)
    worker_parser.add_argument("--nnodes", type=int, default=1)
    worker_parser.add_argument("--node-rank", type=int, default=0)
    worker_parser.add_argument(
        "--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1")
    )
    worker_parser.add_argument("--run-id", help=argparse.SUPPRESS)
    worker_parser.set_defaults(handler=worker)

    launch_parser = subparsers.add_parser("launch", help="launch locally or over SSH")
    add_common_arguments(launch_parser)
    launch_parser.add_argument(
        "--hosts",
        help="comma-separated SSH hosts; required as host0,host1 for 27b",
    )
    launch_parser.add_argument(
        "--master-addr",
        help="distributed rendezvous address (defaults to the first host)",
    )
    launch_parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        metavar="OPTION",
        help="repeatable option passed verbatim to ssh, e.g. -oBatchMode=yes",
    )
    launch_parser.set_defaults(handler=launcher)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
