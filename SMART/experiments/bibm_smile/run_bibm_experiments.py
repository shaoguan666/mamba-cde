"""Run the SMILE BIBM experiment grid through SMART/run_all_experiments.py.

This script is intentionally a thin wrapper: it keeps the experiment grid in
configs/bibm_smile_experiments.json and delegates actual training to the
repository runner so that dataset loading, checkpoints, and evaluation remain
identical across variants.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SMART_DIR = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs" / "bibm_smile_experiments.json"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Display names from config, e.g. Backbone SMILE-Full.")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--use-torchrun", action="store_true")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--finetune-only", action="store_true")
    parser.add_argument("--pretrain-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant_map = cfg["implemented_variants"]
    selected_names = args.models or list(variant_map.keys())
    missing = [name for name in selected_names if name not in variant_map]
    if missing:
        raise SystemExit(f"Unknown or not implemented variant names: {missing}")

    models = [variant_map[name] for name in selected_names]
    datasets = args.datasets or cfg["datasets"]
    seeds = args.seeds or cfg["seeds"]
    defaults = cfg["runner_defaults"]

    cmd = [
        args.python_executable,
        "run_all_experiments.py",
        "--python-executable",
        args.python_executable,
        "--models",
        *models,
        "--datasets",
        *datasets,
        "--seeds",
        *[str(s) for s in seeds],
        "--pretrain-epochs",
        str(defaults["pretrain_epochs"]),
        "--finetune-epochs",
        str(defaults["finetune_epochs"]),
        "--batch-size",
        str(defaults["batch_size"]),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.use_torchrun:
        cmd.extend(["--use-torchrun", "--nproc-per-node", str(args.nproc_per_node)])
    if args.devices:
        cmd.extend(["--devices", args.devices])
    if args.force:
        cmd.append("--force")
    if args.finetune_only:
        cmd.append("--finetune-only")
    if args.pretrain_only:
        cmd.append("--pretrain-only")

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=SMART_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
