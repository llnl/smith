#!/usr/bin/env python3
"""Run one contact experiment and append its structured summary to a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one contact executable invocation and append its run_summary.json to a summary CSV."
    )
    parser.add_argument("--name", default="", help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument(
        "--exe",
        default="./build-llvm19-noomp-enzyme-release/examples/contact_ironing_2D",
        help="Executable to run.",
    )
    parser.add_argument("--output-root", default="experiments", help="Directory where experiment folders are written.")
    parser.add_argument("--summary-csv", default="", help="CSV to append. Defaults to OUTPUT_ROOT/summary.csv.")
    parser.add_argument(
        "experiment_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the executable. Put them after --.",
    )
    args = parser.parse_args()
    if args.experiment_args and args.experiment_args[0] == "--":
        args.experiment_args = args.experiment_args[1:]
    return args


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            flatten(f"{prefix}{key}.", nested_value, out)
    elif isinstance(value, list):
        out[prefix[:-1]] = json.dumps(value, separators=(",", ":"))
    else:
        out[prefix[:-1]] = value


def append_csv(path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    rows.append({key: row.get(key, "") for key in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    name = args.name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary.csv"
    run_summary = run_dir / "run_summary.json"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    command_path = run_dir / "command.txt"

    command = [args.exe, *args.experiment_args]
    if "--run-summary-json" not in args.experiment_args:
        command.extend(["--run-summary-json", str(run_summary)])

    command_path.write_text(shell_join(command) + "\n")
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)

    row: dict[str, Any] = {
        "name": name,
        "return_code": completed.returncode,
        "command": shell_join(command),
        "run_dir": str(run_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "run_summary_json": str(run_summary),
    }

    if run_summary.exists():
        with run_summary.open() as stream:
            summary = json.load(stream)
        flatten("", {key: value for key, value in summary.items() if key != "cycles"}, row)
    else:
        row["summary.converged"] = False
        row["summary.missing_run_summary"] = True

    append_csv(summary_csv, row)
    print(f"Wrote {run_dir}")
    print(f"Appended {summary_csv}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
