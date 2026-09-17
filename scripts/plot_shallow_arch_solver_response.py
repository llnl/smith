#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOLVERS = (
    ("nls", "Newton-LS"),
    ("petsc_cp", "Newton-CP"),
    ("tr", "Trust-region"),
    ("tr_subspace", "Trust-region + subspace"),
)
PRECONDITIONERS = ("HypreAMG", "HypreJacobi")
HISTORY_NAME = "paper_shallow_arch_fast_load_displacement.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot shallow-arch solver response and timing comparisons.")
    parser.add_argument("run_directory", help="Directory created by run_shallow_arch_response_matrix.sh.")
    parser.add_argument("output_directory", help="Directory for comparison PNG files.")
    parser.add_argument(
        "--additional-run-directory",
        action="append",
        default=[],
        help="Additional matrix directory whose entries are merged into the plots.",
    )
    return parser.parse_args()


def load_history(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid response history in {path}")
    return data


def load_timing(path):
    match = re.search(r"02/shallow_arch\s+([0-9.]+) s\s+(?:ok|solver_failed)", path.read_text())
    if not match:
        raise ValueError(f"Missing shallow-arch timing in {path}")
    return float(match.group(1))


def load_status(path):
    with path.open(newline="") as stream:
        return {
            (row["solver"], row["preconditioner"]): row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def load_entries(run_directories):
    entries = {}
    for run_directory in run_directories:
        for key, row in load_status(run_directory / "status.tsv").items():
            if key in entries:
                raise ValueError(f"Duplicate shallow-arch entry {key} in {run_directory}")
            entries[key] = (row, run_directory)
    return entries


def plot_response(output_directory, preconditioner, entries):
    fig, axis = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    plotted = False
    for solver, label in SOLVERS:
        if (solver, preconditioner) not in entries:
            continue
        row, run_directory = entries[(solver, preconditioner)]
        history = load_history(run_directory / f"{solver}_{preconditioner}" / HISTORY_NAME)
        display_label = label if row["outcome"] == "completed" else f"{label} (partial)"
        line, = axis.plot(
            -history[:, 1],
            history[:, 2],
            marker="o",
            linewidth=1.4,
            markersize=3,
            label=display_label,
        )
        if row["outcome"] != "completed":
            axis.scatter(
                -history[-1, 1],
                history[-1, 2],
                color=line.get_color(),
                marker="X",
                s=70,
                zorder=5,
            )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axis.set_xlabel("Downward average top-boundary displacement magnitude")
    axis.set_ylabel("Downward applied-force magnitude")
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(f"Shallow arch — {preconditioner}")
    output_path = output_directory / f"shallow_arch_solver_response_{preconditioner}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_timings(output_directory, entries):
    selected_solvers = [
        (solver, label)
        for solver, label in SOLVERS
        if any((solver, preconditioner) in entries for preconditioner in PRECONDITIONERS)
    ]
    labels = [label for _, label in selected_solvers]
    positions = np.arange(len(selected_solvers))
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)

    for index, preconditioner in enumerate(PRECONDITIONERS):
        timings = []
        outcomes = []
        for solver, _ in selected_solvers:
            if (solver, preconditioner) not in entries:
                timings.append(np.nan)
                outcomes.append("")
                continue
            row, run_directory = entries[(solver, preconditioner)]
            timings.append(load_timing(run_directory / f"{solver}_{preconditioner}" / "run.log"))
            outcomes.append(row["outcome"])
        offsets = positions + (index - 0.5) * width
        bars = axis.bar(offsets, timings, width, label=preconditioner)
        bar_labels = [
            "" if not np.isfinite(timing) else f"{timing:.1f}{'†' if outcome != 'completed' else ''}"
            for timing, outcome in zip(timings, outcomes)
        ]
        axis.bar_label(bars, labels=bar_labels, padding=2, fontsize=7)

    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylabel("Max-rank wall time (s)")
    axis.set_yscale("log")
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    axis.legend(frameon=False)
    axis.set_title("† nonlinear failure before all increments completed", fontsize=9)
    output_path = output_directory / "shallow_arch_solver_timing.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_directories = [run_directory, *(Path(path) for path in args.additional_run_directory)]
    entries = load_entries(run_directories)

    for preconditioner in PRECONDITIONERS:
        plot_response(output_directory, preconditioner, entries)
    plot_timings(output_directory, entries)


if __name__ == "__main__":
    main()
