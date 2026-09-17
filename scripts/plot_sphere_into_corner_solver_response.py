#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SOLVERS = (
    ("nls", "Newton-LS"),
    ("petsc_cp", "Newton-CP"),
    ("tr", "Trust-region"),
    ("tr_subspace", "Trust-region + subspace"),
)
PRECONDITIONERS = ("HypreAMG", "HypreJacobi")
HISTORY_NAME = "paper_sphere_corner_fast_load_displacement.csv"
OUTCOME_STYLE = {
    "completed": {"hatch": "", "alpha": 1.0},
    "failed": {"hatch": "//", "alpha": 0.8},
    "timeout": {"hatch": "xx", "alpha": 0.65},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot sphere-into-corner response and timing comparisons.")
    parser.add_argument("run_directory", help="Directory created by run_sphere_into_corner_response_matrix.sh.")
    parser.add_argument("output_directory", help="Directory for comparison PNG files.")
    parser.add_argument(
        "--supplemental-run-directory",
        action="append",
        default=[],
        help="Directory whose entries override matching entries in the primary matrix.",
    )
    return parser.parse_args()


def load_history(path):
    if not any(line.strip() and not line.lstrip().startswith("#") for line in path.read_text().splitlines()):
        return np.empty((0, 3))
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid response history in {path}")
    return data


def load_status(path):
    with path.open(newline="") as stream:
        return {
            (row["solver"], row["preconditioner"]): row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def load_time_steps(path):
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "time_steps":
            return int(value)
    raise ValueError(f"Missing time_steps in {path}")


def load_solver_time(path, fallback):
    match = re.search(r"07/sphere_into_corner\s+([0-9.]+) s\s+(?:ok|solver_failed)", path.read_text())
    return float(match.group(1)) if match else float(fallback)


def plot_response(entry_directories, output_directory, preconditioner, status, time_steps):
    fig, axis = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    plotted = False
    for solver, label in SOLVERS:
        entry = (solver, preconditioner)
        if entry not in status:
            continue
        history = load_history(entry_directories[entry] / HISTORY_NAME)
        if len(history) == 0:
            continue
        outcome = status[entry]["outcome"]
        axis.plot(
            -history[:, 1],
            history[:, 2],
            marker="o",
            linewidth=1.4,
            markersize=3,
            label=f"{label} ({outcome}, {len(history) - 1}/{time_steps})",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axis.set_xlabel("Load-patch drive displacement magnitude")
    axis.set_ylabel("Applied load resultant")
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.legend(frameon=False, fontsize=7)
    axis.set_title(f"Sphere into corner — {preconditioner}")
    output_path = output_directory / f"sphere_into_corner_solver_response_{preconditioner}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_timings(entry_directories, output_directory, status):
    labels = [label for _, label in SOLVERS]
    positions = np.arange(len(SOLVERS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)

    for index, preconditioner in enumerate(PRECONDITIONERS):
        timings = []
        outcomes = []
        for solver, _ in SOLVERS:
            row = status[(solver, preconditioner)]
            timings.append(
                load_solver_time(
                    entry_directories[(solver, preconditioner)] / "run.log",
                    row["elapsed_seconds"],
                )
            )
            outcomes.append(row["outcome"])

        offsets = positions + (index - 0.5) * width
        bars = axis.bar(offsets, timings, width, label=preconditioner)
        for bar, outcome in zip(bars, outcomes):
            style = OUTCOME_STYLE[outcome]
            bar.set_hatch(style["hatch"])
            bar.set_alpha(style["alpha"])
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

    outcome_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=style["hatch"], label=outcome)
        for outcome, style in OUTCOME_STYLE.items()
    ]
    preconditioner_legend = axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.add_artist(preconditioner_legend)
    axis.legend(handles=outcome_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 0.76), fontsize=8)
    axis.set_xticks(positions, labels, rotation=12, ha="right")
    axis.set_ylabel("Max-rank wall time or timeout (s)")
    axis.set_yscale("log")
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    output_path = output_directory / "sphere_into_corner_solver_timing.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    status = load_status(run_directory / "status.tsv")
    time_steps = load_time_steps(run_directory / "effective_settings.txt")
    entry_directories = {
        entry: run_directory / f"{entry[0]}_{entry[1]}"
        for entry in status
    }
    for supplemental_name in args.supplemental_run_directory:
        supplemental_directory = Path(supplemental_name)
        supplemental_status = load_status(supplemental_directory / "status.tsv")
        status.update(supplemental_status)
        entry_directories.update(
            {
                entry: supplemental_directory / f"{entry[0]}_{entry[1]}"
                for entry in supplemental_status
            }
        )

    for preconditioner in PRECONDITIONERS:
        plot_response(entry_directories, output_directory, preconditioner, status, time_steps)
    if len(status) == len(SOLVERS) * len(PRECONDITIONERS):
        plot_timings(entry_directories, output_directory, status)


if __name__ == "__main__":
    main()
