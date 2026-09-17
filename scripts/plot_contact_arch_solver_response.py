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
HISTORY_NAME = "paper_contact_arch_fast_load_displacement.csv"
OUTCOME_STYLE = {
    "completed": {"hatch": "", "alpha": 1.0},
    "failed": {"hatch": "//", "alpha": 0.8},
    "timeout": {"hatch": "xx", "alpha": 0.65},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot contact-arch solver response and timing comparisons.")
    parser.add_argument("run_directory", help="Directory created by run_contact_arch_response_matrix.sh.")
    parser.add_argument("output_directory", help="Directory for comparison PNG files.")
    return parser.parse_args()


def load_history(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 5 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid response history in {path}")
    return data


def load_status(path):
    with path.open(newline="") as stream:
        return {
            (row["solver"], row["preconditioner"]): row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def load_solver_time(path, fallback):
    match = re.search(r"06/contact_arch\s+([0-9.]+) s\s+(?:ok|solver_failed)", path.read_text())
    return float(match.group(1)) if match else float(fallback)


def plot_response(run_directory, output_directory, preconditioner, status):
    fig, axis = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    histories = {}
    for solver, label in SOLVERS:
        history = load_history(run_directory / f"{solver}_{preconditioner}" / HISTORY_NAME)
        histories[solver] = (label, history)

    reference_solver = max(histories, key=lambda solver: len(histories[solver][1]))
    reference_label, reference_history = histories[reference_solver]
    axis.plot(
        reference_history[:, 2],
        reference_history[:, 4],
        color="black",
        linewidth=1.6,
        label=f"Common accepted branch ({reference_label})",
    )

    for solver, label in SOLVERS:
        history = histories[solver][1]
        outcome = status[(solver, preconditioner)]["outcome"]
        axis.scatter(
            history[-1, 2],
            history[-1, 4],
            s=34,
            label=f"{label} ({outcome}, {len(history) - 1}/50)",
        )

    axis.set_xlabel("Support inset")
    axis.set_ylabel("Support reaction")
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.legend(frameon=False, fontsize=7)
    axis.set_title(f"Contact arch — {preconditioner}")
    output_path = output_directory / f"contact_arch_solver_response_{preconditioner}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_timings(run_directory, output_directory, status):
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
                    run_directory / f"{solver}_{preconditioner}" / "run.log",
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
    output_path = output_directory / "contact_arch_solver_timing.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    status = load_status(run_directory / "status.tsv")

    for preconditioner in PRECONDITIONERS:
        plot_response(run_directory, output_directory, preconditioner, status)
    plot_timings(run_directory, output_directory, status)


if __name__ == "__main__":
    main()
