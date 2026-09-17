#!/usr/bin/env python3

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot matched Euler and cylinder solver response histories.")
    parser.add_argument("run_directory", help="Matrix directory created by run_euler_cylinder_response_matrix.sh.")
    parser.add_argument("output_directory", help="Directory for four comparison PNG files.")
    return parser.parse_args()


def load_history(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite response value in {path}")
    return data


def plot_case(run_directory, output_directory, case_name, filename, x_column, y_column, xlabel, ylabel):
    for preconditioner in PRECONDITIONERS:
        fig, axis = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
        histories_found = 0
        for solver, label in SOLVERS:
            path = run_directory / case_name / f"{solver}_{preconditioner}" / filename
            if not path.exists():
                print(f"Skipping missing history: {path}")
                continue
            history = load_history(path)
            axis.plot(
                history[:, x_column],
                history[:, y_column],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=label,
            )
            histories_found += 1

        if histories_found == 0:
            plt.close(fig)
            continue

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, linewidth=0.4, alpha=0.4)
        axis.legend(frameon=False, fontsize=8)
        axis.set_title(f"{case_name.capitalize()} — {preconditioner}")
        output_path = output_directory / f"{case_name}_solver_response_{preconditioner}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Wrote {output_path}")


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    plot_case(
        run_directory,
        output_directory,
        "euler",
        "paper_euler_fast_load_displacement.csv",
        1,
        4,
        "Average lateral top displacement",
        "Applied compressive force",
    )
    plot_case(
        run_directory,
        output_directory,
        "cylinder",
        "paper_cylinder_crush_fast_load_displacement.csv",
        1,
        3,
        "Average top displacement",
        "Top reaction",
    )


if __name__ == "__main__":
    main()
