#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import numpy as np


SOLVERS = ("nls", "petsc_cp", "tr", "tr_subspace")
PRECONDITIONERS = ("HypreAMG", "HypreJacobi")
HISTORY_NAME = "paper_sphere_corner_fast_load_displacement.csv"
NUM_TIME_STEPS = 32
TIME_STEP = 2.0 / NUM_TIME_STEPS


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a sphere-into-corner solver response matrix.")
    parser.add_argument("run_directory", help="Directory created by run_sphere_into_corner_response_matrix.sh.")
    parser.add_argument("--render-directory", default="", help="Optional directory containing rendered PNGs.")
    parser.add_argument("--output", default="", help="Output TSV path. Defaults to RUN_DIRECTORY/validation.tsv.")
    return parser.parse_args()


def load_status(path):
    with path.open(newline="") as stream:
        return {
            f'{row["solver"]}_{row["preconditioner"]}': row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def load_history(path):
    if path.stat().st_size == 0:
        return np.empty((0, 3))
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 3:
        raise ValueError(f"Expected three response columns in {path}, found {data.shape[1]}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite response value in {path}")
    expected_time = np.arange(len(data)) * TIME_STEP
    if not np.allclose(data[:, 0], expected_time, rtol=0.0, atol=5.0e-7):
        raise ValueError(f"Unexpected time grid in {path}")
    if np.any(np.diff(data[:, 1]) > 5.0e-7):
        raise ValueError(f"Signed drive displacement is not monotonically decreasing in {path}")
    if np.any(data[:, 2] < -1.0e-12) or np.any(np.diff(data[:, 2]) < -5.0e-7):
        raise ValueError(f"Load resultant is negative or nonmonotone in {path}")
    return data


def solver_iterations(log_text, solver):
    if solver == "petsc_cp":
        values = [int(value) for value in re.findall(r"PETSc SNES summary: iterations=(\d+)", log_text)]
    else:
        prefix = "TrustRegion" if solver.startswith("tr") else "Newton"
        values = [int(value) for value in re.findall(rf"^{prefix} iteration\s+(\d+)\s+:", log_text, re.MULTILINE)]
        values = [value for value in values if value > 0]
    return sum(values), max(values, default=0)


def solver_wall_seconds(log_text, fallback):
    match = re.search(r"07/sphere_into_corner\s+([0-9.]+) s\s+(?:ok|solver_failed)", log_text)
    return float(match.group(1)) if match else float(fallback)


def failure_step(log_text):
    match = re.search(r"nonlinear solve failed at step (\d+)", log_text)
    return match.group(1) if match else ""


def visualization_cycles(run_directory):
    cycle_root = run_directory / "paper_sphere_corner_fast" / "paper_sphere_corner_fast"
    return len(list(cycle_root.glob("Cycle*/data.pvtu")))


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_path = Path(args.output) if args.output else run_directory / "validation.tsv"
    render_directory = Path(args.render_directory) if args.render_directory else None
    status = load_status(run_directory / "status.tsv")

    fieldnames = (
        "solver",
        "preconditioner",
        "outcome",
        "solver_wall_seconds",
        "response_states",
        "accepted_increments",
        "visualization_cycles",
        "rendered_images",
        "nonlinear_iterations",
        "maximum_iterations_per_increment",
        "indefinite_cg_diagnostics",
        "inner_cg_cap_hits",
        "final_time",
        "final_signed_drive_displacement",
        "final_load_resultant",
        "failed_step",
    )

    rows = []
    for solver in SOLVERS:
        for preconditioner in PRECONDITIONERS:
            entry = f"{solver}_{preconditioner}"
            if entry not in status:
                continue
            status_row = status[entry]
            entry_directory = run_directory / entry
            history = load_history(entry_directory / HISTORY_NAME)
            log_text = (entry_directory / "run.log").read_text(errors="replace")
            nonlinear_iterations, maximum_iterations = solver_iterations(log_text, solver)
            rendered_images = len(list((render_directory / entry).glob("*.png"))) if render_directory else 0
            cycle_count = visualization_cycles(entry_directory)
            if len(history) > 0 and cycle_count != len(history):
                raise ValueError(f"Response/field count mismatch for {entry}: {len(history)} states, {cycle_count} cycles")
            if len(history) == 0 and cycle_count > 1:
                raise ValueError(f"Unexpected fields without response states for {entry}: {cycle_count} cycles")
            expected_states = NUM_TIME_STEPS + 1
            if status_row["outcome"] == "completed" and (
                len(history) != expected_states or cycle_count != expected_states
            ):
                raise ValueError(
                    f"Completed entry {entry} does not contain all {expected_states} response states and field cycles"
                )
            final = history[-1] if len(history) > 0 else ("", "", "")
            rows.append(
                {
                    "solver": solver,
                    "preconditioner": preconditioner,
                    "outcome": status_row["outcome"],
                    "solver_wall_seconds": f'{solver_wall_seconds(log_text, status_row["elapsed_seconds"]):.3f}',
                    "response_states": len(history),
                    "accepted_increments": max(len(history) - 1, 0),
                    "visualization_cycles": cycle_count,
                    "rendered_images": rendered_images,
                    "nonlinear_iterations": nonlinear_iterations,
                    "maximum_iterations_per_increment": maximum_iterations,
                    "indefinite_cg_diagnostics": log_text.count("operator is not positive definite"),
                    "inner_cg_cap_hits": len(
                        re.findall(r"PCG: Number of iterations: (?:600|60000)$", log_text, re.MULTILINE)
                    ),
                    "final_time": f"{final[0]:.8g}" if len(history) > 0 else "",
                    "final_signed_drive_displacement": f"{final[1]:.8g}" if len(history) > 0 else "",
                    "final_load_resultant": f"{final[2]:.8g}" if len(history) > 0 else "",
                    "failed_step": failure_step(log_text),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
