#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path


HISTORY_NAME = "paper_euler_fast_load_displacement.csv"
EXPECTED_RESPONSE_STATES = 36
EXPECTED_GLOBAL_ELEMENTS = 11200
EXPECTED_PRECRITICAL_FORCE = 2.74e-3
EXPECTED_FINAL_FORCE = 2.75e-3
LATERAL_SMALL_THRESHOLD = 1.0e-2
LATERAL_LARGE_THRESHOLD = 1.0e-1


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the refined Euler analytic-verification run.")
    parser.add_argument("run_directory", help="Directory containing the run log and response CSV.")
    parser.add_argument("--outcome", choices=("completed", "timeout", "failed"), required=True)
    parser.add_argument("--output", default="", help="Output TSV path. Defaults to RUN_DIRECTORY/validation.tsv.")
    return parser.parse_args()


def load_history(path):
    if not path.is_file():
        return []
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) != 6:
                raise ValueError(f"Expected 6 columns at {path}:{line_number}, found {len(values)}")
            row = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"Nonfinite response value at {path}:{line_number}")
            rows.append(row)
    return rows


def closest_force_row(history, target):
    return min(history, key=lambda row: abs(row[4] - target))


def parse_global_elements(log_text):
    match = re.search(r"euler: global elements = (\d+)", log_text)
    return int(match.group(1)) if match else 0


def parse_solver_wall_seconds(log_text):
    match = re.search(r"paper_euler_fast wall = ([0-9.]+) s", log_text)
    return float(match.group(1)) if match else math.nan


def yes_no(value):
    return "yes" if value else "no"


def format_value(value):
    return "" if not math.isfinite(value) else f"{value:.12g}"


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_path = Path(args.output) if args.output else run_directory / "validation.tsv"
    history = load_history(run_directory / HISTORY_NAME)
    log_path = run_directory / "run.log"
    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""

    youngs_modulus = 10.0
    weak_axis_moment = 0.1
    length = 30.0
    analytic_load = math.pi**2 * youngs_modulus * weak_axis_moment / (2.0 * length) ** 2

    global_elements = parse_global_elements(log_text)
    solver_wall_seconds = parse_solver_wall_seconds(log_text)
    monotone_force = bool(history) and all(
        right[4] >= left[4] - 1.0e-12 for left, right in zip(history, history[1:])
    )
    maximum_force_balance_error = max((abs(row[5] - row[4]) for row in history), default=math.nan)

    precritical = closest_force_row(history, EXPECTED_PRECRITICAL_FORCE) if history else None
    final = history[-1] if history else None
    precritical_force = precritical[4] if precritical else math.nan
    precritical_lateral_displacement = precritical[1] if precritical else math.nan
    final_force = final[4] if final else math.nan
    final_lateral_displacement = final[1] if final else math.nan
    final_axial_displacement = final[3] if final else math.nan
    initial_force = history[0][4] if history else math.nan
    final_time = final[0] if final else math.nan
    bracket_contains_analytic_load = (
        precritical is not None and final is not None and precritical_force < analytic_load < final_force
    )
    branch_transition_observed = (
        precritical is not None
        and final is not None
        and abs(precritical_lateral_displacement) < LATERAL_SMALL_THRESHOLD
        and abs(final_lateral_displacement) > LATERAL_LARGE_THRESHOLD
    )

    checks = []
    if args.outcome == "completed":
        checks.extend(
            (
                (len(history) == EXPECTED_RESPONSE_STATES, f"response states: {len(history)}"),
                (global_elements == EXPECTED_GLOBAL_ELEMENTS, f"global elements: {global_elements}"),
                (monotone_force, "applied force is not monotone"),
                (
                    math.isclose(initial_force, 0.0, rel_tol=0.0, abs_tol=1.0e-14),
                    f"initial force: {initial_force}",
                ),
                (
                    math.isclose(final_time, 1.0, rel_tol=0.0, abs_tol=5.0e-7),
                    f"final time: {final_time}",
                ),
                (
                    math.isclose(precritical_force, EXPECTED_PRECRITICAL_FORCE, rel_tol=0.0, abs_tol=5.0e-9),
                    f"precritical force: {precritical_force}",
                ),
                (
                    math.isclose(final_force, EXPECTED_FINAL_FORCE, rel_tol=0.0, abs_tol=5.0e-9),
                    f"final force: {final_force}",
                ),
                (bracket_contains_analytic_load, "analytic load is not inside the final force interval"),
                (branch_transition_observed, "lateral-displacement transition was not resolved"),
                (
                    math.isfinite(maximum_force_balance_error) and maximum_force_balance_error <= 1.0e-8,
                    f"maximum force-balance error: {maximum_force_balance_error}",
                ),
            )
        )

    failed_checks = [message for passed, message in checks if not passed]
    if args.outcome == "completed":
        validation_state = "passed" if not failed_checks else "failed"
    else:
        validation_state = "partial"

    fieldnames = (
        "run_outcome",
        "validation_state",
        "response_states",
        "global_elements",
        "solver_wall_seconds",
        "monotone_applied_force",
        "maximum_force_balance_error",
        "analytic_euler_load",
        "initial_force",
        "final_time",
        "precritical_force",
        "precritical_lateral_displacement",
        "final_force",
        "final_lateral_displacement",
        "final_axial_displacement",
        "bracket_contains_analytic_load",
        "branch_transition_observed",
        "failed_checks",
    )
    row = {
        "run_outcome": args.outcome,
        "validation_state": validation_state,
        "response_states": len(history),
        "global_elements": global_elements,
        "solver_wall_seconds": format_value(solver_wall_seconds),
        "monotone_applied_force": yes_no(monotone_force),
        "maximum_force_balance_error": format_value(maximum_force_balance_error),
        "analytic_euler_load": format_value(analytic_load),
        "initial_force": format_value(initial_force),
        "final_time": format_value(final_time),
        "precritical_force": format_value(precritical_force),
        "precritical_lateral_displacement": format_value(precritical_lateral_displacement),
        "final_force": format_value(final_force),
        "final_lateral_displacement": format_value(final_lateral_displacement),
        "final_axial_displacement": format_value(final_axial_displacement),
        "bracket_contains_analytic_load": yes_no(bracket_contains_analytic_load),
        "branch_transition_observed": yes_no(branch_transition_observed),
        "failed_checks": "; ".join(failed_checks),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {output_path}")

    if failed_checks:
        raise SystemExit("Validation failed: " + "; ".join(failed_checks))


if __name__ == "__main__":
    main()
