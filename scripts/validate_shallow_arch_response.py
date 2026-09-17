#!/usr/bin/env python3

import argparse
import csv
import math
import re
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SOLVERS = ("nls", "petsc_cp", "tr", "tr_subspace")
PRECONDITIONERS = ("HypreAMG", "HypreJacobi")
HISTORY_NAME = "paper_shallow_arch_fast_load_displacement.csv"
OUTPUT_NAME = "paper_shallow_arch_fast"
EXPECTED_STATES = 211
EXPECTED_ELEMENTS = 12288
EXPECTED_RANKS = 16
TIME_STEP = 1.0 / 210.0
PRECOMPRESSION_END_TIME = 10.0 / 210.0
EXPECTED_FINAL_FORCE = 0.182103084
REQUIRED_FIELDS = {
    "mesh_shape_displacement",
    "mesh_shape_displacement_dual",
    "paper_shallow_arch_fast_displacement",
    "paper_shallow_arch_fast_reactions",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a unified shallow-arch response matrix.")
    parser.add_argument("run_directory", help="Matrix directory containing status.tsv and entry directories.")
    parser.add_argument("--output", default="", help="Output TSV path. Defaults to RUN_DIRECTORY/validation.tsv.")
    parser.add_argument("--require-all", action="store_true", help="Require all eight solver/preconditioner entries.")
    return parser.parse_args()


def load_status(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {f'{row["solver"]}_{row["preconditioner"]}': row for row in rows}


def load_history(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(f"Expected four response columns in {path}, found {data.shape[1]}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Nonfinite response value in {path}")
    return data


def parse_command(path):
    return set(shlex.split(path.read_text()))


def parse_global_elements(log_text):
    match = re.search(r"shallow arch: global elements = (\d+)", log_text)
    return int(match.group(1)) if match else 0


def parse_rank_count(log_text):
    match = re.search(r"MPI Rank Count: (\d+)", log_text)
    return int(match.group(1)) if match else 0


def solver_iterations(log_text, solver):
    if solver == "petsc_cp":
        values = [int(value) for value in re.findall(r"PETSc SNES summary: iterations=(\d+)", log_text)]
    else:
        prefix = "TrustRegion" if solver.startswith("tr") else "Newton"
        values = [int(value) for value in re.findall(rf"^{prefix} iteration\s+(\d+)\s+:", log_text, re.MULTILINE)]
        values = [value for value in values if value > 0]
    return sum(values), max(values, default=0)


def solver_wall_seconds(log_text, fallback):
    match = re.search(r"02/shallow_arch\s+([0-9.]+) s\s+(?:ok|solver_failed)", log_text)
    return float(match.group(1)) if match else float(fallback)


def field_summary(entry_directory):
    cycle_root = entry_directory / OUTPUT_NAME / OUTPUT_NAME
    cycle_directories = sorted(cycle_root.glob("Cycle*"))
    missing_fields = set()
    minimum_piece_count = EXPECTED_RANKS if cycle_directories else 0
    for cycle_directory in cycle_directories:
        pvtu_path = cycle_directory / "data.pvtu"
        if not pvtu_path.is_file():
            missing_fields.add("data.pvtu")
            minimum_piece_count = 0
            continue
        root = ET.parse(pvtu_path).getroot()
        names = {element.attrib["Name"] for element in root.iter() if "Name" in element.attrib}
        missing_fields.update(REQUIRED_FIELDS - names)
        minimum_piece_count = min(minimum_piece_count, len(list(cycle_directory.glob("proc*.vtu"))))
    return len(cycle_directories), minimum_piece_count, ",".join(sorted(missing_fields))


def yes_no(value):
    return "yes" if value else "no"


def format_value(value):
    return f"{value:.12g}" if math.isfinite(value) else ""


def main():
    args = parse_args()
    run_directory = Path(args.run_directory)
    output_path = Path(args.output) if args.output else run_directory / "validation.tsv"
    status = load_status(run_directory / "status.tsv")
    expected_entries = {f"{solver}_{preconditioner}" for solver in SOLVERS for preconditioner in PRECONDITIONERS}
    if args.require_all and set(status) != expected_entries:
        missing = sorted(expected_entries - set(status))
        extra = sorted(set(status) - expected_entries)
        raise ValueError(f"Matrix entry mismatch; missing={missing}, extra={extra}")

    histories = {}
    for entry in status:
        history_path = run_directory / entry / HISTORY_NAME
        if history_path.is_file():
            histories[entry] = load_history(history_path)
    reference = histories.get("tr_HypreAMG")

    fieldnames = (
        "solver",
        "preconditioner",
        "outcome",
        "validation_state",
        "solver_wall_seconds",
        "response_states",
        "field_cycles",
        "minimum_vtu_pieces_per_cycle",
        "missing_required_fields",
        "field_output",
        "global_elements",
        "mpi_ranks",
        "nonlinear_iterations",
        "maximum_iterations_per_step",
        "no_bsr_spmv",
        "no_assembled_bsr",
        "warm_start",
        "nonlinear_tolerance_1e_11",
        "linear_tolerance_1e_14",
        "maximum_time_error",
        "maximum_applied_force_error",
        "maximum_force_balance_error",
        "maximum_displacement_increment",
        "maximum_displacement_difference_vs_tr_amg",
        "maximum_reaction_difference_vs_tr_amg",
        "final_time",
        "final_displacement",
        "final_applied_force",
        "final_reaction",
        "nonlinear_failure_recorded",
        "failed_checks",
    )

    rows = []
    any_failed = False
    for solver in SOLVERS:
        for preconditioner in PRECONDITIONERS:
            entry = f"{solver}_{preconditioner}"
            if entry not in status:
                continue
            status_row = status[entry]
            outcome = status_row["outcome"]
            entry_directory = run_directory / entry
            history = histories.get(entry, np.empty((0, 4)))
            log_path = entry_directory / "run.log"
            log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            command_path = entry_directory / "command.txt"
            command = parse_command(command_path) if command_path.is_file() else set()
            field_cycles, minimum_piece_count, missing_fields = field_summary(entry_directory)
            field_output = "--paraview" in command
            nonlinear_iterations, maximum_iterations = solver_iterations(log_text, solver)

            if len(history):
                expected_times = np.arange(len(history)) * TIME_STEP
                load_scale = np.clip(
                    (history[:, 0] - PRECOMPRESSION_END_TIME) / (1.0 - PRECOMPRESSION_END_TIME), 0.0, 1.0
                )
                expected_forces = load_scale * EXPECTED_FINAL_FORCE
                maximum_time_error = float(np.max(np.abs(history[:, 0] - expected_times)))
                maximum_applied_force_error = float(np.max(np.abs(history[:, 2] - expected_forces)))
                maximum_force_balance_error = float(np.max(np.abs(history[:, 3] - history[:, 2])))
                maximum_displacement_increment = float(np.max(np.abs(np.diff(history[:, 1])))) if len(history) > 1 else 0.0
                final_time, final_displacement, final_force, final_reaction = history[-1]
            else:
                maximum_time_error = math.nan
                maximum_applied_force_error = math.nan
                maximum_force_balance_error = math.nan
                maximum_displacement_increment = math.nan
                final_time = final_displacement = final_force = final_reaction = math.nan

            if reference is not None and len(history):
                common_states = min(len(history), len(reference))
                maximum_displacement_difference = float(
                    np.max(np.abs(history[:common_states, 1] - reference[:common_states, 1]))
                )
                maximum_reaction_difference = float(
                    np.max(np.abs(history[:common_states, 3] - reference[:common_states, 3]))
                )
            else:
                maximum_displacement_difference = math.nan
                maximum_reaction_difference = math.nan

            nonlinear_failure_recorded = "nonlinear solve failed" in log_text
            field_checks = (
                [
                    (field_cycles == len(history), f"field cycles={field_cycles}, response states={len(history)}"),
                    (minimum_piece_count == EXPECTED_RANKS, f"minimum VTU pieces={minimum_piece_count}"),
                    (not missing_fields, f"missing fields={missing_fields}"),
                ]
                if field_output
                else [(field_cycles == 0, f"unexpected field cycles={field_cycles}")]
            )
            common_checks = [
                (len(history) >= 1, f"response states={len(history)}"),
                (parse_global_elements(log_text) == EXPECTED_ELEMENTS, "unexpected global element count"),
                (parse_rank_count(log_text) == EXPECTED_RANKS, "unexpected MPI rank count"),
                ("--no-use-bsr-spmv" in command, "missing --no-use-bsr-spmv"),
                ("--no-assemble-bsr" in command, "missing --no-assemble-bsr"),
                ("--use-warm-start" in command, "missing --use-warm-start"),
                ("--nonlinear-tol=1e-11" in command, "missing --nonlinear-tol=1e-11"),
                ("--linear-tol=1e-14" in command, "missing --linear-tol=1e-14"),
                (maximum_time_error <= 1.0e-6, f"maximum time error={maximum_time_error}"),
                (maximum_applied_force_error <= 1.0e-6, f"maximum applied-force error={maximum_applied_force_error}"),
                (maximum_force_balance_error <= 1.0e-6, f"maximum force-balance error={maximum_force_balance_error}"),
            ]
            if outcome == "completed":
                outcome_checks = [
                    (len(history) == EXPECTED_STATES, f"completed response states={len(history)}"),
                    (maximum_displacement_increment > 0.1, f"maximum displacement increment={maximum_displacement_increment}"),
                ]
            elif outcome == "nonlinear_failure":
                outcome_checks = [
                    (len(history) < EXPECTED_STATES, f"failure response states={len(history)}"),
                    (nonlinear_failure_recorded, "nonlinear failure marker missing"),
                ]
            else:
                outcome_checks = [(False, f"unaccepted process outcome={outcome}")]
            failed_checks = [message for passed, message in field_checks + common_checks + outcome_checks if not passed]
            validation_state = "passed" if not failed_checks else "failed"
            any_failed = any_failed or bool(failed_checks)

            rows.append(
                {
                    "solver": solver,
                    "preconditioner": preconditioner,
                    "outcome": outcome,
                    "validation_state": validation_state,
                    "solver_wall_seconds": f'{solver_wall_seconds(log_text, status_row["elapsed_seconds"]):.3f}',
                    "response_states": len(history),
                    "field_cycles": field_cycles,
                    "minimum_vtu_pieces_per_cycle": minimum_piece_count,
                    "missing_required_fields": missing_fields,
                    "field_output": yes_no(field_output),
                    "global_elements": parse_global_elements(log_text),
                    "mpi_ranks": parse_rank_count(log_text),
                    "nonlinear_iterations": nonlinear_iterations,
                    "maximum_iterations_per_step": maximum_iterations,
                    "no_bsr_spmv": yes_no("--no-use-bsr-spmv" in command),
                    "no_assembled_bsr": yes_no("--no-assemble-bsr" in command),
                    "warm_start": yes_no("--use-warm-start" in command),
                    "nonlinear_tolerance_1e_11": yes_no("--nonlinear-tol=1e-11" in command),
                    "linear_tolerance_1e_14": yes_no("--linear-tol=1e-14" in command),
                    "maximum_time_error": format_value(maximum_time_error),
                    "maximum_applied_force_error": format_value(maximum_applied_force_error),
                    "maximum_force_balance_error": format_value(maximum_force_balance_error),
                    "maximum_displacement_increment": format_value(maximum_displacement_increment),
                    "maximum_displacement_difference_vs_tr_amg": format_value(maximum_displacement_difference),
                    "maximum_reaction_difference_vs_tr_amg": format_value(maximum_reaction_difference),
                    "final_time": format_value(final_time),
                    "final_displacement": format_value(final_displacement),
                    "final_applied_force": format_value(final_force),
                    "final_reaction": format_value(final_reaction),
                    "nonlinear_failure_recorded": yes_no(nonlinear_failure_recorded),
                    "failed_checks": "; ".join(failed_checks),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")
    if any_failed:
        raise SystemExit("One or more matrix entries failed evidence validation")


if __name__ == "__main__":
    main()
