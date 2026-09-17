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
HISTORY_NAME = "paper_viscoelastic_buckling_fast_force_displacement.csv"
OUTPUT_NAME = "paper_viscoelastic_buckling_fast"
EXPECTED_STATES = 81
EXPECTED_ELEMENTS = 1097
EXPECTED_RANKS = 16
TIME_STEP = 0.3
LOAD = 0.6
LOAD_REMOVAL_TIME = 6.0
REQUIRED_FIELDS = {
    "mesh_shape_displacement",
    "mesh_shape_displacement_dual",
    "paper_viscoelastic_buckling_fast_displacement",
    "paper_viscoelastic_buckling_fast_reactions",
    "paper_viscoelastic_buckling_fast_temperature",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a unified viscoelastic response matrix.")
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


def expected_force(time):
    return time / LOAD_REMOVAL_TIME * LOAD if time <= LOAD_REMOVAL_TIME else 0.0


def parse_command(path):
    args = shlex.split(path.read_text())
    return set(args)


def parse_global_elements(log_text):
    match = re.search(r"viscoelastic buckling: global elements = (\d+)", log_text)
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
    match = re.search(r"04/viscoelastic_buckling\s+([0-9.]+) s\s+(?:ok|solver_failed)", log_text)
    return float(match.group(1)) if match else float(fallback)


def field_summary(entry_directory):
    cycle_root = entry_directory / OUTPUT_NAME / OUTPUT_NAME
    cycle_directories = sorted(cycle_root.glob("Cycle*"))
    missing_fields = set()
    minimum_piece_count = EXPECTED_RANKS
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
        "global_elements",
        "mpi_ranks",
        "nonlinear_iterations",
        "maximum_iterations_per_step",
        "no_bsr_spmv",
        "no_assembled_bsr",
        "cold_start",
        "nonlinear_tolerance_1e_11",
        "linear_tolerance_1e_14",
        "maximum_time_error",
        "maximum_applied_force_error",
        "maximum_loaded_force_balance_error",
        "maximum_unloaded_reaction",
        "maximum_displacement_difference_vs_tr_amg",
        "maximum_reaction_difference_vs_tr_amg",
        "peak_displacement",
        "peak_applied_force",
        "peak_reaction",
        "unload_displacement",
        "unload_reaction",
        "final_displacement",
        "final_reaction",
        "monotone_relaxation_after_unload",
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
            entry_directory = run_directory / entry
            history = histories.get(entry, np.empty((0, 4)))
            log_path = entry_directory / "run.log"
            log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            command_path = entry_directory / "command.txt"
            command = parse_command(command_path) if command_path.is_file() else set()
            field_cycles, minimum_piece_count, missing_fields = field_summary(entry_directory)
            nonlinear_iterations, maximum_iterations = solver_iterations(log_text, solver)

            if len(history):
                expected_times = np.arange(len(history)) * TIME_STEP
                expected_forces = np.array([expected_force(time) for time in history[:, 0]])
                loaded = history[:, 2] > 0.0
                unloaded = ~loaded
                post_unload = history[:, 0] > LOAD_REMOVAL_TIME
                post_unload_displacement = history[post_unload, 1]
                maximum_time_error = float(np.max(np.abs(history[:, 0] - expected_times)))
                maximum_applied_force_error = float(np.max(np.abs(history[:, 2] - expected_forces)))
                maximum_loaded_force_balance_error = float(np.max(np.abs(history[loaded, 3] - history[loaded, 2])))
                maximum_unloaded_reaction = float(np.max(np.abs(history[unloaded, 3])))
                peak_index = int(np.argmax(history[:, 2]))
                unload_index = int(np.flatnonzero(post_unload)[0])
                monotone_relaxation = bool(np.all(np.diff(post_unload_displacement) >= -1.0e-10))
                peak_displacement, peak_force, peak_reaction = history[peak_index, 1:4]
                unload_displacement = history[unload_index, 1]
                unload_reaction = history[unload_index, 3]
                final_displacement = history[-1, 1]
                final_reaction = history[-1, 3]
            else:
                maximum_time_error = math.nan
                maximum_applied_force_error = math.nan
                maximum_loaded_force_balance_error = math.nan
                maximum_unloaded_reaction = math.nan
                monotone_relaxation = False
                peak_displacement = peak_force = peak_reaction = math.nan
                unload_displacement = unload_reaction = math.nan
                final_displacement = final_reaction = math.nan

            if reference is not None and len(history) == len(reference):
                maximum_displacement_difference = float(np.max(np.abs(history[:, 1] - reference[:, 1])))
                maximum_reaction_difference = float(np.max(np.abs(history[:, 3] - reference[:, 3])))
            else:
                maximum_displacement_difference = math.nan
                maximum_reaction_difference = math.nan

            no_bsr_spmv = "--no-use-bsr-spmv" in command
            no_assembled_bsr = "--no-assemble-bsr" in command
            cold_start = "--no-warm-start" in command
            nonlinear_tolerance = "--nonlinear-tol=1e-11" in command
            linear_tolerance = "--linear-tol=1e-14" in command
            checks = []
            if status_row["outcome"] == "completed":
                checks = [
                    (len(history) == EXPECTED_STATES, f"response states={len(history)}"),
                    (field_cycles == EXPECTED_STATES, f"field cycles={field_cycles}"),
                    (minimum_piece_count == EXPECTED_RANKS, f"minimum VTU pieces={minimum_piece_count}"),
                    (not missing_fields, f"missing fields={missing_fields}"),
                    (parse_global_elements(log_text) == EXPECTED_ELEMENTS, "unexpected global element count"),
                    (parse_rank_count(log_text) == EXPECTED_RANKS, "unexpected MPI rank count"),
                    (no_bsr_spmv, "missing --no-use-bsr-spmv"),
                    (no_assembled_bsr, "missing --no-assemble-bsr"),
                    (cold_start, "missing --no-warm-start"),
                    (nonlinear_tolerance, "missing --nonlinear-tol=1e-11"),
                    (linear_tolerance, "missing --linear-tol=1e-14"),
                    (maximum_time_error <= 1.0e-10, f"maximum time error={maximum_time_error}"),
                    (maximum_applied_force_error <= 1.0e-6, f"maximum applied-force error={maximum_applied_force_error}"),
                    (
                        maximum_loaded_force_balance_error <= 1.0e-6,
                        f"maximum loaded force-balance error={maximum_loaded_force_balance_error}",
                    ),
                    (
                        maximum_unloaded_reaction <= 1.0e-8,
                        f"maximum unloaded reaction={maximum_unloaded_reaction}",
                    ),
                    (monotone_relaxation, "post-unload displacement is not monotone toward zero"),
                ]
            failed_checks = [message for passed, message in checks if not passed]
            if status_row["outcome"] == "completed":
                validation_state = "passed" if not failed_checks else "failed"
            else:
                validation_state = "partial"
            any_failed = any_failed or bool(failed_checks)

            rows.append(
                {
                    "solver": solver,
                    "preconditioner": preconditioner,
                    "outcome": status_row["outcome"],
                    "validation_state": validation_state,
                    "solver_wall_seconds": f'{solver_wall_seconds(log_text, status_row["elapsed_seconds"]):.3f}',
                    "response_states": len(history),
                    "field_cycles": field_cycles,
                    "minimum_vtu_pieces_per_cycle": minimum_piece_count,
                    "missing_required_fields": missing_fields,
                    "global_elements": parse_global_elements(log_text),
                    "mpi_ranks": parse_rank_count(log_text),
                    "nonlinear_iterations": nonlinear_iterations,
                    "maximum_iterations_per_step": maximum_iterations,
                    "no_bsr_spmv": yes_no(no_bsr_spmv),
                    "no_assembled_bsr": yes_no(no_assembled_bsr),
                    "cold_start": yes_no(cold_start),
                    "nonlinear_tolerance_1e_11": yes_no(nonlinear_tolerance),
                    "linear_tolerance_1e_14": yes_no(linear_tolerance),
                    "maximum_time_error": format_value(maximum_time_error),
                    "maximum_applied_force_error": format_value(maximum_applied_force_error),
                    "maximum_loaded_force_balance_error": format_value(maximum_loaded_force_balance_error),
                    "maximum_unloaded_reaction": format_value(maximum_unloaded_reaction),
                    "maximum_displacement_difference_vs_tr_amg": format_value(maximum_displacement_difference),
                    "maximum_reaction_difference_vs_tr_amg": format_value(maximum_reaction_difference),
                    "peak_displacement": format_value(peak_displacement),
                    "peak_applied_force": format_value(peak_force),
                    "peak_reaction": format_value(peak_reaction),
                    "unload_displacement": format_value(unload_displacement),
                    "unload_reaction": format_value(unload_reaction),
                    "final_displacement": format_value(final_displacement),
                    "final_reaction": format_value(final_reaction),
                    "monotone_relaxation_after_unload": yes_no(monotone_relaxation),
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
        raise SystemExit("One or more completed entries failed validation")


if __name__ == "__main__":
    main()
