#!/usr/bin/env python3

import argparse
import csv
import math
import re
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


SOLVERS = ("petsc_cp", "tr", "tr_subspace", "nls")
PRECONDITIONERS = ("HypreAMG", "HypreJacobi")
HISTORY_NAME = "paper_contact_arch_fast_load_displacement.csv"
OUTPUT_NAME = "paper_contact_arch_fast"
REFERENCE_ENTRY = "tr_subspace_HypreAMG"
EXPECTED_STATES = 51
EXPECTED_RANKS = 8
TIME_STEP = 0.16 / 50
SUPPORT_INSET_RATE = 0.05
PLANE_TRAVEL_RATE = 0.2
REQUIRED_FIELDS = {
    "mesh_shape_displacement",
    "mesh_shape_displacement_dual",
    "paper_contact_arch_fast_displacement",
    "paper_contact_arch_fast_reactions",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a contact-arch solver response matrix.")
    parser.add_argument("run_directory", help="Directory created by run_contact_arch_response_matrix.sh.")
    parser.add_argument("--render-directory", default="", help="Optional directory containing rendered PNGs.")
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
    if data.shape[1] != 5:
        raise ValueError(f"Expected five response columns in {path}, found {data.shape[1]}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Nonfinite response value in {path}")
    return data


def parse_command(path):
    return set(shlex.split(path.read_text()))


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
    match = re.search(r"06/contact_arch\s+([0-9.]+) s\s+(?:ok|solver_failed)", log_text)
    return float(match.group(1)) if match else float(fallback)


def failure_step(log_text):
    match = re.search(r"nonlinear solve failed at step (\d+)", log_text)
    return match.group(1) if match else ""


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
    render_directory = Path(args.render_directory) if args.render_directory else None
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
    reference = histories.get(REFERENCE_ENTRY)

    fieldnames = (
        "solver",
        "preconditioner",
        "outcome",
        "validation_state",
        "solver_wall_seconds",
        "response_states",
        "accepted_increments",
        "field_cycles",
        "minimum_vtu_pieces_per_cycle",
        "missing_required_fields",
        "rendered_images",
        "mpi_ranks",
        "nonlinear_iterations",
        "maximum_iterations_per_increment",
        "indefinite_cg_diagnostics",
        "inner_cg_cap_hits",
        "no_bsr_spmv",
        "no_assembled_bsr",
        "cold_start",
        "nonlinear_tolerance_1e_11",
        "linear_tolerance_1e_14",
        "maximum_time_error",
        "maximum_support_inset_error",
        "maximum_plane_travel_error",
        "minimum_support_reaction",
        "final_time",
        "final_contact_displacement",
        "final_support_inset",
        "final_plane_travel",
        "final_support_reaction",
        "failed_step",
        "max_contact_displacement_diff_vs_subspace_tr_amg",
        "max_support_reaction_diff_vs_subspace_tr_amg",
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
            history = histories.get(entry, np.empty((0, 5)))
            log_path = entry_directory / "run.log"
            log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            command_path = entry_directory / "command.txt"
            command = parse_command(command_path) if command_path.is_file() else set()
            field_cycles, minimum_piece_count, missing_fields = field_summary(entry_directory)
            nonlinear_iterations, maximum_iterations = solver_iterations(log_text, solver)
            rendered_images = len(list((render_directory / entry).glob("*.png"))) if render_directory else 0

            if len(history):
                expected_time = np.arange(len(history)) * TIME_STEP
                maximum_time_error = float(np.max(np.abs(history[:, 0] - expected_time)))
                maximum_support_inset_error = float(np.max(np.abs(history[:, 2] - SUPPORT_INSET_RATE * history[:, 0])))
                maximum_plane_travel_error = float(np.max(np.abs(history[:, 3] - PLANE_TRAVEL_RATE * history[:, 0])))
                minimum_support_reaction = float(np.min(history[:, 4]))
                final_time, final_contact_displacement, final_support_inset, final_plane_travel, final_support_reaction = history[-1]
            else:
                maximum_time_error = math.nan
                maximum_support_inset_error = math.nan
                maximum_plane_travel_error = math.nan
                minimum_support_reaction = math.nan
                final_time = final_contact_displacement = final_support_inset = math.nan
                final_plane_travel = final_support_reaction = math.nan

            if reference is not None and len(history):
                common_states = min(len(history), len(reference))
                maximum_contact_difference = float(
                    np.max(np.abs(history[:common_states, 1] - reference[:common_states, 1]))
                )
                maximum_reaction_difference = float(
                    np.max(np.abs(history[:common_states, 4] - reference[:common_states, 4]))
                )
            else:
                maximum_contact_difference = math.nan
                maximum_reaction_difference = math.nan

            expected_max_cg = "600" if preconditioner == "HypreAMG" else "60000"
            expected_subspace = "2" if solver == "tr_subspace" else "0"
            failed_checks = [
                message
                for passed, message in (
                    (len(history) >= 1, f"response states={len(history)}"),
                    (field_cycles == len(history), f"field cycles={field_cycles}, response states={len(history)}"),
                    (minimum_piece_count == EXPECTED_RANKS, f"minimum VTU pieces={minimum_piece_count}"),
                    (not missing_fields, f"missing fields={missing_fields}"),
                    (parse_rank_count(log_text) == EXPECTED_RANKS, "unexpected MPI rank count"),
                    ("--case=06" in command, "missing --case=06"),
                    ("--paraview" in command, "missing --paraview"),
                    ("--no-use-bsr-spmv" in command, "missing --no-use-bsr-spmv"),
                    ("--no-assemble-bsr" in command, "missing --no-assemble-bsr"),
                    ("--warm-start" not in command, "warm start enabled"),
                    ("--nonlinear-tol=1e-11" in command, "missing --nonlinear-tol=1e-11"),
                    ("--linear-tol=1e-14" in command, "missing --linear-tol=1e-14"),
                    (f"--max-cg-iterations={expected_max_cg}" in command, "unexpected CG iteration cap"),
                    (f"--trust-subspace-option={expected_subspace}" in command, "unexpected subspace option"),
                    (maximum_time_error <= 5.0e-7, f"maximum time error={maximum_time_error}"),
                    (maximum_support_inset_error <= 5.0e-7, f"maximum support-inset error={maximum_support_inset_error}"),
                    (maximum_plane_travel_error <= 5.0e-7, f"maximum plane-travel error={maximum_plane_travel_error}"),
                    (minimum_support_reaction >= -1.0e-10, f"minimum support reaction={minimum_support_reaction}"),
                    (
                        outcome != "completed" or (len(history) == EXPECTED_STATES and field_cycles == EXPECTED_STATES),
                        f"completed entry has {len(history)} response states and {field_cycles} field cycles",
                    ),
                    (
                        outcome != "failed" or "nonlinear solve failed" in log_text,
                        "failed entry lacks nonlinear-failure record",
                    ),
                )
                if not passed
            ]
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
                    "accepted_increments": max(0, len(history) - 1),
                    "field_cycles": field_cycles,
                    "minimum_vtu_pieces_per_cycle": minimum_piece_count,
                    "missing_required_fields": missing_fields,
                    "rendered_images": rendered_images,
                    "mpi_ranks": parse_rank_count(log_text),
                    "nonlinear_iterations": nonlinear_iterations,
                    "maximum_iterations_per_increment": maximum_iterations,
                    "indefinite_cg_diagnostics": log_text.count("operator is not positive definite"),
                    "inner_cg_cap_hits": len(re.findall(r"PCG: Number of iterations: (?:600|60000)$", log_text, re.MULTILINE)),
                    "no_bsr_spmv": yes_no("--no-use-bsr-spmv" in command),
                    "no_assembled_bsr": yes_no("--no-assemble-bsr" in command),
                    "cold_start": yes_no("--warm-start" not in command),
                    "nonlinear_tolerance_1e_11": yes_no("--nonlinear-tol=1e-11" in command),
                    "linear_tolerance_1e_14": yes_no("--linear-tol=1e-14" in command),
                    "maximum_time_error": format_value(maximum_time_error),
                    "maximum_support_inset_error": format_value(maximum_support_inset_error),
                    "maximum_plane_travel_error": format_value(maximum_plane_travel_error),
                    "minimum_support_reaction": format_value(minimum_support_reaction),
                    "final_time": format_value(final_time),
                    "final_contact_displacement": format_value(final_contact_displacement),
                    "final_support_inset": format_value(final_support_inset),
                    "final_plane_travel": format_value(final_plane_travel),
                    "final_support_reaction": format_value(final_support_reaction),
                    "failed_step": failure_step(log_text),
                    "max_contact_displacement_diff_vs_subspace_tr_amg": format_value(maximum_contact_difference),
                    "max_support_reaction_diff_vs_subspace_tr_amg": format_value(maximum_reaction_difference),
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
