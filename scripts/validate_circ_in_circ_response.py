#!/usr/bin/env python3

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


FIELDS = [
    "solver",
    "preconditioner",
    "outcome",
    "elapsed_seconds",
    "response_states",
    "accepted_increments",
    "visualization_cycles",
    "vtu_pieces_per_cycle",
    "required_fields_present",
    "response_field_pairing",
    "nonlinear_iterations",
    "maximum_iterations_per_increment",
    "indefinite_cg_diagnostics",
    "inner_cg_cap_hits",
    "final_displacement",
    "final_applied_traction",
    "failed_step",
    "max_displacement_diff_vs_subspace_tr_amg",
    "first_inverted_cycle",
    "minimum_det_f",
    "minimum_det_f_cycle",
    "final_inverted_cells",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a concentric-circle solver response matrix.")
    parser.add_argument("--input", required=True, type=Path, help="Matrix output directory.")
    parser.add_argument("--output", required=True, type=Path, help="Validation TSV path.")
    return parser.parse_args()


def read_response(path):
    response = []
    if not path.exists():
        return response
    with path.open() as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            response.append(tuple(map(float, line.split())))
    return response


def validate_fields(run_dir):
    cycle_root = run_dir / "paper_circ_in_circ_fast" / "paper_circ_in_circ_fast"
    cycle_files = sorted(cycle_root.glob("Cycle*/data.pvtu")) if cycle_root.exists() else []
    required_fields = {
        "paper_circ_in_circ_fast_displacement",
        "paper_circ_in_circ_fast_reactions",
        "attribute",
    }
    piece_counts = []
    all_fields_present = True
    for path in cycle_files:
        tree = ET.parse(path)
        names = {element.attrib.get("Name", "") for element in tree.iter()}
        all_fields_present = all_fields_present and required_fields.issubset(names)
        piece_counts.append(sum(1 for element in tree.iter() if element.tag.endswith("Piece")))
    piece_summary = ""
    if piece_counts:
        piece_summary = str(piece_counts[0]) if len(set(piece_counts)) == 1 else f"{min(piece_counts)}-{max(piece_counts)}"
    return cycle_files, piece_summary, all_fields_present


def deformation_summary(path):
    if not path.exists():
        return "", "", "", ""
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    inverted_rows = [row for row in rows if int(row["inverted_cells"]) > 0]
    first_inverted = inverted_rows[0]["cycle"] if inverted_rows else ""
    minimum_row = min(rows, key=lambda row: float(row["minimum_det_f"]))
    return first_inverted, minimum_row["minimum_det_f"], minimum_row["cycle"], rows[-1]["inverted_cells"]


def main():
    args = parse_args()
    root = args.input.resolve()
    with (root / "status.tsv").open() as stream:
        status_rows = list(csv.DictReader(stream, delimiter="\t"))

    reference = {
        time: displacement
        for time, displacement, _ in read_response(
            root / "tr_subspace_HypreAMG" / "paper_circ_in_circ_fast_load_displacement.csv"
        )
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for status in status_rows:
            entry = f"{status['solver']}_{status['preconditioner']}"
            run_dir = root / entry
            log_text = (run_dir / "run.log").read_text(errors="replace")
            response = read_response(run_dir / "paper_circ_in_circ_fast_load_displacement.csv")
            cycle_files, piece_summary, required_fields_present = validate_fields(run_dir)

            iteration_values = [
                int(value)
                for value in re.findall(r"(?:TrustRegion|Newton) iteration\s+(\d+)\s+:", log_text)
                if int(value)
            ]
            if entry.startswith("petsc_cp_"):
                iteration_values = [
                    int(value) for value in re.findall(r"PETSc SNES summary: iterations=(\d+)", log_text)
                ]
            failed_match = re.search(r"failed at step (\d+)", log_text)
            differences = [
                abs(displacement - reference[time])
                for time, displacement, _ in response
                if time in reference
            ]
            first_inverted, minimum_det_f, minimum_cycle, final_inverted = deformation_summary(
                run_dir / "deformation_validity.tsv"
            )
            cap = 600 if status["preconditioner"] == "HypreAMG" else 60000
            writer.writerow(
                {
                    "solver": status["solver"],
                    "preconditioner": status["preconditioner"],
                    "outcome": status["outcome"],
                    "elapsed_seconds": status["elapsed_seconds"],
                    "response_states": len(response),
                    "accepted_increments": max(0, len(response) - 1),
                    "visualization_cycles": len(cycle_files),
                    "vtu_pieces_per_cycle": piece_summary,
                    "required_fields_present": str(required_fields_present).lower(),
                    "response_field_pairing": str(len(response) == len(cycle_files)).lower(),
                    "nonlinear_iterations": sum(iteration_values),
                    "maximum_iterations_per_increment": max(iteration_values, default=0),
                    "indefinite_cg_diagnostics": log_text.count("The operator is not positive definite"),
                    "inner_cg_cap_hits": log_text.count(f"Number of iterations: {cap}"),
                    "final_displacement": f"{response[-1][1]:.9g}" if response else "",
                    "final_applied_traction": f"{response[-1][2]:.9g}" if response else "",
                    "failed_step": failed_match.group(1) if failed_match else "",
                    "max_displacement_diff_vs_subspace_tr_amg": f"{max(differences):.9g}" if differences else "",
                    "first_inverted_cycle": first_inverted,
                    "minimum_det_f": minimum_det_f,
                    "minimum_det_f_cycle": minimum_cycle,
                    "final_inverted_cells": final_inverted,
                }
            )


if __name__ == "__main__":
    main()
