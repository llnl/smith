#!/usr/bin/env python3
"""Sweep EnergyMortar smoothing parameters for the 2D ironing example.

The script runs each case/radius/start-angle combination sequentially and writes
``timesteps.csv`` and ``summary.csv`` under the selected output directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_RADII = [0.001, 0.01, 0.025, 0.1, 0.25]
DEFAULT_START_ANGLES = [45.0]
BEGIN_RE = re.compile(r"IRONING_TIMESTEP_BEGIN\s+(\d+)")
END_RE = re.compile(r"IRONING_TIMESTEP_END\s+(\d+)")
ITERATION_RE = re.compile(r"TrustRegion iteration\s+(\d+)\s*:\s*\|\|r\|\|\s*=\s*(\S+)")
TIMESTEP_FIELDS = [
    "case",
    "smoothing_radius",
    "normal_smoothing_start_angle_degrees",
    "timestep",
    "iterations",
    "converged",
    "complete",
    "initial_residual_norm",
    "final_residual_norm",
    "log",
]
SUMMARY_FIELDS = [
    "case",
    "smoothing_radius",
    "normal_smoothing_start_angle_degrees",
    "requested_timesteps",
    "observed_nonlinear_solves",
    "converged_nonlinear_solves",
    "failed_nonlinear_solves",
    "total_iterations_all_solves",
    "total_iterations_converged_solves",
    "average_iterations_per_nonlinear_solve",
    "max_iterations_per_nonlinear_solve",
    "all_solves_converged",
    "wall_seconds",
    "return_code",
    "timed_out",
    "status",
    "log",
    "command",
]


@dataclass
class TimestepResult:
    case: str
    smoothing_radius: float
    normal_smoothing_start_angle_degrees: float
    timestep: int
    iterations: int
    converged: bool
    complete: bool
    initial_residual_norm: float | None
    final_residual_norm: float | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exe",
        type=Path,
        default=root / "build-llvm19-noomp-enzyme-release/examples/contact_ironing_2D",
        help="Path to the contact_ironing_2D executable.",
    )
    parser.add_argument("--cases", nargs="+", choices=["square", "circle"], default=["square", "circle"])
    parser.add_argument("--radii", nargs="+", type=float, default=DEFAULT_RADII)
    parser.add_argument(
        "--normal-smoothing-start-angles",
        nargs="+",
        type=float,
        default=DEFAULT_START_ANGLES,
        help="Normal smoothing start angles in degrees; 90 disables attenuation.",
    )
    parser.add_argument("--num-steps", type=int, default=175)
    parser.add_argument("--gap-mode", choices=["nodal", "quadrature-point"], default="quadrature-point")
    parser.add_argument("--mpi-tasks", type=int, default=4)
    parser.add_argument("--launcher", default="mpirun", help="MPI launcher used when --mpi-tasks is greater than one.")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Per-run timeout; zero disables it.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "smoothing_runs" / f"ironing_2D-{timestamp}",
    )
    parser.add_argument("--keep-visualization-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional executable arguments, placed after --.",
    )
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if args.mpi_tasks <= 0:
        parser.error("--mpi-tasks must be positive")
    if any(radius < 0.0 or radius > 0.5 for radius in args.radii):
        parser.error("all --radii values must be in [0, 0.5]")
    if any(angle < 0.0 or angle > 90.0 for angle in args.normal_smoothing_start_angles):
        parser.error("all --normal-smoothing-start-angles values must be in [0, 90]")
    return args


def radius_tag(radius: float) -> str:
    return f"{radius:g}".replace(".", "p")


def angle_tag(angle: float) -> str:
    return f"{angle:g}".replace(".", "p")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_timestep_results(log_text: str, case: str, radius: float, start_angle: float) -> list[TimestepResult]:
    results: list[TimestepResult] = []
    current: dict[str, object] | None = None

    def finish(complete: bool) -> None:
        nonlocal current
        if current is None:
            return
        iterations = current["iterations"]
        results.append(
            TimestepResult(
                case=case,
                smoothing_radius=radius,
                normal_smoothing_start_angle_degrees=start_angle,
                timestep=int(current["timestep"]),
                iterations=int(iterations) if iterations is not None else 0,
                converged=bool(current["converged"]) and complete,
                complete=complete,
                initial_residual_norm=current["initial_residual_norm"],
                final_residual_norm=current["final_residual_norm"],
            )
        )
        current = None

    for line in log_text.splitlines():
        if match := BEGIN_RE.search(line):
            timestep = int(match.group(1))
            if current is None:
                current = {
                    "timestep": timestep,
                    "iterations": None,
                    "converged": True,
                    "initial_residual_norm": None,
                    "final_residual_norm": None,
                }
            elif current["timestep"] != timestep:
                finish(complete=False)
                current = {
                    "timestep": timestep,
                    "iterations": None,
                    "converged": True,
                    "initial_residual_norm": None,
                    "final_residual_norm": None,
                }
            continue

        if current is None:
            continue

        if match := ITERATION_RE.search(line):
            iteration = int(match.group(1))
            residual_norm = float(match.group(2))
            if current["initial_residual_norm"] is None:
                current["initial_residual_norm"] = residual_norm
            current["iterations"] = iteration
            current["final_residual_norm"] = residual_norm
            continue

        if "TrustRegion: No convergence!" in line:
            current["converged"] = False
            continue

        if match := END_RE.search(line):
            if int(match.group(1)) != current["timestep"]:
                continue
            finish(complete=True)

    finish(complete=False)
    return results


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    header = (
        f"{'case':<8} {'radius':>8} {'theta':>7} {'solves':>7} {'conv':>6} {'failed':>7} "
        f"{'total(conv)':>11} {'avg/solve':>10} {'max':>6} {'wall(s)':>9} {'status':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<8} {row['smoothing_radius']:>8g} "
            f"{row['normal_smoothing_start_angle_degrees']:>7g} {row['observed_nonlinear_solves']:>7} "
            f"{row['converged_nonlinear_solves']:>6} {row['failed_nonlinear_solves']:>7} "
            f"{row['total_iterations_converged_solves']:>11} "
            f"{row['average_iterations_per_nonlinear_solve']:>10.3f} "
            f"{row['max_iterations_per_nonlinear_solve']:>6} {row['wall_seconds']:>9.2f} {row['status']:>12}"
        )


def main() -> int:
    args = parse_args()
    executable = args.exe.expanduser().resolve()
    if not args.dry_run and not executable.is_file():
        raise SystemExit(f"executable not found: {executable}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestep_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case in args.cases:
        for radius in args.radii:
            for start_angle in args.normal_smoothing_start_angles:
                run_dir = output_dir / (
                    f"{case}-radius-{radius_tag(radius)}-theta-start-{angle_tag(start_angle)}"
                )
                run_dir.mkdir(parents=True, exist_ok=False)
                log_path = run_dir / "run.log"
                command_path = run_dir / "command.txt"

                command: list[str] = []
                if args.mpi_tasks > 1:
                    command.extend([args.launcher, "-np", str(args.mpi_tasks)])
                command.extend(
                    [
                        str(executable),
                        "--case",
                        case,
                        "--energy-mortar-gap-mode",
                        args.gap_mode,
                        "--energy-mortar-smoothing-radius",
                        f"{radius:.17g}",
                        "--energy-mortar-normal-smoothing-start-angle",
                        f"{start_angle:.17g}",
                        "--num-steps",
                        str(args.num_steps),
                        "--print-timestep-markers",
                    ]
                )
                if not args.keep_visualization_output:
                    command.append("--disable-output")
                command.extend(args.extra_args)
                command_path.write_text(shell_join(command) + "\n")
                print(
                    f"Running {case}, smoothing radius {radius:g}, start angle {start_angle:g} degrees",
                    flush=True,
                )

                return_code = 0
                timed_out = False
                start = time.perf_counter()
                if args.dry_run:
                    print(f"  {shell_join(command)}")
                    log_path.write_text("")
                else:
                    environment = os.environ.copy()
                    environment.setdefault("OMP_NUM_THREADS", "1")
                    try:
                        with log_path.open("w") as log:
                            completed = subprocess.run(
                                command,
                                cwd=run_dir,
                                env=environment,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                text=True,
                                check=False,
                                timeout=args.timeout_seconds or None,
                            )
                        return_code = completed.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        return_code = 124
                wall_seconds = time.perf_counter() - start

                timestep_results = parse_timestep_results(log_path.read_text(), case, radius, start_angle)
                timestep_rows.extend(asdict(result) | {"log": str(log_path)} for result in timestep_results)
                completed_solves = [result for result in timestep_results if result.complete]
                converged_solves = [result for result in completed_solves if result.converged]
                failed_solves = [result for result in timestep_results if not result.converged]
                total_iterations = sum(result.iterations for result in completed_solves)
                total_converged_iterations = sum(result.iterations for result in converged_solves)
                average_iterations = total_iterations / len(completed_solves) if completed_solves else 0.0
                max_iterations = max((result.iterations for result in completed_solves), default=0)

                if args.dry_run:
                    status = "dry-run"
                elif timed_out:
                    status = "timeout"
                elif return_code != 0:
                    status = "process-failed"
                elif len(completed_solves) != args.num_steps:
                    status = "parse-failed"
                elif failed_solves:
                    status = "nonconverged"
                else:
                    status = "ok"

                summary_rows.append(
                    {
                        "case": case,
                        "smoothing_radius": radius,
                        "normal_smoothing_start_angle_degrees": start_angle,
                        "requested_timesteps": args.num_steps,
                        "observed_nonlinear_solves": len(completed_solves),
                        "converged_nonlinear_solves": len(converged_solves),
                        "failed_nonlinear_solves": len(failed_solves),
                        "total_iterations_all_solves": total_iterations,
                        "total_iterations_converged_solves": total_converged_iterations,
                        "average_iterations_per_nonlinear_solve": average_iterations,
                        "max_iterations_per_nonlinear_solve": max_iterations,
                        "all_solves_converged": status == "ok",
                        "wall_seconds": wall_seconds,
                        "return_code": return_code,
                        "timed_out": timed_out,
                        "status": status,
                        "log": str(log_path),
                        "command": shell_join(command),
                    }
                )

                write_csv(output_dir / "timesteps.csv", timestep_rows, TIMESTEP_FIELDS)
                write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)

    print()
    print_summary(summary_rows)
    print(f"\nWrote {output_dir / 'timesteps.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    return 0 if all(row["status"] in {"ok", "dry-run"} for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
