#!/usr/bin/env python3
"""Run a curated option sweep over the hyperelastic performance suite."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PRESETS = [
    ("baseline", []),
    ("left1_prev4", ["--trust-num-leftmost", "1", "--trust-num-previous-steps", "4"]),
    ("left2_prev0", ["--trust-num-leftmost", "2", "--trust-num-previous-steps", "0"]),
    ("left4_prev8", ["--trust-num-leftmost", "4", "--trust-num-previous-steps", "8"]),
    ("schwarz", ["--deflation-coarse-mode", "schwarz"]),
    ("local", ["--deflation-coarse-mode", "local"]),
    ("quadratic", ["--deflation-order", "quadratic"]),
    ("quadratic_schwarz", ["--deflation-order", "quadratic", "--deflation-coarse-mode", "schwarz"]),
    ("bsr", ["--use-bsr-spmv"]),
    ("schwarz_bsr", ["--deflation-coarse-mode", "schwarz", "--use-bsr-spmv"]),
    ("lanczos2", ["--trust-num-lanczos", "2", "--trust-num-lanczos-iters", "8"]),
    ("lanczos4", ["--trust-num-lanczos", "4", "--trust-num-lanczos-iters", "12"]),
    ("cg500", ["--max-cg-iterations", "500"]),
    ("cg2000", ["--max-cg-iterations", "2000"]),
    ("stagnation", ["--cg-model-energy-stagnation-reltol", "1e-3", "--cg-stagnation-window", "5"]),
    ("simpson", ["--trust-work-quadrature", "3"]),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def read_summary(path: Path, preset: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["preset"] = preset
    return rows


def write_aggregate(rows: list[dict[str, str]], path: Path) -> None:
    preferred = [
        "preset",
        "problem",
        "status",
        "wall_s",
        "reported_step_wall_s",
        "trust_total_s",
        "assemble_jacobian_s",
        "in_cg_h_mult_s",
        "in_cg_dot_many_s",
        "in_cg_p_mult_s",
        "cg_total_iters",
        "cg_hit_max",
        "defl_mult_coarse_s",
        "defl_allgather_s",
        "wtaw_cholesky_failures_per_rank",
        "log",
    ]
    fields = preferred + sorted({key for row in rows for key in row.keys()} - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    totals: dict[str, float] = {}
    failures: dict[str, int] = {}
    for row in rows:
        preset = row["preset"]
        totals[preset] = totals.get(preset, 0.0) + float(row.get("reported_step_wall_s") or row.get("wall_s") or 0.0)
        failures[preset] = failures.get(preset, 0) + (0 if row.get("status") == "ok" else 1)

    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Hyperelastic Option Sweep\n\n")
        stream.write("| preset | total reported wall s | failures |\n")
        stream.write("| --- | ---: | ---: |\n")
        for preset, _ in PRESETS:
            if preset in totals:
                stream.write(f"| {preset} | {totals[preset]:.3f} | {failures.get(preset, 0)} |\n")

        stream.write("\n## Per Problem\n\n")
        stream.write("| preset | problem | status | wall s | CG iters | hit max | H.Mult s | dots s |\n")
        stream.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            stream.write(
                f"| {row['preset']} | {row['problem']} | {row['status']} | "
                f"{float(row.get('reported_step_wall_s') or row.get('wall_s') or 0.0):.3f} | "
                f"{row.get('cg_total_iters', '')} | {row.get('cg_hit_max', '')} | "
                f"{row.get('in_cg_h_mult_s', '')} | {row.get('in_cg_dot_many_s', '')} |\n"
            )


def write_relative_summary(rows: list[dict[str, str]], path: Path, baseline_preset: str = "baseline") -> None:
    """Score presets by geometric-mean speedup over problems solved correctly by both
    the preset and the baseline. Failed / wrong-answer runs are excluded from the
    timing score and reported separately, so a timeout or a wrong equilibrium branch
    can never masquerade as a (fixed, arbitrary) slowdown factor.
    """

    def wall(row: dict[str, str]) -> float | None:
        value = row.get("reported_step_wall_s") or row.get("wall_s")
        return float(value) if value else None

    by_preset: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_preset.setdefault(row["preset"], {})[row["problem"]] = row

    baseline = by_preset.get(baseline_preset, {})
    base_ok = {name: row for name, row in baseline.items() if row.get("status") == "ok" and wall(row)}
    problems = sorted({row["problem"] for row in rows})

    entries = []
    for preset, problem_rows in by_preset.items():
        speedups: dict[str, float] = {}
        bad: dict[str, str] = {}
        for name, row in problem_rows.items():
            status = row.get("status", "")
            if status != "ok":
                bad[name] = status
            elif name in base_ok and wall(row):
                speedups[name] = wall(base_ok[name]) / wall(row)
        geom = math.exp(sum(math.log(s) for s in speedups.values()) / len(speedups)) if speedups else float("nan")
        total = sum(wall(row) or 0.0 for row in problem_rows.values())
        entries.append((preset, geom, len(speedups), bad, speedups, total))

    entries.sort(key=lambda e: (-(e[1] if e[1] == e[1] else -1.0)))

    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Hyperelastic Option Sweep Relative Summary\n\n")
        stream.write(
            "Speedup = baseline_time / preset_time, geometric mean over problems where both the\n"
            "preset and the baseline finished with the correct answer. Problems that failed,\n"
            "timed out, or converged to a wrong answer are listed in `bad problems` and do not\n"
            "contribute to the speedup.\n\n"
        )
        header = ["preset", "geom speedup", "scored problems", "bad problems", *problems, "total wall s"]
        stream.write("| " + " | ".join(header) + " |\n")
        stream.write("| --- | ---: | ---: | --- | " + " | ".join(["---:"] * len(problems)) + " | ---: |\n")
        for preset, geom, scored, bad, speedups, total in entries:
            bad_text = "; ".join(f"{name}:{status}" for name, status in sorted(bad.items())) or "-"
            per_problem = [f"{speedups[name]:.3f}" if name in speedups else "-" for name in problems]
            geom_text = f"{geom:.3f}" if geom == geom else "-"
            stream.write(
                f"| {preset} | {geom_text} | {scored} | {bad_text} | " + " | ".join(per_problem) + f" | {total:.3f} |\n"
            )

        stream.write("\n## Baseline Times\n\n")
        for name, row in sorted(base_ok.items()):
            stream.write(f"- {name}: {wall(row):.3f} s\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--procs", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problems", nargs="+", default=["arch", "block", "contact", "twist"])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (root / "performance_runs" / f"option-sweep-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = PRESETS[args.start :]
    if args.limit is not None:
        presets = presets[: args.limit]

    suite = root / "scripts" / "run_hyperelastic_suite.py"
    all_rows: list[dict[str, str]] = []
    for index, (name, options) in enumerate(presets, start=args.start):
        run_dir = output_dir / f"{index:02d}-{name}"
        cmd = [
            str(suite),
            "--output-dir",
            str(run_dir),
            "--procs",
            str(args.procs),
            "--timeout-sec",
            str(args.timeout_sec),
            "--jobs",
            str(args.jobs),
            "--problems",
            *args.problems,
            *options,
        ]
        if args.skip_build or index != args.start:
            cmd.append("--skip-build")
        if args.dry_run:
            cmd.append("--dry-run")
        rc = run(cmd, root)
        if rc != 0:
            print(f"preset {name} exited with return code {rc}; continuing", flush=True)
        summary = run_dir / "summary.csv"
        if summary.exists():
            all_rows.extend(read_summary(summary, name))
            write_aggregate(all_rows, output_dir / "aggregate.csv")
            write_markdown(all_rows, output_dir / "aggregate.md")
            write_relative_summary(all_rows, output_dir / "relative_summary.md")
        elif not args.dry_run:
            print(f"missing summary for preset {name}: {summary}", flush=True)

    print(f"wrote {output_dir / 'aggregate.csv'}")
    print(f"wrote {output_dir / 'aggregate.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
