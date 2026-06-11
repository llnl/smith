#!/usr/bin/env python3
"""Run the nonlinear hyperelastic performance suite and summarize timings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROBLEMS = ("arch", "block", "contact", "twist")


@dataclass(frozen=True)
class Problem:
    name: str
    executable: str
    hyper_case: str | None = None
    # Per-problem load-step override. Twist needs finer stepping (plus the in-code
    # imperfection) to stay on one post-buckling branch across proc counts.
    default_steps: int | None = None
    # Per-problem refinement calibrated so each case runs ~60 s at np=6 with the
    # current best options (see performance_plan.md). Global --mesh-scale overrides.
    default_mesh_scale: float | None = None


PROBLEM_TABLE = {
    "arch": Problem("arch", "tests/shallow_arch_buckling", default_mesh_scale=1.2),
    "block": Problem("block", "tests/hyperelastic_benchmarks", "block", default_mesh_scale=1.6),
    "contact": Problem("contact", "tests/hyperelastic_benchmarks", "contact", default_mesh_scale=0.8),
    "twist": Problem("twist", "tests/hyperelastic_benchmarks", "twist", default_steps=6, default_mesh_scale=0.9),
}


TRUST_LABELS = {
    "total solve": "trust_total_s",
    "assembleJacobian": "assemble_jacobian_s",
    "precond SetOperator": "precond_setop_s",
    "hess_vec total": "hess_vec_total_s",
    "precond Mult total": "precond_mult_total_s",
    "batched hess_vec": "batched_hess_vec_s",
    "subspace prepare": "subspace_prepare_s",
    "subspace solve": "subspace_solve_s",
    "augment-direction": "augment_direction_s",
    "in-CG H.Mult": "in_cg_h_mult_s",
    "in-CG P.Mult": "in_cg_p_mult_s",
    "in-CG dot_many (+Allred)": "in_cg_dot_many_s",
}

DEFLATION_LABELS = {
    "assembleWtAW (total)": "defl_assemble_wtaw_s",
    "halo (pack+isend/irecv+wait)": "defl_halo_s",
    "diag SpMV+gemm": "defl_diag_spmv_gemm_s",
    "offd SpMV+gemm": "defl_offd_spmv_gemm_s",
    "Allreduce(m*m)": "defl_allreduce_s",
    "factor (Cholesky/LU)": "defl_factor_s",
    "smoother setup": "defl_smoother_setup_s",
    "total": "defl_mult_total_s",
    "smoother": "defl_mult_smoother_s",
    "coarse": "defl_mult_coarse_s",
    "W^T * r": "defl_wt_r_s",
    "Allgather / nbr-exchange": "defl_allgather_s",
    "factor-solve": "defl_factor_solve_s",
    "W * alpha": "defl_w_alpha_s",
}

SUMMARY_FIELDS = [
    "problem",
    "status",
    "answer_rel_err",
    "returncode",
    "wall_s",
    "reported_step_wall_s",
    "trust_total_s",
    "assemble_jacobian_s",
    "precond_setop_s",
    "hess_vec_total_s",
    "precond_mult_total_s",
    "batched_hess_vec_s",
    "subspace_prepare_s",
    "subspace_solve_s",
    "augment_direction_s",
    "in_cg_h_mult_s",
    "in_cg_p_mult_s",
    "in_cg_dot_many_s",
    "cg_outers",
    "cg_total_iters",
    "cg_max_iters",
    "cg_hit_max",
    "line_search_retries_total",
    "defl_assemble_wtaw_s",
    "defl_halo_s",
    "defl_diag_spmv_gemm_s",
    "defl_offd_spmv_gemm_s",
    "defl_allreduce_s",
    "defl_factor_s",
    "defl_smoother_setup_s",
    "defl_mult_total_s",
    "defl_mult_smoother_s",
    "defl_mult_coarse_s",
    "defl_wt_r_s",
    "defl_allgather_s",
    "defl_factor_solve_s",
    "defl_w_alpha_s",
    "wtaw_cholesky_failures_per_rank",
    "wtaw_pp_cholesky_failures_per_rank",
    "contact_active_area",
    "contact_energy",
    "final_displacement_l2",
    "log",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def references_path(args=None) -> Path:
    if args is not None and args.references_file is not None:
        return args.references_file
    return repo_root() / "scripts" / "hyperelastic_references.json"


def load_references(args=None) -> dict:
    path = references_path(args)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def check_answer(row: dict, references: dict) -> None:
    """Compare final_displacement_l2 against the stored reference; flag wrong answers.

    A run that converges fast to the wrong equilibrium branch must not score as a
    performance win, so it is demoted to status=wrong-answer.
    """
    ref = references.get(row["problem"])
    if not ref:
        return
    value = row.get("final_displacement_l2")
    if value in ("", None):
        if row.get("status") == "ok":
            row["status"] = "no-answer"
        return
    rel_err = abs(float(value) - ref["final_displacement_l2"]) / abs(ref["final_displacement_l2"])
    row["answer_rel_err"] = rel_err
    if row.get("status") == "ok" and rel_err > ref["rel_tol"]:
        row["status"] = "wrong-answer"


def update_references(rows: list[dict], default_rel_tols: dict[str, float], args=None) -> None:
    references = load_references(args)
    for row in rows:
        value = row.get("final_displacement_l2")
        if row.get("status") == "ok" and value not in ("", None):
            old = references.get(row["problem"], {})
            references[row["problem"]] = {
                "final_displacement_l2": float(value),
                "rel_tol": old.get("rel_tol", default_rel_tols.get(row["problem"], 0.01)),
            }
    references_path(args).write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    print(f"updated {references_path(args)}")


def default_build_dir(root: Path) -> Path:
    candidates = sorted(root.glob("build-Michael*-release"))
    if not candidates:
        candidates = sorted(root.glob("build-*-release"))
    if not candidates:
        candidates = sorted(root.glob("build-*"))
    if not candidates:
        raise SystemExit("No build directory found; pass --build-dir")
    return candidates[0]


def discover_dyld_path(root: Path) -> str:
    search_roots = [root.parent / "smith-tpls", root / "smith-tpls"]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matches = sorted(search_root.glob("**/netlib-lapack-*/lib/liblapack.3.dylib"))
        if matches:
            return str(matches[0].parent)
    return ""


def append_env_path(env: dict[str, str], key: str, value: str) -> None:
    if not value:
        return
    old = env.get(key, "")
    parts = [value]
    if old:
        parts.append(old)
    env[key] = os.pathsep.join(parts)


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    log: Path | None = None,
    timeout_sec: int | None = None,
) -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return 0
    if log is None:
        return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {printable}\n")
        stream.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            stream.write(f"\nTIMEOUT after {timeout_sec} seconds\n")
            stream.flush()
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return 124


def build_targets(build_dir: Path, jobs: int, dry_run: bool) -> None:
    for target in ("shallow_arch_buckling", "hyperelastic_benchmarks"):
        cmd = ["cmake", "--build", str(build_dir), "--target", target, "--parallel", str(jobs)]
        rc = run_command(cmd, repo_root(), os.environ.copy(), dry_run)
        if rc != 0:
            raise SystemExit(f"Build failed for target {target} with return code {rc}")


def problem_mesh_scale(args: argparse.Namespace, problem: Problem) -> float | None:
    """Global --mesh-scale replaces the per-problem default; --mesh-scale-factor multiplies it
    (proportional screening meshes for parameter searches)."""
    if args.mesh_scale is not None:
        return args.mesh_scale
    base = problem.default_mesh_scale if problem.default_mesh_scale is not None else 1.0
    if args.mesh_scale_factor is not None:
        return base * args.mesh_scale_factor
    return problem.default_mesh_scale


def solver_args(args: argparse.Namespace, problem: Problem) -> list[str]:
    common = [
        f"--preconditioner={args.preconditioner}",
        f"--deflation-order={args.deflation_order}",
        f"--deflation-coarse-mode={args.deflation_coarse_mode}",
        f"--trust-subspace-option={args.trust_subspace_option}",
        f"--trust-num-leftmost={args.trust_num_leftmost}",
        f"--trust-num-previous-steps={args.trust_num_previous_steps}",
        f"--trust-work-quadrature={args.trust_work_quadrature}",
        f"--trust-num-lanczos={args.trust_num_lanczos}",
        f"--trust-num-lanczos-iters={args.trust_num_lanczos_iters}",
        f"--print-level={args.print_level}",
        f"--nonlinear-max-iterations={args.nonlinear_max_iterations}",
    ]
    for flag, value in (
        ("--cg-forcing-rel", args.cg_forcing_rel),
        ("--residual-growth-cap", args.residual_growth_cap),
        ("--tr-decrease-factor", args.tr_decrease_factor),
        ("--tr-increase-factor", args.tr_increase_factor),
        ("--tr-eta1", args.tr_eta1),
        ("--tr-eta2", args.tr_eta2),
        ("--tr-eta3", args.tr_eta3),
        ("--tr-eta4", args.tr_eta4),
        ("--mesh-scale", problem_mesh_scale(args, problem)),
    ):
        if value is not None:
            common.append(f"{flag}={value}")
    if args.cg_stagnation_tol is not None:
        common.append(f"--cg-stagnation-tol={args.cg_stagnation_tol}")
    if args.cg_stagnation_window is not None:
        common.append(f"--cg-stagnation-window={args.cg_stagnation_window}")
    if args.cg_eisenstat_walker:
        common.append("--cg-eisenstat-walker")
    if args.deflation_smoother is not None:
        common.append(f"--deflation-smoother={args.deflation_smoother}")
    if args.assemble_bsr:
        common.append("--assemble-bsr")

    if problem.name == "arch":
        arch = [
            f"--solver={args.nonlinear_solver}",
            f"--max-cg-iterations={args.max_cg_iterations}",
        ]
        if args.use_bsr_spmv:
            arch.append("--use-bsr-spmv")
        if args.use_exact_energy:
            arch.append("--use-exact-energy")
        return common + arch + args.extra_arch_arg

    hyper = [
        f"--case={problem.hyper_case}",
        f"--nonlinear-solver={args.nonlinear_solver}",
        f"--linear-solver={args.linear_solver}",
        f"--linear-max-iterations={args.max_cg_iterations}",
        f"--steps={args.hyper_steps if args.hyper_steps is not None else (problem.default_steps or 3)}",
    ]
    if args.use_bsr_spmv:
        hyper.append("--use-bsr-spmv")
    return common + hyper + args.extra_hyper_arg


def parse_timing_log(text: str) -> dict[str, float | int | str]:
    data: dict[str, float | int | str] = {}

    for label, field in TRUST_LABELS.items():
        values = [float(x) for x in re.findall(rf"^\s*{re.escape(label)}\s*:\s*([0-9.eE+-]+)\s+s", text, re.M)]
        if values:
            data[field] = sum(values)

    for label, field in DEFLATION_LABELS.items():
        values = [float(x) for x in re.findall(rf"^\s*{re.escape(label)}\s*:\s*([0-9.eE+-]+)\s+s", text, re.M)]
        if values:
            data[field] = sum(values)

    total_advance_walls = [float(x) for x in re.findall(r"Total advanceTimestep wall = ([0-9.eE+-]+) s", text)]
    if total_advance_walls:
        data["reported_step_wall_s"] = sum(total_advance_walls)
    else:
        step_walls = [float(x) for x in re.findall(r"step [0-9]+/[0-9]+ wall = ([0-9.eE+-]+) s", text)]
        if step_walls:
            data["reported_step_wall_s"] = sum(step_walls)

    outer_values = [int(x) for x in re.findall(r"CG iter histogram \(per outer; ([0-9]+) outers\)", text)]
    if outer_values:
        data["cg_outers"] = sum(outer_values)

    total_iter_values = [int(x) for x in re.findall(r"mean = [0-9.eE+-]+\s+max = [0-9]+\s+total = ([0-9]+)", text)]
    if total_iter_values:
        data["cg_total_iters"] = sum(total_iter_values)

    max_iter_values = [int(x) for x in re.findall(r"mean = [0-9.eE+-]+\s+max = ([0-9]+)\s+total =", text)]
    if max_iter_values:
        data["cg_max_iters"] = max(max_iter_values)

    cg_hit_max = [int(x) for x in re.findall(r"cg_hit_max\s*:\s*([0-9]+)", text)]
    if cg_hit_max:
        data["cg_hit_max"] = sum(cg_hit_max)

    retries = [int(x) for x in re.findall(r"line-search retries ---\s*\n\s*total = ([0-9]+)", text)]
    if retries:
        data["line_search_retries_total"] = sum(retries)

    wtaw_fail = [int(x) for x in re.findall(r"WtAW Cholesky failures:\s*([0-9]+)\s*/ rank", text)]
    if wtaw_fail:
        data["wtaw_cholesky_failures_per_rank"] = sum(wtaw_fail)

    wtaw_pp_fail = [int(x) for x in re.findall(r"WtAW_pp Cholesky failures:\s*([0-9]+)\s*/ rank", text)]
    if wtaw_pp_fail:
        data["wtaw_pp_cholesky_failures_per_rank"] = sum(wtaw_pp_fail)

    contact = re.findall(r"active area = ([0-9.eE+-]+), contact energy = ([0-9.eE+-]+)", text)
    if contact:
        data["contact_active_area"] = float(contact[-1][0])
        data["contact_energy"] = float(contact[-1][1])

    disp = re.findall(r"final displacement l2 = ([0-9.eE+-]+)", text)
    if disp:
        data["final_displacement_l2"] = float(disp[-1])

    return data


def run_problem(args: argparse.Namespace, problem: Problem, build_dir: Path, env: dict[str, str], out_dir: Path) -> dict:
    exe = build_dir / problem.executable
    log = out_dir / f"{problem.name}.log"
    cmd = [args.mpirun, "-np", str(args.procs), str(exe)] + solver_args(args, problem)
    start = time.monotonic()
    rc = run_command(cmd, build_dir, env, args.dry_run, log, args.timeout_sec)
    wall = time.monotonic() - start
    row: dict[str, float | int | str] = {
        "problem": problem.name,
        "status": "dry-run" if args.dry_run else ("ok" if rc == 0 else "failed"),
        "returncode": rc,
        "wall_s": wall,
        "log": str(log),
    }
    if log.exists():
        row.update(parse_timing_log(log.read_text(encoding="utf-8", errors="replace")))
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def top_bottlenecks(row: dict, limit: int = 5) -> str:
    keys = [
        "assemble_jacobian_s",
        "precond_setop_s",
        "hess_vec_total_s",
        "precond_mult_total_s",
        "batched_hess_vec_s",
        "subspace_prepare_s",
        "subspace_solve_s",
        "augment_direction_s",
        "in_cg_h_mult_s",
        "in_cg_p_mult_s",
        "in_cg_dot_many_s",
        "defl_assemble_wtaw_s",
        "defl_allreduce_s",
        "defl_mult_smoother_s",
        "defl_mult_coarse_s",
        "defl_allgather_s",
    ]
    items = [(key, float(row.get(key, 0.0))) for key in keys if row.get(key, "") != ""]
    items = sorted(items, key=lambda item: item[1], reverse=True)[:limit]
    return "; ".join(f"{key}={value:.3f}s" for key, value in items)


def write_markdown(rows: list[dict], path: Path, args: argparse.Namespace) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Hyperelastic Performance Suite\n\n")
        stream.write(f"- procs: {args.procs}\n")
        stream.write(f"- preconditioner: `{args.preconditioner}`\n")
        stream.write(f"- deflation order: `{args.deflation_order}`\n")
        stream.write(f"- deflation coarse mode: `{args.deflation_coarse_mode}`\n")
        stream.write(f"- leftmost / previous steps: {args.trust_num_leftmost} / {args.trust_num_previous_steps}\n\n")
        cols = [
            "problem",
            "status",
            "wall_s",
            "reported_step_wall_s",
            "trust_total_s",
            "hess_vec_total_s",
            "precond_mult_total_s",
            "in_cg_dot_many_s",
            "cg_total_iters",
            "cg_hit_max",
        ]
        stream.write("| " + " | ".join(cols) + " |\n")
        stream.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            stream.write("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |\n")
        stream.write("\n## Top Parsed Bottlenecks\n\n")
        for row in rows:
            stream.write(f"- `{row['problem']}`: {top_bottlenecks(row)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--procs", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dyld-library-path", default=None)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--problems", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument(
        "--update-references",
        action="store_true",
        help="Write final_displacement_l2 from this run into scripts/hyperelastic_references.json",
    )

    parser.add_argument("--nonlinear-solver", default="TrustRegion")
    parser.add_argument("--linear-solver", default="CG")
    parser.add_argument("--preconditioner", default="Deflation")
    parser.add_argument("--deflation-order", choices=("affine", "quadratic"), default="affine")
    parser.add_argument("--deflation-coarse-mode", choices=("global", "local", "schwarz"), default="global")
    parser.add_argument("--trust-subspace-option", type=int, default=2)
    parser.add_argument("--trust-num-leftmost", type=int, default=1)
    parser.add_argument("--trust-num-previous-steps", type=int, default=4)
    parser.add_argument("--trust-work-quadrature", type=int, default=2)
    parser.add_argument("--trust-num-lanczos", type=int, default=0)
    parser.add_argument("--trust-num-lanczos-iters", type=int, default=0)
    parser.add_argument("--max-cg-iterations", type=int, default=723)
    parser.add_argument("--cg-forcing-rel", type=float, default=1.512e-05)
    parser.add_argument("--residual-growth-cap", type=float, default=5.654)
    parser.add_argument("--tr-decrease-factor", type=float, default=0.437)
    parser.add_argument("--tr-increase-factor", type=float, default=1.786)
    parser.add_argument("--tr-eta1", type=float, default=None)
    parser.add_argument("--tr-eta2", type=float, default=0.1534)
    parser.add_argument("--tr-eta3", type=float, default=0.59996)
    parser.add_argument("--tr-eta4", type=float, default=3.408)
    parser.add_argument("--mesh-scale", type=float, default=None)
    parser.add_argument("--mesh-scale-factor", type=float, default=None,
                        help="Multiply each problem's default mesh scale (screening meshes)")
    parser.add_argument("--references-file", type=Path, default=None,
                        help="Alternate references JSON (e.g. screening-scale references)")
    parser.add_argument("--cg-stagnation-tol", type=float, default=0.000753)
    parser.add_argument("--cg-stagnation-window", type=int, default=2)
    parser.add_argument("--cg-eisenstat-walker", action="store_true")
    parser.add_argument("--use-bsr-spmv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deflation-smoother", choices=("hypre", "jacobi", "block"), default="jacobi")
    parser.add_argument("--assemble-bsr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-exact-energy", action="store_true")
    parser.add_argument("--nonlinear-max-iterations", type=int, default=300000)
    parser.add_argument("--hyper-steps", type=int, default=None,
                        help="Override load steps for all hyper cases (default: per-problem, twist=6, others=3)")
    parser.add_argument("--print-level", type=int, default=1)
    parser.add_argument("--extra-arch-arg", action="append", default=[])
    parser.add_argument("--extra-hyper-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    build_dir = args.build_dir or default_build_dir(root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or (root / "performance_runs" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    dyld = args.dyld_library_path if args.dyld_library_path is not None else discover_dyld_path(root)
    append_env_path(env, "DYLD_LIBRARY_PATH", dyld)

    if dyld:
        print(f"DYLD_LIBRARY_PATH prefix: {dyld}", flush=True)
    print(f"build dir: {build_dir}", flush=True)
    print(f"output dir: {out_dir}", flush=True)

    if not args.skip_build:
        build_targets(build_dir, args.jobs, args.dry_run)

    rows = []
    for problem_name in args.problems:
        problem = PROBLEM_TABLE[problem_name]
        rows.append(run_problem(args, problem, build_dir, env, out_dir))

    # twist is bifurcation-prone, so its tolerance is looser; tighten once branch
    # selection is controlled.
    default_rel_tols = {"arch": 0.01, "block": 0.01, "contact": 0.01, "twist": 0.05}
    if args.update_references:
        update_references(rows, default_rel_tols, args)
    references = load_references(args)
    for row in rows:
        check_answer(row, references)

    write_csv(rows, out_dir / "summary.csv")
    write_markdown(rows, out_dir / "summary.md", args)
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    bad = [row for row in rows if row["status"] not in ("ok", "dry-run")]
    for row in bad:
        print(f"{row['problem']}: status={row['status']}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
