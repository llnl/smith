#!/usr/bin/env python3
"""E* / s(n,M) study for piecewise-affine deflation (see performance_plan.md item 2b).

Grid over (mesh scale -> M, procs -> n, pieces -> s, polynomial order p in {1,2})
on the branch-stable block case (+ a small contact validation set). Records CG
iterations, outers, walls, and elements-per-piece; the collapse hypothesis is that
iters/outer is a function of M/(n*s) alone (per order). Combos with predicted
elements/piece < 16 are skipped (hang regime).
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "scripts" / "run_hyperelastic_suite.py"
OUT = REPO / "performance_runs" / "estar-study"
RESULTS = OUT / "results.csv"

# block base mesh 16x5x5 = 400 elements at scale 1.0 (scaled() rounds per axis).
# contact base 26x26x8 = 5408 at scale 1.0.
BASE_ELEMS = {"block": 400, "contact": 5408}

FIELDS = ["problem", "order", "scale", "np", "s", "elems", "el_per_piece", "status",
          "rel_err", "wall_s", "cg_total_iters", "cg_outers", "iters_per_outer",
          "defl_assemble_wtaw_s", "precond_setop_s", "in_cg_h_mult_s", "final_displacement_l2"]


def predicted_elems(problem: str, scale: float) -> int:
    nx = {"block": (16, 5, 5), "contact": (26, 26, 8)}[problem]
    from math import floor
    dims = [max(1, round(n * scale)) for n in nx]
    return dims[0] * dims[1] * dims[2]


def run_case(problem: str, order: int, scale: float, nproc: int, s: int, writer, stream) -> None:
    elems_pred = predicted_elems(problem, scale)
    epp = elems_pred / (nproc * s)
    if epp < 16:
        print(f"SKIP {problem} p{order} scale={scale} np={nproc} s={s} (predicted {epp:.0f} el/piece)", flush=True)
        return
    tag = f"{problem}-p{order}-m{scale}-np{nproc}-s{s}"
    out_dir = OUT / tag
    cmd = [sys.executable, str(SUITE), "--skip-build", "--problems", problem,
           f"--mesh-scale={scale}", f"--procs={nproc}", f"--deflation-pieces={s}",
           "--timeout-sec=420", f"--output-dir={out_dir}"]
    if order != 2:
        cmd.append(f"--extra-hyper-arg=--order={order}")
    subprocess.run(cmd, cwd=REPO, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    summary = out_dir / "summary.csv"
    row_out = {"problem": problem, "order": order, "scale": scale, "np": nproc, "s": s,
               "elems": elems_pred, "status": "missing"}
    if summary.exists():
        for row in csv.DictReader(summary.open()):
            if row["problem"] != problem:
                continue
            row_out["status"] = row["status"]
            row_out["rel_err"] = row.get("answer_rel_err", "")
            for k in ("wall_s", "cg_total_iters", "cg_outers", "defl_assemble_wtaw_s",
                      "precond_setop_s", "in_cg_h_mult_s", "final_displacement_l2"):
                row_out[k] = row.get(k, "")
            # true element count from the log when available
            log = out_dir / f"{problem}.log"
            if log.exists():
                import re
                m = re.search(r"global elements = (\d+)", log.read_text(errors="ignore"))
                if m:
                    row_out["elems"] = int(m.group(1))
            try:
                row_out["iters_per_outer"] = float(row["cg_total_iters"]) / max(1.0, float(row["cg_outers"]))
            except (ValueError, KeyError):
                pass
    row_out["el_per_piece"] = round(int(row_out["elems"]) / (nproc * s), 1)
    writer.writerow(row_out)
    stream.flush()
    print(f"{tag}: status={row_out['status']} wall={row_out.get('wall_s','')} "
          f"iters={row_out.get('cg_total_iters','')} outers={row_out.get('cg_outers','')} "
          f"el/pc={row_out['el_per_piece']}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        # p=2 main grid (M x n x s)
        for scale in (1.0, 1.3, 1.6):
            for nproc in (2, 4, 6):
                for s in (1, 2, 4, 8):
                    run_case("block", 2, scale, nproc, s, writer, stream)
        # larger-M extension at np=6
        for s in (1, 2, 4, 8):
            run_case("block", 2, 2.0, 6, s, writer, stream)
        # extra n point (np=8: walls noisy on this machine, iteration counts valid)
        for scale in (1.0, 1.6):
            for s in (1, 2, 4, 8):
                run_case("block", 2, scale, 8, s, writer, stream)
        # p=1 set (same meshes; tests whether E* lives in elements or dofs)
        for scale in (1.0, 1.6, 2.0):
            for nproc in (4, 6):
                for s in (1, 2, 4, 8):
                    run_case("block", 1, scale, nproc, s, writer, stream)
        # contact validation (held-out problem)
        for scale in (0.8, 1.1):
            for s in (1, 2, 4):
                run_case("contact", 2, scale, 6, s, writer, stream)
    print(f"results: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
