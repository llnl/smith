#!/usr/bin/env python3
"""Plot Deflation vs HypreAMG wall time across proc counts from precond-scaling runs."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "performance_runs" / "precond-scaling"
PROCS = (4, 6, 8)
PROBLEMS = ("arch", "block", "contact", "twist")


def load(tag: str, np_: int) -> dict[str, dict]:
    path = ROOT / f"{tag}-np{np_}" / "summary.csv"
    if not path.exists():
        return {}
    return {r["problem"]: r for r in csv.DictReader(path.open())}


def wall(row: dict | None) -> float | None:
    if not row or row.get("status") != "ok":
        return None
    v = row.get("reported_step_wall_s") or row.get("wall_s")
    return float(v) if v else None


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    speedup_rows = []
    for np_ in PROCS:
        defl = load("deflation", np_)
        amg = load("hypreamg", np_)
        speedups = {}
        for prob in PROBLEMS:
            wd, wa = wall(defl.get(prob)), wall(amg.get(prob))
            if wd and wa:
                speedups[prob] = wa / wd  # >1 means Deflation faster
        geom = (
            math.exp(sum(math.log(s) for s in speedups.values()) / len(speedups))
            if speedups
            else float("nan")
        )
        bad = [
            f"{prob}:{tbl.get(prob, {}).get('status', 'missing')}"
            for tbl, name in ((defl, "defl"), (amg, "amg"))
            for prob in PROBLEMS
            if tbl.get(prob, {}).get("status") != "ok"
        ]
        speedup_rows.append((np_, geom, speedups, bad))

    for prob in PROBLEMS:
        for tag, style in (("deflation", "-o"), ("hypreamg", "--s")):
            xs, ys = [], []
            for np_ in PROCS:
                w = wall(load(tag, np_).get(prob))
                if w:
                    xs.append(np_)
                    ys.append(w)
            if xs:
                axes[0].plot(xs, ys, style, label=f"{prob} ({tag})")
    axes[0].set_xlabel("MPI procs")
    axes[0].set_ylabel("wall s")
    axes[0].set_yscale("log")
    axes[0].set_title("Wall time per problem")
    axes[0].legend(fontsize=7)

    xs = [r[0] for r in speedup_rows]
    axes[1].plot(xs, [r[1] for r in speedup_rows], "-o", color="black", label="geom mean")
    for prob in PROBLEMS:
        ys = [r[2].get(prob, float("nan")) for r in speedup_rows]
        axes[1].plot(xs, ys, "--", alpha=0.6, label=prob)
    axes[1].axhline(1.0, color="gray", lw=0.8)
    axes[1].set_xlabel("MPI procs")
    axes[1].set_ylabel("HypreAMG wall / Deflation wall")
    axes[1].set_title("Deflation speedup over HypreAMG (>1 = Deflation faster)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = ROOT / "precond_scaling.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    for np_, geom, speedups, bad in speedup_rows:
        print(f"np={np_} geom_speedup={geom:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in speedups.items())
              + (f"  excluded: {', '.join(bad)}" if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
