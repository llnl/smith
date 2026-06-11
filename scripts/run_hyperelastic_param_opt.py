#!/usr/bin/env python3
"""Black-box parameter optimization (Track B) for the hyperelastic suite.

Differential evolution over solver parameters, scoring each candidate by the
geometric mean of per-problem wall time with hard multiplicative penalties for
failed / wrong-answer / no-answer runs (score = geomean * 4^n_bad). Evaluations
run the regular suite script on reduced "screening" meshes (--mesh-scale-factor)
against screening-scale references; promote winners with --confirm afterwards.

Usage:
  # screening search (run on an otherwise idle machine; each eval ~30-60 s)
  python3 scripts/run_hyperelastic_param_opt.py --generations 12 --population 12

  # confirm the top screening configs at full size
  python3 scripts/run_hyperelastic_param_opt.py --confirm 5 \
      --log performance_runs/paramopt-<ts>/evals.jsonl

The baseline configuration is always member 0 of the initial population, so the
search can only improve on it. All evals are appended to evals.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "scripts" / "run_hyperelastic_suite.py"
SCREEN_REFS = REPO / "scripts" / "hyperelastic_references_screening.json"

PROBLEMS = ("arch", "block", "contact", "twist")

# name: (kind, low, high, baseline)   kind: lin | log | int | cat
PARAM_SPACE = {
    "cg_stagnation_tol": ("log", 10**-4.5, 10**-1.5, 1e-3),
    "cg_stagnation_window": ("int", 2, 12, 5),
    "trust_num_leftmost": ("int", 0, 4, 2),
    "trust_num_previous_steps": ("int", 0, 8, 4),
    "max_cg_iterations": ("int", 100, 3000, 1000),
    "cg_forcing_rel": ("log", 1e-6, 1e-3, 5e-5),
    "residual_growth_cap": ("lin", 1.2, 10.0, 3.0),
    "tr_decrease_factor": ("lin", 0.1, 0.5, 0.25),
    "tr_increase_factor": ("lin", 1.2, 3.5, 1.75),
    "tr_eta2": ("lin", 0.01, 0.3, 0.1),
    "tr_eta3": ("lin", 0.3, 0.95, 0.6),
    "tr_eta4": ("lin", 1.5, 10.0, 4.2),
    "deflation_smoother": ("cat", ("jacobi", "block"), None, "jacobi"),
}


def to_unit(name: str, value):
    kind, lo, hi, _ = PARAM_SPACE[name]
    if kind == "cat":
        choices = lo
        return choices.index(value) / max(1, len(choices) - 1) if len(choices) > 1 else 0.0
    if kind == "log":
        return (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return (value - lo) / (hi - lo)


def from_unit(name: str, u: float):
    kind, lo, hi, _ = PARAM_SPACE[name]
    u = min(1.0, max(0.0, u))
    if kind == "cat":
        choices = lo
        idx = min(len(choices) - 1, int(u * len(choices)))
        return choices[idx]
    if kind == "log":
        return 10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo)))
    value = lo + u * (hi - lo)
    return int(round(value)) if kind == "int" else value


def repair(params: dict) -> dict:
    # TR acceptance thresholds must be ordered
    if params["tr_eta2"] >= params["tr_eta3"]:
        params["tr_eta2"], params["tr_eta3"] = (
            min(params["tr_eta2"], params["tr_eta3"]) * 0.9,
            max(params["tr_eta2"], params["tr_eta3"]),
        )
    if params["tr_increase_factor"] <= 1.0:
        params["tr_increase_factor"] = 1.2
    return params


def baseline_params() -> dict:
    return {name: spec[3] for name, spec in PARAM_SPACE.items()}


def suite_command(params: dict, args, out_dir: Path, screening: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(SUITE),
        "--skip-build",
        "--use-bsr-spmv",
        "--assemble-bsr",
        f"--procs={args.np}",
        f"--timeout-sec={args.timeout_sec}",
        f"--output-dir={out_dir}",
        f"--cg-stagnation-tol={params['cg_stagnation_tol']:.6g}",
        f"--cg-stagnation-window={params['cg_stagnation_window']}",
        f"--trust-num-leftmost={params['trust_num_leftmost']}",
        f"--trust-num-previous-steps={params['trust_num_previous_steps']}",
        f"--max-cg-iterations={params['max_cg_iterations']}",
        f"--cg-forcing-rel={params['cg_forcing_rel']:.6g}",
        f"--residual-growth-cap={params['residual_growth_cap']:.4g}",
        f"--tr-decrease-factor={params['tr_decrease_factor']:.4g}",
        f"--tr-increase-factor={params['tr_increase_factor']:.4g}",
        f"--tr-eta2={params['tr_eta2']:.4g}",
        f"--tr-eta3={params['tr_eta3']:.4g}",
        f"--tr-eta4={params['tr_eta4']:.4g}",
        f"--deflation-smoother={params['deflation_smoother']}",
    ]
    if screening:
        cmd += [f"--mesh-scale-factor={args.mesh_scale_factor}", f"--references-file={SCREEN_REFS}"]
    return cmd


def evaluate(params: dict, args, out_dir: Path, screening: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = suite_command(params, args, out_dir, screening)
    subprocess.run(cmd, cwd=REPO, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    walls, statuses = {}, {}
    summary = out_dir / "summary.csv"
    if summary.exists():
        with summary.open() as stream:
            for row in csv.DictReader(stream):
                statuses[row["problem"]] = row["status"]
                try:
                    walls[row["problem"]] = float(row["wall_s"])
                except ValueError:
                    walls[row["problem"]] = float(args.timeout_sec)
    n_bad = sum(1 for p in PROBLEMS if statuses.get(p) != "ok")
    geomean = math.exp(
        sum(math.log(max(1e-3, walls.get(p, float(args.timeout_sec)))) for p in PROBLEMS) / len(PROBLEMS)
    )
    score = geomean * (4.0**n_bad)
    return {"score": score, "geomean": geomean, "n_bad": n_bad, "walls": walls, "statuses": statuses}


def log_eval(log_path: Path, tag: str, params: dict, result: dict, out_dir: Path) -> None:
    record = {"tag": tag, "params": params, **result, "dir": str(out_dir), "time": datetime.now().isoformat()}
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    walls = " ".join(f"{p}={result['walls'].get(p, float('nan')):.1f}" for p in PROBLEMS)
    print(f"[{tag}] score={result['score']:.2f} bad={result['n_bad']} {walls}", flush=True)


def run_search(args) -> None:
    rng = random.Random(args.seed)
    names = list(PARAM_SPACE)
    dim = len(names)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = REPO / "performance_runs" / f"paramopt-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "evals.jsonl"
    print(f"optimization root: {root}")

    # initial population: baseline + perturbations
    population: list[list[float]] = []
    base_u = [to_unit(name, baseline_params()[name]) for name in names]
    population.append(base_u)
    while len(population) < args.population:
        population.append([min(1.0, max(0.0, u + rng.gauss(0.0, 0.15))) for u in base_u])

    scores: list[float] = []
    eval_count = 0
    for i, member in enumerate(population):
        params = repair({name: from_unit(name, u) for name, u in zip(names, member)})
        result = evaluate(params, args, root / f"eval-{eval_count:04d}", screening=True)
        log_eval(log_path, f"gen0/{i}", params, result, root / f"eval-{eval_count:04d}")
        scores.append(result["score"])
        eval_count += 1

    F, CR = 0.6, 0.85
    for gen in range(1, args.generations + 1):
        for i in range(args.population):
            a, b, c = rng.sample([j for j in range(args.population) if j != i], 3)
            trial = list(population[i])
            j_rand = rng.randrange(dim)
            for j in range(dim):
                if j == j_rand or rng.random() < CR:
                    trial[j] = min(1.0, max(0.0, population[a][j] + F * (population[b][j] - population[c][j])))
            params = repair({name: from_unit(name, u) for name, u in zip(names, trial)})
            result = evaluate(params, args, root / f"eval-{eval_count:04d}", screening=True)
            log_eval(log_path, f"gen{gen}/{i}", params, result, root / f"eval-{eval_count:04d}")
            if result["score"] < scores[i]:
                population[i] = trial
                scores[i] = result["score"]
            eval_count += 1
        best = min(range(args.population), key=lambda k: scores[k])
        print(f"== gen {gen} best score {scores[best]:.2f} ==", flush=True)

    best = min(range(args.population), key=lambda k: scores[k])
    best_params = repair({name: from_unit(name, u) for name, u in zip(names, population[best])})
    print("best screening params:")
    print(json.dumps(best_params, indent=2))
    print(f"all evals: {log_path}; confirm with --confirm N --log {log_path}")


def run_confirm(args) -> None:
    records = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    screening = [r for r in records if r["tag"].startswith("gen")]
    # unique by params, best score first
    seen, candidates = set(), []
    for r in sorted(screening, key=lambda r: r["score"]):
        key = json.dumps(r["params"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(r)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.log).parent / f"confirm-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "evals.jsonl"
    confirm_args = argparse.Namespace(**vars(args))
    confirm_args.timeout_sec = args.confirm_timeout_sec
    # always include the baseline for a fair same-day comparison
    for i, candidate in enumerate([{"params": baseline_params(), "tag": "baseline"}] + candidates[: args.confirm]):
        result = evaluate(candidate["params"], confirm_args, root / f"confirm-{i:02d}", screening=False)
        log_eval(log_path, f"confirm/{candidate.get('tag', i)}", candidate["params"], result, root / f"confirm-{i:02d}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--np", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout-sec", type=int, default=120, help="per-problem timeout for screening evals")
    parser.add_argument("--mesh-scale-factor", type=float, default=0.6)
    parser.add_argument("--confirm", type=int, default=0, help="confirm top-N configs from --log at full size")
    parser.add_argument("--confirm-timeout-sec", type=int, default=240)
    parser.add_argument("--log", type=Path, default=None, help="evals.jsonl from a screening run (for --confirm)")
    args = parser.parse_args()

    if args.confirm:
        if not args.log:
            raise SystemExit("--confirm requires --log <evals.jsonl>")
        run_confirm(args)
        return 0
    if args.population < 4:
        raise SystemExit("--population must be >= 4 (DE mutation samples 3 distinct partners)")
    run_search(args)
    return 0


if __name__ == "__main__":
    main()
