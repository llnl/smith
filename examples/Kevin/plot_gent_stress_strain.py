#!/usr/bin/env python3

"""Plot Gent tension stress-strain CSV output from Smith build directories."""

import argparse
import csv
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from error


CURVE_LABELS = {
    "gent_planar_tension_stress_strain": "Planar tension",
    "gent_dumbbell_tension_stress_strain": "Dumbbell tension",
}


def read_curve(csv_path: Path) -> tuple[list[float], list[float]]:
    """Read engineering strain and nominal stress columns from one CSV file."""
    strain = []
    stress = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = csv.DictReader(csv_file)
        required_columns = {"engineering_strain", "nominal_stress"}
        if rows.fieldnames is None or not required_columns.issubset(rows.fieldnames):
            raise ValueError(f"{csv_path} does not contain {sorted(required_columns)}")

        for row in rows:
            strain.append(float(row["engineering_strain"]))
            stress.append(float(row["nominal_stress"]))

    return strain, stress


def plot_build(examples_directory: Path, output_name: str) -> Path | None:
    """Plot all Gent stress-strain curves found in one build's examples directory."""
    csv_paths = sorted(examples_directory.glob("gent_*_tension_stress_strain.csv"))
    if not csv_paths:
        return None

    figure, axes = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for csv_path in csv_paths:
        strain, stress = read_curve(csv_path)
        label = CURVE_LABELS.get(csv_path.stem, csv_path.stem.replace("_", " ").title())
        axes.plot(strain, stress, marker="o", linewidth=2, markersize=4, label=label)

    axes.set_xlabel("Engineering strain")
    axes.set_ylabel("Nominal stress")
    axes.set_title("Gent material stress-strain response")
    axes.grid(True, alpha=0.3)
    axes.legend()

    output_path = examples_directory / output_name
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-name",
        default="gent_stress_strain.png",
        help="figure filename created in each build examples directory",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    examples_directories = sorted(path for path in repository_root.glob("build*/examples") if path.is_dir())

    output_paths = []
    for examples_directory in examples_directories:
        output_path = plot_build(examples_directory, args.output_name)
        if output_path is not None:
            output_paths.append(output_path)
            print(f"Wrote {output_path}")

    if not output_paths:
        raise SystemExit(f"No Gent stress-strain CSV files found under {repository_root}/build*/examples")


if __name__ == "__main__":
    main()
