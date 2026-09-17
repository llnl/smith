#!/usr/bin/env pvpython

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


CASE_PRESETS = {
    "euler": {"camera_axis": "y"},
    "shallow_arch": {
        "color_mode": "component",
        "color_component": "Y",
        "color_array": "y_displacement",
        "scalar_bar_title": "Y displacement",
        "camera_time": "initial",
        "camera_tighten": 0.92,
    },
    "cylinder_crush_benchmark": {"camera_axis": "x"},
    "viscoelastic_buckling": {},
    "circ_in_circ": {"color_mode": "attribute"},
    "contact_arch": {},
    "sphere_into_corner": {},
}

TAG_TO_CASE = {
    "paper_euler_fast": "euler",
    "paper_shallow_arch_fast": "shallow_arch",
    "paper_cylinder_crush_fast": "cylinder_crush_benchmark",
    "paper_viscoelastic_buckling_fast": "viscoelastic_buckling",
    "paper_circ_in_circ_fast": "circ_in_circ",
    "paper_contact_arch_fast": "contact_arch",
    "paper_sphere_corner_fast": "sphere_into_corner",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render warped displacement-magnitude PNG snapshots from Smith ParaView output."
    )
    parser.add_argument("--input", required=True, help="Case output directory or cycle root.")
    parser.add_argument("--output", required=True, help="PNG output directory.")
    parser.add_argument("--case", default="", help="Case key for presets.")
    parser.add_argument("--cycles", default="", help="Comma-separated cycle numbers.")
    parser.add_argument("--cycle-step", type=int, default=1, help="Render every Nth cycle.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Render this many evenly spaced snapshots instead of every cycle.",
    )
    parser.add_argument("--width", type=int, default=1800, help="Image width.")
    parser.add_argument("--height", type=int, default=1200, help="Image height.")
    parser.add_argument(
        "--warp-scale",
        type=float,
        default=1.0,
        help="Displacement scale used for displayed coordinates. Default: 1 (physical deformation).",
    )
    parser.add_argument("--show-edges", action="store_true", help="Render element edges on the warped surface.")
    return parser.parse_args()


def find_cycle_root(path_str):
    path = Path(path_str).resolve()

    if path.is_file() and path.name == "data.pvtu":
        return path.parent.parent

    if list(path.glob("Cycle*/data.pvtu")):
        return path

    for child in sorted(path.iterdir()):
        if child.is_dir() and list(child.glob("Cycle*/data.pvtu")):
            return child

    raise FileNotFoundError(f"No Cycle*/data.pvtu files found under {path}")


def infer_case_key(case_arg, cycle_root):
    if case_arg:
        return case_arg

    for part in [cycle_root.name, cycle_root.parent.name]:
        if part in TAG_TO_CASE:
            return TAG_TO_CASE[part]

    lower = str(cycle_root).lower()
    for case_key in CASE_PRESETS:
        if case_key in lower:
            return case_key

    return "case"


def cycle_number(path):
    match = re.search(r"Cycle(\d+)", str(path))
    return int(match.group(1)) if match else -1


def parse_displacement_field(first_pvtu):
    root = ET.parse(first_pvtu).getroot()
    names = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag not in {"PDataArray", "DataArray"}:
            continue
        name = elem.attrib.get("Name", "")
        if not name:
            continue
        names.append(name)

    preferred = [
        name
        for name in names
        if name.endswith("_displacement") and not name.startswith("mesh_shape_") and not name.endswith("_dual")
    ]
    if not preferred:
        raise RuntimeError(f"Could not infer displacement field from {first_pvtu}")
    return preferred[0]


def needs_2d_warp_field(case_key):
    return case_key in {"shallow_arch", "circ_in_circ"}


def set_camera_from_axis(view, source, camera_axis):
    if not camera_axis:
        view.ResetCamera()
        return

    view.ResetCamera()

    data_info = source.GetDataInformation()
    bounds = data_info.GetBounds()
    if not bounds or len(bounds) != 6:
        return

    center = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ]
    extent = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )
    offset = 2.5 * extent

    if camera_axis == "x":
        position = [center[0] + offset, center[1], center[2]]
        view_up = [0.0, 0.0, 1.0]
    elif camera_axis == "y":
        position = [center[0], center[1] + offset, center[2]]
        view_up = [0.0, 0.0, 1.0]
    else:
        position = [center[0], center[1], center[2] + offset]
        view_up = [0.0, 1.0, 0.0]

    view.CameraPosition = position
    view.CameraFocalPoint = center
    view.CameraViewUp = view_up


def set_optional_property(proxy, name, value):
    try:
        setattr(proxy, name, value)
    except Exception:
        pass


def set_white_background(view):
    set_optional_property(view, "UseColorPaletteForBackground", 0)
    set_optional_property(view, "UseGradientBackground", 0)
    set_optional_property(view, "Background", [1.0, 1.0, 1.0])
    set_optional_property(view, "Background2", [1.0, 1.0, 1.0])


def tighten_camera(view, factor):
    if not factor or factor == 1.0:
        return
    try:
        view.CameraParallelScale *= factor
    except Exception:
        pass


def get_array_range(source, association, array_name):
    data_info = source.GetDataInformation()
    if association == "CELLS":
        array_info = data_info.GetCellDataInformation().GetArrayInformation(array_name)
    else:
        array_info = data_info.GetPointDataInformation().GetArrayInformation(array_name)
    if array_info is None:
        raise RuntimeError(f"Could not find {association} array '{array_name}'")
    return array_info.GetComponentRange(0)


def configure_displacement_lut(lut, scalar_range):
    scalar_min, scalar_max = scalar_range
    if scalar_max <= scalar_min:
        scalar_max = scalar_min + 1.0
    scalar_mid = 0.5 * (scalar_min + scalar_max)
    lut.RGBPoints = [
        scalar_min, 1.0, 1.0, 1.0,
        scalar_mid, 0.992, 0.682, 0.380,
        scalar_max, 0.647, 0.000, 0.149,
    ]
    lut.ColorSpace = "RGB"
    lut.NanColor = [1.0, 1.0, 1.0]


def configure_component_lut(lut, scalar_range):
    scalar_min, scalar_max = scalar_range
    if scalar_max <= scalar_min:
        scalar_max = scalar_min + 1.0
    if scalar_min < 0.0 < scalar_max:
        lut.RGBPoints = [
            scalar_min, 0.231, 0.298, 0.753,
            0.0, 1.0, 1.0, 1.0,
            scalar_max, 0.706, 0.016, 0.150,
        ]
    elif scalar_max <= 0.0:
        lut.RGBPoints = [
            scalar_min, 0.231, 0.298, 0.753,
            scalar_max, 1.0, 1.0, 1.0,
        ]
    else:
        lut.RGBPoints = [
            scalar_min, 1.0, 1.0, 1.0,
            scalar_max, 0.706, 0.016, 0.150,
        ]
    lut.ColorSpace = "RGB"
    lut.NanColor = [1.0, 1.0, 1.0]


def configure_attribute_lut(lut):
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = [
        "1", "outer ring",
        "2", "jelly 1",
        "3", "middle ring",
        "4", "jelly 2",
        "5", "center",
    ]
    lut.IndexedColors = [
        0.137, 0.306, 0.592,
        0.741, 0.812, 0.337,
        0.875, 0.486, 0.173,
        0.325, 0.573, 0.349,
        0.702, 0.208, 0.192,
    ]
    lut.IndexedOpacities = [1.0] * 5


def select_cycle_files(all_files, args):
    if not all_files:
        return []

    if args.cycles:
        requested = {int(token.strip()) for token in args.cycles.split(",") if token.strip()}
        return [path for path in all_files if cycle_number(path) in requested]

    if args.sample_count and args.sample_count > 0:
        if args.sample_count == 1:
            return [all_files[-1]]
        count = min(args.sample_count, len(all_files))
        indices = {
            round(i * (len(all_files) - 1) / (count - 1))
            for i in range(count)
        }
        return [all_files[i] for i in sorted(indices)]

    step = max(1, args.cycle_step)
    selected = all_files[::step]
    if selected[-1] != all_files[-1]:
        selected.append(all_files[-1])
    return selected


def main():
    args = parse_args()
    cycle_root = find_cycle_root(args.input)
    case_key = infer_case_key(args.case, cycle_root)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cycle_files = sorted(cycle_root.glob("Cycle*/data.pvtu"), key=cycle_number)
    selected_files = select_cycle_files(cycle_files, args)
    if not selected_files:
        raise RuntimeError("No cycle files selected for rendering")

    displacement_field = parse_displacement_field(selected_files[0])
    case_preset = CASE_PRESETS.get(case_key, {})
    warp_scale = args.warp_scale
    color_mode = case_preset.get("color_mode", "displacement")

    try:
        from paraview.simple import (
            Calculator,
            ColorBy,
            GetActiveViewOrCreate,
            GetAnimationScene,
            GetColorTransferFunction,
            GetScalarBar,
            Hide,
            OpenDataFile,
            Render,
            ResetSession,
            SaveScreenshot,
            Show,
            UpdatePipeline,
            WarpByVector,
            _DisableFirstRenderCameraReset,
        )
    except ImportError as exc:
        raise RuntimeError("Run this script with pvpython") from exc

    ResetSession()
    _DisableFirstRenderCameraReset()

    reader = OpenDataFile([str(path) for path in selected_files])
    UpdatePipeline()

    warp_input = reader
    warp_vector_field = displacement_field
    if needs_2d_warp_field(case_key):
        warp_vector = Calculator(Input=reader)
        warp_vector.ResultArrayName = "warp_displacement"
        warp_vector.Function = f"{displacement_field}_X*iHat + {displacement_field}_Y*jHat"
        warp_input = warp_vector
        warp_vector_field = "warp_displacement"

    calculator = Calculator(Input=warp_input)
    if color_mode == "component":
        calculator.ResultArrayName = case_preset.get("color_array", "displacement_component")
        calculator.Function = f"{displacement_field}_{case_preset.get('color_component', 'Y')}"
    else:
        calculator.ResultArrayName = "displacement_magnitude"
        calculator.Function = f'mag("{warp_vector_field}")'

    warp = WarpByVector(Input=calculator)
    warp.Vectors = ["POINTS", warp_vector_field]
    warp.ScaleFactor = warp_scale

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [args.width, args.height]
    set_white_background(view)

    display = Show(warp, view)
    Hide(reader, view)
    display.Representation = "Surface With Edges" if args.show_edges else "Surface"
    if args.show_edges:
        display.EdgeColor = [0.0, 0.0, 0.0]
    if color_mode == "attribute":
        color_assoc = "CELLS"
        color_array = "attribute"
        scalar_bar_title = "Block"
        ColorBy(display, (color_assoc, color_array))
    else:
        color_assoc = "POINTS"
        color_array = calculator.ResultArrayName
        scalar_bar_title = case_preset.get("scalar_bar_title", "Displacement magnitude")
        ColorBy(display, (color_assoc, color_array))
    display.SetScalarBarVisibility(view, True)

    lut = GetColorTransferFunction(color_array)
    scalar_bar = GetScalarBar(lut, view)
    scalar_bar.Title = scalar_bar_title
    scalar_bar.ComponentTitle = ""
    scalar_bar.TitleColor = [0.0, 0.0, 0.0]
    scalar_bar.LabelColor = [0.0, 0.0, 0.0]
    scalar_bar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    time_values = list(scene.TimeKeeper.TimestepValues) if hasattr(scene.TimeKeeper, "TimestepValues") else []
    if time_values:
        scalar_ranges = []
        for time_value in time_values:
            scene.AnimationTime = time_value
            UpdatePipeline(time_value, warp)
            scalar_ranges.append(get_array_range(warp, color_assoc, color_array))
        scalar_range = (
            min(value[0] for value in scalar_ranges),
            max(value[1] for value in scalar_ranges),
        )
    else:
        UpdatePipeline(proxy=warp)
        scalar_range = get_array_range(warp, color_assoc, color_array)
    if color_mode == "attribute":
        configure_attribute_lut(lut)
        lut.RescaleTransferFunction(*scalar_range)
    elif color_mode == "component":
        lut.RescaleTransferFunction(*scalar_range)
        configure_component_lut(lut, scalar_range)
    else:
        lut.RescaleTransferFunction(*scalar_range)
        configure_displacement_lut(lut, scalar_range)

    if case_preset.get("camera_time") == "initial" and time_values:
        scene.AnimationTime = time_values[0]
        UpdatePipeline(time_values[0], warp)
    set_camera_from_axis(view, warp, case_preset.get("camera_axis", ""))
    tighten_camera(view, case_preset.get("camera_tighten", 1.0))

    for index, path in enumerate(selected_files):
        if time_values:
            scene.AnimationTime = time_values[min(index, len(time_values) - 1)]
            UpdatePipeline(scene.AnimationTime, warp)
        else:
            UpdatePipeline(index, warp)
        Render()
        png_name = f"{case_key}_cycle{cycle_number(path):06d}.png"
        SaveScreenshot(str(output_dir / png_name), view)
        print(f"Wrote {output_dir / png_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
