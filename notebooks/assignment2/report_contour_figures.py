"""Render the matched report comparison from the notebooks and verify CSV results.

Run after executing 06_contour_detection.ipynb. Only the illustrated image is
recomputed; full-dataset measurements are read without modifying them.
"""
import ast
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from contourpy import contour_generator

import report_blob_figures

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/assignment2/contour_detection"
FIGURE_DIR = ROOT / "figures/assignment2/contour_detection"
REPORT_IMAGES = ROOT / "reports/assignment2/Images"


def notebook_functions():
    notebook = json.loads((Path(__file__).parent / "06_contour_detection.ipynb").read_text(encoding="utf-8"))
    functions = {"polygon_statistics", "detect_contours", "draw_contours"}
    constants = {"CONTOUR_COLUMNS", "CONTOUR_BASELINE"}
    declarations = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                declarations.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets
            ):
                declarations.append(node)
    namespace = dict(globals())
    exec(compile(ast.Module(body=declarations, type_ignores=[]), "contour_notebook_declarations", "exec"), namespace)
    assert functions | constants <= namespace.keys()
    return namespace


def main():
    blob_implementation = report_blob_figures.notebook_functions()
    contour_implementation = notebook_functions()
    image_id = "sedan/PIC_0.jpg"
    record = blob_implementation["load_image"](image_id)
    blobs = blob_implementation["detect_blobs"](record["gray"], **blob_implementation["BASELINE"])
    contours = contour_implementation["detect_contours"](record["gray"], **contour_implementation["CONTOUR_BASELINE"])

    exported = pd.read_csv(RESULT_DIR / "baseline_contours.csv")
    expected = exported.loc[exported.image == image_id, contours["statistics"].columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(contours["statistics"], expected, check_dtype=False, rtol=1e-10, atol=1e-10)
    comparison = pd.read_csv(RESULT_DIR / "blob_contour_comparison.csv").set_index("image")
    assert len(blobs) == comparison.loc[image_id, "n_blobs"]
    assert len(contours["contours"]) == comparison.loc[image_id, "n_contours"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    report_blob_figures.draw(axes[0], record, blobs, f"LoG blobs: {len(blobs)}")
    contour_implementation["draw_contours"](axes[1], record, contours, f"Closed contours: {len(contours['contours'])}")
    # Contour 3 encloses the illustrated rear-window region. The same box is a
    # visual guide in both panels, not a ground-truth boundary or a matching score.
    window = contours["contours"][2]
    left, top = window.min(axis=0) - 3
    right, bottom = window.max(axis=0) + 3
    for ax in axes:
        ax.add_patch(Rectangle((left, top), right - left, bottom - top,
                               fill=False, edgecolor="white", linestyle="--", linewidth=1.5))
        ax.text(left, top - 3, "Rear-window region", color="white", fontsize=9,
                va="bottom", bbox=dict(facecolor="black", alpha=.75, edgecolor="none", pad=1.5))
    fig.legend(handles=[Line2D([], [], color=color, lw=2, label=label.capitalize())
                        for label, color in report_blob_figures.COLORS.items()],
               loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1), pad=.8, w_pad=1.5)
    name = "blob_contour_window_comparison.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(FIGURE_DIR / name, REPORT_IMAGES / name)
    parameter_name = "contour_parameters_pickup_PIC_0.png"
    shutil.copy2(FIGURE_DIR / "parameters_pickup_PIC_0.png", REPORT_IMAGES / parameter_name)
    print(f"Verified report comparison: {image_id}, {len(blobs)} blobs and {len(contours['contours'])} contours.")


if __name__ == "__main__":
    main()
