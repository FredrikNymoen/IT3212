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
    image_id = "pickup/PIC_1.jpg"
    record = blob_implementation["load_image"](image_id)
    blobs = blob_implementation["detect_blobs"](record["gray"], **blob_implementation["BASELINE"])
    contours = contour_implementation["detect_contours"](record["gray"], **contour_implementation["CONTOUR_BASELINE"])

    exported = pd.read_csv(RESULT_DIR / "baseline_contours.csv")
    expected = exported.loc[exported.image == image_id, contours["statistics"].columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(contours["statistics"], expected, check_dtype=False, rtol=1e-10, atol=1e-10)
    exported_blobs = pd.read_csv(ROOT / "results/assignment2/blob_detection/baseline_blobs.csv")
    expected_blobs = exported_blobs.loc[exported_blobs.image == image_id, blobs.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(blobs, expected_blobs, check_dtype=False, rtol=1e-10, atol=1e-10)
    comparison = pd.read_csv(RESULT_DIR / "blob_contour_comparison.csv").set_index("image")
    assert len(blobs) == comparison.loc[image_id, "n_blobs"]
    assert len(contours["contours"]) == comparison.loc[image_id, "n_contours"]

    def save_report_figure(fig, name):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_IMAGES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        shutil.copy2(FIGURE_DIR / name, REPORT_IMAGES / name)

    # Task 2: a single original-image overlay showing only colored contours.
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    contour_implementation["draw_contours"](ax, record, contours, "", annotate=False)
    fig.tight_layout(pad=.2)
    save_report_figure(fig, "contour_report_baseline.png")

    # Task 4: identical zoomed views, without resizing/re-running the detectors.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    report_blob_figures.draw(axes[0], record, blobs, "LoG blobs")
    contour_implementation["draw_contours"](axes[1], record, contours, "Closed contours")
    for ax in axes:
        ax.set_xlim(130, record["width"] - .5)
        ax.set_ylim(151, 10)
        # These manually chosen boxes are common visual guides only.
        for left, top, width, height, label in [
            (140, 19, 183, 64, "Windshield frame"),
            (175, 103, 70, 40, "Headlamp"),
        ]:
            ax.add_patch(Rectangle((left, top), width, height, fill=False,
                                   edgecolor="white", linestyle="--", linewidth=1.2))
            label_y = top - 2 if label == "Windshield frame" else top + height + 2
            ax.text(left, label_y, label, color="white", fontsize=9,
                    va="bottom" if label == "Windshield frame" else "top",
                    bbox=dict(facecolor="black", alpha=.8, edgecolor="none", pad=1.0))
    fig.legend(handles=[Line2D([], [], color=color, lw=2, label=label.capitalize())
                        for label, color in report_blob_figures.COLORS.items()],
               loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, 1), pad=.5, w_pad=1.5)
    save_report_figure(fig, "blob_contour_detail_comparison.png")
    parameter_name = "contour_parameters_pickup_PIC_0.png"
    shutil.copy2(FIGURE_DIR / "parameters_pickup_PIC_0.png", REPORT_IMAGES / parameter_name)
    print(f"Verified report comparison: {image_id}, {len(blobs)} blobs and {len(contours['contours'])} contours.")


if __name__ == "__main__":
    main()
