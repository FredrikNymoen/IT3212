"""Render report figures using the blob notebook's exact implementation.

Run with the project Python after executing 05_blob_detection.ipynb.
Only two display images are recomputed; dataset-wide results are read, not changed.
"""
import ast
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/vehicle-type-detection"
RESULT_DIR = ROOT / "results/assignment2/blob_detection"
FIGURE_DIR = ROOT / "figures/assignment2/blob_detection"
REPORT_IMAGES = ROOT / "reports/assignment2/Images"
MAX_SIDE = 512
COLORS = {"bright": "#ffdc00", "dark": "#00e5ff"}


def notebook_functions():
    """Load only required declarations; do not execute notebook analysis cells."""
    notebook = json.loads((Path(__file__).parent / "05_blob_detection.ipynb").read_text(encoding="utf-8"))
    functions = {"load_image", "circle_overlap_fraction", "overlaps_any", "detect_blobs"}
    constants = {"BLOB_COLUMNS", "BASELINE"}
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
    exec(compile(ast.Module(body=declarations, type_ignores=[]), "blob_notebook_declarations", "exec"), namespace)
    assert functions | constants <= namespace.keys()
    return namespace


def draw(ax, record, blobs=None, title="", label_first_three=False):
    ax.imshow(record["rgb"])
    if blobs is not None:
        for index, row in enumerate(blobs.itertuples(index=False), 1):
            color = COLORS[row.polarity]
            ax.add_patch(Circle((row.x_px, row.y_px), row.radius_px,
                                fill=False, color=color, linewidth=1.0))
            if label_first_three and index <= 3:
                ax.text(row.x_px, row.y_px, str(index), fontsize=9, color="white",
                        ha="center", va="center",
                        bbox=dict(facecolor="black", alpha=.85, edgecolor="none", pad=1.2))
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-.5, record["width"] - .5)
    ax.set_ylim(record["height"] - .5, -.5)
    ax.axis("off")


def save(fig, name):
    fig.legend(handles=[Line2D([], [], color=color, lw=2, label=label.capitalize())
                        for label, color in COLORS.items()], loc="lower center", ncol=2,
               frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, 1), pad=.9, h_pad=3.5, w_pad=1.2)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)
    destination = FIGURE_DIR / name
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(destination, REPORT_IMAGES / name)


def main():
    implementation = notebook_functions()
    baseline = implementation["BASELINE"]
    examples = ["pickup/PIC_0.jpg", "sedan/PIC_0.jpg"]
    manifest = json.loads((Path(__file__).parent / "detection_images.json").read_text())
    assert set(examples) <= set(manifest)
    settings = [("lower threshold", .075), ("baseline", .15), ("higher threshold", .225)]
    exported_blobs = pd.read_csv(RESULT_DIR / "baseline_blobs.csv")
    experiments = pd.read_csv(RESULT_DIR / "parameter_experiments.csv")
    outputs, records = {}, {}
    for image_id in examples:
        record = records[image_id] = implementation["load_image"](image_id)
        cache = {}
        for setting, threshold in settings:
            blobs = implementation["detect_blobs"](
                record["gray"], **{**baseline, "threshold": threshold}, response_cache=cache)
            expected = experiments[(experiments.image == image_id) & (experiments.setting == setting)]
            assert len(expected) == 1 and len(blobs) == int(expected.iloc[0].n_blobs)
            if setting == "baseline":
                exported = exported_blobs[exported_blobs.image == image_id][blobs.columns].reset_index(drop=True)
                pd.testing.assert_frame_equal(blobs, exported, check_exact=False, rtol=1e-10, atol=1e-10)
            outputs[image_id, setting] = blobs
            print(image_id, setting, len(blobs))

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4))
    for row, image_id in enumerate(examples):
        draw(axes[row, 0], records[image_id], title=f"{image_id} | original")
        blobs = outputs[image_id, "baseline"]
        draw(axes[row, 1], records[image_id], blobs, f"Baseline | {len(blobs)} blobs",
             label_first_three=(row == 0))
    save(fig, "blob_report_baseline.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for row, image_id in enumerate(examples):
        for col, (setting, threshold) in enumerate(settings):
            blobs = outputs[image_id, setting]
            draw(axes[row, col], records[image_id], blobs,
                 f"{image_id.split('/')[0]} | threshold {threshold:g}\n{len(blobs)} blobs")
    save(fig, "blob_report_thresholds.png")

    summary = pd.read_csv(RESULT_DIR / "baseline_summary.csv")
    print("Baseline per-image quartiles:", summary.n_blobs.quantile([.25, .5, .75]).to_dict())
    counts = experiments.pivot(index="image", columns="setting", values="n_blobs")
    assert len(counts) == 1310 and counts.notna().all().all()
    for setting in ["lower threshold", "higher threshold"]:
        difference = counts[setting] - counts.baseline
        print(setting, "fewer/equal/more", int((difference < 0).sum()),
              int((difference == 0).sum()), int((difference > 0).sum()),
              "median count", counts[setting].median())
    print("All six displayed results match the existing experiment exports.")


if __name__ == "__main__":
    main()
