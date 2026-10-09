# Detection sections for Overleaf

All prose, equations, tables, and figure environments are inline in two files:

- `blob_detection.tex`: paste into `Sections/Blob_detection.tex`.
- `contour_detection.tex`: paste into `Sections/Contour_detection.tex`.

Upload these four PNGs from `Images/` into the Overleaf project's `Images/` folder:

- `blob_report_baseline.png`
- `blob_report_thresholds.png`
- `blob_contour_window_comparison.png`
- `contour_parameters_pickup_PIC_0.png`

The matched comparison replaces the earlier `sedan_PIC_0_contours.png` in the report.
It shows both methods on the same sedan, with the rear-window region marked by identical dashed boxes.
The boxes are visual guides, not ground-truth boundaries or an accuracy assessment.

Load the standard packages `graphicx`, `booktabs`, `amsmath`, `placeins`, and `float`.
Figure paths are literal `Images/filename.png`, relative to `main.tex`. Include the sections with:

```tex
\input{Sections/Blob_detection.tex}
\input{Sections/Contour_detection.tex}
```

There are no custom path macros or external table inputs. Headings use `\subsubsection{}` in the
assignment's question order; figures use `[H]`. Compile twice to resolve references. The sections
occur late in the full report and avoid repeating the general dataset introduction. Keep them concise,
with a target of 7-8 pages together in the user's Overleaf layout; local preview pagination can differ.

## Analysis and display scope

Both detectors' baselines, all parameter experiments, and their matched comparison cover all
1,310 discovered images. The ten paths in `detection_images.json` are display examples only, selected
as the first two alphabetically sorted filenames per class. The report's blob image table is an excerpt;
the contour table summarizes all images by class. Detailed per-image and per-contour statistics remain
in CSV exports. Areas and perimeters are measurements of mask boundaries, not vehicle sizes or accuracy.
The dataset was downloaded from Canvas.

Run `06_contour_detection.ipynb` from its assignment directory in a fresh kernel. It independently
recomputes LoG on every image and checks against prior blob exports when those are available.
Full contour outputs under `results/assignment2/contour_detection/` include:

- `baseline_summary.csv`: counts, exclusions, and mean area/perimeter for every image, including zeros.
- `baseline_contours.csv`: individual contour areas, perimeters, centroids, and vertex counts.
- `baseline_vertices/<class>/<image>.json.gz`: complete retained baseline boundary coordinates.
- `class_summary.csv`: full-dataset baseline aggregates by class.
- `parameter_experiments.csv` and `parameter_aggregate.csv`: per-image and aggregate results for all six settings.
- `contours_<setting>.csv`: individual measurements for each non-baseline parameter setting.
- `blob_contour_comparison.csv`: matched full-dataset statistics for the two detectors.

Per-loop means are weighted over individual retained contours, not over per-image means. Empty-image
means are undefined. Inner and outer loops are retained separately, so enclosed areas must not be summed
as net foreground area. Update inline report values and discussion when experiments change.

## Regenerate report figures

After executing the notebooks, run from the repository root:

```powershell
.venv/Scripts/python.exe notebooks/assignment2/report_blob_figures.py
.venv/Scripts/python.exe notebooks/assignment2/report_contour_figures.py
```

Both scripts load required function declarations from the notebooks, recompute only illustrated inputs,
and check results against CSV exports. They copy the report PNGs to `Images/`. The contour script also
copies the notebook's pickup parameter figure. Neither script modifies dataset-wide measurements.
The algorithm schematic is not included in the report.
