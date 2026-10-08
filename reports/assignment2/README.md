# Detection sections for Overleaf

All text, equations, tables, and figure environments are contained in two files:

- `blob_detection.tex`: paste into `Sections/Blob_detection.tex`.
- `contour_detection.tex`: paste into `Sections/Contour_detection.tex`.

Upload the four PNGs from this folder's `Images/` directly into your Overleaf project's `Images/` folder. Paths are literal `Images/filename.png`, relative to `main.tex`. Parameter figures have distinct filenames to avoid collisions.

Ensure these standard packages are loaded in `packages.sty` or your preamble:

```tex
\usepackage{graphicx,booktabs,amsmath,placeins,float}
```

Include the sections from your main document or existing image-processing section:

```tex
\input{Sections/Blob_detection.tex}
\input{Sections/Contour_detection.tex}
```

There are no custom path commands or external table files. Each section contains its own image-statistics table and introduces its own analysis scope. Compile twice to resolve references.

The two contour figures are notebook outputs. The two blob figures are regenerated from the exact notebook implementation and checked against the existing CSV exports. The algorithm schematic is not included in the report. Blob results cover 1,310 images; contour results and the matched comparison cover 10. The user confirmed that vehicle-type-detection was downloaded from Canvas. These sections appear late in the full report and begin with methods and results without repeating the general dataset introduction. Update inline tables and discussion if experiments change.

Headings use numbered `\subsubsection{}` in the order of the assignment questions. All figures use `[H]`, which requires the standard `float` package. The text uses impersonal reporting of methods, observations, and limitations.

## Current figure files

Upload these four files from `Images/`:

- `blob_report_baseline.png`
- `blob_report_thresholds.png`
- `sedan_PIC_0_contours.png`
- `contour_parameters_pickup_PIC_0.png`

The blob baseline figure shows two overlays side by side without separate original-image panels. The blob figures show pickup/PIC_0.jpg and sedan/PIC_0.jpg, selected from the existing manifest to contrast local structures. The threshold figure holds the two inputs fixed across three thresholds. This display selection does not limit the 1,310-image baseline or seven-setting experiment. The ten-image table is a reproducible display subset; the contour analysis currently covers only those ten images. These different scopes are explicitly explained in the report.

Regenerate the two blob figures with `.venv/Scripts/python.exe notebooks/assignment2/report_blob_figures.py`. The script reads the required function declarations from the blob notebook, recomputes six displayed results, checks baseline measurements and parameter counts against the CSV exports, and copies the figures to `Images/`. It also checks full-dataset per-image threshold changes and reports baseline quartiles. It does not change the notebook or the dataset-wide exports.

The sections have been condensed for the full report: all four blob questions and seven contour questions remain, with four result figures and five inline tables in total. Blob baseline parameters are introduced in the method text; parameter experiments retain their measured results and interpretation.
