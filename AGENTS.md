# AGENTS.md

This file provides guidance to coding agents working with this repository.

## Project overview

IT3212 (Data-Driven Software Engineering) is a course repository for all four assignments.
Assignment 1 has an existing preprocessing implementation. Assignment 2 is the current focus and
covers image processing. Assignments 3 and 4 will be added later; their requirements are not yet available.
Do not assume that later assignments use the student graduation dataset or the Assignment 1 workflow.

The assignment PDFs in the repository root are the source of truth for grading and submission requirements.
Read the relevant PDF before implementing an assignment. Distinguish assignment requirements from local
implementation choices, and update this file when new briefs or repository structure are added.

## Repository structure and conventions

- `IT3212 - Assignment 1.pdf` - preprocessing brief, rubric, and submission requirements.
- `IT3212 - Assignment 2.pdf` - image-processing brief and rubric.
- `notebooks/assignment1/student_graduation.ipynb` - populated Assignment 1 implementation, including optional PCA.
  Preserve this work when starting later assignments.
- `notebooks/assignment1/` through `notebooks/assignment4/` - one directory per assignment.
  Assignment 3-4 directories currently contain `.gitkeep` placeholders; remove each when adding content.
- `notebooks/assignment2/05_blob_detection.ipynb` - implemented and executed multiscale LoG blob
  detection on the shared 10-image sample, with synthetic checks, circle overlays, per-blob/per-image
  statistics, parameter experiments, and discussion. Numerical exports are gitignored under
  `results/assignment2/blob_detection/`; plots are in `figures/assignment2/blob_detection/`.
- `notebooks/assignment2/06_contour_detection.ipynb` - starter with working dataset inventory and
  shared image selection; contour implementation and comparison remain pending. Both notebooks must
  run independently. Match the blob notebook's EXIF orientation, aspect-preserving resize to maximum
  side 512 (no upscaling), and grayscale preparation when implementing the comparison.
- `notebooks/assignment2/detection_images.json` - shared image paths relative to the detection dataset.
  Initially two images per class, selected by sorted filename. Update this manifest for both notebooks
  when refining the sample. Put the blob/contour comparison in the contour notebook.
  Other planned notebooks are `01_fourier.ipynb`, `02_pca.ipynb`, `03_hog.ipynb`, and `04_lbp.ipynb`;
  create them when work on those topics begins.
- `data/` - original datasets only; everything except `.gitkeep` is gitignored.
  Assignment 1 uses `data/graduation_dataset.csv`. Use assignment-specific subdirectories for new
  datasets and document their source and placement. The user-provided Assignment 2 detection dataset
  is `data/vehicle-type-detection/`, with hatchback, motorcycle, pickup, sedan, and suv class folders.
- `results/` - generated numerical results, derived datasets, and experiment configurations, organized
  by assignment and task (for example, `results/assignment2/blob_detection/`). Everything except
  `.gitkeep` is gitignored. Do not write generated results into `data/`.
- `figures/` - tracked plot outputs. Existing Assignment 1 plots are `01_continuous_boxplots.png`
  and `02_pca_explained_variance.png`. Use assignment-specific subdirectories for new figures.
- `requirements.txt` - shared Python dependencies with minimum versions.
- `README.md` - currently empty; use for course navigation, setup, and dataset instructions when needed.

Do not rename, move, or overwrite previous deliverables just to start the next assignment.
Keep notebooks runnable from a fresh kernel, document working directories and data paths, and create
output directories before saving figures. Run notebooks from their assignment directory; paths to root-level
data and figures are `../../data/` and `../../figures/`. Assignment 1 uses this convention.
Never commit local datasets unless explicitly requested.

## Environment and validation

`requirements.txt` contains pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.12,
scikit-learn>=1.3, jupyter>=1.0, ipykernel>=6.25, scipy>=1.11, and Pillow>=10.0.
These are minimum constraints, not exact pins. SciPy and Pillow support the image-processing notebook.
Add new implementation dependencies (such as scikit-image, OpenCV, or Pillow) when introduced rather
than installing them ad hoc. PDF-reading tools used only to inspect briefs are not runtime dependencies.

There is no configured build, lint, or test suite. For notebook changes, execute the affected notebook
from top to bottom in a fresh kernel when the required data and dependencies are available. Check outputs,
array shapes, numerical ranges, and saved plots; report missing inputs or unexecuted work honestly.
Use fixed random seeds for sampling, noise generation, and splitting. Keep interpretations alongside
results; do not invent observations before running code.

## Assignment 1: Data preprocessing

### Dataset and existing implementation

The input is the UCI "Predict Students' Dropout and Academic Success" dataset: 4,424 rows and 35 columns.
The target is `Target`, with classes `Dropout`, `Enrolled`, and `Graduate`. Predictors are numeric,
but several represent nominal integer codes (for example, marital status, course, and occupation).
Do not treat those codes as continuous measurements.

The notebook explores the data, implements median/mode missing-value handling, retains plausible outliers,
applies `log1p` to strongly skewed curricular-unit counts, one-hot encodes nominal predictors, and uses
a stratified 80/20 split with `random_state=42`. Continuous predictors are standardized using training
statistics. The optional PCA section compares variance thresholds and selects 95%.

For future modeling, split before fitting data-dependent preprocessing, including imputation, category
discovery, skewness-based transform selection, scaling, and PCA. The existing notebook fits some earlier
preprocessing steps on the full dataset; do not assume the entire workflow is leakage-free because its
scaler and PCA use training data. Fit preprocessing inside training folds when cross-validating.
Preserve the previous assignment unless changes to it are part of the requested task.

### Rubric

1. **Data exploration (10):** first rows, summary statistics, data types, missing values, outliers,
   and unique categorical values.
2. **Data cleaning (20):** choose and implement missing-value handling and justify the choices.
3. **Handling outliers (20):** detect outliers (for example, IQR or Z-score), decide whether to remove,
   cap, or transform them, and justify the decisions.
4. **Data transformation (30):** encode categorical data and scale features; justify the encoding
   and explain why scaling is necessary and how it affects models.
5. **Data splitting (10):** create training/test sets and explain their role in evaluating
   generalization and detecting overfitting.
6. **Optional bonus (10):** apply dimensionality reduction such as PCA and discuss its effects.

### Submission

The brief requires a **PDF report**, not only a notebook. Include results and code only where necessary
for the explanation. The word limit is **3,000**, excluding code and references but including table and
figure captions. Be precise and concise. Missing a requested justification incurs a 10-mark deduction
from that task. Assignment 1 datasets must come from those uploaded to Blackboard.

## Assignment 2: Image processing (current focus)

Passing requires **65 points**. Each task is assessed on **clarity of explanation (30%)**, **quality of
results (30%)**, and **insight in discussion (40%)**. Include implementation, visual results, and discussion;
code alone is insufficient. The supplied Assignment 2 PDF specifies neither a report word limit nor a
submission file format; do not automatically carry over Assignment 1's submission rules.

### Dataset requirements

The general brief allows Blackboard image datasets or outside datasets with a basic description.
However, blob detection specifically requires a provided Blackboard image dataset, and contour detection
must use the **same dataset**. Follow this more specific requirement for those tasks.
Document sources, selected examples, grayscale conversion, resizing, and normalization. If required
Blackboard images are missing, request their location rather than silently substituting another dataset.
Use multiple equally sized grayscale images for PCA.

The selected local detection dataset contains 1,310 JPG images: hatchback (181), motorcycle (122),
pickup (478), sedan (400), and suv (129). Its original source/Blackboard attribution still needs to be
documented. Class labels are vehicle types, not ground-truth blob or contour annotations; do not equate
detected regions with vehicles or claim detection accuracy from these labels alone.
Save detection figures under `figures/assignment2/blob_detection/` and
`figures/assignment2/contour_detection/` respectively.

### Fourier transform (20 points)

1. Apply the 2D DFT to a grayscale image. Show the original and magnitude spectrum with an explanation.
2. Implement a frequency-domain low-pass filter to remove high-frequency noise. Compare original
   and filtered images and analyze the results.
3. Implement a high-pass filter to enhance edges. Show the filtered image and discuss its effects.
4. Compress by retaining selected percentages of Fourier coefficients. Reconstruct at multiple
   percentages and discuss image quality and compression ratio.

Use an appropriate spectrum visualization (for example, centered log magnitude). Explain coefficient
selection and filter masks, handle conjugate symmetry for real reconstructions, and distinguish
retained-coefficient fractions from actual stored-file compression ratios.

### PCA (25 points)

Normalize grayscale pixels to `[0, 1]`. Write a Python PCA function implementing the requested steps:

1. Form a matrix with one image per row and one pixel per column.
2. Center the data and compute the covariance matrix.
3. Calculate its eigenvalues and eigenvectors.
4. Sort eigenvectors by descending eigenvalue.
5. Select the top `k` eigenvectors.
6. Project images into the lower-dimensional space.

Reconstruct images (restoring the mean), compare originals and reconstructions for multiple `k` values,
plot explained variance, and justify a component count balancing compression and quality. Compute MSE
and discuss reconstruction error, visual information loss, and compression trade-offs.
Do not replace the required manual steps with only a call to `sklearn.decomposition.PCA`.
Choose a manageable image resolution: the pixel covariance matrix grows quadratically with pixel count.
Document any resizing.

### Image processing (55 points)

- **HOG (12):** compute descriptors using a library such as OpenCV or scikit-image. Apply to at least
  three images spanning simple and complex scenes. Show originals, gradient images, and HOG images.
  Compare descriptors and discuss changes to cell size, block size, and number of orientation bins.
- **LBP (13):** write a function for basic 8-neighbor grayscale LBP and a function for its histogram.
  Produce LBP images and histograms for at least three different grayscale images (for example, a
  natural scene, texture, and face). Explain texture information and compare histogram differences.
  State neighbor order, threshold convention, and image-border handling.
- **Blob detection (15):** implement a blob-detection algorithm on a Blackboard dataset. Mark detections
  with circles or bounding boxes. Report per-image counts, sizes, and positions, and evaluate how
  parameter choices affect detection.
- **Contour detection (15):** implement contour detection on the same dataset. Mark contours with
  different colors and report per-image counts, areas, and perimeters. Compare blob and contour results,
  advantages and limitations, effects of parameters such as thresholds and filter sizes, and examples
  where each method is more suitable.

## Assignments 3 and 4

Requirements are pending. When their PDFs become available, read them and add accurate summaries here.
Keep each assignment's implementation, datasets, figures, and submission requirements identifiable,
while reusing shared dependencies and utilities where useful.
