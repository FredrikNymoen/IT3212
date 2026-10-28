# AGENTS.md

This file provides guidance to coding agents working with this repository.

## Project overview

IT3212 (Data-Driven Software Engineering) is a course repository for all four assignments.
Assignment 1 has an existing preprocessing implementation. Assignment 2 covers image processing.
Assignment 3 is the current focus and covers a justified machine-learning workflow; the user selected
the student graduation dataset for this assignment. Assignment 4 requirements are not yet available.
Do not assume that Assignment 4 uses the same dataset or workflow.

The assignment PDFs in the repository root are the source of truth for grading and submission requirements.
Read the relevant PDF before implementing an assignment. Distinguish assignment requirements from local
implementation choices, and update this file when new briefs or repository structure are added.

## Repository structure and conventions

- `IT3212 - Assignment 1.pdf` - preprocessing brief, rubric, and submission requirements.
- `IT3212 - Assignment 2.pdf` - image-processing brief and rubric.
- `IT3212-2026-Assignment 3.pdf` - machine-learning brief, 200 points with 130 required to pass.
- `workshops/workshop1/Monday-Workshop (1).pdf` - classification workshop brief;
  start with UCI Dry Bean (602), then repeat with another listed dataset.
  The original CSV and UCI metadata are gitignored at `workshops/workshop1/dry_bean.csv`
  and `workshops/workshop1/dry_bean_metadata.json` (the user's chosen location).
  Source, loading instructions, and
  verified dimensions (13,611 rows, 16 numerical features, seven classes; target
  `Class`) are documented in `workshops/workshop1/README.md`.
  `workshops/workshop1/workshop1_classification.ipynb` follows all 11 workshop
  points with PyTorch neural networks, controlled hyperparameter grids, a scaling
  comparison, random-forest tuning, and the repeated workflow on bundled Wine.
  The CSV retains all 13,611 rows; modeling removes 68 exact duplicates before
  splitting (13,543 unique rows) to prevent identical records across partitions.
  Run from `workshops/workshop1/`; use training-only preprocessing and validation
  for selection, reserving test evaluation until each dataset's settings are fixed.
- `notebooks/assignment1/student_graduation.ipynb` - populated Assignment 1 implementation, including optional PCA.
  Preserve this work when starting later assignments.
- `notebooks/assignment1/` through `notebooks/assignment4/` - one directory per assignment.
  Assignment 4 currently contains a `.gitkeep` placeholder; remove it when adding content.
- `notebooks/assignment3/student_dropout_combined.ipynb` - combined binary Assignment 3 workflow
  on branch `assignment3-combined`, with English answers to points 1-10. This is the current
  synthesis; the earlier three-class notebook is retained as a separate reference. Positive class
  is `Dropout=1`, negative is `Graduate=0`; 794 unresolved `Enrolled` records are excluded from
  supervised fitting/evaluation and receive only explicitly unvalidated scores after development.
  Excludes `Debtor` and `Tuition fees up to date` from the main experiment because their recording
  time is unverified. Uses an approximately 80/20 resolved-outcome development/test split and five-fold
  stratified group CV: identical retained-predictor profiles stay in one partition even with different
  outcomes (one such pair appears after administrative fields are excluded). No records are deleted.
  Preprocessing, log selection, rare-category pooling and reduction are fitted inside training folds.
  Compares explicit ANOVA counts, PCA and binary LDA jointly with classifier settings. Required
  ensemble pair is tree AdaBoost and KNN bagging; extra tree bagging has a single-tree control.
  Transfer adapts source-course network weights to the largest development course, with a matched
  scratch arm and a separate target-course test table. PyTorch DNN tuning uses internal stopping
  splits with training-only preprocessing, then refits on every outer training row for the selected
  epoch count. AP selects models; 20% capacity and OOF F2 thresholds are distinct policies fixed
  before testing. Calibration, conditional uncertainty, costs and limited subgroup diagnostics
  are reported. Results and figures use `results/assignment3/combined/` (gitignored) and
  `figures/assignment3/combined/`, preserving earlier outputs. See `notebooks/assignment3/README.md`.
- `notebooks/assignment3/student_dropout_classification.ipynb` - self-contained English answers
  to Assignment 3 points 1-10. Uses the original `data/graduation_dataset.csv`, both semesters,
  three-class prediction and dropout ranking for an illustrative 20% student-support capacity.
  Run from `notebooks/assignment3/`. Results go to `results/assignment3/` (gitignored), figures
  to `figures/assignment3/`. A stratified 60/20/20 train/validation/test split fits all learned
  preprocessing and feature ranking on training data only; validation selects models and neural
  stopping epochs; test evaluation occurs only in point 10 after choices are locked.
  Three basic algorithms are KNN, decision tree and RBF SVM. Boosting uses shallow trees;
  bagging uses KNN, satisfying the requirement for different base algorithms. Point 8 interprets
  the unspecified transfer comparison as training-only autoencoder encoder transfer to classification,
  with a matched scratch control; it explicitly distinguishes this from cross-domain transfer.
  Point 9 tunes a three-hidden-layer PyTorch DNN. The CSV has no dropout dates or timestamped
  administrative fields: this is retrospective feasibility analysis, not validated prospective
  warning before dropout. Keep `Enrolled` distinct from confirmed graduation.
- `notebooks/assignment2/05_blob_detection.ipynb` - multiscale LoG blob detection over the entire
  discovered image dataset for both baseline and all seven parameter settings. The 10-image manifest
  selects display examples only. Full per-image/per-blob statistics and original-resolution numbered
  overlays are gitignored under `results/assignment2/blob_detection/` (overlays in `overlays/`).
  Tracked report figures show display examples and dataset-wide aggregates in
  `figures/assignment2/blob_detection/`. Images are loaded per worker; parameter variants of the same
  image reuse LoG responses, and vectorized overlap preserves the original suppression rule.
- `notebooks/assignment2/06_contour_detection.ipynb` - threshold-based contour detection using
  ContourPy, with full-dataset baseline statistics, all six parameter settings, and a matched LoG
  comparison on every discovered image. The ten manifest images select illustrations only.
  Results are in `results/assignment2/contour_detection/` and plots in
  `figures/assignment2/contour_detection/`. Both notebooks run independently; the contour notebook
  repeats the exact blob helpers and recomputes the full baseline. Keep these functions synchronized.
  Preparation matches EXIF orientation, aspect-preserving resize to maximum side 512 (no upscaling),
  and grayscale normalization. Retained contours include closed outer/hole loops; open paths and
  loops below the minimum area are excluded and counted separately. Images are loaded per worker;
  full per-image/per-contour CSVs and per-image gzip JSON boundary vertices are exported without
  retaining all image arrays or experiment paths in memory. Only display paths stay in memory.
- `notebooks/assignment2/detection_images.json` - shared display-image paths relative to the detection
  dataset, initially two images per class selected by sorted filename. It must not limit either
  full-dataset analysis. Update it when refining illustrations. Put the blob/contour comparison in
  the contour notebook. Other planned notebooks are `01_fourier.ipynb`, `02_pca.ipynb`, `03_hog.ipynb`,
  and `04_lbp.ipynb`; create them when work on those topics begins.
- `data/` - original datasets only; everything except `.gitkeep` is gitignored.
  Assignment 1 uses `data/graduation_dataset.csv`. Use assignment-specific subdirectories for new
  datasets and document their source and placement. The user-provided Assignment 2 detection dataset
  is `data/vehicle-type-detection/`, with hatchback, motorcycle, pickup, sedan, and suv class folders.
- `results/` - generated numerical results, derived datasets, and experiment configurations, organized
  by assignment and task (for example, `results/assignment2/blob_detection/`). Everything except
  `.gitkeep` is gitignored. Do not write generated results into `data/`.
- `figures/` - tracked plot outputs. Existing Assignment 1 plots are `01_continuous_boxplots.png`
  and `02_pca_explained_variance.png`. Use assignment-specific subdirectories for new figures.
- `reports/assignment2/` - exactly two English LaTeX section files for Overleaf, with question headings
  matching the PDF and all tables written inline. Use literal `Images/filename.png` figure paths;
  the accompanying `Images/` folder contains two blob figures, one contour-only overlay,
  one matched blob/contour detail comparison, and one contour parameter figure with unique filenames.
  `notebooks/assignment2/report_blob_figures.py` regenerates the blob figures using notebook functions
  and checks displayed results against CSV exports. `report_contour_figures.py` regenerates and
  checks the pickup/PIC_1.jpg contour-only overlay and detail comparison, marking the same windshield-frame
  and headlamp regions in both comparison panels. Keep the pure contour figure under question 2
  and the matched comparison under question 4. Questions 5 and 7 discuss measured examples and
  suitable applications, with enough explanation to distinguish representation from segmentation.
  Both methods and all parameter settings cover the full dataset; ten fixed paths select
  illustrations only. The contour report table summarizes each class rather than listing all images. Keep methods concise and
  use figures for measured results; the algorithm schematic is not included in the report.
  The sections are condensed for the larger report: retain the four blob and seven contour question
  headings, five result figures, and five inline tables. Introduce blob baseline parameters in prose
  and avoid repeating limitations or narrating every table entry.
  Do not introduce custom path macros or external table inputs. See its README for integration.
  Report scope is 1,310 images for both methods and their comparison; update inline tables and
  prose when experiments change. The user confirmed that vehicle-type-detection was downloaded
  from Canvas. These sections occur late in a larger report: start with methods and results,
  without repeating the general dataset introduction or adding provenance reminders.
  The user confirmed that only the report will be submitted, with no code or CSV attachments.
  Keep the report self-contained and do not refer readers to exported files or notebooks for results.
  Local CSV exports and code remain internal analysis and reproducibility artifacts.
- `requirements.txt` - shared Python dependencies with minimum versions.
- `README.md` - currently empty; use for course navigation, setup, and dataset instructions when needed.

Do not rename, move, or overwrite previous deliverables just to start the next assignment.
Keep notebooks runnable from a fresh kernel, document working directories and data paths, and create
output directories before saving figures. Run notebooks from their assignment directory; paths to root-level
data and figures are `../../data/` and `../../figures/`. Assignment 1 uses this convention.
Never commit local datasets unless explicitly requested.

Notebook headings must follow the assignment PDF's numbered tasks and subquestions so readers can
identify exactly which requirement each section answers. Use supporting subsections within those
points rather than assigning new task numbers to setup, preprocessing, or conclusions. In Assignment 2,
blob detection uses points 1-4 and contour detection uses points 1-7. Apply the same convention to all
other notebooks and retain point letters where the brief uses them.

## Environment and validation

`requirements.txt` contains pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.12,
scikit-learn>=1.3, jupyter>=1.0, ipykernel>=6.25, scipy>=1.11, Pillow>=10.0,
contourpy>=1.1, torch>=2.6 (Workshop 1 and Assignment 3 neural networks), and
threadpoolctl>=3.1 (bounded numerical threads for Assignment 3 timing comparisons).
These are minimum constraints, not exact pins. SciPy, Pillow, and ContourPy support the image-processing notebooks.
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

## Assignment 2: Image processing

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
pickup (478), sedan (400), and suv (129). The user confirmed that vehicle-type-detection was obtained
from Canvas; use Canvas as its course download source even though the PDF names Blackboard.
Class labels are vehicle types, not ground-truth blob or contour annotations; do not equate
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

## Assignment 3: Machine learning (current focus)

Source of truth: `IT3212-2026-Assignment 3.pdf`. Total 200 points, passing mark 130.
The supplied one-page brief specifies neither submission format nor word limit. The point allocations
shown below follow the PDF; point 2 has no separate displayed allocation.

1. Develop a real-world and machine-learning problem statement, connecting the selected dataset
   and learning task to a real scenario (10).
2. Implement and justify preprocessing; Assignment 1 preprocessing may be reused.
3. Extract features and justify the methods (20; 10 without justification).
4. Select features or reduce dimensionality and justify the method, or justify using neither
   (20; 10 without justification).
5. Implement and justify three of KNN, linear/polynomial regression, additive model, decision trees,
   Naive Bayes, kernel SVM, or neural network (30; 10 without justification).
6. Compare and explain the three algorithms' performance in relation to the problem
   (20; 10 without explanation).
7. Implement boosting and bagging, explain the steps, and use different base algorithms with
   justification (20).
8. Compare basic, boosting, bagging and transfer approaches in relation to the learning problem
   (20; 10 without explanation). The brief does not define the transfer source/task; state the
   chosen interpretation explicitly rather than silently omitting it or claiming external transfer.
9. Implement a CNN or deep neural network with justified hyperparameter tuning (30). The brief
   explicitly permits using data directly after preprocessing for this point.
10. Compare complexity, computation time and classification performance, relating the results
    to both the machine-learning and real-world problems (30).

The selected problem concerns advisor prioritization using background and first-year results.
The combined workflow uses resolved binary outcomes and explicitly limits inference to that
population; it must not treat unresolved students as known non-dropouts. The earlier notebook
remains a three-class registered-status analysis. Maintain the distinction between recorded
status and future dropout after an intervention date.
No evidence that support reduces dropout can be inferred from classification metrics alone.
Retain Assignment 1 deliverables; fix preprocessing leakage in the new workflow rather than
assuming its earlier full-dataset transformations are safe for model evaluation.

## Assignment 4

Requirements are pending. When the PDF becomes available, read it and add an accurate summary here.
Keep its implementation, datasets, figures, and submission requirements identifiable.
