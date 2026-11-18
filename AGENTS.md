# AGENTS.md

This file provides guidance to coding agents working with this repository.

## Project overview

IT3212 (Data-Driven Software Engineering) is a course repository for all four assignments.
Assignment 1 has an existing preprocessing implementation. Assignment 2 covers image processing.
Assignment 3 is the current focus and covers a justified machine-learning workflow; the user selected
the student graduation dataset for this assignment. Assignment 4 covers reinforcement learning;
its brief is now available. Do not assume it uses Assignment 3's dataset or workflow.

The assignment PDFs in the repository root are the source of truth for grading and submission requirements.
Read the relevant PDF before implementing an assignment. Distinguish assignment requirements from local
implementation choices, and update this file when new briefs or repository structure are added.

## Repository structure and conventions

- `IT3212 - Assignment 1.pdf` - preprocessing brief, rubric, and submission requirements.
- `IT3212 - Assignment 2.pdf` - image-processing brief and rubric.
- `IT3212-2026-Assignment 3.pdf` - machine-learning brief, 200 points with 130 required to pass.
- `IT3212 - Assignment 4.pdf` - reinforcement-learning brief, Q-learning versus PPO,
  with six assessed parts and separate prediction, experiment, evidence and review records.
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
- `notebooks/assignment3/check_lda_vs_adaboost.py` - development-only review of
  LDA as a classifier against the original locked AdaBoost. Reuses exact notebook
  split/preprocessing definitions, compares eight LDA settings on the same five
  grouped folds, and verifies that AdaBoost reproduces its original fold AP.
  Separate outputs under `results/assignment3/model_choice_review/` include OOF
  capacity/alert results, coefficients, fresh batch timings and conditional paired
  bootstrap differences. No held-out test is scored and the original notebook,
  model registry, test exports and report choice are not changed by this script.
  This is an exploratory follow-up after original test inspection, not a new
  independent test or proof of equivalence. See the assignment README.
- `notebooks/assignment3/finalize_lda_report.py` - current report finalization
  after the user chose LDA for explainability and development contact yield.
  Repeats the eight-setting LDA search on the same folds, freezes settings and
  OOF F2 threshold, and adds LDA's evaluation on the original test population.
  This is an exploratory revision after the original test report was known.
  Results go to `results/assignment3/lda_selected/`; vector figures are saved directly
  in `report/Images/`, with local gitignored PNG previews under
  `figures/assignment3/lda_selected/`. LDA/AdaBoost refit and batch timings are
  measured together; other costs retain the original-run provenance.
  The figure helper prefers finalized LDA exports when present. Report sections
  7--10 now recommend LDA, with AP selecting settings within families, rather
  than claiming LDA has the highest AP. Original combined exports are preserved.
- `notebooks/assignment3/report_model_comparison.py` refreshes the section-8
  `08_model_comparison_AP.pdf` from the same finalized development exports.
  It preserves the model-family bar-chart layout, uses actual mean AP/fold SD,
  and writes the PDF directly to `report/Images/`, with a local gitignored PNG preview.
  The finalizer calls it after the eight
  task-9/10 plots. Section 8 uses this figure instead of the duplicate AP table.
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
- `report/` - Assignment 3 English LaTeX report project requested by the user, separate from
  the existing Assignment 2 report. `main.tex` imports ten files from `Sections/` in task order,
  using the supplied frontmatter/mainmatter layout. Includes a minimal compatible `packages.sty`,
  placeholder title page, bibliography, optional appendix and figure copies in `Images/`.
  Keep the structure compact: shared evaluation protocol in point 2, methods in 5/7,
  development comparisons in 6/8 and one final test/cost comparison in 10. Sections 9/10 are
  completed English report text with five main figures and inline tables. Three
  separate comparison plots that repeat tables are in the disabled optional
  `Appendices/task_09_10_extra_figures.tex`.
  Each task-9/10 figure contains one plot with its own number/caption; do not combine
  these plots into lettered subpanels. At the user's request, show the DNN training-loss
  and internal-validation plots side by side, retaining separate figure numbers/captions.
  Make requirement coverage explicit: 9.1 implementation,
  9.2 justified tuning, 9.3 measured tuning/training results; 10.1 classification,
  10.2 complexity, 10.3 computation time, 10.4 machine-learning/real-world implications.
  Interpret measured trade-offs rather than only listing scores; do not claim a guaranteed grade.
  Keep tasks 9/10 concise and use plain English. State the model, evaluation split,
  metric and unit when quoting a number. Explain AP as a ranking score, distinguish
  alert precision/recall from the appointment budget, and avoid unnecessary formulas
  or repeated diagnostics. Retain separate single-plot figures and criterion headings.
  Explain the alert rule before its confusion matrix: predicted Dropout raises
  an alert when the model score is at least the development-selected threshold
  (0.0614 rounded for the recommended LDA). This is separate from the top-20% contact list.
  Distinguish the reasons for the scores: AP selects ranking models; F2 selects
  alert thresholds because the scenario prioritizes finding dropout cases over
  avoiding every false alert. State this as a project assumption, not measured costs.
  The cutoff maximizes F2 on development cross-validation predictions, then stays fixed on test.
  Matrix rows are recorded outcomes and columns are predicted Graduate/Dropout;
  show student counts with correct/false-alert/missed-dropout labels, without
  row percentages that can be confused with precision.
  `tasks_09_10.tex`
  imports those same sections for a standalone preview build. Obsolete preview PDFs and
  Overleaf ZIP snapshots were removed during cleanup; use current sources and assets. Sections 1--6
  remain a scaffold whose italic prompts must be replaced before submission. Point 8
  documents the common CV comparison and the group's decision to omit transfer
  learning as outside the pooled-population scenario. The revised recommendation
  is LDA, with its added classification, cost and contact-budget results. Explain
  the trade-off and exploratory revision from the initial AP-only AdaBoost choice.
  Sections 9/10 omit transfer results, costs and cross-references. Use fewer
  semicolons and prefer clear sentences. The earlier notebook transfer experiment
  and its exports remain internal analysis unless the user requests code changes.
  Keep the computation-time environment description to one brief sentence naming
  the shared Intel Core Ultra 7 258V CPU. Software versions, thread settings and
  timing provenance go in the enabled computational-environment appendix.
  The user requests LaTeX and updated figure assets only. Do not compile report
  PDFs or create report ZIP snapshots unless requested.
  Seven figures are enabled, including the section-8 model comparison,
  in the main text and three older supporting figures are in the disabled appendix.
  `notebooks/assignment3/report_tasks_09_10.py` reads fixed exports, checks advisor counts,
  renders the new PDF/PNG figures and refreshes report image copies without retraining or
  selecting models. The notebook calls it after all numerical exports; DNN metadata and
  stopping/refit histories are saved separately. Keep report tables synchronized after reruns;
  the figure script does not overwrite authored report text. Keep all required numerical evidence in the report itself,
  and refresh figure copies/numbers from the combined binary workflow when experiments change.
  See `report/README.md` for integration into Overleaf or the user's existing NTNU template.
- `IT3212___Assignment_3.pdf` - user-supplied assembled group report (20 physical pages),
  preserved unchanged. It is a review input, not a new assignment brief. Its full Overleaf
  sources are absent locally. The review found missing AdamW/AP bibliography entries,
  an undocumented transfer comparison, LDA/AdaBoost recommendation disagreement,
  stale figure-7 image content and inconsistent preprocessing descriptions.
  `report/tasks_08_10.tex` previews coordinated corrected sections; `report/integration/`
  contains bibliography additions, a shared-protocol replacement and merge instructions.
  Do not claim the complete group PDF was rebuilt just because the local excerpt compiles.
  The revised notebook records actual CPU/software/threads, 25 prediction repeats after
  3 warm-ups with median/IQR, three development-only DNN seeds (42/7/2026), and full
  transfer/paired-bootstrap metadata. Original locked choices and test quality metrics
  are unchanged; all reported costs must use the newly measured timing tables.
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
- `README.md` - course navigation, Python setup, local dataset paths and housekeeping policy.

### Post-Assignment 3 cleanup

Obsolete Assignment 3 notebook-build/review scratch files, report render/build files,
temporary PDF tools, package-download caches, local report previews and ZIP snapshots
were removed. Keep scientific Python dependencies in the selected environment, not
under `tmp/`; the maintained scripts no longer add scratch packages to `sys.path`.
The section-8 figure renderer no longer creates a one-off backup in `tmp/report_build/`.
Two unused report image copies (`03_deep_network_learning.png` and
`04_test_classification.png`) were removed from `report/Images/`; original plots
remain under `figures/assignment3/combined/`. Do not regenerate unused report copies.
LaTeX intermediates and local report preview/bundle names are now gitignored.
Generated task-9/10 plots and the section-8 comparison now have one versioned
vector PDF per figure in `report/Images/`. Duplicate PDFs under `figures/`,
obsolete task-9/10 plots from the initial combined report and duplicate report
PNGs were removed. Renderers save PDFs directly in the report and regenerate
gitignored PNG notebook previews under `figures/assignment3/combined/` (09/10 only)
or `figures/assignment3/lda_selected/`. Keep the earlier six tracked combined
overview plots and their referenced report assets. Do not commit regenerated
preview copies as additional report assets.
Original data, numerical experiment exports, notebooks, report sources, required
figure assets, the supplied group PDF, lecture/workshop work and previous assignment
deliverables are preserved. Keep the numerical exports: timings and revision provenance
cannot be recovered exactly by rerunning an experiment.

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

Source of truth: `IT3212 - Assignment 4.pdf` (six pages). Compare off-policy
Q-learning with on-policy actor-critic PPO. Listed options are MsPacMan, Breakout
and SpaceInvaders. The brief also refers to maps, terminal success and compatible
state/action representations; clarify how these apply to the team's option before
choosing an environment or altering the required settings.

1. **Predictions before training (30):** mandatory timestamped prediction sheet,
   stated due date Friday 13 November. Main submission is not considered without
   it. Predict budget efficiency, seed variation, stochastic behaviour, step-penalty
   effects, training/evaluation differences and the initial recommendation. Include
   three states with expected actions, reasons and falsifying observations. Preserve
   original predictions; never reconstruct them after seeing results.
2. **Learning settings (20):** A deterministic transitions/terminal success reward;
   B stochastic transitions with other conditions fixed; C add a step penalty to B;
   D evaluate B agents on an unseen map without further training. Justify a common
   interaction budget and equal tuning allowance. Use common training seeds where
   practical, separate evaluation seeds and at least 200 evaluation episodes per
   trained agent. Disable evaluation updates, specify the evaluation policy and keep
   held-out evaluation out of tuning. Record success/failure rates, mean return,
   successful/failed episode lengths, interactions, runtime/hardware and variation
   across trained seeds. Plot learning against interactions and document smoothing
   and evaluation frequency. Do not treat pooled episodes as independent training
   replications or compare returns directly across different reward definitions.
3. **Mechanisms (20):** vary one main factor at a time: Q-learning epsilon schedule,
   discount factor, PPO entropy, learning rate and PPO update epochs. Use at least
   five seeds per tested configuration. Revisit predictions, inspect a contrary
   case, propose an alternative explanation and run a diagnostic that distinguishes
   explanations rather than merely increasing the budget.
4. **Episode inspection (20):** select ten recorded episodes: three typical successes,
   three typical failures, two surprising successes and two surprising failures.
   Justify selection using outcome distributions. Annotate three decisions per
   episode with state/actions, chosen action and Q-values or policy probabilities,
   observed transition/outcome and an alternative action with its uncertainty.
   Distinguish bad actions from unlucky outcomes and explain an aggregate limitation.
5. **Human review (30):** team review identifies strongest evidence, an overreaching
   claim and an alternative explanation, and compares these with original predictions.
6. **Recommendations (30):** choose an approach/configuration for a rapid explainable
   demonstration, repeated uncertain operation where failures matter, and changed
   environments. Each needs two experimental supports, an observed weakness, a
   reversal condition and another experiment before use beyond simulation. A conclusion
   that neither approach is supported is allowed; avoid universal superiority claims.

Submit executable notebooks/scripts with a documented reproduction command; experiment
configurations, raw logs, seeds, software versions and episode traces; original and
annotated prediction sheets; an evidence brief of at most 2,000 words excluding tables
and figure captions; and a review record of at most 1,000 words. Every figure must
identify source runs. A short record of consequential AI assistance, accepted/rejected
suggestions and verification is mandatory but ungraded. Do not fabricate runs,
predictions, peer feedback or observations. Keep Assignment 4's implementation,
datasets, figures and submission records distinct from prior assignments.
