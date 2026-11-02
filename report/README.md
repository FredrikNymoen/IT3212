# Assignment 3 report structure

This is an English LaTeX **writing scaffold**, based on
`notebooks/assignment3/student_dropout_combined.ipynb` and the assignment PDF.
The ten imported sections follow tasks 1--10. The italic writing prompts are
instructions for the authors and must be replaced before submission.

```text
report/
  main.tex
  packages.sty
  title.tex
  references.bib
  Sections/
    01_problem_statement.tex
    02_preprocessing.tex
    03_feature_extraction.tex
    04_feature_selection.tex
    05_classification_algorithms.tex
    06_basic_model_comparison.tex
    07_boosting_and_bagging.tex
    08_transfer_and_ensemble_comparison.tex
    09_deep_neural_network.tex
    10_final_evaluation.tex
  Images/
    01_features_and_representations.png
    02_basic_development_pr.png
    03_deep_network_learning.png
    04_test_classification.png
    05_cost_and_input_dependence.png
    06_probability_reliability.png
  Appendices/
    additional_results.tex
```

## Use in Overleaf

Upload the contents of `report/`, select `main.tex` as the main document and use
pdfLaTeX with Biber bibliography processing. Replace the title-page names and date.
The supplied `packages.sty` is a minimal compatible setup: it provides the
`frontmatter` and `mainmatter` commands missing from the standard article class.

If continuing in the existing NTNU template, retain its own `packages.sty` and
title page, copy `Sections/`, `Images/` and `references.bib`, and use the ten
import lines from this `main.tex`. Ensure that `import`, `graphicx`, `amsmath`
and `biblatex` are available and that the bibliography resource is registered.
Sections do not depend on custom path or writing-prompt macros.

## Writing and result integration

- Keep the report compact: explain the evaluation protocol once in Section 2;
  use Sections 5/7 for methods, 6/8 for development comparisons and 10 for the
  final test/cost comparison. Cross-reference related sections rather than
  repeating explanations. Use compact tables for feature definitions and tuning.
- Only two figures are enabled in the main text. Four supporting figures are
  in the optional, disabled appendix; include them only if they add useful evidence.
- Keep task numbering 1--10. Supporting introductions and conclusions belong
  inside the relevant sections so that they do not shift the task numbers.
- Section 1 includes the research question; all remaining italic prompts are
  writing guidance, not finished report prose.
- Put definitions, implementation choices, measured results and interpretation
  alongside the task they answer. Avoid pasting notebook cells as explanations.
- Internal source filenames are listed in LaTeX comments. Obtain numbers from
  `results/assignment3/combined/` and write tables inline in the section files.
  The final report must contain the necessary results itself.
- The six images are exact copies from `figures/assignment3/combined/`. Their
  paths are literal `Images/filename.png`, suitable for a standalone Overleaf upload.
  Refresh these copies and all affected numbers together after experiments change.
- Development CV results and held-out results must be identified separately.
  Section 8's transfer comparison uses only the target-course test subset; it
  must not be ranked directly against full-population metrics.
- The appendix is optional and commented out. Core answers stay in the main report.

The structure is a writing scaffold, not a completed report. The main document
has been validated locally with pdfLaTeX and Biber; build outputs stay in the
gitignored `tmp/report_build/` directory.
