# Assignment 3 report structure

**Current revision: LDA is the preferred classifier.** Sections 7--10 and their
figure assets use `results/assignment3/lda_selected/`, produced by
`notebooks/assignment3/finalize_lda_report.py` after the development-only
LDA/AdaBoost review. The original combined notebook is preserved as the initial
AP-only analysis. The recommendation now balances development contact yield and
explainability, accepting LDA's slightly lower AP. Section 8 documents the
exploratory chronology. Final vector plots are saved once in `Images/`.
Reproducible PNG notebook previews under `figures/assignment3/lda_selected/`
are local and gitignored. The computational-environment appendix is enabled.

Obsolete report preview PDFs, ZIP snapshots and LaTeX build intermediates were
removed during repository cleanup. Current LaTeX sources and figure assets
are retained. Copy the current files directly; no report PDF was rebuilt.

This is an English LaTeX report project, based on
`notebooks/assignment3/student_dropout_combined.ipynb` and the assignment PDF.
The ten imported sections follow tasks 1--10. **Tasks 9 and 10 contain completed
report text, inline result tables and five separate main figures. Three additional
plots repeat the comparison tables and are available in an optional appendix. Section 8 now
contains the coordinated comparison and a brief explanation of why transfer
learning is outside the chosen problem scope. Tasks 1--6 remain
a writing scaffold:** their italic prompts must be replaced before submission.

For the user's contribution alone, use **`tasks_09_10.tex`**. For the whole
group-report scaffold, use `main.tex`.
Both entry points import the same section files, keeping one source of report text.
For the review corrections across sections 8--10, use `tasks_08_10.tex`.
See `integration/README.md` for the exact
changes to merge into the supplied group report, including bibliography entries
and a replacement shared-preprocessing section. The supplied full group PDF is
preserved; its Overleaf sources are not available locally.
For the standalone contribution, upload `tasks_09_10.tex`, `packages.sty`,
`references.bib`, sections 9/10, `Appendices/computational_environment.tex` and
the figures referenced by those sections. Select `tasks_09_10.tex` as the main
document. Include `Appendices/task_09_10_extra_figures.tex` and its three figures
only if enabling the optional appendix. No dataset or internal CSV is needed.

```text
report/
  main.tex
  tasks_08_10.tex
  tasks_09_10.tex
  packages.sty
  title.tex
  references.bib
  integration/
    README.md
    02_shared_protocol.tex
    missing_references.bib
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
    05_cost_and_input_dependence.png
    06_probability_reliability.png
    08_model_comparison_AP.pdf
    09_dnn_hyperparameter_comparison.pdf
    09_dnn_training_loss.pdf
    09_dnn_stopping_performance.pdf
    10_test_average_precision.pdf
    10_model_search_time.pdf
    10_model_inference_time.pdf
    10_advisor_contact_capacity.pdf
    10_alert_confusion_matrix.pdf
  Appendices/
    computational_environment.tex
    additional_results.tex
    task_09_10_extra_figures.tex
```

## Use in Overleaf

Upload the contents of `report/`, select `main.tex` as the main document and use
pdfLaTeX with Biber bibliography processing. Replace the title-page names and date.
The supplied `packages.sty` is a minimal compatible setup: it provides the
`frontmatter` and `mainmatter` commands missing from the standard article class.

If continuing in the existing NTNU template, retain its own `packages.sty` and
title page, copy `Sections/`, `Images/` and `references.bib`, and use the ten
import lines from this `main.tex`. Ensure that `import`, `graphicx`, `amsmath`,
`booktabs`, `tabularx`, `float` and `biblatex` are available and that the
bibliography resource is registered. Figures and tables are placed within the task or evaluation criterion they
answer, rather than floating into a later section.
Sections do not depend on custom path or writing-prompt macros.

## Writing and result integration

The completed contribution is organized around the wording of the brief:

| Requirement | Report location |
|---|---|
| Implement a DNN | 9.1: architecture, loss and training choices |
| Tune and justify hyperparameters | 9.2: grid, rationale and selection protocol |
| Show what tuning achieved | 9.3: controlled configuration comparisons and training behaviour |
| Compare classification performance | 10.1: held-out metrics and explanations on the common test population |
| Compare complexity | 10.2: fitted structures, prediction work and interpretability |
| Compare computation time | 10.3: search, refit, inference and budget caveats |
| Relate results to both problems | 10.4: discrimination, advisor capacity, alert errors and recommendation |

- Use plain English and identify the model, validation/test stage, metric and unit
  for quoted results. Test AP is a ranking score, not a percentage. The main test
  table shows alert precision/recall as percentages; appointment capacity is a
  separate rule. Advanced diagnostics remain in the notebook rather than being
  repeated throughout the report.
- Keep the report compact: explain the evaluation protocol once in Section 2;
  use Sections 5/7 for methods, 6/8 for development comparisons and 10 for the
  final test/cost comparison. Cross-reference related sections rather than
  repeating explanations. Use compact tables for feature definitions and tuning.
- The DNN training-loss and internal-validation plots are placed side by side,
  with independent figure numbers and captions, without lettered subpanels.
- Seven figures are enabled in the main text: the feature comparison, section-8
  model comparison and five
  separate task-9/10 figures. Each has one plot with its own number and caption.
  Three task-9/10 comparison plots remain available in
  `Appendices/task_09_10_extra_figures.tex`; the standalone entry point has
  commented import lines for this optional appendix. The older group-report
  appendix also remains disabled. Include extra plots only if they add evidence.
- Keep task numbering 1--10. Supporting introductions and conclusions belong
  inside the relevant sections so that they do not shift the task numbers.
- Section 1 includes the research question. Remaining italic prompts in tasks
  1--6 are author guidance, not finished report prose.
- Put definitions, implementation choices, measured results and interpretation
  alongside the task they answer. Avoid pasting notebook cells as explanations.
- Internal source filenames are listed in LaTeX comments. Obtain numbers from
  `results/assignment3/lda_selected/` and write tables inline in the section files.
  The final report must contain the necessary results itself.
- Task-9/10 vector figures are saved directly to `Images/`, with literal
  `Images/filename.pdf` paths suitable for Overleaf. Duplicate PDFs and report PNGs
  were removed. Refresh the figures and affected numbers together after experiments
  change; local PNG notebook previews are regenerated separately and are gitignored.
- Development CV results and held-out results must be identified separately.
  At the group's request, Section 8 omits the transfer experiment and explains
  its scope decision. Sections 9 and 10 contain no transfer results or costs.
  The earlier experiment remains in the notebook as internal analysis.
- The computational-environment appendix is enabled. Extra figure appendices
  remain optional and commented out. Core answers stay in the main report.

## Reproduce tasks 9 and 10

1. Execute the combined notebook from a fresh kernel. It exports DNN metadata,
   the internal stopping and final-refit histories, CV searches, fixed test scores,
   complexity/timing tables and uncertainty diagnostics.
   The revised execution also records CPU/software/thread provenance, three
   fixed-setting DNN initialization seeds (42, 7, 2026) and 25 raw
   prediction-timing repeats after three warm-up calls. Report prediction medians
   and interquartile ranges together. Earlier transfer exports are retained
   internally but are excluded from the current report.
2. Run `check_lda_vs_adaboost.py`, then `finalize_lda_report.py` to reproduce the
   exploratory LDA revision. Preserve the original combined exports alongside
   `results/assignment3/model_choice_review/` and `results/assignment3/lda_selected/`.
   Both scripts use the selected Python environment with the root dependencies
   installed; no temporary package directory is required.
3. The notebook's final presentation cell runs
   `notebooks/assignment3/report_tasks_09_10.py`. That script can also be run
   separately from any directory after the notebook. It reads fixed exports,
   checks contact counts against saved predictions and renders PDF/PNG figures
   without refitting or selecting models. The LDA finalizer refreshes these figures
   and the section-8 chart from finalized LDA exports. The scripts write each vector
   PDF directly into `Images/` and keep PNG notebook previews local.
4. If experiments change, update the inline numbers in the two section files
   alongside the figures. The plotting script deliberately does not overwrite
   authored report prose or tables. Keep development and test results distinct.
5. To build a preview yourself, use pdfLaTeX, Biber, then pdfLaTeX twice. For the standalone preview,
   use job name `tasks_09_10`; for the complete scaffold use `main`.

Keep generated preview PDFs and auxiliary files in the gitignored
`tmp/report_build/` directory. Historical previews were removed because they
predated the LDA revision. The current revision was not compiled during cleanup.
The supplied full group report is retained unchanged and has not been rebuilt
from its unavailable sources.
