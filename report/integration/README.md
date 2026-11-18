# Integrating the review corrections

The supplied `IT3212___Assignment_3.pdf` has 20 physical pages (printed report
pages 1--16 after front matter). Its full Overleaf sources are not in this
workspace. The local `report/main.tex` still has author prompts in sections
1--6; do not mistake it for a corrected rebuild of that full PDF.

Use the revised section files and figures to update the group project, then
rebuild the complete group PDF. `tasks_08_10.tex` previews the coordinated
replacement sections only; `tasks_09_10.tex` previews the user's contribution.

## Required changes in the group project

1. **References, printed page 10 / physical page 14:** merge the entries from
   `missing_references.bib` into the project's existing bibliography exactly
   once. Keep `\cite{loshchilov2019adamw}` and `\cite{sklearn_ap}` in the text.
   Check that the bibliography filename in `\addbibresource{...}` matches the
   uploaded resource and use Biber. Recompile from scratch. The resulting PDF
   must contain the two bibliography entries and no literal citation keys.
   Do not replace the existing bibliography with a file that drops other authors'
   references. The standalone previews include their own complete resource.
2. **Scope and recommendation, printed pages 9--10:** the group has decided
   to omit the transfer experiment and explain why adaptation between populations
   is outside its pooled-population advisor scenario. The revised section 8
   contains the common development comparison and this scope statement.
   Sections 9 and 10 omit all transfer results, timing claims and cross-references.
   The current recommendation is LDA. Replace sections 7--10 together, including
   section 8's development-only LDA comparison and the revised preference for
   explainability. The notebook's earlier transfer experiment remains internal
   analysis and is not part of the current report.
3. **LDA evidence:** distinguish section 4's dimensionality-reduction probe
   from the newly evaluated LDA classifier. The classifier has CV AP 0.9397,
   test AP 0.9445 and reaches 146 recorded dropouts within 146 test contacts.
   Its settings and F2 cutoff were fixed from development predictions before
   its test scoring. The original test report was already known when the
   follow-up was proposed, which section 8 states. This is an exploratory
   revision, not a new independent test or a claim that LDA has the highest AP.
4. **DNN and timing:** replace sections 9 and 10, not just their figures.
   These contain split counts, seeds, class-weight calculation and the repeated
   timing protocol. Also copy `Appendices/computational_environment.tex` and
   its import. The body briefly names the Intel Core Ultra 7 258V CPU, while
   the appendix gives software versions, threads and timing provenance.
   Replace the
   timing table and timing plots together; old single-run numbers must not be
   mixed with the new timing distribution.
5. **Figure 7, printed page 15 / physical page 19:** upload the current
   `10_alert_confusion_matrix.pdf` and use the figure include line from the
   revised section 10. The new image shows LDA: 371 correct graduates,
   71 false alerts, 28 missed dropout cases and 256 correct dropout predictions.
   Its revised cutoff is 0.0614. Cells show counts and classification labels.
   Replacing only the caption or only
   the TeX will not update a stale image asset.
6. **Shared protocol:** the supplied section 2 says log transformation is used
   and only the target changes from Assignment 1. That conflicts with the
   combined implementation's no-log selection, excluded administrative fields
   and fold-fitted transformations. `02_shared_protocol.tex` is a concrete
   replacement for the group to integrate. Ensure sections 3--6 use the same
   split, retained population and feature availability assumptions.

## Check the rebuilt full PDF

- No raw citation keys, `??`, missing bibliography entries or missing figures.
- One final model-selection policy across sections 6, 8 and 10.
- Keep development CV and held-out test results distinct. Remove the earlier
  target-course transfer results and costs from the submitted report.
- Check the earlier section-8 bagged-tree AP against the revised results: its
  earlier chart reported 0.9406, while the combined export reports about 0.9415.
- Distinguish the tuned decision tree from the matched single-tree bagging
  control when describing bagging gains.
- Do not describe all retained first-year features as verified early-warning
  measurements. Dropout dates and administrative timestamps are absent.
- Check each figure caption against the actual image in the final compiled PDF.
- Reviewer-added seed checks are development-only, at fixed selected settings;
  they do not establish that small architecture differences are robust.

The supplied group PDF is preserved unchanged. A successful local excerpt build
does not prove that these corrections have been incorporated into Overleaf.

Only LaTeX and figure assets were updated for the LDA revision. Obsolete preview
PDFs and ZIP snapshots were subsequently removed during repository cleanup.
Use the current section files, bibliography, appendix and fresh Images/ assets.
The supplied group PDF remains unchanged; no report PDF was rebuilt during cleanup.
