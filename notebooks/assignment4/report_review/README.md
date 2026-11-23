# Current report checks

With the original local experiment records available, run from the repository root:

```powershell
python notebooks/assignment4/report_review/analyse.py
python notebooks/assignment4/report_review/verify_quality.py
```

The analysis uses the same five trained seeds, with no training or selection.
The verifier checks the three-table report against raw episode CSVs, source
and model hashes, paired-seed statements, original predictions, LaTeX references
and the absence of visible file paths in Part2. Outputs go to the ignored
`results/assignment4/report_review_current/` directory.

Required inputs are the original `three_hour_20261122`, `transfer_boundary_v1`
and the archived pre-edit table under `report_review_current/before/`.
These records remain local and must accompany the assignment submission.
Previous authoring/audit versions remain in the original results directories
and in Git checkpoint `714c0cb` where applicable.
