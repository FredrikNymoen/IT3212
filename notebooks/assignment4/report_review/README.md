# Current report checks

With the original local experiment records available, run from the repository root:

```powershell
python notebooks/assignment4/report_review/analyse.py
python notebooks/assignment4/report_review/verify_quality.py
```

The analysis uses the same five trained seeds, with no training or selection.
The verifier checks all five report tables and the complete single-file Part 2
against raw episode CSVs, source
and model hashes, paired-seed statements, original predictions, LaTeX references
and the absence of visible file paths in Part2. Outputs go to the ignored
`results/assignment4/report_review_current/` directory.

The rationale audit checks the actual search-job counts/budgets, all development
seed ranges, registered ALE frame/sticky defaults and the penalty/discount
arithmetic. A feature-level check confirms that Q drops remaining time/score/
lives while PPO's actor retains them. Design reasons do not establish optimality
or satisfy Part 3's one-factor mechanism study. The local Part 2 prose allowance
is 1,250 words; the brief's 2,000-word limit applies to the whole evidence brief.

Required inputs are the original `three_hour_20261122`, `transfer_boundary_v1`
and the archived pre-edit table under `report_review_current/before/`.
These records remain local and must accompany the assignment submission.
Previous authoring/audit versions remain in the original results directories
and in Git checkpoint `714c0cb` where applicable.
