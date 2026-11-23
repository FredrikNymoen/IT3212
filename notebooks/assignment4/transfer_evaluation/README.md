# Current frozen transfer evaluation

All ten original B checkpoints (Q/PPO at seeds17/29/101/314/2718) receive
200 fresh D episodes, seeds1250000--1250199, without training or reselection.
The source models, original selection locks and results must be available
locally under `results/assignment4/three_hour_20261122/`.

```powershell
python notebooks/assignment4/transfer_evaluation/check.py
python notebooks/assignment4/transfer_evaluation/evaluate.py --workers 8
python notebooks/assignment4/transfer_evaluation/verify.py
```

Outputs and source snapshots live in the ignored
`results/assignment4/transfer_boundary_v1/` directory. Copy experiment records
separately when preparing the submission; models/traces are not stored in Git.

Native transitions/rewards and observations before a map exit are unchanged.
At any map exit the evaluation-only Gymnasium wrapper returns `truncated=True`,
records censoring and permits no next action. A censored episode is neither
verified success nor known failure. Observed success is S/N and the unresolved
completed-task fraction is bounded by [S/N,(S+C)/N], not a confidence interval.
Its return covers the last verified in-map prefix, excluding the crossing reward.

All2000 stopped episodes and1325990 native decisions were verified;21 episodes
were replayed directly in Gymnasium, including the sole censored case.

| Method | Observed success, mean ± seed SD | Success bound | Known failure | Censored |
|---|---:|---:|---:|---:|
| Ordinary Q | 4.5% ± 2.9 pp | 4.5--4.6% | 95.4% | 1/1000 |
| PPO | 14.7% ± 17.1 pp | 14.7--14.7% | 85.3% | 0/1000 |

This is a post-inspection measurement repair, not improved learning or a new
independent domain. The old interruption, original predictions and model
hashes remain intact. Current report checks are in [report_review](../report_review/README.md).
Obsolete two-table authoring helpers are available in Git checkpoint `714c0cb`.
