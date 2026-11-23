# Assignment 4: current MsPacMan workflow

Run commands from the repository root after installing `requirements.txt`.
The report compares ordinary single-table Q-learning with PPO. Original
predictions remain unchanged. A/B/C use the five-seed November 22 study;
D uses the same frozen B models with the November 23 boundary protocol.

## Entry points

On `assignment4-part2-synthesis`, start with
[the common Part 2 notebook](02_part2_synthesis.ipynb) and
[the four-branch comparison/inspection guide](synthesis/README.md).
The common interface combines reviewed ideas from Johan, Elling and Mats
while preserving the registered ordinary-Q/PPO experiment and original
predictions. No DQN implementation or results are imported.

| Purpose | Guide / implementation |
|---|---|
| Matched search and A/B/C training/evaluation | [three_hour](three_hour/README.md) |
| Frozen transfer evaluation with explicit censoring | [transfer_evaluation](transfer_evaluation/README.md) |
| Current report/analysis validation | [report_review](report_review/README.md) |
| Common notebook, evidence overview and native recorded-episode viewer | [synthesis](synthesis/README.md) |
| Report sources and submission limits | [Report guide](../../reports/assignment4/README.md) |

The `current/environment.py`, `improvements/agents.py`, `pilot/` core modules
and `q_motion/motion_q.py` are shared dependencies. Their historical directory
names remain because checkpoints and source-hash protocols depend on those
paths. They are not alternative current workflows.

## Local experiment records

`results/assignment4/` contains the local configurations, models, per-episode
CSVs and decision traces. Results, datasets and Python environments are
ignored by Git. Copy the required experiment records separately when preparing
the assignment submission or moving to another computer; the repository alone
contains source and report assets, not all trained models/traces.

No evaluation outcome selects models or parameters. D uses every original B
checkpoint, 200 fresh episodes per model, and no additional training. A map
exit is censored rather than silently classified as agent failure. Current
D observed success is Q4.5% and PPO14.7%, with Q's logical bound4.5--4.6%.
These rates do not establish reliable transfer or universal algorithm ranking.

## Historical work

Earlier notebooks, failed variants, one-off migration/recovery helpers and
unused figures were removed from the working tree after being preserved in
Git commit `714c0cb`. See [ARCHIVE.md](ARCHIVE.md). Numerical results and exact
source snapshots remain in their original ignored results directories.
Parts3--6 still require their own experiments, episode analysis and real human
review; the current Part2 report is not the complete assignment submission.
