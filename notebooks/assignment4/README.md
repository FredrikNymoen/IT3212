# Assignment 4: current MsPacMan workflow

Run commands from the repository root after installing `requirements.txt`.
The report compares ordinary single-table Q-learning with PPO. Original
predictions remain unchanged. A/B/C use the five-seed November 22 study;
D uses the same frozen B models with the November 23 boundary protocol.

## Entry points

The two readable algorithm files are **[q_learning.py](q_learning.py)** and
**[ppo.py](ppo.py)**. Each contains its actual agent implementation and a
training command, with the same selected settings as the report. Q has its
own table, state encoding, action selection, Bellman update and Gymnasium
training loop. PPO has its actor/critic, schedules, SB3 configuration and
training loop. PPO imports common environment/evaluation/logging helpers
from the Q file; both reuse the audited native Gymnasium environment.

```powershell
python notebooks/assignment4/q_learning.py check
python notebooks/assignment4/ppo.py check
python notebooks/assignment4/q_learning.py train --setting B --seed 17 --output results/assignment4/readable_Q_B17_NEW
python notebooks/assignment4/ppo.py train --setting B --seed 17 --output results/assignment4/readable_PPO_B17_NEW
```

Training defaults to 262,144 interactions with ten source-development
episodes at each checkpoint. It saves models, curves and a development-only
`selected.json`; it does not automatically inspect held-out test outcomes.
Use the same setting, cap and training seeds17/29/101/314/2718 for both methods.
Each invocation trains one model, so repeat for each seed and A/B/C. D is
excluded from training. To evaluate a locked selection:

```powershell
python notebooks/assignment4/q_learning.py evaluate --selection results/assignment4/readable_Q_B17_NEW/selected.json --setting B --output results/assignment4/readable_Q_B17_test_NEW
python notebooks/assignment4/ppo.py evaluate --selection results/assignment4/readable_PPO_B17_NEW/selected.json --setting D --output results/assignment4/readable_PPO_D17_test_NEW
```

Evaluation defaults to 200 frozen episodes and writes episode CSVs and full
traces. D requires B-trained weights and uses the existing explicit-censoring
wrapper. Greedy Q uses seeded uniform ties; PPO uses argmax. Saved selections
verify source/model hashes and existing output directories are never replaced.
Evaluation seeds reproduce the original ranges, so these reruns are not new
independent tests. The readable files do not rerun the earlier tuning search
or overwrite the report. The registered implementations/checkpoints below
remain the source of the existing measured report results.

On `assignment4-part2-synthesis`, start with
[the common Part 2 notebook](02_part2_synthesis.ipynb) and
[the four-branch comparison/inspection guide](synthesis/README.md).
The common interface combines reviewed ideas from Johan, Elling and Mats
while preserving the registered ordinary-Q/PPO experiment and original
predictions. No DQN implementation or results are imported.

| Purpose | Guide / implementation |
|---|---|
| Readable ordinary-Q agent, training and evaluation | [q_learning.py](q_learning.py) |
| Readable PPO actor/critic, training and evaluation | [ppo.py](ppo.py) |
| Matched search and A/B/C training/evaluation | [three_hour](three_hour/README.md) |
| Frozen transfer evaluation with explicit censoring | [transfer_evaluation](transfer_evaluation/README.md) |
| Current report/analysis validation | [report_review](report_review/README.md) |
| Part 3 mechanism experiments and diagnostic | [part3](part3/README.md) |
| Unified Part 2 A/C/D runner (clarification-compliant) | [run_unified_part2.py](run_unified_part2.py) |
| Part 3 mechanism notebook, sweeps and extra diagnostics | [03_part3_mechanisms.ipynb](03_part3_mechanisms.ipynb), [part3_experiments.py](part3_experiments.py), [run_part3_extras.py](run_part3_extras.py) |
| Part 4 episode inspection, RGB frames and counterfactual replays | [inspect_part4.py](inspect_part4.py), [run_part4_replays_and_frames.py](run_part4_replays_and_frames.py) |
| Final report verification (word limits, refs, figures, hashes) | [verify_final_report.py](verify_final_report.py) |
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
Use the Part 3 runner for the mechanism experiments. Parts 4--6 cover episode
analysis, human review and recommendations.
