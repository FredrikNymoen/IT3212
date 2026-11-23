# Short development pilot: improve both agents before a large run

This file preserves the first unsuccessful short investigation. For the later
corrected-geometry experiments and successful frozen-policy checks, see
[FURTHER_EXPLORATION.md](FURTHER_EXPLORATION.md). The original local-feature
package below inherited geometry errors; its negative result must not be used
to reject the separately corrected implementation.

The user authorised a bounded diagnostic, not another full A--D experiment.
All runs below use Gymnasium, setting B, the original 2,000-point success
threshold and 1,000-decision horizon. Only development seeds 650000--650019 are
evaluated. No earlier held-out evaluation seeds or D layout are scored, and
the registered Part 2 source files, original predictions and report are unchanged.
This is exploratory follow-up after the original results were known.

## Results

### Saved policies: evaluation behaviour, no training

`policy_check.py` evaluates the three original B checkpoints (training seeds
42, 7, 2026), 20 development episodes each. Means below average the three
trained-agent means.

| Policy | Mean game score | Successes / episodes |
|---|---:|---:|
| Q-learning, original greedy policy | 690.5 | 0 / 60 |
| PPO, highest-probability action | 179.8 | 0 / 60 |
| PPO, actions sampled from learned probabilities | 681.2 | 0 / 60 |

PPO's stationary-position fraction falls from about 80% to 46%. This measure
also includes startup/death animations and is not a certified wall-collision
count. Sampling clearly helps score in this small diagnostic but does not
establish reliable completion of the task.

`q_exploration_check.py` evaluates the original Q checkpoints for training
seeds 42 and 7. The greedy reference below uses those same two checkpoints
and the same 20 development episodes each.

| Q evaluation policy | Mean game score | Successes / episodes |
|---|---:|---:|
| Original greedy | 694.5 | 0 / 40 |
| Epsilon 0.05 | 768.2 | 1 / 40 |
| Epsilon 0.10 | 741.0 | 0 / 40 |

These are different evaluation policies, not improved training. Q-values and
PPO weights were checked unchanged. One success is too little to establish a
robust gain; do not treat evaluation episodes as independent training seeds.

### Relative corridor-feature package: short retraining

`local_pilot.py` trains both algorithms for 65,536 interactions at seeds 42
and 7, with 20 development episodes per trained seed and policy mode. It tests
relative food-path/ghost proximity features, coarse local Q states, and a shared
action filter derived from neighbors in the initial food lattice. NOOP is
excluded. PPO also uses learning rate 0.0003 instead of 0.001. These changes are
a feasibility package, not a one-factor experiment or a certified ALE legal
action mask. The graph is checked only for training layout 0; it is not a
validated transfer representation.

| Trained local package | Mean game score | Successes / episodes |
|---|---:|---:|
| Q-learning, greedy | 416.0 | 1 / 40 |
| PPO, highest-probability action | 282.0 | 0 / 40 |
| PPO, sampled actions | 391.5 | 1 / 40 |

The Q seed scores are 661 and 171, so the result is not stable. On the same
two original checkpoints/development seeds, original greedy Q scores 694.5
and sampled PPO scores 611.8. The local package therefore does not improve
both workflows and is **not adopted**.

Untrained filtered controls at seed 42 score 155.5 for Q, 107.0 for greedy
PPO and 169.5 for sampled PPO, all with zero successes. Learning improves on
these controls but the final package still fails the comparison against the
original agents. The `train_seconds` field on those untrained-control rows is
the associated job's subsequent training time, not time spent training the
untrained control; their `training_interactions` is correctly zero.

A separate five-episode hand-coded food-direction diagnostic scored 1640,
2080, 2000, 1000 and 570. It indicates that the path information can sometimes
support navigation. It is not an RL agent and its successes are never counted
as Q-learning or PPO results.

## Decision and next step

The short test supports investigating PPO action selection and occasional
non-greedy Q actions. It does **not** show that both agents have become much
better at reaching 2,000 points. A larger run of the local package is not
justified by this pilot.

The next isolated test should retain the original information, add verified
wall/corridor and motion features rather than replacing the state wholesale,
and check a reference PPO implementation. Any larger training budget should
follow a demonstrated development improvement. Keep original predictions and
the existing Part 2 results as the baseline. A graded Part 3 mechanism study
still needs at least five seeds per tested configuration; this pilot does not
fulfil that requirement.

## Reproduce

From the repository root, using the project Python environment:

```powershell
python notebooks/assignment4/pilot/policy_check.py
python notebooks/assignment4/pilot/q_exploration_check.py
python notebooks/assignment4/pilot/local_pilot.py
```

The first two commands only evaluate saved B agents. The third performs the
bounded retraining. Local records, protocols and episode CSVs are under
`results/assignment4/quick_pilot/`, separately from the main experiment.
