# Frozen transfer evaluation with explicit map boundaries

This is a post-inspection repair of the interrupted November 22 D evaluation.
It evaluates **all ten original B checkpoints**, Q/PPO at training seeds
17/29/101/314/2718. Their original model hashes and source-development choices
are retained. No model is selected, modified or trained on D. The old run and
its missing Q/D cell remain historical records.

Run from the repository root:

```powershell
python notebooks/assignment4/transfer_evaluation/check.py
python notebooks/assignment4/transfer_evaluation/evaluate.py --workers 8
python notebooks/assignment4/transfer_evaluation/verify.py
python notebooks/assignment4/transfer_evaluation/update_report.py
```

Results, the pre-evaluation protocol/model lock, engineering checks, source
snapshots and all episode/decision traces are stored separately under
`results/assignment4/transfer_boundary_v1/`. The renderer inserts verified D
rows into the reviewed section while retaining A/B/C measurements.

Every model gets 200 fresh episode seeds, 1250000--1250199. These values are
arbitrary identifiers in a disjoint range, not seeds selected for performance.
Q uses the original greedy policy with episode-seeded uniform ties; PPO uses
the original highest-probability action. Both remain frozen. Stickiness is
0.25, target 2,000 points, horizon 1,000 decisions; the native Gymnasium game
and pre-boundary observations/rewards are unchanged.

The old guard correctly rejected a natural change from the transfer maze to
the training maze, but aborted the entire evaluation. The new evaluation-only
Gymnasium wrapper stops at the same transition using `truncated=True`. A map
exit is **censored**, not a goal success or known agent failure. No subsequent
action is allowed. Its terminal observation is explicitly a placeholder based
on the last in-map features; it is never passed to a policy or used to train.

Three disjoint outcomes are recorded for every episode: verified success,
known failure (game over/horizon), or censoring. With S successes and C censored
episodes out of N, observed in-map success is S/N; success of the original
uncensored task is bounded by [S/N,(S+C)/N]. Bounds describe unresolved outcomes,
not confidence intervals. Rates are first computed per trained agent, then
averaged over five agents with sample SD. Censoring remains visible.

At a map exit, the observed return excludes the crossing action's reward and
covers the prefix ending at the last verified in-map observation. Full native
rewards and before/after RAM are retained independently. Thus the report does
not attribute potentially off-map points to transfer performance. Uncensored
episodes have their ordinary full return.

`check.py` compares 400 native steps in each A/B/C/D against the old wrapper,
then independently reproduces the exact old exit: evaluation seed1150094,
step975, score1760, lives2, no native termination. All earlier observations
and transitions agree. It verifies that the censoring result permits no next
action. `verify.py` checks every trace/score/reward, original B hashes and
outcome partition, then directly replays typical successes/failures and every
censored episode through native Gymnasium without the custom wrappers.

The current D domain was already inspected. Fresh episode seeds do not make
this an independent new-domain test. The boundary rule repairs measurement;
it is not a learning improvement, a five-seed mechanism experiment, or a
retrospective rewrite of the original predictions. No PDF/ZIP is compiled.

## Verified results

All ten models completed 200 stopped episodes: **2,000 episodes and
1,325,990 native decisions**. The verifier independently replayed 21 episodes,
including the sole censored episode. Every logged before/after native RAM,
reward, action, score and model/source hash passed its checks.

| Method | Observed success, mean ± seed SD | Success bound | Known failure | Censored |
|---|---:|---:|---:|---:|
| Ordinary Q | 4.5% ± 2.9 pp | 4.5--4.6% | 95.4% | 1/1,000 |
| PPO | 14.7% ± 17.1 pp | 14.7--14.7% | 85.3% | 0/1,000 |

The Q training-seed314 agent accounts for the one censoring event. All five
models remain in each method's result. These rates describe transfer of the
same B models and are not improvements in learning. A/B/C numerical outcomes
are unchanged. The report retains return and removes the duplicate game-score
column; C's raw scores remain in prose. `verify_report.py` checks all forty
A--D table cases against their corresponding raw CSVs, LaTeX structure,
original predictions, and the absence of visible file references.
