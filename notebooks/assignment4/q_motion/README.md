# Isolated Q state experiment: observed relative closing motion

Completed exploration: the added closing bit helped one A training seed but
did not improve transfer. Locked confirmation used 200 new episodes per model:

| Condition | Original Q, 196,608 updates | Motion-bit Q, 196,608 updates |
|---|---:|---:|
| A | 0.0% | 50.0% |
| D, frozen B-to-new-map | 4.5% | 2.5% |

The two motion-bit A models scored 100%/0%; the D models both scored 2.5%.
The original full-budget D reference scored 3.25% on these same new episodes.
The representation is not promoted as a general Q improvement. It remains
an opt-in experimental candidate; the earlier B-focused selected-agent loader,
PPO models and complete v3 report are unchanged. D confirmation is not used
to choose another representation or checkpoint. Two trained seeds limit the
conclusion about training variability; the episodes are not retraining runs.

To load the locked experimental candidate with its compatible Gymnasium wrapper:

```powershell
python notebooks/assignment4/q_motion/selected_q.py --condition A --seed 42
```

`load_selected(condition, seed)` returns the frozen agent, environment and
registered model metadata. It supports A, B and frozen B-to-D transfer.
Choose seeds for demonstrations explicitly; do not present the successful
A seed as evidence that retraining reliably succeeds.

Run from the repository root with root requirements installed:

```powershell
python notebooks/assignment4/q_motion/check.py
python notebooks/assignment4/q_motion/experiment.py --workers 3
```

The user asked to explore better Q behaviour on A and changed environments.
Earlier multi-factor Q packages did not establish whether motion information
alone helped. This experiment changes exactly one representation feature:
one bit appended to the original compact Q key. The bit is one when the
nearest observed ghost is within 48 Manhattan pixels, the inherited power
proxy is zero, and observed relative distance is closing faster than 0.8
pixels per decision. Relative velocity subtracts observed player motion from
observed nearest-ghost motion; it uses no future transition or simulation.
Everything else in the compact key is identical. The encoding is invariant
to rotation of physical positions and velocities. It can at most split each
original key into two codes, although actual visited table sizes also depend
on which routes the policies visit.

Both control and variant use ordinary one-table Q-learning, alpha 0.15,
gamma 0.99, and epsilon 1 to 0.05 over the first 80% of 262,144 interactions,
then 0.05. Every new transition has one Bellman update. No Double Q, replay,
eligibility traces, action masks or extra reward is introduced. Both use
Gymnasium `reset`/`step`, the same native score rewards, five actions, horizon
1,000 and success at 2,000 points. The existing 49-value observation wrapper
is shared; only the coarsening into a Q-table key changes.

Four new agents cover A/B and training seeds 42/7. Identical-config baseline
checkpoints are reused from the registered v3 runs, with their original
source/model hashes checked. Primary comparison is at the full 262,144
updates. A predeclared secondary comparison at 196,608 updates checks the
previous stopping-rule choice without changing the epsilon schedule.
Each checkpoint receives 50 development episodes, seeds 890000--890049,
shared across methods and separate from previous tuning/final/confirmation
seeds. Evaluation is greedy with seeded uniform ties, read-only Q lookup,
unchanged learned parameters and restored training RNG. All 800 development
episodes are logged individually. No D data select this representation.

`results/assignment4/q_motion_v1/` holds the frozen protocol, source snapshots,
reused baseline hashes, new checkpoints, training logs, episode CSVs and
per-seed/aggregate comparisons. Diagnostics report table size, unknown-key
fraction and how often the closing flag activates. Completed jobs can resume
only with identical sources/configuration; partial directories are preserved.
Source/config changes require a new run ID. No report PDF or registered v3
report is rewritten. Original predictions stay unchanged.

Engineering checks cover approaching/receding ghosts, a player escaping
faster than an approaching ghost, powered/far cases, rotation, exact retention
of the original key prefix, numerical Bellman targets and read-only actions.
The power signal is still a capped proxy, not a validated precise timer.
Nearest-ghost selection uses inherited Manhattan distance, so a nearby ghost
behind a wall may be less relevant than another one on the same corridor.

This is development after prior result inspection, with two training seeds.
It is not an independent final A--D comparison or the five-seed Part 3 study.
Read A and B separately; a gain in one with a loss in the other is a trade-off.
A promising configuration must be fixed before fresh confirmation or a frozen
B-to-D transfer test. Neither more state detail nor improved D performance is
assumed before measurement.

The completed motion-bit development pilot found a trade-off:

| Checkpoint | A original | A motion bit | B original | B motion bit |
|---|---:|---:|---:|---:|
| 196,608 | 0% | 50% | 33% | 30% |
| 262,144 (primary) | 0% | 31% | 25% | 21% |

These are means across two trained agents, each with 50 development episodes.
At the primary checkpoint, motion-bit A successes were 62%/0% for seeds 42/7;
there is considerable training-seed variation. The secondary checkpoint's
100%/0% split is not evidence of reliable performance across retraining.
The closing bit does not establish a gain on B or D. Its primary tables were
about 1.4 times the original size, and unknown-key fractions stayed below 1%
on these development episodes. This is not simply a case of most evaluation
states being completely unknown.

## Separate tunnel-connectivity experiment

Inspection of the training-map graph found four degree-one endpoints at
grid rows 4/8, columns 0/17. These are native wrap tunnels: the original B
traces contain 41 observed boundary crossings there (9 for seed 42, 32 for
seed 7). `tunnel_audit.json` records examples and counts. Treating these nodes
as dead ends is an error in the observed graph, not a change in native gameplay.

`TunnelTask` in `tunnel_features.py` separately connects paired left/right
degree-one boundary endpoints. It recalculates only corridor availability,
food/power path regret and path-based threat (observation fields 15--30).
Graph distances remain unweighted hop counts, not exact travel times.
The original Q key and learning settings are unchanged; this candidate has
no closing bit. The frozen motion experiment never imports this repair.

```powershell
python notebooks/assignment4/q_motion/check_tunnels.py
python notebooks/assignment4/q_motion/tunnel_experiment.py --workers 3
```

The engineering test checks 1,244 identical native transitions/rewards across
A/B, including a replayed actual tunnel crossing. Only graph-derived features
change. Random-action tests alone did not visit a tunnel; the targeted replay
is therefore necessary to exercise the repair.

The second experiment waits for the first study to finish and reuses its
identical baseline development records, with record/source/model hashes.
Four newly trained original-Q agents use the repaired features on A/B,
seeds 42/7, the same budget/checkpoints and the same 890000--890049
development episodes. Records are under `results/assignment4/q_tunnels_v2/`.
The first `q_tunnels_v1` attempt stopped at worker startup before training
because Windows resolved an ambiguous `experiment` module to another file.
The revised runner loads its study by absolute path and a spawned-worker
engineering check covers this failure. The failed attempt is preserved.
There are 400 new and 400 reused reference development episodes in its table;
they are not an independent validation or D-based selection. The native
game engine, rewards, five actions, horizon and success threshold are fixed.

The tunnel-only pilot completed without a measured learning gain:

| Checkpoint | A original | A tunnels | B original | B tunnels |
|---|---:|---:|---:|---:|
| 196,608 | 0% | 0% | 33% | 17% |
| 262,144 | 0% | 0% | 25% | 16% |

These use the same two seeds and 50 development episodes per model. The
connectivity repair is supported by native transitions, but this unweighted
feature-graph variant is not promoted as a performance improvement. Graph
hop distances are only approximate travel costs; the study does not isolate
whether that approximation, changed training routes or seed variability
explains the drop.

An offline power-signal audit is also retained in
`power_signal_observations.json`. It describes byte sequences and events only;
no decoder or precise-time claim is changed as a result of this inspection.

## Locked A/D confirmation after development

```powershell
python notebooks/assignment4/q_motion/confirm_transfer.py --workers 3
```

This runs only after both development studies complete. It ranks the original,
motion-bit and tunnel-only configurations by equal-weight mean A/B success
across seeds 42/7, then game score, then earlier checkpoint. This is explicit
development selection across the two predeclared checkpoints; the primary
one-factor result remains the full-budget comparison above.

The script locks the representation, checkpoint, source hashes and model hashes
before accessing new confirmation seeds 960000--960199. A uses A-trained
models; D uses unchanged B-trained models with zero further training. Matched
original-feature controls use the same checkpoint. When an earlier checkpoint
is selected, a full-budget original D reference is also included. Each model
has 200 episodes per tested condition. Results are separate under
`results/assignment4/q_representation_confirmation_v1/`.
No A/D confirmation outcome reselects a representation or checkpoint. This is
still only two trained seeds, not Part 3's five-seed evidence or a full new
A--D/PPO report.

Development selection chose the motion-bit representation at 196,608 updates
(mean success 50% on A and 30% on B). The lock was written before any new
A/D confirmation episodes. The old representation at the same checkpoint,
and the old full-budget D models, are retained as controls.

The complete confirmation contains 2,000 episodes: A and D controls/candidates
for both training seeds, plus the two full-budget D references. Mean game
scores were A 884.7 to 1,755.0 and D 1,011.6 to 782.8 points for matched
196,608-update controls versus motion-bit candidates. These are results for
the selected fixed models, not an independent comparison across many retrainings.
The tunnel-only variant was not selected or tested on D. Neither pilot
establishes improved generalisation to changed environments.
