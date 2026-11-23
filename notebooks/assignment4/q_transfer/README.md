# Q transfer follow-up after independent result verification

The completed valid D confirmation found a modest conditional gain:

| Frozen B model | Mean D success | Successes / episodes |
|---|---:|---:|
| Original, 196,608 updates | 2.5% | 10 / 400 |
| Original, 262,144 updates | 2.0% | 8 / 400 |
| Power-route candidate, 262,144 updates | 4.5% | 18 / 400 |

Each mean covers two trained agents, each with 200 identical fresh D episode
seeds. At matched full budget, seed 42 was unchanged (2% versus 2%); seed 7
improved (7% versus 2%). The gain is therefore small and training-seed evidence
is limited. 95.5% of candidate episodes still fail. This is not a generally
reliable Q or transfer method, and the old report is not replaced.

The user asked to confirm the earlier numbers and try to improve transfer.
`../q_motion/verify_native.py` independently recounted all 2,000 earlier
confirmation episodes and replayed 16 outcome cases. Every step's cumulative
Gymnasium reward matched the native BCD score registers (RAM 120--122).
Those earlier A/D counts are correct. The replay record is
`results/assignment4/q_representation_confirmation_v1/independent_native_replay.json`.

Run from the repository root:

```powershell
python notebooks/assignment4/q_transfer/check.py
python notebooks/assignment4/q_transfer/run_study.py --workers 3
python notebooks/assignment4/q_transfer/run_resources.py --workers 2
python notebooks/assignment4/q_transfer/run_power_route.py --workers 2
```

This new study trains on B only, with seeds 42/7 and 262,144 decisions per
candidate. Alpha 0.15, gamma 0.99, epsilon 1 to 0.05 over 80% of training,
native score rewards, five actions, target 2,000 and horizon 1,000 are unchanged.
Every real transition makes one ordinary single-table Bellman update.
No Double Q, replay, eligibility traces, masks or teacher actions are used.
All environment interaction is Gymnasium `reset`/`step`.

The first two candidates each change one original key component:

- `threat`: directional danger uses three levels at observed threat values
  0.13/0.36, retaining the original near-danger boundary. Other fields are
  unchanged, including the original binary power state. This coarsens the
  existing path-based threat values rather than adding a map-specific ID.
- `power`: the existing capped power proxy has zero/low/high bins, split at
  16/63. All other fields, including binary danger, stay unchanged. No RAM
  decoder is changed. These are observed proxy bins, not validated seconds
  of remaining protection.

Neither candidate uses the previous closing bit or tunnel-graph repair.
The common audited 49-value observation is retained. The inherited graph and
power approximation remain limitations, not hidden changes to native gameplay.

An additional B-only `resources` package retains the original key and appends
two bits from the already observed 49-vector: last remaining life, and at
least 1,500 earned points. The original Q key discards both score and lives,
although task termination depends on them. This package tests that information
loss without reward shaping or a new game engine. It is a two-variable
representation package, not a single-factor causal mechanism study.
`run_resources.py` uses the same trainer, budget, seeds, schedule and development
episodes through an explicitly extended class registry. Its separate records
and source snapshots are under `results/assignment4/q_resources_v1/`.

A separate `power_route` candidate appends the existing best-power-route
direction mask only when the original key indicates nearby danger and the
power proxy is zero. The original Q representation discards this route field;
PPO's actor already receives it. The mask is a state feature, not an action
filter or preferred-action rule. The same Q update, rewards and training
schedule are retained. `run_power_route.py` records two B training seeds under
`results/assignment4/q_power_route_v1/`.

Both checkpoints (196,608 and 262,144) are evaluated on the same 50 new
B-development episodes, seeds 1000000--1000049. Matching original Q models
are reused with source/model hashes. Per-seed results and coverage diagnostics
are stored under `results/assignment4/q_transfer_v1/`; failed candidates are
preserved. The full-budget comparison is primary; earlier weights are an
explicit development stopping candidate. The run consumes 262,144 decisions
even if an earlier checkpoint is eventually selected.

The completed first B-development study found no source-task improvement:

| Checkpoint | Original | Threat levels | Power-proxy levels |
|---|---:|---:|---:|
| 196,608 | 24% | 22% | 20% |
| 262,144 | 41% | 12% | 21% |

Each entry is a mean across two trained agents, each evaluated on 50 identical
development episodes. These differ from the old confirmation seed set and
must not be substituted for the old report's values.

The additional B studies also completed:

| Checkpoint | Resources | Power-route direction |
|---|---:|---:|
| 196,608 | 5% | 25% |
| 262,144 | 2% | 27% |

These use the same seeds and development episodes. The strongest new candidate
is therefore power-route direction at 262,144 updates, while the original
full-budget model remains stronger on B (41%). It was locked for the D test
as a transfer hypothesis, not presented as a demonstrated B improvement.

The planned transfer confirmation fixes the strongest **new** representation
using mean B-development success across seeds, then game score and earlier
checkpoint. It compares that fixed candidate against the original checkpoints
on 200 fresh D episodes per model, without further training. Original 196,608
and 262,144 checkpoints are retained as controls. Choosing the best new candidate
does not assert that it beat the baseline in B. D results never revise the
selected configuration.
The selection pool contains threat, power, resource and power-route candidates
after all B studies finish. No valid new D confirmation is run before this
choice is locked.

```powershell
python notebooks/assignment4/q_transfer/confirm_d.py --workers 3
```

Complete all three B training commands first; confirmation can wait for the
registered studies, but never trains a model itself.
It writes model/source hashes and the chosen configuration before any new
D episode. It uses seeds 1060000--1060199, 200 episodes per model, and checks
native BCD score agreement on every decision. Two candidate models and the
two original models at each of 196,608/262,144 updates give 1,200 D episodes.
Records are separate under `results/assignment4/q_transfer_confirmation_v2/`.
There is no learning on D, and no post-confirmation reselection.

The first waiting confirmation process compiled an earlier selection plan
before the power-route extension was added. That process was stopped and its
directory is marked aborted. Its outcomes were not inspected or used. The v2
process uses a fresh episode set and verifies that its loaded source files
still match their hashes after waiting. Do not resume or report v1 as valid
confirmation, and do not edit the registered source while a run is active.

This follows inspected D outcomes. New episode seeds do not make the same map
an independent unseen domain for the investigators; it remains untrained for
the agents. Keep this chronology explicit, do not reuse old final episodes
for tuning, and do not silently replace the complete v3 report. Two training
seeds are not the required five-seed Part 3 mechanism study. Prior cumulative
search effort is unequal.
