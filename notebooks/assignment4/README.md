# Assignment 4: MsPacMan, Part 2

The current comparison uses the [November 22 five-seed study](three_hour/README.md)
for A/B/C and the [new frozen-transfer evaluation](transfer_evaluation/README.md)
for D. All ten original B checkpoints receive 200 fresh D episodes without
training or reselection. D observed success is Q4.5% and PPO14.7%; one Q map
exit is censored, giving a4.5--4.6% success bound. The original interrupted
test is preserved. The report explicitly distinguishes censoring from failure.
Historical descriptions and reproduction commands below retain their original
scope; their renderers must not overwrite the new report.

## Historical studies

The [further Q transfer study](q_transfer/README.md) independently verified the
earlier A/D numbers against native score RAM, then tested four B-trained
representation packages. Fresh frozen D confirmation gave 4.5% for the
power-route candidate versus 2.0% for matched full-budget Q controls. Only
one of two training seeds improved; this remains weak transfer evidence.
The November 19 report included these locked confirmations alongside the
original benchmark; that text is now archived. Baseline records and predictions
are preserved, and the report guide identifies the latest evidence.

The [isolated Q representation study](q_motion/README.md) investigated A and
transfer after the B-only improvements. Locked confirmation found an A gain
with strong seed variation (0% to 50%, candidate seeds 100%/0%) but worse D
transfer (4.5% to 2.5% at matched checkpoints). It is not a general replacement
for Q. A separate tunnel-graph correction was validated geometrically but
did not improve the development scores. All results and reproduction commands
are separately labelled in the report.

The earlier [B-only improvement study](improvements/README.md) confirmed a modest
ordinary-Q gain through development-selected stopping (31.0% to 37.5%) and a
larger PPO gain (74.25% to 94.5%) on matched new episodes. It includes commands
to load the selected agents. At that stage revised PPO A/C/D remained unverified; the later
Q A/D studies above have separate evidence and limits.

The [complete v3 A--D workflow](current/README.md) reproduces the original
benchmark, not the newer report synthesis:

```powershell
python notebooks/assignment4/current/experiment.py run --workers 3
python notebooks/assignment4/current/verify.py
```

Run these historical commands from the repository root. The v3 comparison uses
ordinary single-table Q-learning and PPO, corrected shared observations,
262,144 training interactions, matched additional tuning and 200 frozen
evaluation episodes per trained agent/condition. Its historical report renderer
does not include the follow-ups and its overwrite guard rejects the current
reviewed synthesis. Use the report guide for the current evidence sources.
The original prediction sheet is preserved and no report PDF is compiled.

The [ordinary-Q B pilot](pilot/SINGLE_Q.md) preceded this full study. Older
Double Q/replay experiments are archived side investigations and are not the
ordinary-Q result required for the main comparison.

## Historical v2 baseline

The remaining instructions and the top-level notebook/runner below document the
preserved old baseline. Their legacy report command renders old results;
neither historical renderer should overwrite the current report synthesis.

The registered experiment below is a preserved historical baseline. Later
investigation found incorrectly transcribed pellet-layout cells and an omitted
horizontal coordinate offset. Do not treat its feature geometry as verified or
submit its results as the final improved implementation. See
[pilot/FURTHER_EXPLORATION.md](pilot/FURTHER_EXPLORATION.md) for the corrected,
isolated development investigation, and [pilot/README.md](pilot/README.md) for
the earlier unsuccessful short pilot. Original predictions and baseline sources
are preserved so the chronology and old numerical results remain auditable.

Run from `notebooks/assignment4/` with the root `requirements.txt` installed.
Use [02_learning_settings.ipynb](02_learning_settings.ipynb) with **Restart
Kernel and Run All**, or run the documented command:

```powershell
python part2_experiment.py run
```

From the repository root, the equivalent is:

```powershell
.venv/Scripts/python.exe notebooks/assignment4/part2_experiment.py run
```

The default run ID is `part2_controlled_v2`. The runner resumes completed jobs only
when the protocol and recorded source hashes match. Use a new `--run-id` for a
changed experiment. Do not edit code during a running experiment. For example,
`--run-id part2_larger_budget --budget 131072` is a new experiment, not a
replacement for the first run. Smaller engineering runs are not main results.

## What follows Part 1

The original predictions in `reports/assignment4/Sections/01_predictions.tex`
are preserved. The selected option is **MsPacMan**. Tabular Q-learning uses a
coarse discretisation of object features; PPO uses the same information in a
finer, continuous representation. Success is at least **2,000 newly earned game
points** before all lives are lost or the episode limit is reached. This score
threshold does not establish that a whole level was completed.

The team confirmed in the conversation that **game-score rewards in A/B were
agreed with the instructor**, adapting the brief's terminal-only success reward.
This is explicitly documented in the report and protocol. The original
predictions are not rewritten to hide this difference.

## Implemented design

- Environment: `gymnasium.make('ALE/MsPacman-v5', ...)`, RAM observations,
  fixed frame skip 4, mode 0. Agents interact only through Gymnasium
  `reset(seed=...)` and `step(action)` with separate terminal/truncation flags.
- Actions: no-op, up, right, left, down. The wrapper's five-action mapping is
  unchanged across algorithms and settings.
- A: sticky-action probability 0; score-increment reward.
- B: sticky-action probability 0.25 at emulator-frame level; same score reward.
- C: B plus a **0.1-point penalty per decision**. This is a declared project
  choice, not a measured real-world cost.
- D: frozen B agents on a different native ROM maze selected through RAM byte 0
  before the first action. Training/development use selector 0; D uses 1.
  Changing `mode` alone was checked and did not change the initial maze.
  Natural level transitions raise an error instead of exposing the holdout.
  ROM layout-dependent behaviour may change too; this is not a geometry-only test.
- Horizon: **1,000 decisions**, chosen to bound computation. The agent observes
  time remaining. Since Part 1 defines failure at the horizon, it is a terminal
  finite-horizon task failure, not a continuing-task truncation to bootstrap over.
  Losing a single life does not end an episode. No hidden no-op warm-up is used.
- Reward scaling: both agents divide rewards by 100 internally. All score and
  task-return exports use unscaled points. C returns cannot be compared directly
  with A/B returns; compare success/failure and common game scores instead.

The feature wrapper supplies position, nearest ghost and pellet offsets and
Manhattan distances, pellet counts in four directions, power-pellet countdown,
orientation and time left (15 values). Q-learning replaces fine values with
coarse bins/directions and a dominant pellet direction. The baseline caps the
internal RAM-116 signal at 63 and normalises it. Later ghost-bonus replays show
values above 63, so the earlier engineering check did not establish a precise
remaining-power countdown. Retain this limitation when interpreting the saved
baseline. Features are fixed, with no fitted preprocessing. Both
representations omit parts of the emulator state, and their different resolutions
prevent an algorithm-only interpretation.
The distances are Manhattan coordinate distances, not shortest paths through
the maze. Wall/corridor geometry is not supplied explicitly, so a nearby pellet
can be separated by a wall. Position can help learn the training layout but
does not guarantee useful navigation on the held-out layout.

## Selection and evaluation

For **each algorithm and A/B/C setting**, two learning-rate candidates receive
16,384 training interactions each, training seed 31415 and 50 development
evaluation episodes. Mean development success across A/B/C, then mean game
score, selects **one common learning rate per algorithm**, breaking final ties
by candidate index. All hyperparameters therefore stay fixed across A/B/C;
only sticky actions and the specified step penalty change. Both algorithms receive equal search
allowance. The search is deliberately small and does not establish an optimum.

The initial `part2_score_v1` pilot tuned each setting separately. This was
corrected after partial evaluation inspection because the brief requires other
conditions to stay fixed. Only development data choose the common rates. The
evaluation seeds had already been used, so this revision is **not a new
independent test**. Twelve unchanged final agents and the original development
search can be reused with explicit source-run records; six changed agents are
retrained. The original measurements and exact original Python source snapshots
remain separate in the pilot directory.

To reproduce the controlled experiment from scratch, run the default command.
To reuse the compatible pilot jobs explicitly, run
`python part2_experiment.py run --reuse-from part2_score_v1`. The reuse check
requires identical environment/agent sources, predictions, task definitions,
seeds and budget. Changed hyperparameters are never relabelled as reused runs.
The reuse manifest records original directories/protocol hashes, and each
reused checkpoint retains its original measured timing provenance.

Each final agent trains for
**65,536 interactions**, using seeds **42, 7 and 2026**. The three seeds assess
Part 2 variation; they do not satisfy Part 3's separate requirement of at least
five seeds per tested mechanism configuration.

Learning curves evaluate 20 development episodes at 0, 16,384, 32,768, 49,152 and
65,536 training decisions. No smoothing is used. Final performance uses **200
held-out evaluation episodes per trained agent/condition**. D reuses each B
model with zero new training interactions. Training, tuning, curve and final
evaluation seeds are disjoint. Shared seeds do not imply identical trajectories.

The final Q policy is greedy with seeded uniform tie-breaking; PPO takes its
highest-probability action. Evaluation uses read-only Q lookups, no network
updates, and restores the training policy RNG. Parameter fingerprints are checked
before/after evaluation and match between B and D. Do not choose new settings
from the final results. Report means and sample SD across trained agents rather
than treating all evaluation episodes as independent training runs.

## Files and provenance

| Source | Role |
|---|---|
| `mspacman_env.py` | Gymnasium task wrapper and fixed features |
| `rl_agents.py` | Tabular Q-learning and PyTorch clipped actor-critic PPO |
| `part2_experiment.py` | Audits, equal-budget search, training and frozen evaluation |
| `part2_report.py` | Figures and inline report tables from finished exports |

`results/assignment4/part2_controlled_v2/` contains the protocol, source/ROM hashes,
CPU/software/thread provenance, selected settings and per-seed summaries.
Each tuning/final run directory contains its configuration, training episode
records, final checkpoint and evaluation episode table. Final runs additionally
include learning curves, PPO update histories where applicable, and compressed
per-decision evaluation traces. B runs also contain D's episode table/traces.

Trace archives contain native RAM, action, Q-values or policy probabilities,
reward, episode ID and step ID. Episode seeds and actions permit replay of
recorded trajectories for later Part 4 inspection; do not invent alternative
outcomes. Record which configuration/run/episode supplied every later figure.

CPU training times exclude curve evaluations. Evaluation times include the
200-episode loop and trace collection, but exclude archive compression. Up to
three workers share the CPU, each with one numerical thread. These are measured
shared-compute runtimes, not isolated inference benchmarks. Checkpoints pickle
locally generated agents and are intended for this recorded software environment.

The renderer writes the measured Part 2 section and one vector figure directly
to `reports/assignment4/`. A local PNG preview remains under the ignored run
directory. It does not compile the report or create a ZIP. Refresh presentation
without training using `python part2_experiment.py report`. Authored discussion
is separate in `reports/assignment4/Sections/02_discussion.tex`; review it when
results change.

## Validation and sources

Run `python part2_experiment.py audit` for the Gymnasium contract, finite-horizon
and success rules, reward isolation in B/C, actual map change, frozen evaluation,
training updates and NumPy/Torch PPO inference agreement. Audit records and
images are separate under `results/assignment4/engineering_audit/`.

Primary references used to implement and check the experiment:

- [Gymnasium environment API](https://gymnasium.farama.org/api/env/).
- [ALE MsPacMan specification](https://ale.farama.org/environments/ms_pacman/).
- [OCAtari MsPacMan RAM/object definitions](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py).
- [HackAtari MsPacMan maze-selector modifications](https://github.com/k4ntz/HackAtari/blob/master/hackatari/games/mspacman.py).
- [PPO's clipped objective](https://arxiv.org/abs/1707.06347).

The assignment brief and team's supplied predictions define the task. Budgets,
episode horizon, penalty size, feature bins and small tuning grid are documented
implementation choices. No original prediction text or human review is generated
by the experiment.
