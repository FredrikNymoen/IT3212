# Part 2: MsPacMan, Q-learning versus PPO

The current protocol is `mspacman_maze_pellets_walls_v8`. The completed
`results/assignment4_v7` study used older observations. Keep its raw results:
they describe that implementation, and cannot measure the effect of these fixes.
Old models are refused by the current replay command. Replaying them requires
the matching original code; their source hashes are in their manifests.

## Validate and run

From `øv4ny/code`, using Python with the dependencies in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -X utf8 -m unittest discover -s tests -v
python -X utf8 part2_run.py --selfcheck
```

Then use this single command for a new, complete study:

```powershell
python -X utf8 part2_run.py --tune --results ..\results\assignment4_v8
```

It performs the equal-allowance search once, then runs all six algorithm/setting
combinations with five training seeds. B also produces D transfer evaluations.
The command runs in the foreground: check its exit status, not the count of all
Python processes on the machine. Use a fresh directory for each new study;
existing runs are intentionally not overwritten or resumed. Do not change the
source while a study is running.

```powershell
python -X utf8 part2_tables.py --results ..\results\assignment4_v8
```

This writes tables and learning curves beside the raw logs. The result directory
is mandatory; there is no automatic fallback to an older experiment. To render
the existing study without retraining:

```powershell
python -X utf8 part2_tables.py --results ..\results\assignment4_v7
```

These commands do not establish that a larger training budget would improve
performance. A longer study, if chosen, needs its own directory and the same
budget for both algorithms, for example `--budget 500000 --results
..\results\assignment4_v8_500k`.

## Fixed protocol

| Item | Value |
|---|---|
| Budget | 100,000 environment interactions per trained agent, startup included |
| Training seeds | 42, 7, 2026, 123, 999 for both algorithms |
| Primary evaluation | 200 episodes per agent, seeds 300,000-300,199, updates off |
| Curves | Every 25,000 interactions; 20 episodes from seed 200,000; no smoothing |
| Search | Four candidates per algorithm; each trains on five seeds for 25,000 interactions on B |
| Search evaluation | 20 episodes per candidate/seed, from seed 100,000; final episodes excluded |
| Primary policies | Greedy Q with uniform ties; sampled PPO |
| Additional policy | Greedy PPO, recorded and reported separately |
| Success | Raw game score >= 2,000; a one-time +100 terminal bonus |
| Settings | A sticky 0; B sticky 0.25; C B plus -0.5 per decision; D unseen maze 1 |

Both algorithms receive 17 features. Q discretises every feature, including
pellet distance; PPO uses their continuous values. Discretisation still loses
information, so conclusions concern these implementations and representations.
The 100,000-step budget measures learning under limited interactions, not
convergence. Report hardware and recorded training/evaluation times as well.

## Observation fixes and limitations

- Corridor probes examine every intervening pixel. Vertical exits are blocked;
  horizontal exits follow the maze's side-tunnel convention. This remains a
  screen-based probe, not a verified legal-action mask for the ROM.
- Regular pellet locations are estimated from the initial screen. Only raw
  rewards of 10 or 20 remove one or two nearest estimated dots. Ghost/fruit
  rewards do not remove unrelated dots. Power pellets and mixed rewards are not
  updated; occlusion and ambiguous nearest matches can leave the map stale.
  A correct removed count does not prove correct removed locations.
- The frightened timer and RAM-to-screen offset are empirically checked.
- D modifies the starting maze byte. Describe that modified start state and
  the limited transfer information in absolute coordinates and local features.
- The original predictions must be preserved. Annotate representation changes
  and do not claim nearest-pellet features describe larger pellet clusters.

## Replay a new-protocol episode

```powershell
python -X utf8 part2_run.py --replay ..\results\assignment4_v8\main_Q_B_42\model.npz --condition D --eval-seed 300001
```

Add `--greedy` for the additional PPO mode. Replay requires the matching protocol
and experiment manifest. Configurations, raw episode logs, software versions,
source hashes and the first two primary traces are saved with each study.

## Watch the best saved model choose actions

From `code/`:

```powershell
python -X utf8 watch_best.py
```

The viewer defaults to the strongest checkpoint from the group with the highest
mean success in the completed common test. Currently this is PPO in B, training
seed 2026, with 299 successes in 300 episodes (99.7%). Playback uses the navigation
observations, sampling temperature 0.2 and episode seed 996300 by default. If
the common test is unavailable, the viewer falls back to the cooler PPO study.
The recorded success is for the selected checkpoint and evaluation episodes.

The left panel shows the game state before the action. The right panel shows
all action probabilities (or Q-values), highlights the chosen action in green,
and displays the score, reward and outcome after the transition. Sampled PPO
can choose an action other than its most probable one. Sticky actions can make
the emulator repeat earlier input despite the current request.

Controls: Space pauses/plays; Right advances one decision; N starts the next
episode; R replays the current seed; Escape closes the window. The default is
three episodes, matching the selected experiment's action RNG.

```powershell
python -X utf8 watch_best.py --paused --seed 996301
python -X utf8 watch_best.py --condition B
python -X utf8 watch_best.py --results ..\results\assignment4_v8
python -X utf8 watch_best.py --model ..\results\assignment4_v8\main_Q_B_42\model.npz --condition D
```

Use `--greedy` to inspect the additional highest-probability PPO mode. A PNG
preview and a JSON decision trace can also be saved without opening a window:

```powershell
python -X utf8 watch_best.py --headless --steps 25 --output ..\results\viewer_preview.png --trace ..\results\viewer_preview.json
```

Only use matching-protocol models: the viewer verifies both protocol and source
hashes. It performs no learning and does not overwrite the study's logs.

## Run 500k training concurrently on Windows

From `code/`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_parallel.ps1
```

The launcher selects one Python executable (even if multiple installations are
on PATH), checks dependencies, tunes both algorithms once, starts the remaining
five shards, checks process exit codes and generates the report. Logs are in
`results/assignment4_v8_500k_parallel`. An existing empty directory is accepted;
a directory containing files is refused. For another study, supply a fresh path:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\train_parallel.ps1 -Results ..\results\assignment4_v8_500k_parallel_2
```

`-PythonExecutable` can select a specific Python installation. `-Smoke` tests all
six launches with the short technical protocol and does not run tuning or the
500k study. Stop any earlier broken interactive wait loop with Ctrl+C before
starting the launcher. Do not target the completed v7 or v8 experiment folders.

## Test navigation in stochastic Setting B

From `code_wall_fix/`, evaluate the hand-written reference policy and run the
paired PPO comparison:

```powershell
python -X utf8 reference_policy.py --results ..\results\assignment4_reference_B
python -X utf8 compare_navigation.py --results ..\results\assignment4_navigation_ppo_100k_B
```

The reference uses a corridor graph, dot and power-pellet targets, and a heuristic
for nearby ghosts. Its fallback uses seeded random cardinal actions. It was
piloted on seeds 910000–910015. The reported run uses seeds 300000–300199, matching
the baseline evaluation. Its outcomes describe a hand-written controller.

The PPO comparison trains five paired seeds at 100,000 interactions in B. Both
variants use the wall fix and the same architecture, learning rate and evaluation
seeds. The navigation variant adds corridor distances to food and ghosts, recent
player and nearest-ghost movement, and the previous requested action. Ghost
status is approximated using the existing global frightened timer. Interpret any
change as the effect of the combined feature set. Each run saves its model,
episode logs and traces. The directory also contains a source manifest and a
paired summary.

From the same directory, run the imitation and fine-tuning experiment:

```powershell
python -X utf8 imitate_reference.py --results ..\results\assignment4_imitation_B
```

It gathers 64 episodes from the reference controller, holds out eight entire
episodes to select the cloning checkpoint, and initializes PPO from the cloned
weights with a fresh optimizer. Three student seeds each receive 30,000 online
PPO interactions. The reference and both student stages are evaluated on fresh
seeds 930000–930199. The run saves demonstrations, checkpoints, evaluation CSVs,
and a comparison. Use a fresh output directory and add `--smoke` for a short
integration check.

To test DAgger using the saved imitation checkpoints, run this from
`code_wall_fix/`:

```powershell
python -X utf8 dagger_reference.py --results ..\results\assignment4_dagger_B
```

Each of the three students collects two rounds of 12 labeled episodes. The
reference labels every decision and takes the action with probability 25% in
round one and 10% in round two. Two episodes per round are reserved for
validation. The remaining decisions are added to the original demonstrations.
Each student then receives 30,000 PPO interactions. The original fine-tuned
checkpoints, DAgger checkpoints, and DAgger plus PPO checkpoints all play
200 sampled episodes on fresh seeds 950000–950199. This compares the full
DAgger workflow with the previous workflow, which used fewer interactions.
Use `--source` to choose the imitation run and `--smoke` for an integration check.

To compare longer PPO fine-tuning budgets from the saved DAgger weights, run
this command from `code_wall_fix/`:

```powershell
python -X utf8 extend_dagger_ppo.py --results ..\results\assignment4_dagger_budget_B
```

The study reuses the saved 30,000-interaction checkpoints and trains independent
60,000- and 100,000-interaction runs from the same DAgger weights. The three
training seeds share one selected budget based on mean success over 100
validation episodes per model. Raw score and shorter budget break ties.
The selected budget and the original checkpoints then play the same 200 fresh
test episodes per model. The manifest records the source hashes and seed ranges.

To compare uniform and targeted DAgger corrections from the saved 60k PPO
checkpoints in B, run from `code_wall_fix/`:

```powershell
python -X utf8 targeted_dagger.py --results ..\results\assignment4_targeted_dagger_B
```

Both correction strategies use one shared pool of 20 student episodes per
seed, with four entire episodes reserved for cloning validation. Each strategy
uses the same number of labels and exactly 60,000 additional PPO interactions.
The targeted strategy gives priority to decisions at 1,500 points or higher,
or when a nearby ghost is approaching and the global frightened timer is low.
Both updated policies and the starting checkpoint play the same 200 fresh
episodes per seed. The report includes the fraction of episodes reaching
2,000 among those that reached 1,500. Use `--smoke` for an integration check.

Run the three follow-up score experiments from `code_wall_fix/`:

```powershell
python -X utf8 score_diagnostics.py --results ..\results\assignment4_score_diagnostics_B
```

The script uses the saved targeted DAgger checkpoints. It selects an action
temperature from 1.0, 0.75 and 0.5 using 60 validation episodes per model.
The selected temperature and 1.0 then play the same fresh test seeds. It also
replays four failed episodes per training seed with final scores from 1,800
to 1,999, testing each first action at 1, 4 and 8 decisions before the end.
Each branch restores the emulator and policy random state. Survival is measured
over the next 24 decisions. Finally, it compares 30,000 more PPO interactions
with the original training reward and with a 25-point training penalty per life
lost. Final evaluation uses 200 fresh episodes per model and the original
success rule. Add `--smoke` for a short integration check.

The late collision checks above usually fall inside the death animation. To
test choices before the animation, use the same episodes with:

```powershell
python -X utf8 replay_before_freeze.py --diagnostics ..\results\assignment4_score_diagnostics_B
```

This reconstructs the original action prefix from reset to preserve the
emulator's sticky-action state. It verifies the recorded failure on the original
branch and tests alternative first inputs 1, 4, 8 and 16 decisions before the
terminal freeze, with up to 64 decisions of follow-up.

To compare PPO trained at temperature 0.5 with the saved 30k control trained
at temperature 1.0, run from `code_wall_fix/`:

```powershell
python -X utf8 train_cool_ppo.py --results ..\results\assignment4_cool_ppo_B
```

Both training groups start from the same targeted DAgger checkpoints and use
30,000 interactions and the original reward. The colder variant uses
temperature 0.5 in both action collection and the PPO objective. The starting
checkpoints and both trained groups are evaluated at 0.5 on fresh seeds
996000–996199. The script checks the gradient through the temperature scaling.

To run a timed PPO search in stochastic Setting B from those checkpoints:

```powershell
python -X utf8 improve_ppo_session.py --minutes 60 --results ..\results\assignment4_ppo_hour_B
```

The search tests continuation budgets, learning rates, rollout lengths,
sampling temperatures, and a KL penalty against each continuation's starting
policy. It screens candidates on seeds 997000–997039 and confirms two
finalists per training seed plus the original temperature-0.5 baseline on
998000–998079. The selected checkpoint per training seed and the baseline
then play 200 fresh paired episodes from seed 999000. The runner reserves
time for evaluation and records intermediate results in `progress.json`.
The completed run and its follow-up experiments are summarized in
[the one-hour PPO results](results/assignment4_ppo_hour_B/summary.md).
