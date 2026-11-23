# Post-v3 development improvements

The completed B-only confirmation on the same 200 new episodes per model
(860000--860199) supports the following changes:

| Method | Original mean success | Revised mean success | Change used |
|---|---:|---:|---|
| Ordinary Q-learning | 31.0% | 37.5% | Development-selected earlier checkpoint, 196,608 updates |
| PPO | 74.25% | 94.5% | Motion-aware actor and `stable` update settings |

Means are across two trained agents (42/7), not independent training runs for
each episode. Q improved modestly and remains weak. These results do not
establish improved A/C/D performance. The complete matched v3 A--D report is
preserved rather than mixing the new B results into its old A/C/D tables.

Load and run the selected frozen agents through Gymnasium:

```powershell
python notebooks/assignment4/improvements/selected_agents.py --algorithm Q-learning
python notebooks/assignment4/improvements/selected_agents.py --algorithm PPO
```

`SelectedAgent` also exposes `action(observation)` for the common 49-value
`ImprovedTask` observation. It reads the model hashes locked before confirmation.
The power-time signal remains a declared limitation. Five unsuccessful Q
training packages and the mixed execution-policy diagnostics remain recorded.

Run from the repository root with the root requirements installed:

```powershell
python notebooks/assignment4/improvements/check.py
python notebooks/assignment4/improvements/experiment.py screen --workers 3
python notebooks/assignment4/improvements/replicate.py --algorithm PPO --variant stable --seed 7 --setting B
```

This is development after inspecting the v3 results, including its weak A/D
performance. It is not a new independent final test or a five-seed Part 3
mechanism study. D is never run or used to rank candidates here. The registered
v3 source, report, predictions, final seeds and model files remain unchanged.
Outputs and source snapshots are under `results/assignment4/improvements_v1/`.

Both methods receive the same 49 observed values: the original corrected 45,
nearest-ghost velocity (two values), earned score / 2,000 and native lives / 5.
The last four are observed game information, not future outcomes or additional
rewards. Power remains the inherited capped signal, not a verified exact timer.
All interaction uses Gymnasium `reset`/`step`. Rewards, five available actions,
horizon 1,000, score goal 2,000 and native dynamics are unchanged. No teacher,
action mask, simulation search, Double Q, replay or eligibility traces is used.

The screen gives each algorithm two fresh B candidates, each with 262,144
training decisions, seed 42 and the same 50 development episodes
(810000--810049). Selection uses final-budget success, then raw game score.
The original frozen models are evaluated on those same development episodes.
Confirmation adds seed 7 for the selected configuration and the original
baseline. A confirmation trains A models separately; it does not relabel B
models evaluated with sticky actions disabled as A-trained models.
`selected.json` ranks the new candidates, even if they lose to the baseline.
It is not an automatic recommendation: both first-screen Q candidates were
rejected. Use `replicate.py` to replicate a promising individual candidate
without retraining rejected ones. `experiment.py confirm` remains an optional
bulk command, not a quality gate.

Q variants both retain ordinary one-table, one-transition/one-update Q-learning.
Both use gamma 0.995 and epsilon 1 to 0.02 over the first 40% of training,
then 0.02. `credit` keeps the original compact state with alpha 0.10. `context`
uses alpha 0.15, three levels of directional ghost proximity, a coarse
approaching/receding category when a ghost is near, and a two-bin score phase.
The change bundles are not single-factor causal experiments.

Both PPO candidates use a learned shared action scorer with 23 inputs per
action and two 32-unit tanh layers, and a 49-input two-64-unit value network.
The actor adds nearest-ghost motion, directional food density, score and lives
to its previous inputs. `motion` retains the baseline update settings.
`stable` decreases learning rate 0.0003 to 0.00005 and entropy coefficient
0.02 to 0.002 over the single uninterrupted training run. It uses GAE 0.97,
up to six epochs, clip 0.15 and target KL 0.015. Both retain gamma 0.995,
four 512-step environments and minibatches of 256.

Engineering checks compare 900 native transitions/rewards against the old
wrapper, Gymnasium reset/step determinism and spaces (skipping reconstruction
of the frozen wrapper's unsupported constructor spec), numerical ordinary-Q
targets, read-only action selection and NumPy/Torch PPO probabilities.
Frozen evaluation checks learned parameters and model-file SHA-256 values.
Development episodes do not update model parameters. Training times use shared
CPU compute with at most three workers and one numerical thread per worker.

Only measured replicated gains should be described as improvements. These
commands do not automatically replace the main report or its controlled A--D
results. A new complete locked comparison is needed before such replacement.

## Follow-up Q screen

The first two Q packages failed on seed 42: `credit` achieved 12% and `context`
6%, versus 42% for the original model on the same 50 development episodes.
They remain recorded as failed candidates. `q_refinement.py` therefore tests
two additional ordinary-Q variants while retaining the original compact state,
gamma 0.99 and epsilon schedule (1 to 0.05 over 80% of training):

- `smooth`: constant alpha 0.05.
- `decay`: per-state/action alpha `max(0.02, 0.25/sqrt(1 + visits/10))`.

```powershell
python notebooks/assignment4/improvements/q_refinement.py --workers 1
```

These additional Q-only attempts make cumulative search effort unequal again.
Records and source snapshots are separate under `results/assignment4/q_refinement_v1/`.
They use the same 262,144-decision budget and development episode seeds, not D
or the registered v3 final episodes. Neither smaller updates nor added state
detail is assumed to help before results are measured.

The constant-alpha-0.05 follow-up also failed (2% on seed 42). A further
`fast_q.py --alpha 0.3` test retains the original state, gamma and epsilon
schedule and tests faster ordinary updates. Its records are under
`results/assignment4/q_alpha_followup_v1/`. No result is assumed in advance.

`policy_check.py` separately diagnoses frozen execution on A/B development
seeds 830000--830029: Q epsilon 0/0.01/0.03 and PPO argmax or categorical
temperature 0.5/1. It changes execution only, with zero additional training.
These results must not be labelled as training improvements or mixed into v3.

## Locked confirmation

The PPO `stable` package reached 50/50 development successes with each of
training seeds 42 and 7. After these models were fixed, `validate.py` locked
their hashes and the baseline hashes before evaluating 200 unused B episodes
per model (860000--860199):

```powershell
python notebooks/assignment4/improvements/validate.py --algorithm PPO --workers 2
```

Records are separate under `results/assignment4/improvement_confirmation_v1/`.
The source/seed/model registry is written before any confirmation episodes.
No confirmation result selects a configuration or intermediate checkpoint.
This remains B-only and two-seed evidence; A/C/D results are not implied.

The completed locked PPO comparison is:

| Training seed | Original PPO | Revised PPO (`stable`) |
|---|---:|---:|
| 42 | 66.0% | 96.0% |
| 7 | 82.5% | 93.0% |
| Mean across trained agents | 74.25% | 94.5% |

Each cell is 200 episodes on the identical 860000--860199 seeds. This is a
fresh comparison for these fixed models, not the old v3 episode set or 400
independent training runs. The model/change bundle improved both training
seeds; this does not isolate a causal contribution from one hyperparameter.

## Q stopping-rule revision

The larger-alpha-0.30 test also failed (0/50 development successes). The
per-visit decay candidate achieved 34%, versus the baseline's 42% on those
50 episodes. These changes are not adopted.

The original B development curves instead favour the common checkpoint at
196,608 updates: mean success 40%, versus 32.5% at 262,144, on the original
20 development episodes per trained agent. `select_q_checkpoint.py` ranks
those development curves, locks one common checkpoint and its model hashes,
then compares both selected Q models against the final v3 models on the
unused 860000--860199 episodes. It uses neither v3 final episodes nor D to
choose the checkpoint. This is a stopping-rule change after inspection.

```powershell
python notebooks/assignment4/improvements/select_q_checkpoint.py --workers 3
```

Records are in `results/assignment4/q_checkpoint_confirmation_v1/`. The original
run still consumed the full 262,144-decision budget; the deployed earlier
checkpoint has 196,608 updates. Do not describe those weights as having all
262,144 updates, or describe the stopping point as selected on confirmation.
No new Q training or change of Q-learning's Bellman update is involved.

The locked Q confirmation completed:

| Training seed | Final-budget Q | Development-selected earlier Q |
|---|---:|---:|
| 42 | 32.0% | 36.0% |
| 7 | 30.0% | 39.0% |
| Mean across trained agents | 31.0% | 37.5% |

Each cell uses 200 identical confirmation seeds. This modest gain supports
retaining a development-selected checkpoint instead of assuming the last
checkpoint is best; it is not evidence that Q now solves the task reliably.
