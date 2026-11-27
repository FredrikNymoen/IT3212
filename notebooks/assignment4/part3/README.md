# Part 3 — Mechanism experiments

Run from the repository root using the same Python environment as Part 2.
The Part 2 report records an Intel Core Ultra 7 258V with Python 3.13.7,
PyTorch 2.14.1, Gymnasium 1.4.0, ALE 0.12.1 and Stable-Baselines3 2.9.2.
Each baseline is trained again alongside its variants. Use the same machine,
interpreter and worker count for the entire run. The runner records the versions
and CPU actually used for interpreting comparisons with Part 2.

To set up an environment with the reported RL package versions, run this in
an activated Python 3.13 virtual environment.

```powershell
python -m pip install torch==2.14.1 gymnasium==1.4.0 ale-py==0.12.1 stable-baselines3==2.9.2 matplotlib
```

```powershell
python notebooks/assignment4/part3/experiment.py run --output results/assignment4/part3_v1 --workers 4
```

This trains **45 models** (nine configurations × five seeds) for a total of
11,796,480 training decisions. Each model gets 262,144 interactions, the Part 2
budget. Running the same command again verifies and reuses completed jobs.
Interrupted jobs restart from their original seed. Use one invocation per
output directory. To check installation and the pipeline first, add
`--smoke` and use a separate output name. Smoke results are engineering records.

## Controlled comparisons

All main comparisons use Setting B on maze 0 with sticky probability 0.25,
the same observations, actions, score rewards and success/horizon rules as
Part 2. Each algorithm keeps its Part 2 state representation, update rule and
other hyperparameters. Both algorithms use training seeds 17, 29, 101, 314
and 2718. Within each algorithm, each row compares a variant with the same
fresh baseline, paired by training seed. The saved configuration records every value.

| Mechanism | Variant | Baseline → variant |
| --- | --- | --- |
| Exploration | `q-fast-epsilon` | Linear epsilon 1 → 0.05 over 80% → 20% of budget, then held at 0.05 |
| Planning horizon | `q-short-gamma`, `ppo-short-gamma` | Discount 0.995 → 0.95 in separate comparisons |
| Policy exploration | `ppo-no-entropy` | Linear entropy coefficient 0.02 → 0.002 versus zero throughout |
| Update size | `q-low-alpha` | Q learning rate 0.15 → 0.05 |
| Update size | `ppo-low-lr` | PPO initial learning rate 0.0003 → 0.0001, common endpoint 0.00005 |
| PPO data reuse | `ppo-two-epochs` | Maximum epochs per rollout 6 → 2, with target KL held at 0.015 |

We compare schedule endpoints and the two parameter values as controlled
contrasts. For example, lower gamma changes a reward 100 decisions later from
a weight of about 0.61 to about 0.006. PPO's entropy contrast changes the loss
term while retaining action sampling during training. The epoch contrast
changes the maximum, with SB3's existing KL early stopping still enabled.

Development curves use ten separate episodes at 0, 65,536, 131,072, 196,608
and 262,144 interactions. The plots use unsmoothed measurements with faint
seed-level lines. Final evaluation uses every model's full-budget checkpoint
and 200 other episodes per model. Greedy Q uses seeded uniform ties and PPO
uses argmax. No configuration or checkpoint is selected by the final outcomes.

## Read the evidence

- `plan.json` and `sources/` record the protocol, seed lists, packages, CPU,
  source versions and original prediction text.
- `agents.csv` shows individual agents. `summary.csv` gives equal weight to
  each trained agent with sample standard deviations across seeds.
  Outcome-specific lengths use only agents for which the outcome occurred.
  `paired_deltas.csv` shows variant minus baseline for each training seed.
- `curves.csv` and `learning_curves.png` show learning against interactions.
  Each agent's `completed.json` identifies its successful `attempt-*` folder
  and records file hashes. That folder holds the checkpoints, raw training
  logs, development and final episodes, and final per-decision observations,
  actions, native RAM, rewards and Q-values or action probabilities.
- For the planning comparison, inspect first-100-decision score gain, count
  of reward events of at least 200 points and success together. These measure
  observed behaviour. For the entropy comparison, inspect the policy's
  probability entropy on its own frozen evaluation trajectories. This is
  distinct from the diversity of actions taken by argmax.

After the run, follow the original prediction text with clearly labelled
annotations. Compare the
early baseline curves to the predicted early Q advantage and the per-agent
success standard deviations to the predicted Q seed sensitivity. Check mean
training returns alongside frozen success for mismatches. Use the paired rows
and their episode traces to identify a seed or outcome that challenges your
explanation. Cite configuration, training seed and episode seed.

## Follow-up diagnostic

If a variant's success changes unexpectedly in Setting B, one competing
explanation is sensitivity to repeated actions. Record the observed case,
your explanation and the competing explanation before running the diagnostic.
The example below uses the faster epsilon decay. Substitute the variant
relevant to your observation.

```powershell
python notebooks/assignment4/part3/experiment.py diagnose --run results/assignment4/part3_v1 --variant q-fast-epsilon --output results/assignment4/part3_diagnostic_v1
```

The command freezes the two configurations' models for all five training
seeds. It evaluates each policy with sticky probability 0 and 0.25 on the
same 20 fresh episode seeds per condition. Reward, maze, model weights and
evaluation action rule are fixed. The diagnostic saves its plan and checkpoint
hashes before evaluation, then writes raw episodes, traces and per-seed
variant-minus-baseline gaps. A gap that shrinks when sticky actions are
removed supports an execution-sensitivity explanation. A similar gap in both
conditions is more consistent with a general difference in the learned
policies. The diagnostic tests execution, so differences in training data or
representations require their own evidence. Interpret the 20-episode gaps
alongside their variation across seeds.
