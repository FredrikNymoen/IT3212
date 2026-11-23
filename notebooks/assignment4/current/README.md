# Historical v3 Part 2: ordinary Q-learning versus PPO, settings A--D

The latest report uses the [November 22 five-seed workflow](../three_hour/README.md).
Its Q/D aggregate is explicitly incomplete after a native map-guard event.
This directory preserves the older two-seed v3 benchmark and its source hashes.
Its report renderer does not reproduce the latest comparison.


This registered v3 benchmark and the later B-only/A/D studies are historical
records. Their models, numerical exports and source hashes remain intact.
The report guide points to the latest complete study. Do not use this old
renderer to replace the latest report or bypass its overwrite guard.

Run from the repository root with `requirements.txt` installed:

```powershell
python notebooks/assignment4/current/audit.py
python notebooks/assignment4/current/experiment.py run --workers 3
python notebooks/assignment4/current/verify.py
```

The main command performs the matched prospective learning-rate search, trains
A/B/C, evaluates 200 frozen episodes per agent/condition, and evaluates B weights
on D without training. Its final historical rendering step targets
`reports/assignment4/Sections/02_learning_settings.tex` and the learning-curve
PNG; the overwrite guard now rejects the reviewed synthesis. Do not bypass it.
The authored `02_discussion.tex` is reviewed separately. No report PDF or old
ZIP is compiled/refreshed. Original predictions and historical source/results
are preserved; report text before replacement is archived with the new run.

Historical command for the original benchmark report only (do not use to
overwrite the current synthesis):

```powershell
python notebooks/assignment4/current/experiment.py report
```

## Fixed comparison

- Native MsPacMan through Gymnasium. Five actions, 45 corrected RAM-derived
  observations, horizon 1,000 decisions and success at 2,000 earned points.
- Ordinary Q: one table, one Bellman update per real transition, gamma 0.99,
  epsilon 1 to 0.05 over 80% of the fixed 262,144-interaction budget. No replay,
  Double Q or eligibility traces. PPO: SB3 clipped PPO, four environments,
  512 decisions/environment/rollout, gamma 0.995, GAE 0.95, four epochs maximum,
  batches 256, clip 0.2, entropy 0.01, target KL 0.02, shared 2x32 actor and
  separate 2x64 value network. Both scale training rewards by 1/100.
- Training seeds 42 and 7; same configuration per algorithm across A/B/C.
  Matching completed B checkpoints can be reused only with source compatibility
  checks. Their original training time/provenance is retained, and every
  reported evaluation is newly run. On a fresh checkout without local pilot
  checkpoints, B is trained normally.
- Equal additional tuning allowance: two rates, one training seed 31415,
  65,536 interactions and 50 B-development episodes per candidate. Q candidates
  0.10/0.15; PPO candidates 0.0001/0.0003. Success then game score selects one
  common rate per algorithm. Previous exploratory work was unequal and is
  disclosed; it is not relabelled as this prospective search.
- Tuning seeds 760000--760049, curve seeds 780000--780019, final seeds
  790000--790199. Final data and the D condition never select configurations.
  Q evaluation is greedy with seeded uniform ties; PPO uses highest probability.
  All model parameters remain frozen.
- A has sticky probability zero; B changes it to 0.25 per emulator frame;
  C changes only reward by subtracting 0.1 points per decision. D uses native
  ROM layout 1 and frozen B models, keeping action/feature shapes and scales.
  Maze-0 feature extraction is the exact pilot function; D uses corrected
  lookup data with the same feature formulas. No actions are masked.
- RAM 116 is inherited as a capped power-related signal. It is not a validated
  precise remaining-time measurement, so that Part 1 explanation remains a
  limitation. The original predictions are never rewritten retrospectively.

## Records and validation

`results/assignment4/part2_current_v3/` contains the frozen protocol, source
snapshots, software/hardware record, tuning results, selected rates, checkpoint
models, raw episode records, curves and compressed per-decision traces.
Each trace records native RAM, before/after features, Q-values or policy
probabilities, selected actions and raw/task rewards, indexed by episode/step.
These are reproducibility records for later episode inspection, not a completed
ten-episode Part 4 analysis or a human review.

`comparison.csv` reports per-trained-agent results. The renderer checks all
3,200 final episodes and their trace rewards/lengths, models/source hashes and
the shared seed lists before writing the report. `verify.py` independently
replays one episode per algorithm/condition and checks native RAM, observations,
values/probabilities and outcomes. Two trained agents per algorithm/setting are
two replications; their 400 evaluation episodes are not 400 trained agents.
This does not satisfy Part 3's five-seed mechanism requirement.

Completed job records can be reused only with the same global protocol/source
hashes. Partial evidence is protected from silent overwrite. If source or
protocol changes, use a separate `--run-id` and preserve the original directory.
The legacy top-level runner and its report exports describe the old v2
baseline. For the current reviewed report, use the synthesis and evidence
references in the report guide rather than either historical renderer.
