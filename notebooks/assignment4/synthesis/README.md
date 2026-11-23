# Part 2 synthesis

This is the historical Part 2/map-D interface. For the current full evidence
brief and instructor-clarified observation D, use
[the completion guide](../completion/README.md). Its mechanism study and
episode analysis extend this synthesis; they do not rewrite original predictions.

Branch `assignment4-part2-synthesis` combines reviewed ideas from Johan,
Elling, Mats and Fredrik. It uses **ordinary single-table Q-learning and PPO**.
No DQN implementation or results are imported. The scope is Part 2 only.

Start with [the common notebook](../02_part2_synthesis.ipynb). It runs from
the repository root or `notebooks/assignment4/`, displays the controlled
protocol, all trained seeds, costs and development curves, and builds a
standalone interactive overview. Training is never launched automatically.

The actual readable agent/training implementations are now collected in
[q_learning.py](../q_learning.py) and [ppo.py](../ppo.py), with `train`,
`evaluate` and `check` commands. See the [main guide](../README.md) for examples.
The original registered modules remain the source of the reported results.

## What was combined

See [COMPARISON.md](COMPARISON.md) for the exact four-branch review,
incompatible protocols, accepted/rejected contributions and requirements map.
The implementation decisions are:

| Source | Adopted strength | Concrete result |
|---|---|---|
| Johan | One notebook/session interface with explicit training/evaluation separation | Common notebook and read-only evidence API |
| Elling | Per-agent result exploration and RAM-checked recorded-action replay | Filterable standalone overview and native trace viewer |
| Mats | Observation/model protocol checks, explicit evaluation modes, per-decision inspection | Strict source/model/trace hashes, portable recorded paths, separate values/probabilities, all-seed table |
| Fredrik/current | Matched ordinary-Q/PPO A–D study with frozen transfer and explicit censoring | Preserved registered engines, models, predictions and numerical outcome/cost rows |

These are adaptations of reviewed designs, not a claim that the new tools
were written by the other students. No foreign training code is executed.
Existing models require their original feature extraction; installing another
branch's observation fixes beneath them would invalidate the evidence.

## Evidence and overview

From the repository root, with the local original archive present:

```powershell
python notebooks/assignment4/synthesis/part2_evidence.py build
python notebooks/assignment4/synthesis/part2_evidence.py check --full
python notebooks/assignment4/synthesis/part2_evidence.py dashboard
```

Open `results/assignment4/synthesis/overview.html` in a browser. It needs no
server, downloads or network access. Filters show condition/algorithm summaries,
all five training replications, development curves and per-agent costs.
The new interface never picks a model by final-test performance.

The tracked `evidence.json` is a compact derived snapshot: 40 cells, 8,000
primary episodes, 150 development checkpoints and SHA-256 fingerprints for
the 179 original inputs. It is not a replacement for raw episode logs,
models and traces. It contains no fabricated or newly collected outcomes.
`check` validates available inputs and reports missing ignored records;
`check --full` refuses any missing record. `build` reconstructs the snapshot
from the raw CSVs, verifies the identical B/D model hashes and preserves
every trained seed. Archived absolute Windows paths are rebased only within
this checkout, so moving the complete archive is supported.

Raw records remain ignored under `results/assignment4/three_hour_20261122/`
and `results/assignment4/transfer_boundary_v1/`. They must accompany the
assignment submission. A fresh Git checkout alone contains the compact
overview evidence but cannot replay native screens or load trained agents.

## Inspect a recorded episode

```powershell
python notebooks/assignment4/synthesis/replay.py --condition B --algorithm PPO --training-seed 17 --evaluation-seed 1150000 --output results/assignment4/synthesis/B_PPO17_1150000.html
```

Choose the condition, algorithm, training seed and recorded evaluation seed
explicitly. The output is a self-contained HTML viewer plus a JSON receipt.
It shows the pre-action native RGB screen, recorded Q-values or policy
probabilities, chosen action, reward/score transition and observation. Play,
pause, frame stepping and a slider are available. Default display stride is
four decisions; `--stride 1` displays every decision. Every intervening
decision is still replayed and checked against native RAM/rewards/BCD score.

The viewer replays recorded actions through Gymnasium's five-value step API;
it does not load or change model parameters. It requires matching source,
model and trace hashes, compatible 61-value observations/five-action values,
and the original episode. Q-values use the training reward scale and are
not probabilities. Native map exits must be the final recorded action.
An inspection is neither a new evaluation nor evidence about unchosen actions.
Existing output files are preserved; choose a fresh output filename.

## Validation and reproduction

```powershell
python notebooks/assignment4/report_review/analyse.py
python notebooks/assignment4/report_review/verify_quality.py
python notebooks/assignment4/synthesis/compare_branches.py
python notebooks/assignment4/synthesis/verify.py
```

The full check validates all five report tables, including design rationales,
and recomputes every numerical outcome/cost/seed cell from raw episode CSVs,
checks all source/model hashes, the unchanged prediction sheet and all seed
cells, and verifies that the report contains no visible filenames/paths.
The branch comparison rechecks the exact Git blobs, shared notebook code,
Mats' two complete primary studies and the native mode/reset-maze distinction.
It requires the reviewed commits to be present locally; fetch the named remote
branches first on a fresh clone. It never executes another branch's code.
The synthesis verifier additionally checks portability, incompatible-record
rejection, the derived snapshot, notebook execution and native replay cases.

Original expensive training/search reproduction remains in
[three_hour](../three_hour/README.md), and the separate frozen-D measurement
repair in [transfer_evaluation](../transfer_evaluation/README.md). Use fresh
run IDs; never modify registered runtime sources under old checkpoints.
No new training, altered evaluation mode, report PDF or submission ZIP is
introduced by this synthesis. Parts 3–6 and real human review remain separate.
