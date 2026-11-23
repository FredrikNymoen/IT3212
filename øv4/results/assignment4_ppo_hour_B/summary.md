# One-hour PPO session

The objective was to raise success and raw score in stochastic Setting B.
Success requires reaching 2,000 points in the existing episode rules.

## Main search

The previous temperature-0.5 checkpoints averaged 68.7% success and 1,940
points on the fresh comparison. Checkpoints selected using two separate
validation sets averaged 93.8% and 2,207 points on the same 200 test episodes
per training seed. Individual success rates were 94.5%, 91.5%, and 95.5% for
seeds 7, 42, and 2026. The scheduler launched 81 training jobs, covering 32
distinct combinations of seed, parent weights, and training configuration.
The remaining jobs repeated deterministic configurations after the leading
checkpoints stopped changing.

[Main search](comparison.json)

## Common Setting B comparison

Every group played 300 fresh paired episodes per training seed.

| Group | Success | Mean raw score |
| --- | ---: | ---: |
| Frozen checkpoints, temperature 0.20 | 97.6% | 2,255 |
| Frozen checkpoints, temperature 0.15 | 98.2% | 2,273 |
| Another 30k PPO interactions at temperature 0.20 | **98.3%** | 2,414 |
| Coordinate dropout during 60k PPO interactions | 97.4% | **2,513** |

The 30k group succeeded in 885 of 900 episodes, one more than the
temperature-0.15 group. Its mean score was 141 points higher. Coordinate
dropout produced the highest mean score because winning episodes finished
approximately 534 points above 2,000 on average, compared with 425 for
the 30k group. With scores capped at 2,000 for this secondary analysis,
their means were 1,993 and 1,996, respectively.

[Common test](../assignment4_ppo_hour_frontier_B/comparison.json)

## Independently confirmed individual models

Both models use training seed 2026 and play at temperature 0.20. A new
paired set contained 1,200 episodes per model.

| Model | Success | Mean raw score |
| --- | ---: | ---: |
| [Reliability checkpoint](../assignment4_ppo_hour_cold_continuation_B/training/budget_30000/seed_2026/model.npz) | **1,186 / 1,200 (98.8%)** | 2,838 |
| [Score checkpoint](../assignment4_ppo_hour_dropout/coordinate_dropout/seed_2026/model.npz) | 1,183 / 1,200 (98.6%) | **2,864** |

By default, `code/watch_best.py` loads the reliability checkpoint with the
navigation inputs and temperature 0.20. Playback was verified against the
recorded trace.

[Large confirmation](../assignment4_ppo_hour_champions_B/comparison.json)

## Frozen transfer to unseen maze D

Across 150 fresh episodes per training seed, the same 30k group reached
33.8% success, the frozen temperature-0.20 controls reached 30.9%, and
coordinate dropout reached 26.7%. Horizontal mirroring also fell below its
frozen controls in a separate paired test. The improvement is much larger
in B than on the unseen maze.

[Final D comparison](../assignment4_ppo_hour_frontier_D/comparison.json) ·
[Mirror comparison](../assignment4_ppo_hour_mirror/comparison.json)

## Steering noise checks

The two independently confirmed models were also evaluated on the same
300 episode seeds at three sticky-action probabilities. Neither model was
updated between these evaluations.

| Sticky-action probability | Reliability success | Score checkpoint success |
| --- | ---: | ---: |
| 0.00 (Setting A) | 100.0% | 100.0% |
| 0.25 (Setting B) | 99.0% | 99.3% |
| 0.40 (stress test) | 96.7% | 98.3% |

[Predictable setting](../assignment4_ppo_hour_predictable_A/comparison.json) ·
[Sticky-action stress test](../assignment4_ppo_hour_sticky_stress_B/comparison.json)

## Supporting comparisons

- [Fine temperature sweep](../assignment4_ppo_hour_temperature_B/comparison.json)
  selected 0.20 and measured 97.0% success versus 93.0% at 0.50.
- [Lower-temperature and greedy comparison](../assignment4_ppo_hour_greedy_limit_B/comparison.json)
  selected 0.15 and measured 97.2% versus 96.7% at 0.20 on its paired test.
- [Continuation at temperature 0.20](../assignment4_ppo_hour_cold_continuation_B/comparison.json)
  found a mean score gain at 30k in its first held-out comparison, with a
  small success decline. The larger common test above retested that group.
- [Coordinate dropout](../assignment4_ppo_hour_dropout/comparison.json)
  raised B score while providing no transfer gain over its starting checkpoints.
