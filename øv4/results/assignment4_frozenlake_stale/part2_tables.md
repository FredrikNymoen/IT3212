# Part 2 tables

Source: seed_results.csv, 40 run summaries. Cells are mean ± sample SD across trained seeds. A dash means the measure never occurred, not zero.

## Required evidence, Settings A/B/C

| Setting | Agent | n       | Success rate  | Mean return   | Failure rate  | Len (success) | Len (failure) | Timeout rate  | Interactions | Runtime (s) |
|---------|-------|---------|---------------|---------------|---------------|---------------|---------------|---------------|--------------|-------------|
| A       | Q     | 5 seeds | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 6.0 ± 0.0     | -             | 0.000 ± 0.000 | 200000 ± 0   | 7 ± 2       |
| A       | PPO   | 5 seeds | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 6.0 ± 0.0     | -             | 0.000 ± 0.000 | 200000 ± 0   | 7 ± 2       |
| B       | Q     | 5 seeds | 0.670 ± 0.109 | 0.670 ± 0.109 | 0.279 ± 0.124 | 37.0 ± 1.3    | 36.8 ± 4.5    | 0.051 ± 0.015 | 200000 ± 0   | 8 ± 1       |
| B       | PPO   | 5 seeds | 0.702 ± 0.089 | 0.702 ± 0.089 | 0.246 ± 0.110 | 37.4 ± 2.4    | 37.7 ± 6.3    | 0.052 ± 0.021 | 200000 ± 0   | 5 ± 0       |
| C       | Q     | 5 seeds | 0.223 ± 0.094 | 0.069 ± 0.046 | 0.775 ± 0.097 | 21.4 ± 4.8    | 13.5 ± 5.3    | 0.002 ± 0.003 | 200000 ± 0   | 7 ± 2       |
| C       | PPO   | 5 seeds | 0.298 ± 0.239 | 0.109 ± 0.086 | 0.689 ± 0.258 | 23.0 ± 10.3   | 17.5 ± 15.4   | 0.012 ± 0.020 | 200000 ± 0   | 7 ± 1       |

## Setting B versus Setting C

Step penalty effect, C minus B (paired on shared training seeds; returns are omitted because B and C define the reward differently)

| Agent | seeds | success rate Δ | failure rate Δ | timeout rate Δ |
|-------|-------|----------------|----------------|----------------|
| Q     | 5     | -0.446 ± 0.144 | +0.496 ± 0.155 | -0.049 ± 0.015 |
| PPO   | 5     | -0.403 ± 0.289 | +0.443 ± 0.317 | -0.040 ± 0.028 |

## Setting D, transfer from Setting B

Transfer effect, D minus B (paired on shared training seeds; same agents and same reward, evaluated on the unseen map with no further training)

| Agent | seeds | success rate Δ | failure rate Δ | timeout rate Δ |
|-------|-------|----------------|----------------|----------------|
| Q     | 5     | -0.670 ± 0.109 | +0.721 ± 0.124 | -0.051 ± 0.015 |
| PPO   | 5     | -0.694 ± 0.105 | +0.747 ± 0.126 | -0.052 ± 0.021 |
