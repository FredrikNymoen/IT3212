# v7 and v8 comparison

Both studies use five training seeds, 100,000 training interactions per agent, and 200 primary evaluation episodes per trained agent per condition. Cells summarize trained agents, not independent pooled episodes. V8 changes multiple observation details together; differences do not identify the causal effect of an individual fix. Both selected Q position_bins=4 and PPO lr=0.001.

| Setting | Agent | v7 success, mean ± SD | v8 success, mean ± SD | Change (percentage points) | v7 mean score | v8 mean score |
|---|---|---|---|---|---|---|
| A | Q | 0.0% ± 0.0% | 0.2% ± 0.3% | +0.2 | 352 | 649 |
| A | PPO | 1.8% ± 1.0% | 2.7% ± 1.9% | +0.9 | 773 | 836 |
| B | Q | 0.0% ± 0.0% | 0.2% ± 0.4% | +0.2 | 381 | 478 |
| B | PPO | 2.6% ± 1.9% | 2.9% ± 1.1% | +0.3 | 826 | 808 |
| C | Q | 0.5% ± 0.6% | 0.1% ± 0.2% | -0.4 | 501 | 423 |
| C | PPO | 1.4% ± 1.2% | 4.6% ± 2.2% | +3.2 | 733 | 837 |
| D | Q | 0.0% ± 0.0% | 0.0% ± 0.0% | +0.0 | 255 | 254 |
| D | PPO | 2.3% ± 2.2% | 0.5% ± 0.6% | -1.8 | 712 | 576 |

Sources: ../assignment4_v7/seed_results_*.csv and seed_results_*.csv. Per-run configurations and raw episode logs are in the respective study directories. D is frozen transfer from B and has no additional training.
