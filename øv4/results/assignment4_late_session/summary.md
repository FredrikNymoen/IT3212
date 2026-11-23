# Late-game transfer results

This round starts from the previous balanced PPO checkpoints. That earlier comparison informed the starting models and the focus on late deaths. D remains outside training and validation. Both test rounds use fresh episodes with the original 2,000-point target and original stopping rules.

## Frozen comparison

Each row pools 160 episodes per checkpoint and setting across the same three training seeds.

| PPO setup | B success | D success | D score | D reached 1,500 | D conversion after 1,500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting balanced PPO | 93.3% | 36.9% | 1,700 | 71.9% | 51.3% |
| Plain continuation (240k) | 89.6% | 48.1% | 1,785 | 81.2% | 59.2% |
| Averaged weights with survival continuation | 94.4% | 51.7% | 1,827 | 82.5% | 62.6% |
| Averaged sibling checkpoints | 95.4% | 48.3% | 1,842 | 84.0% | 57.6% |
| Life-loss penalty | 92.5% | 40.8% | 1,772 | 78.1% | 52.3% |
| Survival policy after 1,500 | 93.5% | 37.9% | 1,703 | 71.9% | 52.8% |
| Plain continuation (500k) | 96.0% | 47.9% | 1,786 | 76.5% | 62.7% |

## Fresh D confirmation

| PPO setup | D success | D score | Reached 1,500 | Conversion after 1,500 | Late deaths | Episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting balanced PPO | 36.2% | 1,673 | 68.2% | 53.0% | 377 | 1,200 |
| Averaged weights with survival continuation | 46.0% | 1,786 | 78.7% | 58.5% | 382 | 1,200 |
| Averaged sibling checkpoints | 48.8% | 1,813 | 79.8% | 61.2% | 365 | 1,200 |

The confirmation used 400 new D episodes per checkpoint. The strongest cohort in confirmation changed success by +12.7 percentage points across matched seeds. The paired episode resampling interval was +8.8 to +16.5 percentage points conditional on these three sets of trained weights.

| Training seed | Starting D success | Strongest confirmation cohort D success |
| --- | ---: | ---: |
| 7 | 45.0% | 57.0% |
| 42 | 34.8% | 48.8% |
| 2026 | 28.7% | 40.8% |

## Common training-layout screen

| PPO setup | B | Layout 2 | Layout 3 |
| --- | ---: | ---: | ---: |
| Starting balanced PPO | 92.9% | 58.6% | 66.2% |
| Averaged sibling checkpoints | 94.8% | 71.4% | 72.9% |
| Averaged weights with survival continuation | 91.4% | 79.5% | 71.9% |
| Longer-action branch teacher | 91.9% | 60.0% | 63.8% |
| In-place replay specialist | 94.3% | 59.5% | 67.6% |
| Plain continuation (500k) | 93.8% | 63.8% | 71.9% |
| Averaged weights after 1,500 | 92.9% | 59.5% | 68.1% |
| Survival policy after 1,500 | 92.9% | 63.3% | 69.5% |
| Mixed saved-state replay | 93.8% | 65.7% | 71.4% |
| Replay with life-loss penalty | 94.3% | 61.0% | 70.5% |
| Plain continuation (240k) | 91.4% | 64.3% | 69.5% |
| Late-score and survival rewards | 85.7% | 65.2% | 69.5% |
| Life-loss penalty | 92.4% | 63.3% | 76.7% |
| Late-game specialist | 91.4% | 57.6% | 66.7% |
| Action-branch teacher | 92.4% | 53.8% | 59.5% |

The specialist with decoded ghost state completed its common screen after the D shortlist froze. It reached 93.8% on B, 63.3% on layout 2, and 62.9% on layout 3.

## Verification

Verified checkpoint hashes, matched episode seeds and threshold outcomes for 6,720 frozen test episodes and 3,600 fresh confirmation episodes. The phased policies also matched the opening checkpoints on which episodes reached 1,500.

## Checkpoints and replay

The training audit records 36 trained checkpoints and 9,690,000 interactions across layouts 0, 2 and 3. The averaged checkpoints and policy switches reused existing weights.

Watch Best selects the strongest cohort from the matched confirmation sample for D. In that cohort it selects seed 7, which reached 59.4% in the first test and 57.0% in confirmation. The regular B selection uses the established B comparison.

Run `python watch_best.py --condition D --seed 988501` from the experiment code directory to see a successful episode from the selected D checkpoint. The recorded replay reaches 2,000, and the viewer matches its evaluation decisions.

## Third D sample for the strongest individual

On 250 additional matched episodes, the averaged seed 7 checkpoint reached 55.2%. The same starting checkpoint reached 44.0%. Across the two fresh samples, the averaged checkpoint succeeded in 366 of 650 episodes, or 56.3%.
