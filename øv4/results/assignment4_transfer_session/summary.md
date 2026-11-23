# Setting D transfer results

Frozen test with 3 training seeds and 150 paired episodes per training seed and setting. Training used only ROM layouts 0, 2 and 3. The held-out layout used ROM layout 1.

| PPO setup | B success | B mean score | D success | D mean score |
| --- | ---: | ---: | ---: | ---: |
| Starting B checkpoints | 97.6% | 2,416 | 30.9% | 1,603 |
| B-only continuation | 83.1% | 2,293 | 22.7% | 1,518 |
| Mixed layouts | 97.8% | 2,545 | 35.1% | 1,676 |
| Mixed + power paths and ghost state | 93.3% | 2,153 | 28.7% | 1,534 |
| Ghost state + balanced continuation | 88.7% | 2,113 | 31.8% | 1,652 |
| Ghost state + threshold reward | 79.6% | 2,147 | 35.6% | 1,704 |
| Mixed + side-tunnel paths | 95.1% | 2,279 | 33.6% | 1,692 |
| Mixed + balanced continuation | 93.8% | 2,362 | 38.0% | 1,696 |
| Mixed + balanced continuation and reflection | 97.3% | 2,313 | 34.9% | 1,697 |

## D by training seed

| PPO setup | 7 | 42 | 2026 |
| --- | ---: | ---: | ---: |
| Starting B checkpoints | 40.7% | 23.3% | 28.7% |
| B-only continuation | 24.7% | 25.3% | 18.0% |
| Mixed layouts | 36.7% | 44.0% | 24.7% |
| Mixed + power paths and ghost state | 40.7% | 20.7% | 24.7% |
| Ghost state + balanced continuation | 38.0% | 31.3% | 26.0% |
| Ghost state + threshold reward | 32.0% | 35.3% | 39.3% |
| Mixed + side-tunnel paths | 32.0% | 24.0% | 44.7% |
| Mixed + balanced continuation | 41.3% | 34.0% | 38.7% |
| Mixed + balanced continuation and reflection | 35.3% | 36.7% | 32.7% |

## Training-layout validation

| PPO setup | B | Layout 2 | Layout 3 |
| --- | ---: | ---: | ---: |
| B-only continuation | 82.0% | 26.7% | 33.3% |
| Mixed layouts | 98.7% | 45.0% | 67.7% |
| Mixed + power paths | 95.3% | 37.0% | 52.7% |
| Mixed + power paths and ghost state | 95.3% | 49.7% | 53.7% |
| Mixed + higher training temperature | 98.7% | 41.0% | 64.7% |
| Mixed + coordinate masking | 92.0% | 45.7% | 51.7% |
| Mixed + balanced continuation | 92.3% | 57.3% | 64.0% |
| Mixed + balanced continuation and reflection | 94.3% | 48.3% | 65.7% |
| Ghost state + balanced continuation | 92.1% | 50.8% | 63.7% |
| Ghost state + threshold reward | 82.1% | 60.0% | 53.8% |
| Mixed + side-tunnel paths | 94.0% | 52.0% | 49.3% |

## Verification

Verified 33 training exposure logs and 8,100 individual test episodes. The test pairs the same evaluation seeds across all checkpoints. The three independently trained policies give the variation across training seeds in the table. Setting D is a single layout, so these figures quantify that specific transfer.

Balanced continuation led the first D test with 171 successes in 450 episodes.

## Training changes

The strongest D cohort continued from the mixed-layout checkpoints for 350,000 interactions per model. The learner allocated 26% of its interactions to the original maze and 37% to each extra training maze. It used unreflected observations during that continuation. The simpler mixed training used 200,000 additional interactions per model and retained 97.8% success on B.

The audit found 6,617 steps on which the original feature overstated the remaining frightened countdown. It verified 161 power-pellet events and 170 ghost reward transitions. The variant adding these sensors reached 28.7% success on D at the original continuation budget. The side-tunnel graph variant reached 33.6%.

The fixed-temperature ensemble trial on validation layouts reached 81.0%, 48.0%, and 60.0% on layouts 0, 2, and 3. It stayed outside the D comparison under the preregistered validation rule.

## Independent D confirmation

Each checkpoint used 450 fresh episodes with the same seeds across policies. The balanced cohort and the individual side-tunnel checkpoint were selected from the first D test. All evaluated weights were frozen before that test.

| PPO setup | Success | Mean score | Reached 1,500 | Success after reaching 1,500 | Episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting B checkpoints | 29.3% | 1,612 | 63.3% | 46.3% | 1,350 |
| Mixed layouts | 33.8% | 1,671 | 68.4% | 49.4% | 1,350 |
| Mixed + balanced continuation | 39.3% | 1,731 | 74.4% | 52.8% | 1,350 |
| Side-tunnel checkpoint (seed 2026) | 41.8% | 1,800 | 82.9% | 50.4% | 450 |

The balanced cohort collected 2.52 power pellets and 474 ghost points per episode. The starting cohort recorded 2.41 power pellets and 400 ghost points per episode.

Verified 4,500 individual confirmation episodes. The scoring-event counts use the timer and score transition checks documented above.

## Remaining gap to 50%

Balanced PPO reached 1,500 in 1,004 of 1,350 confirmation episodes. It ended short of 2,000 in 474 of those episodes, including 459 life-loss failures and 15 timeouts. It finished between 1,800 and 1,990 in 169 episodes. The event tracker recorded fewer than four power pellets in 127 of those 169 episodes. Keeping the same 1,500-point reach rate, reaching 50% success would require converting 145 more of the episodes that already got to 1,500.

## Replay

Run `python watch_best.py --condition D --seed 983002` from the `øv4ny/code` directory. The viewer selects the top checkpoint within the strongest D cohort from the frozen comparison. This example earned 2,130 points in 728 decisions.

