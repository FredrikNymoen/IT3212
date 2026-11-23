# Late-game PPO transfer protocol

The previous round's three balanced multi-maze PPO checkpoints are the starting point. The observed D results informed that choice and the focus on the final 500 points. This round uses fresh D episode seeds for evaluation, with D excluded from training, saved-state collection, and screening.

Every test keeps the ordinary threshold of 2,000 raw points, the 1,500-decision horizon, and the original emulator settings. The training layouts are ROM layouts 0, 2 and 3. The shared mixed training schedule targets 26%, 37% and 37% of interactions on those layouts.

## Candidate comparisons

- Reward arms use 240,000 interactions per seed and the same training random seeds. They compare the original reward with a penalty for lost lives that grows near the target and with that same penalty plus extra reward for late score gains.
- Replay arms use the same total budget, including 35,000 interactions to collect saved states from the starting policy. Some training episodes restart from those states. They compare the original reward with the survival penalty.
- Dedicated specialists use 320,000 interactions per seed including 60,000 to collect states. They retain the starting actor's decisions up to 1,500 in evaluation and then switch to a late-game actor trained from saved states. One specialist also observes decoded frightened state and direction-specific power-pellet and ghost distances.
- The branch teacher uses 260,000 interactions per seed including its short branches. From a scored state it tests each of the nine action requests, then follows the parent for up to 24 decisions overall. It trains a specialist to prefer outcomes with fewer lost lives and greater score. At evaluation the trained actor uses the ordinary observations and performs no emulator branching.
- The longer control uses an additional 500,000 interactions from the same parents and with the same random streams as the 240,000-interaction control. The averaged checkpoint combines only siblings with the same parent and adds no training interactions.

## Screening and testing

All candidates use 70 matched screening episodes per training seed and layout, starting at episode seed 980500. The screen ranks candidates by mean success over layouts 2 and 3, followed by conversion after 1,500 and mean score. The shortlist keeps the starting models and the matched continuation control. It ranks new arms with B success within six percentage points of the starting models and includes the highest eligible phased policy.

The frozen test uses episode seeds starting at 987000 on B and D. The strongest newly trained cohort from that test is compared with the same starting policies on independent D episodes beginning at seed 988500. This selection changes which fixed weights are checked in confirmation.

## Added variants

The faster specialist retains the emulator between branch restores and lets its seeded sticky-action random stream advance. Its 240,000-interaction budget includes 45,000 to collect states. The restore benchmark verified the initial features on every branch and produced multiple subsequent trajectories from each saved state.

The first teacher could distinguish life losses across actions on 2.4% to 5.3% of branch states, depending on training seed. Its supervised loss increased on held-out training episodes. The revised teacher repeats each candidate request for the first three decisions of a 32-decision branch and retains the fit with the lowest held-episode loss.

The averaged weights also received a further 150,000-interaction survival continuation. Two other phased variants retain the starting actor through 1,500 and then use either the averaged weights or the best survival-reward weights from the training-layout screen. These phased variants reuse existing checkpoints.
Both matched budgets are included as controls so the D test can compare 240,000 and 500,000 further training interactions on the same seeds. The final shortlist uses completed common screens available before any D episode in this round.
