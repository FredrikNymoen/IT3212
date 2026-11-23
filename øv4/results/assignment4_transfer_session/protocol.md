# Ms. Pac-Man transfer study

The goal is higher success on Setting D, the maze identified by ROM maze byte 1. The success cutoff is 2,000 raw points. The 1,500-decision cap, sticky action probability of 0.25 and start animation accounting follow the existing protocol.

## Comparisons

- Checkpoints start from the same three B-trained cold-continuation policies, with training seeds 7, 42 and 2026.
- The 200,000-interaction comparisons run B only or alternate the original layout and layouts 2 and 3. The observation comparisons add power-pellet paths and decoded frightened state. Other variants change continuation temperature or mask the absolute coordinates on some training episodes.
- Initial validation uses 100 episodes per seed and layout, beginning at seed 982000. Candidate configurations are ranked by average success on validation layouts 2 and 3, with a minimum average success of 90% on layout 0 for refinement. Mean score breaks a tie.
- The longer continuation trials reallocate experience by interaction count and compare a 35% horizontal reflection probability with the same unaugmented schedule. They use separate training seeds and validation seeds beginning at 982150.
- Setting D is excluded from all training, validation and checkpoint choices for the frozen test. Use fresh paired evaluation episodes beginning at seed 983000 for both B and D. The fixed comparison retains the starting checkpoint and matched B-only and mixed checkpoints as controls.
- This uses extra start layouts available in the same ROM and does not change the original assignment experiment. Layouts 2 and 3 are reached by setting the startup maze byte. The held-out result measures transfer to one layout.

Each training arm saves checkpoint hashes, episode-level validation output, and interaction counts by maze.

## Follow-up comparisons

- A paired 250,000-interaction continuation isolates the reward for reaching 2,000. It caps training credit for the raw score at the threshold and increases the success bonus from 100 to 400. Evaluation uses the usual reward and stopping rules.
- The side-tunnel variant adds wrap edges to the corridor graph. A controlled emulator probe confirmed a horizontal wrap of 160 pixels at the open tunnel rows on layouts 0, 2 and 3.
- The final test includes the starting cohort, the matched mixed and B-only cohorts, and the ghost-state and reward comparisons. Other evaluated groups qualify by mean validation success over layouts 2 and 3 subject to the B validation floor. A separate sampling-temperature check uses seeds starting at 982350 on layouts 2 and 3.

## D confirmation

After the frozen comparison, select the strongest newly trained cohort by D success and mean score and compare those same weights to the starting checkpoints on fresh seed 985000 onward. This second sample checks whether the measured gain survives the selection of a winner. Neither pass updates a policy. Count power-pellet events and ghost rewards from the decoded timer and combo signals on the confirmation episodes.
