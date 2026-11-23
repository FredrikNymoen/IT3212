# AGENTS.md

This file provides guidance to coding agents working with this repository.

## Project overview

IT3212 (Data-Driven Software Engineering) is a course repository for all four assignments.
Assignment 1 has an existing preprocessing implementation. Assignment 2 covers image processing.
Assignment 3 covers a justified machine-learning workflow using the student graduation dataset.
Assignment 4 is the current focus and covers reinforcement learning; report scaffolding is
available and the team has already completed Part 1. Do not assume it uses Assignment 3's
dataset or workflow.

The assignment PDFs in the repository root are the source of truth for grading and submission requirements.
Read the relevant PDF before implementing an assignment. Distinguish assignment requirements from local
implementation choices, and update this file when new briefs or repository structure are added.

## Repository structure and conventions

### Active layout after the 23 November repository cleanup

The current Assignment4 entry points are documented in `notebooks/assignment4/README.md`:
`three_hour/` for the reported search/A/B/C workflow, `transfer_evaluation/` for
frozen D, and `report_review/` for the current three-table report audit. Shared
core files retain their original paths and registered byte hashes. Earlier
baselines, rejected experiments, obsolete figure copies and one-off helpers
are preserved in Git checkpoint `714c0cb` and the ignored local archive at
`results/assignment4/repository_cleanup_20261123/`, rather than the active tree.
Historical structure entries below refer to that checkpoint when a file is
no longer present; do not recreate or rewrite them as current implementations.
Use `notebooks/assignment4/report_review/analyse.py` and `verify_quality.py`
for current report checks; their output goes to `results/assignment4/report_review_current/`.
Preserve original results/models and Part1. Old sources can be inspected with
`git show 714c0cb:<original-path>`. See `notebooks/assignment4/ARCHIVE.md`.

- `IT3212 - Assignment 1.pdf` - preprocessing brief, rubric, and submission requirements.
- `IT3212 - Assignment 2.pdf` - image-processing brief and rubric.
- `IT3212-2026-Assignment 3.pdf` - machine-learning brief, 200 points with 130 required to pass.
- `IT3212 - Assignment 4.pdf` - reinforcement-learning brief, Q-learning versus PPO,
  with six assessed parts and separate prediction, experiment, evidence and review records.
- `workshops/workshop1/Monday-Workshop (1).pdf` - classification workshop brief;
  start with UCI Dry Bean (602), then repeat with another listed dataset.
  The original CSV and UCI metadata are gitignored at `workshops/workshop1/dry_bean.csv`
  and `workshops/workshop1/dry_bean_metadata.json` (the user's chosen location).
  Source, loading instructions, and
  verified dimensions (13,611 rows, 16 numerical features, seven classes; target
  `Class`) are documented in `workshops/workshop1/README.md`.
  `workshops/workshop1/workshop1_classification.ipynb` follows all 11 workshop
  points with PyTorch neural networks, controlled hyperparameter grids, a scaling
  comparison, random-forest tuning, and the repeated workflow on bundled Wine.
  The CSV retains all 13,611 rows; modeling removes 68 exact duplicates before
  splitting (13,543 unique rows) to prevent identical records across partitions.
  Run from `workshops/workshop1/`; use training-only preprocessing and validation
  for selection, reserving test evaluation until each dataset's settings are fixed.
- `notebooks/assignment1/student_graduation.ipynb` - populated Assignment 1 implementation, including optional PCA.
  Preserve this work when starting later assignments.
- `notebooks/assignment1/` through `notebooks/assignment4/` - one directory per assignment.
  Assignment 4 contains the Gymnasium MsPacMan Part 2 workflow; its placeholder was removed.
- `notebooks/assignment4/02_learning_settings.ipynb` - historical v2 Part 2 reproduction notebook,
  run from `notebooks/assignment4/`. `part2_experiment.py run` is the equivalent
  command for that archived baseline; use `current/` for the current report.
  Uses `mspacman_env.py` (Gymnasium wrapper/features), `rl_agents.py`
  (tabular Q-learning/PyTorch PPO) and `part2_report.py` (measured report exports).
  The original Part 1 text is preserved. The user confirmed instructor approval
  of score-increment rewards for A/B instead of terminal-only success rewards.
  Success is >=2,000 new points before all lives are lost or 1,000 decisions.
  The finite horizon is a terminal task failure with observable time remaining.
  A has no sticky actions; B uses probability 0.25; C subtracts 0.1 points/decision.
  D freezes B agents and changes ROM maze selector 0 to 1 before the first action.
  Both wrappers retain the same action/observation formats. Native level transitions
  raise an error to protect the map holdout. This ROM intervention may also change
  layout-dependent behaviour; do not call it a geometry-only causal experiment.
  Q-learning coarsens object features; PPO receives 15 finer features. The inherited
  RAM-116 cap is a power-related signal, not a verified precise countdown: later
  successful replays show values above 63 after ghosts are eaten. Comparisons are between
  workflows with different representations, not algorithms in isolation.
  Each algorithm/setting has two learning rates with 16,384 interactions each,
  a separate tuning seed and 50 development episodes. Mean development success
  across A/B/C, then mean game score, selects one common rate per algorithm.
  Other hyperparameters stay fixed across A/B/C. Main runs use 65,536
  decisions and seeds 42/7/2026, with 20 development episodes at five checkpoints
  and 200 final held-out episodes per trained agent/condition. D adds no training.
  This three-seed Part 2 comparison does not satisfy Part 3's five-seed rule.
  Q evaluation is greedy with seeded uniform ties; PPO uses highest-probability
  actions. Frozen evaluation checks parameter fingerprints, uses read-only Q
  lookups and restores training RNG state. Final results never tune parameters.
  Numerical records and per-decision traces are ignored under
  `results/assignment4/part2_controlled_v2/`; engineering checks are separate under
  `results/assignment4/engineering_audit/`. Protocol and source hashes are checked
  when resuming; use a new run ID after code/protocol changes. Raw traces contain
  RAM, Q-values/probabilities, actions, rewards and episode/step IDs for Part 4.
  Report text and a single vector learning curve are generated only after the
  full run. No report PDF/ZIP is compiled. See its README for budgets, sources,
  runtime accounting, Gymnasium audit and reproducibility instructions.
  The initial `part2_score_v1` pilot selected rates separately by setting. This
  was corrected after partial evaluation inspection to satisfy the controlled
  setting comparison. The correction selects only from development data and
  reuses evaluation seeds; never call it a new independent test. Twelve identical
  final agents and the original tuning search are reused with a manifest;
  six changed agents are retrained. Original raw results and exact original
  Python snapshots are preserved separately. Reused costs keep source provenance.
- `notebooks/assignment4/pilot/` - bounded improvement diagnostics authorised
  after the poor main results. `policy_check.py` evaluates saved B checkpoints
  with greedy/sampled PPO policies; `q_exploration_check.py` evaluates frozen Q
  tables with epsilon 0.05/0.10. Development seeds are 650000--650019, separate
  from the main held-out seeds. `local_pilot.py` tries a relative food-lattice
  representation and advisory action filter for both agents, 65,536 interactions
  at seeds 42/7; PPO also uses LR 0.0003. Its graph is only checked for training
  layout 0 and is not an exact ALE legality detector. This is a package feasibility
  pilot, not a controlled one-factor Part 3 experiment. It does not improve both
  workflows and is NOT adopted. Sampled original PPO improves development scores
  (179.8 to 681.2 across three agents) but has 0/60 successes; Q epsilon 0.05
  yields only 1/40 successes across two agents. Do not claim reliable improvement
  or count the separate hand-coded geometry diagnostic as an RL result. Main
  code, report, predictions and held-out exports remain unchanged. Records are
  separate under `results/assignment4/quick_pilot/`. See its README for paired
  comparisons, controls, limitations and the recommended next isolated check.
- `notebooks/assignment4/pilot/FURTHER_EXPLORATION.md` - subsequent investigation.
  The user explicitly reminded that the main implementation must match the
  original prediction sheet. Use ordinary single-table tabular Q-learning,
  updated from each new environment transition, versus PPO. Double Q, replay
  and eligibility-trace variants are separate exploratory deviations. Do not
  promote their improved rates as the original Q-learning baseline or modify
  the prediction sheet to legitimise them. A corrected ordinary-Q rerun is
  required before reporting improved main-comparison Q results. Retain the
  geometry corrections and document the shared underlying feature information.
  The archived locked Double Q results below remain genuine variant results.
  The investigation began after the user rejected the first pilot. Original pellet masks contain manual
  transcription errors and right-side pellet coordinates omit a four-pixel offset.
  The old local pilot inherits them. `maze_geometry.json` is extracted automatically
  from the referenced upstream source; `navigation_features.py` provides corrected
  current geometry and motion (45 values), no forced actions or shaped rewards.
  `validate_navigation.py` checked 1,405 Gymnasium steps for unchanged native RAM,
  rewards and episode outcomes. Preserve the registered top-level source hashes and
  original predictions; do not silently rewrite the historical baseline/report.
  `reference_ppo.py` tests SB3 PPO with original versus rich features; SB3 2.9.2 is
  documented in pilot/requirements.txt. `navigation_q.py` tests one-step relative
  Q and phase-aware Watkins Q(lambda), including failed controls. `canonical_q.py`
  and `shared_policy.py` are separately labelled exploratory packages. Main new
  candidates have a 524,288-interaction maximum; development uses 670000-series
  seeds, frozen confirmation 680000-series. The held-out D map is not evaluated.
  Seed-42 rich PPO at checkpoint 327,680 succeeded in 20/20 selection and 50/50 new
  development episodes (2,500.4 mean points). Replay checks every score against
  native BCD registers/RGB and documents 200/400/800/1,600-point ghost bonuses.
  Seed-42 phase/trace Q reached 8/20 at 458,752 but did not replicate at seed 7
  (0/50 separate development successes). Both can regress with more training;
  never report selected peaks as final independent tests or infer seed stability
  from one trained agent. Shared-action PPO at the final 262,144 checkpoint
  achieved 35/50 and 37/50 separate development successes for seeds 42 and 7.
  `replay_double_q.py` adds tabular Double Q and observed-transition replay;
  `goal_replay_q.py` shares learned table values in food-relative coordinates.
  The latter's final 262,144 checkpoints achieved 17/50 and 22/50 on those
  development episodes. `power_goal_q.py` is a separate short bonus-direction
  observation test, not an adopted change or a hand-coded action policy.
  Four final goal-relative Q/shared-PPO model hashes and 200 untouched seeds
  710000--710199 are frozen in `locked_validation_protocol.json`. Do not use
  this locked pilot validation set for further model/configuration selection.
  The locked 200-episode/model results are Q 59/200 and 131/200, PPO 136/200
  and 158/200 for seeds 42/7 respectively. All four model hashes and seed lists
  were verified. This is a B-only pilot, not the complete new A--D study. Q still
  varies substantially across training seeds. `locked_validation_summary.csv`
  includes raw-score means and episode lengths by outcome; the matching checks
  record verifies all 800 episodes. Use the final fixed checkpoints, not the
  earlier transient 100% PPO peak, when reporting this locked comparison.
  `confirm_candidate.py` freezes tables/actors, `sb3_numpy.py` checks actor inference
  against Torch, and `exploration_summary.py` exports raw comparison tables/plots
  and verifies original source hashes. Records live under
  `results/assignment4/further_exploration/`, plot outputs under the matching
  `figures/assignment4/` subdirectory. The first canonical-Q unseen_fraction metadata
  is wrong (rotated wrapper lookup); it does not affect learning. These are
  exploratory packages with unequal tuning, not the formal five-seed Part 3 or
  the required new controlled A--D/200-episode evaluation. See the investigation
  record before changing the report recommendation or promoting the implementation.
- `notebooks/assignment4/pilot/tabular_q.py` is the Part 1-aligned replacement:
  an independent ordinary single-table Q-learning implementation. It imports
  no Double Q/replay implementation. Each real transition makes one Bellman
  update; terminal states omit bootstrapping. It retains the corrected shared
  observation and compact food-relative bins, with alpha 0.15, gamma 0.99 and
  the declared epsilon schedule. `validate_tabular_q.py` numerically checks the
  target, terminal handling, native action mapping and read-only lookup.
  `single_q_experiment.py` trains seeds 42/7 for a fixed 262,144 decisions and
  asserts that the update count equals the interaction count. Records are
  separate under `results/assignment4/single_q_v1/`; development seeds are
  730000--730019 and final frozen evaluation uses 740000--740199. Do not use
  these final seeds to select parameters or intermediate checkpoints.
  Unchanged PPO checkpoints are separately evaluated on the identical final
  seeds under `further_exploration/confirmation/aligned_ppo_*_740000/`.
  `summarize_single_q.py` validates the model hashes, update counts, paired seeds
  and raw outcomes and exports the new comparison. See `pilot/SINGLE_Q.md`.
  This rerun is complete: ordinary Q achieved 74/200 and 59/200 successes for
  seeds 42/7; unchanged PPO achieved 131/200 and 166/200 on the identical new
  episodes. All ordinary-Q models have one table and exactly 262,144 update
  calls, matching their real training interactions. The final 800 episodes
  were evaluated without parameter updates. The earlier Double Q success rates
  are not ordinary-Q results and are not reused in the new comparison.
  The old Double Q rates remain variant results and must not be copied into
  this ordinary-Q comparison. Historical top-level sources and Part 1 remain
  unchanged; the new B-only rerun is not the full refreshed A--D report.
- `notebooks/assignment4/current/` is the updated full Part 2 A--D workflow.
  Run `python notebooks/assignment4/current/experiment.py run --workers 3` from
  the repository root. `environment.py` preserves exact pilot features on map 0
  and provides compatible map-1 features for D. The audit checks exact B
  equivalence, B/C reward isolation and D shapes/geometry. Ordinary single-table
  Q-learning and SB3 shared-action PPO use 262,144 interactions and seeds 42/7.
  A new matched grid gives each method two 65,536-step candidates and 50 B
  development episodes each, training seed 31415. It selected Q alpha 0.15 and
  PPO LR 0.0003, held fixed across A/B/C. Exactly matching B checkpoints can be
  reused with source checks; their original training-time provenance is retained.
  All reported evaluations are rerun. Tuning seeds 760000--760049, curve seeds
  780000--780019 and final seeds 790000--790199 are separate. D freezes B weights
  without training. Never tune on the held-out final results or D performance.
  Records and source snapshots are under `results/assignment4/part2_current_v3/`.
  The renderer verifies all 3,200 episodes/traces and writes the new section 2
  and `Images/part2_learning_curves_v3.png` only after all 12 trained-model jobs
  complete. It does not compile a report PDF. Review `02_discussion.tex` manually
  against the new measurements; the old 1%/zero-PPO interpretation was replaced.
  The full v3 run completed: B success averages Q 31.0% and PPO 69.5%; C 21.8%
  and 58.5%; D 1.5% and 37.5%. Across-seed variation is large for PPO, including
  A successes of 0%/100%, so do not claim general reliability from these means.
  `verify.py` independently replayed eight saved A--D/algorithm episodes. The
  previous report text is archived in the run. Two seeds are not the five-seed
  Part 3 mechanism study. The capped power signal remains a declared limitation.
  Use this new runner/report command for the current report. The old top-level
  runner, notebook and `part2_controlled_v2` outputs are historical; preserve their
  registered source hashes rather than modifying them to imitate the new run.
- `notebooks/assignment4/improvements/` contains the user-authorised post-v3
  improvement search. Preserve the v3 sources/report/results. `check.py` checks
  unchanged Gymnasium transitions/rewards, ordinary Q updates and PPO inference.
  `experiment.py screen` gives each method two 262,144-decision B candidates,
  seed 42, evaluated on development seeds 810000--810049. `replicate.py` repeats
  an individually promising candidate without retraining rejected variants.
  The bulk `confirm` command ranks new candidates even when all lose to the
  baseline, so it is not an automatic quality gate. Do not select on D or
  reuse old final episodes for tuning.
  Ordinary Q is retained; no Double Q, replay, action masks or reward shaping.
  Changes include common observed motion/score/lives, compact Q state/credit
  variants and PPO actor/update variants. Read the directory README for exact
  settings and chronology. This is development after inspection, not a new
  independent final test or Part 3's five-seed mechanism experiment.
  The PPO `stable` package improved on both seeds: locked B confirmation on
  860000--860199 gave 96%/93%, versus old-model 66%/82.5%, 200 episodes per
  model. All weights were frozen; do not tune on these confirmation results.
  Five newly trained Q packages failed to beat the development baseline and
  are not recommended. `select_q_checkpoint.py` separately locks a common
  earlier checkpoint (196,608) from the original development curves and
  confirmed Q improvements on both seeds: 36%/39% versus 32%/30%, 200 episodes
  per model, mean 37.5% versus 31%. This is a modest gain, not reliable success.
  `selected_agents.py` loads the locked selected Q/PPO models and exposes
  `action(observation)` with the common Gymnasium 49-value wrapper.
  The original training run consumed
  262,144 interactions, but selected weights contain only 196,608 updates.
  These are B-only development revisions with unequal cumulative search, not
  a new matched A--D report; preserve the full v3 report and label scope.
- `notebooks/assignment4/q_motion/` is the user-requested isolated follow-up
  after the earlier Q packages failed. `motion_q.py` appends one relative
  closing-motion bit to the original Q key; all other key fields and Q alpha,
  gamma, epsilon, reward, actions and horizon are unchanged. `experiment.py`
  trains A/B seeds 42/7 for 262,144 decisions and reuses source/hash-matched
  v3 baselines. Primary comparison is full budget; a predefined secondary
  comparison uses 196,608 checkpoints. Fifty development seeds 890000--890049
  per checkpoint are separate from earlier tests. Records are under
  `results/assignment4/q_motion_v1/`; never tune on D or old final outcomes.
  Preserve these registered sources when resuming; use a new ID after changes.
  This remains a two-seed development pilot, not a five-seed mechanism study
  or a replacement of the complete A--D report. Read its README for diagnostic
  counts, motion thresholds and the inherited power/geometry limitations.
  A separate `tunnel_experiment.py` tests an observed graph repair, without
  the closing bit: four apparent dead ends in map 0 are actual wrap portals
  (41 boundary crossings in old B traces). `tunnel_features.py` adds paired
  boundary links only to feature extraction, leaving native dynamics unchanged.
  `check_tunnels.py` verified 1,244 identical A/B transitions/rewards, including
  a recorded wrap replay. The separate Q tunnel pilot uses matched settings
  and reuses the first study's baseline development episodes with hashes;
  outputs go to `results/assignment4/q_tunnels_v2/`. The v1 attempt failed
  before training due to a spawned-worker module-name collision; sources are
  preserved there. The runner now loads its study by explicit file path, with
  a spawned baseline-worker regression check. D never selects either
  representation. Preserve original graph code and frozen model/report records.
  The completed motion-bit development pilot has a trade-off: primary A mean
  success 31% versus 0%, B 21% versus 25%, with A seeds 62%/0%. At 196,608,
  A is 50% versus 0%, B 30% versus 33%. Do not claim a universal improvement.
  `confirm_transfer.py` selects representation/checkpoint only from the two
  A/B development studies, then locks hashes and tests A and frozen B-to-D on
  new seeds960000--960199,200 episodes per model. It includes matched checkpoint
  controls and, when applicable, the old full-budget D reference. Results are
  under `results/assignment4/q_representation_confirmation_v1/`. Never reselect
  from these confirmation results; A uses A-trained models and D adds no training.
  Completed locked confirmation selected motion at 196,608 from A/B development:
  A success 50% versus 0% (candidate seeds 100%/0%); D 2.5% versus 4.5% at
  matched checkpoints. The full-budget old D reference scored 3.25%. All cells
  have 200 new episodes per model, seeds 960000--960199, 2,000 episodes total.
  Do not promote this as a general Q or transfer improvement. `selected_q.py`
  loads the opt-in experimental model with the right Gymnasium wrapper; the
  previous B-focused loader and full report remain unchanged. The tunnel-only
  pilot gave A 0% and B 16% versus original B 25% at full budget, so it was not
  selected. Keep both failed and successful cases and the native graph evidence.
- `notebooks/assignment4/q_transfer/` continues user-authorised Q transfer
  exploration after verification of the motion study. `q_motion/verify_native.py`
  independently recounted 2,000 old A/D episodes and replayed 16 outcome cases
  with native BCD score agreement at every step. The old 50% A/2.5% D result
  is correct; no old confirmation episode becomes tuning data.
  `transfer_q.py` tests two separate one-component state changes: three-level
  directional threat, or zero/low/high bins of the existing capped power proxy.
  No precise-timer claim or RAM decoder change is made. Original Q updates,
  alpha/gamma/epsilon/reward/actions/horizon stay fixed; no closing bit, tunnel
  repair, Double Q or replay enters this study. `run_study.py` trains B seeds
  42/7, 262,144 decisions, checkpoints 196,608/262,144 and new development seeds
  1000000--1000049. Records/snapshots go to `results/assignment4/q_transfer_v1/`.
  Select a new representation only from B development before fresh frozen D
  testing. Previously inspected D layout is not a new independent domain just
  because episode seeds change. Read the README for scope and cumulative search.
  `resources_q.py` adds a separate package with last-life and near-goal score
  bits from the existing observation, leaving the original key/update intact.
  `run_resources.py` trains B seeds 42/7 with the same trainer/schedule/budget,
  storing separate records under `results/assignment4/q_resources_v1/`.
  This package changes two observed variables and is not a one-factor study.
  `confirm_d.py` waits for both B studies, chooses the strongest NEW candidate
  from threat/power/resources/power-route using B data only, then locks hashes
  before seeds 1060000--1060199. It compares 200 frozen D episodes per model
  against both original checkpoints, with native score agreement checked at
  every decision; results go to `results/assignment4/q_transfer_confirmation_v2/`.
  Never reselect from these D outcomes. The old baseline can be stronger in B;
  selecting a new candidate for a transfer hypothesis is not a claim of B gain.
  `power_route_q.py` separately retains the existing power-pellet direction
  field when unpowered and threatened; no action is forced or masked.
  `run_power_route.py` uses the same B trainer/seeds/budget, recording two new
  agents under `results/assignment4/q_power_route_v1/` before valid D locking.
  The first confirmation process had loaded an earlier plan while waiting;
  it was stopped and v1 marked aborted, with no outcomes inspected/used.
  Only v2 with fresh seeds and loaded-source hash checks is valid. Never edit
  registered source files in a running/waiting process.
  All four B candidates completed on seeds42/7. At196,608/262,144, mean success
  is baseline24%/41%, threat22%/12%, power20%/21%, resources5%/2%, power-route
  25%/27%. The D v2 lock selected the strongest NEW candidate, power-route at
  262,144, before its fresh D outcomes. This is not a B improvement over the
  stronger original model. Preserve all failures and do not cherry-pick D.
  Valid v2 D confirmation completed: candidate power-route262,144 mean4.5%
  (seed42=2%,seed7=7%) versus matched original262,144 mean2% (2%/2%). Original
  196,608 mean2.5%. Each model has200 fresh episodes,1,200 total, with native
  BCD score agreement checked every step and no D training. Only one training
  seed improved, so describe this as a small conditional gain, not reliable
  transfer. The new source models were weaker on B; do not replace the complete
  v3 report or original B-focused recommendation with these partial results.
- `notebooks/assignment4/three_hour/` is the user-approved 22 November 2026
  improvement study, bounded to 19:58--22:58 UTC. All native task/reward rules
  and ordinary single-table Q updates remain intact. The 61-value observation
  adds all four ghost offsets/activity to the audited 49 values. Seven matched
  candidate packages per algorithm use A/B, seed 42 and 262,144 decisions per
  setting, then the best two repeat on seed 7. No D result selects a candidate.
  `select.py` locks source-development decisions, `main_cohort.py` trains fresh
  main seeds with a common recorded cap, and `evaluate_final.py` selects
  checkpoints from each A/B/C model's source-condition development episodes
  using the same rule/allowance. Every run consumes the full cap; selected
  earlier weights must not be labelled as containing all those transitions.
  D reuses B's selected frozen model and gets zero additional training. Each
  final model has at least 200 separate episodes, full traces, native BCD score
  checks and hash/fingerprint checks. Native replay bypasses custom wrappers.
  The original v1 runner's duplicate `checkpoints` keyword failed only when
  assembling metadata after successful checkpoint/curve saving. Preserve v1.
  `recover_metadata*.py` checks all saved models and development episodes before
  recovering metadata; `run_study_v2.py` changes only that dictionary assembly.
  Queued-job rescheduling records actual v2 runtime sources separately and never
  overwrites an existing job directory. A small engineering cohort tested the
  corrected end-to-end save/completion path and is excluded from selection.
  `reproduce.py` provides the corrected sequence under a new run prefix; consult
  its README and the actual result locks under
  `results/assignment4/three_hour_20261122/`. Sources are immutable once locked.
  Keep all training seeds, failed candidates and original predictions. Do not
  present post-inspection development as an independent new map-domain test or
  fabricate human review. Only update report LaTeX/PNG after full verification;
  no report PDF/ZIP is compiled. Main algorithm comparisons and separate
  five-seed mechanism studies remain distinct.
  The November 22 training/search completed all 30 fresh main models. Final
  evaluation completed 39/40 cells, 7,800 episodes and 5,195,714 decisions.
  Q/PPO means are A32.9%/80%, B32.2%/67.4%, C26.9%/52.7%; PPO D13.9%.
  Q/D seed314 hit a native maze1->0 change at1760points, episode1150094,
  decision975, lives2 and no native termination. The guard correctly stopped
  evaluation; do not label it agent failure or average only four Q/D agents.
  `interrupted.json` preserves the 40-case lock and the report suppresses the
  incomplete Q/D aggregate. Exact native flags were independently reproduced.
  The user approved up to fifteen extra minutes for diagnosis/report checks,
  with no new training/model changes. Source-selected fixed B policies scored
  Q66.5%/PPO92.5%; their D rates were only2.5%/2%. Do not replace five-seed
  means with these n=1 choices. The earlier fixed PPO42/7 models separately
  verified98.5%/98%; these are different models, not five-seed reliability.
- `notebooks/assignment4/transfer_evaluation/` repairs measurement of the native
  D map-boundary interruption without modifying any learned model. All ten
  original B checkpoints (Q/PPO seeds17/29/101/314/2718) are locked before
  200 fresh D episodes per model, seeds1250000--1250199. `boundary_env.py`
  preserves native dynamics and pre-boundary features; map exits produce
  external truncation/censoring rather than agent failure. No action is
  allowed afterward. A censored return excludes the crossing action's reward
  and is the verified in-map prefix. Report S/N as observed in-map success
  and [S/N,(S+C)/N] as bounds on unresolved original-task outcomes, not CIs.
  Records/snapshots are separate under `results/assignment4/transfer_boundary_v1/`.
  `check.py` verifies1600 unchanged steps and the exact old975-step exit;
  `verify.py` checks all traces/model hashes and native replays. `update_report.py`
  replaces only D numbers/discussion in the reviewed report; `verify_report.py`
  checks old A/B/C and new D against raw CSVs with the reduced four-column
  outcome table. No original source/model/prediction/old-result rewrite is made.
  The2000-episode run measured Q4.5% observed success versus PPO14.7%.
  One Q episode is censored (1/1000), so the mean Q success bound is4.5--4.6%;
  PPO has no censored episodes. This is a post-inspection boundary repair,
  not improved learning, a new independent domain, or Part3's mechanism study.
- `notebooks/assignment3/student_dropout_combined.ipynb` - combined binary Assignment 3 workflow
  on branch `assignment3-combined`, with English answers to points 1-10. This is the current
  synthesis; the earlier three-class notebook is retained as a separate reference. Positive class
  is `Dropout=1`, negative is `Graduate=0`; 794 unresolved `Enrolled` records are excluded from
  supervised fitting/evaluation and receive only explicitly unvalidated scores after development.
  Excludes `Debtor` and `Tuition fees up to date` from the main experiment because their recording
  time is unverified. Uses an approximately 80/20 resolved-outcome development/test split and five-fold
  stratified group CV: identical retained-predictor profiles stay in one partition even with different
  outcomes (one such pair appears after administrative fields are excluded). No records are deleted.
  Preprocessing, log selection, rare-category pooling and reduction are fitted inside training folds.
  Compares explicit ANOVA counts, PCA and binary LDA jointly with classifier settings. Required
  ensemble pair is tree AdaBoost and KNN bagging; extra tree bagging has a single-tree control.
  Transfer adapts source-course network weights to the largest development course, with a matched
  scratch arm and a separate target-course test table. PyTorch DNN tuning uses internal stopping
  splits with training-only preprocessing, then refits on every outer training row for the selected
  epoch count. AP selects models; 20% capacity and OOF F2 thresholds are distinct policies fixed
  before testing. Calibration, conditional uncertainty, costs and limited subgroup diagnostics
  are reported. Results and figures use `results/assignment3/combined/` (gitignored) and
  `figures/assignment3/combined/`, preserving earlier outputs. See `notebooks/assignment3/README.md`.
- `notebooks/assignment3/check_lda_vs_adaboost.py` - development-only review of
  LDA as a classifier against the original locked AdaBoost. Reuses exact notebook
  split/preprocessing definitions, compares eight LDA settings on the same five
  grouped folds, and verifies that AdaBoost reproduces its original fold AP.
  Separate outputs under `results/assignment3/model_choice_review/` include OOF
  capacity/alert results, coefficients, fresh batch timings and conditional paired
  bootstrap differences. No held-out test is scored and the original notebook,
  model registry, test exports and report choice are not changed by this script.
  This is an exploratory follow-up after original test inspection, not a new
  independent test or proof of equivalence. See the assignment README.
- `notebooks/assignment3/finalize_lda_report.py` - current report finalization
  after the user chose LDA for explainability and development contact yield.
  Repeats the eight-setting LDA search on the same folds, freezes settings and
  OOF F2 threshold, and adds LDA's evaluation on the original test population.
  This is an exploratory revision after the original test report was known.
  Results go to `results/assignment3/lda_selected/`; vector figures are saved directly
  in `report/Images/`, with local gitignored PNG previews under
  `figures/assignment3/lda_selected/`. LDA/AdaBoost refit and batch timings are
  measured together; other costs retain the original-run provenance.
  The figure helper prefers finalized LDA exports when present. Report sections
  7--10 now recommend LDA, with AP selecting settings within families, rather
  than claiming LDA has the highest AP. Original combined exports are preserved.
- `notebooks/assignment3/report_model_comparison.py` refreshes the section-8
  `08_model_comparison_AP.pdf` from the same finalized development exports.
  It preserves the model-family bar-chart layout, uses actual mean AP/fold SD,
  and writes the PDF directly to `report/Images/`, with a local gitignored PNG preview.
  The finalizer calls it after the eight
  task-9/10 plots. Section 8 uses this figure instead of the duplicate AP table.
- `notebooks/assignment3/student_dropout_classification.ipynb` - self-contained English answers
  to Assignment 3 points 1-10. Uses the original `data/graduation_dataset.csv`, both semesters,
  three-class prediction and dropout ranking for an illustrative 20% student-support capacity.
  Run from `notebooks/assignment3/`. Results go to `results/assignment3/` (gitignored), figures
  to `figures/assignment3/`. A stratified 60/20/20 train/validation/test split fits all learned
  preprocessing and feature ranking on training data only; validation selects models and neural
  stopping epochs; test evaluation occurs only in point 10 after choices are locked.
  Three basic algorithms are KNN, decision tree and RBF SVM. Boosting uses shallow trees;
  bagging uses KNN, satisfying the requirement for different base algorithms. Point 8 interprets
  the unspecified transfer comparison as training-only autoencoder encoder transfer to classification,
  with a matched scratch control; it explicitly distinguishes this from cross-domain transfer.
  Point 9 tunes a three-hidden-layer PyTorch DNN. The CSV has no dropout dates or timestamped
  administrative fields: this is retrospective feasibility analysis, not validated prospective
  warning before dropout. Keep `Enrolled` distinct from confirmed graduation.
- `notebooks/assignment2/05_blob_detection.ipynb` - multiscale LoG blob detection over the entire
  discovered image dataset for both baseline and all seven parameter settings. The 10-image manifest
  selects display examples only. Full per-image/per-blob statistics and original-resolution numbered
  overlays are gitignored under `results/assignment2/blob_detection/` (overlays in `overlays/`).
  Tracked report figures show display examples and dataset-wide aggregates in
  `figures/assignment2/blob_detection/`. Images are loaded per worker; parameter variants of the same
  image reuse LoG responses, and vectorized overlap preserves the original suppression rule.
- `notebooks/assignment2/06_contour_detection.ipynb` - threshold-based contour detection using
  ContourPy, with full-dataset baseline statistics, all six parameter settings, and a matched LoG
  comparison on every discovered image. The ten manifest images select illustrations only.
  Results are in `results/assignment2/contour_detection/` and plots in
  `figures/assignment2/contour_detection/`. Both notebooks run independently; the contour notebook
  repeats the exact blob helpers and recomputes the full baseline. Keep these functions synchronized.
  Preparation matches EXIF orientation, aspect-preserving resize to maximum side 512 (no upscaling),
  and grayscale normalization. Retained contours include closed outer/hole loops; open paths and
  loops below the minimum area are excluded and counted separately. Images are loaded per worker;
  full per-image/per-contour CSVs and per-image gzip JSON boundary vertices are exported without
  retaining all image arrays or experiment paths in memory. Only display paths stay in memory.
- `notebooks/assignment2/detection_images.json` - shared display-image paths relative to the detection
  dataset, initially two images per class selected by sorted filename. It must not limit either
  full-dataset analysis. Update it when refining illustrations. Put the blob/contour comparison in
  the contour notebook. Other planned notebooks are `01_fourier.ipynb`, `02_pca.ipynb`, `03_hog.ipynb`,
  and `04_lbp.ipynb`; create them when work on those topics begins.
- `data/` - original datasets only; everything except `.gitkeep` is gitignored.
  Assignment 1 uses `data/graduation_dataset.csv`. Use assignment-specific subdirectories for new
  datasets and document their source and placement. The user-provided Assignment 2 detection dataset
  is `data/vehicle-type-detection/`, with hatchback, motorcycle, pickup, sedan, and suv class folders.
- `results/` - generated numerical results, derived datasets, and experiment configurations, organized
  by assignment and task (for example, `results/assignment2/blob_detection/`). Everything except
  `.gitkeep` is gitignored. Do not write generated results into `data/`.
- `figures/` - tracked plot outputs. Existing Assignment 1 plots are `01_continuous_boxplots.png`
  and `02_pca_explained_variance.png`. Use assignment-specific subdirectories for new figures.
- `report/` - Assignment 3 English LaTeX report project requested by the user, separate from
  the existing Assignment 2 report. `main.tex` imports ten files from `Sections/` in task order,
  using the supplied frontmatter/mainmatter layout. Includes a minimal compatible `packages.sty`,
  placeholder title page, bibliography, optional appendix and figure copies in `Images/`.
  Keep the structure compact: shared evaluation protocol in point 2, methods in 5/7,
  development comparisons in 6/8 and one final test/cost comparison in 10. Sections 9/10 are
  completed English report text with five main figures and inline tables. Three
  separate comparison plots that repeat tables are in the disabled optional
  `Appendices/task_09_10_extra_figures.tex`.
  Each task-9/10 figure contains one plot with its own number/caption; do not combine
  these plots into lettered subpanels. At the user's request, show the DNN training-loss
  and internal-validation plots side by side, retaining separate figure numbers/captions.
  Make requirement coverage explicit: 9.1 implementation,
  9.2 justified tuning, 9.3 measured tuning/training results; 10.1 classification,
  10.2 complexity, 10.3 computation time, 10.4 machine-learning/real-world implications.
  Interpret measured trade-offs rather than only listing scores; do not claim a guaranteed grade.
  Keep tasks 9/10 concise and use plain English. State the model, evaluation split,
  metric and unit when quoting a number. Explain AP as a ranking score, distinguish
  alert precision/recall from the appointment budget, and avoid unnecessary formulas
  or repeated diagnostics. Retain separate single-plot figures and criterion headings.
  Explain the alert rule before its confusion matrix: predicted Dropout raises
  an alert when the model score is at least the development-selected threshold
  (0.0614 rounded for the recommended LDA). This is separate from the top-20% contact list.
  Distinguish the reasons for the scores: AP selects ranking models; F2 selects
  alert thresholds because the scenario prioritizes finding dropout cases over
  avoiding every false alert. State this as a project assumption, not measured costs.
  The cutoff maximizes F2 on development cross-validation predictions, then stays fixed on test.
  Matrix rows are recorded outcomes and columns are predicted Graduate/Dropout;
  show student counts with correct/false-alert/missed-dropout labels, without
  row percentages that can be confused with precision.
  `tasks_09_10.tex`
  imports those same sections for a standalone preview build. Obsolete preview PDFs and
  Overleaf ZIP snapshots were removed during cleanup; use current sources and assets. Sections 1--6
  remain a scaffold whose italic prompts must be replaced before submission. Point 8
  documents the common CV comparison and the group's decision to omit transfer
  learning as outside the pooled-population scenario. The revised recommendation
  is LDA, with its added classification, cost and contact-budget results. Explain
  the trade-off and exploratory revision from the initial AP-only AdaBoost choice.
  Sections 9/10 omit transfer results, costs and cross-references. Use fewer
  semicolons and prefer clear sentences. The earlier notebook transfer experiment
  and its exports remain internal analysis unless the user requests code changes.
  Keep the computation-time environment description to one brief sentence naming
  the shared Intel Core Ultra 7 258V CPU. Software versions, thread settings and
  timing provenance go in the enabled computational-environment appendix.
  The user requests LaTeX and updated figure assets only. Do not compile report
  PDFs or create report ZIP snapshots unless requested.
  Seven figures are enabled, including the section-8 model comparison,
  in the main text and three older supporting figures are in the disabled appendix.
  `notebooks/assignment3/report_tasks_09_10.py` reads fixed exports, checks advisor counts,
  renders the new PDF/PNG figures and refreshes report image copies without retraining or
  selecting models. The notebook calls it after all numerical exports; DNN metadata and
  stopping/refit histories are saved separately. Keep report tables synchronized after reruns;
  the figure script does not overwrite authored report text. Keep all required numerical evidence in the report itself,
  and refresh figure copies/numbers from the combined binary workflow when experiments change.
  See `report/README.md` for integration into Overleaf or the user's existing NTNU template.
- `IT3212___Assignment_3.pdf` - user-supplied assembled group report (20 physical pages),
  preserved unchanged. It is a review input, not a new assignment brief. Its full Overleaf
  sources are absent locally. The review found missing AdamW/AP bibliography entries,
  an undocumented transfer comparison, LDA/AdaBoost recommendation disagreement,
  stale figure-7 image content and inconsistent preprocessing descriptions.
  `report/tasks_08_10.tex` previews coordinated corrected sections; `report/integration/`
  contains bibliography additions, a shared-protocol replacement and merge instructions.
  Do not claim the complete group PDF was rebuilt just because the local excerpt compiles.
  The revised notebook records actual CPU/software/threads, 25 prediction repeats after
  3 warm-ups with median/IQR, three development-only DNN seeds (42/7/2026), and full
  transfer/paired-bootstrap metadata. Original locked choices and test quality metrics
  are unchanged; all reported costs must use the newly measured timing tables.
- `reports/assignment2/` - exactly two English LaTeX section files for Overleaf, with question headings
  matching the PDF and all tables written inline. Use literal `Images/filename.png` figure paths;
  the accompanying `Images/` folder contains two blob figures, one contour-only overlay,
  one matched blob/contour detail comparison, and one contour parameter figure with unique filenames.
  `notebooks/assignment2/report_blob_figures.py` regenerates the blob figures using notebook functions
  and checks displayed results against CSV exports. `report_contour_figures.py` regenerates and
  checks the pickup/PIC_1.jpg contour-only overlay and detail comparison, marking the same windshield-frame
  and headlamp regions in both comparison panels. Keep the pure contour figure under question 2
  and the matched comparison under question 4. Questions 5 and 7 discuss measured examples and
  suitable applications, with enough explanation to distinguish representation from segmentation.
  Both methods and all parameter settings cover the full dataset; ten fixed paths select
  illustrations only. The contour report table summarizes each class rather than listing all images. Keep methods concise and
  use figures for measured results; the algorithm schematic is not included in the report.
  The sections are condensed for the larger report: retain the four blob and seven contour question
  headings, five result figures, and five inline tables. Introduce blob baseline parameters in prose
  and avoid repeating limitations or narrating every table entry.
  Do not introduce custom path macros or external table inputs. See its README for integration.
  Report scope is 1,310 images for both methods and their comparison; update inline tables and
  prose when experiments change. The user confirmed that vehicle-type-detection was downloaded
  from Canvas. These sections occur late in a larger report: start with methods and results,
  without repeating the general dataset introduction or adding provenance reminders.
  The user confirmed that only the report will be submitted, with no code or CSV attachments.
  Keep the report self-contained and do not refer readers to exported files or notebooks for results.
  Local CSV exports and code remain internal analysis and reproducibility artifacts.
- `requirements.txt` - shared Python dependencies with minimum versions.
- `reports/assignment4/` - English LaTeX writing project for Assignment 4.
  `main.tex` includes six section files in the brief's Part 1--6 order and an
  `ai_assistance.tex` record. The team's original Part 1 answer is now supplied in
  `Sections/01_predictions.tex`; preserve it unchanged and keep later annotations
  separate. The latest Part 2 workflow is `three_hour/`: fresh five-training-seed
  A/B/C comparison and frozen D tests of ordinary Q `gamma995` and PPO `stable`.
  Both use 262,144 interactions and the same source-development checkpoint
  rule; D inherits B. A/B/C retain the original five-seed results; current D
  uses `transfer_evaluation/` with all ten original B models and explicitly
  censored map exits. The original Q/D interruption remains historical. A
  separate source-selected operating-policy table has n=1 policy per cell
  and must not replace training-seed variability. The old v3 benchmark and
  later two-seed follow-ups are historical. In particular, PPO's historical
  94.5% is a fixed two-model B result, not five-seed A--D reliability.
  `three_hour/render_results.py` verifies complete frozen traces before writing
  numerical LaTeX/PNG; `refresh_discussion.py` updates interpretation and
  `validate_delivery.py` independently checks table arithmetic/LaTeX. Previous
  text is archived with the final run. Historical `current/report.py` cannot
  reproduce the new report; retain its sources and overwrite guard.
  Parts 3--6 remain prompts and empty placeholders, not results or peer feedback.
  `Sections/02_discussion.tex` contains authored interpretation after the measured
  Part 2 section; it keeps reviewed discussion separate from numerical rendering.
  Recheck this text when experiments change and include both Part 2 input lines
  when extracting the evidence brief.
  The 23 November Part 2 editorial review preserves all numerical table values,
  removes visible file/path references, explicitly records failure rates in
  prose and explains A/B, B/C and transfer findings. The figure identifies the
  22 November main experiment by date/settings/configuration/seeds instead of
  a filename. The subsequent requested edit explains arbitrary seed values
  and native maze identifiers, keeps only return in the outcome table, and
  uses the new censor-aware D results. The old interrupted run is not rewritten.
  The latest quality revision compresses Part 2 to about1100 prose words,
  retains outcome/cost Tables1/2 and adds configuration Table3. It answers
  each A--D question directly and audits paired-seed differences without
  retraining/reselecting agents. Q's almost unchanged A/B mean hides changes
  from-71to+48pp. PPO's B mean advantage survives omission of any paired seed
  (29.25--44.5pp); this is descriptive sensitivity, not significance or new
  independent validation. Two seeds per algorithm improve under C despite
  lower means, and the92.5%B PPO model achieves2%D. Preserve the one-censored
  Q transfer bound4.5--4.6%. Current archived text, analysis and verifier are
  under `results/assignment4/part2_quality_review_20261123/`; run its
  `verify_quality.py` for the three-table revision. Earlier two-table report
  generators/checkers are historical and must not replace this text.
  Earlier review records and archived pre-edit text are under
  `results/assignment4/section2_review_20261123/`. Do not run historical report
  or discussion generators over this reviewed text; their numerical exports
  and old report hashes remain historical provenance. Part 1 remains unchanged.
  The combined document is a writing layout; evidence
  brief (Parts 2/3/4/6, editorial grouping), original predictions and Part-5 review
  remain distinct submission components. Respect the separate 2,000-word evidence
  and 1,000-word review limits. See its README for Overleaf and extraction details.
  The selected environment is MsPacMan. Keep actual run IDs on all figures.
- `README.md` - course navigation, Python setup, local dataset paths and housekeeping policy.

### Post-Assignment 3 cleanup

Obsolete Assignment 3 notebook-build/review scratch files, report render/build files,
temporary PDF tools, package-download caches, local report previews and ZIP snapshots
were removed. Keep scientific Python dependencies in the selected environment, not
under `tmp/`; the maintained scripts no longer add scratch packages to `sys.path`.
The section-8 figure renderer no longer creates a one-off backup in `tmp/report_build/`.
Two unused report image copies (`03_deep_network_learning.png` and
`04_test_classification.png`) were removed from `report/Images/`; original plots
remain under `figures/assignment3/combined/`. Do not regenerate unused report copies.
LaTeX intermediates and local report preview/bundle names are now gitignored.
Generated task-9/10 plots and the section-8 comparison now have one versioned
vector PDF per figure in `report/Images/`. Duplicate PDFs under `figures/`,
obsolete task-9/10 plots from the initial combined report and duplicate report
PNGs were removed. Renderers save PDFs directly in the report and regenerate
gitignored PNG notebook previews under `figures/assignment3/combined/` (09/10 only)
or `figures/assignment3/lda_selected/`. Keep the earlier six tracked combined
overview plots and their referenced report assets. Do not commit regenerated
preview copies as additional report assets.
Original data, numerical experiment exports, notebooks, report sources, required
figure assets, the supplied group PDF, lecture/workshop work and previous assignment
deliverables are preserved. Keep the numerical exports: timings and revision provenance
cannot be recovered exactly by rerunning an experiment.

Do not rename, move, or overwrite previous deliverables just to start the next assignment.
Keep notebooks runnable from a fresh kernel, document working directories and data paths, and create
output directories before saving figures. Run notebooks from their assignment directory; paths to root-level
data and figures are `../../data/` and `../../figures/`. Assignment 1 uses this convention.
Never commit local datasets unless explicitly requested.

Notebook headings must follow the assignment PDF's numbered tasks and subquestions so readers can
identify exactly which requirement each section answers. Use supporting subsections within those
points rather than assigning new task numbers to setup, preprocessing, or conclusions. In Assignment 2,
blob detection uses points 1-4 and contour detection uses points 1-7. Apply the same convention to all
other notebooks and retain point letters where the brief uses them.

## Environment and validation

`requirements.txt` contains pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.12,
scikit-learn>=1.3, jupyter>=1.0, ipykernel>=6.25, scipy>=1.11, Pillow>=10.0,
contourpy>=1.1, torch>=2.6 (Workshop 1 and Assignment 3/4 neural networks),
threadpoolctl>=3.1 (bounded numerical threads), gymnasium>=1.1, ale-py>=0.12.1
(Assignment 4 Atari backend), stable-baselines3==2.9.2 (current Assignment 4 PPO),
plus IPython.
These are minimum constraints, not exact pins. SciPy, Pillow, and ContourPy support the image-processing notebooks.
Add new implementation dependencies (such as scikit-image, OpenCV, or Pillow) when introduced rather
than installing them ad hoc. PDF-reading tools used only to inspect briefs are not runtime dependencies.

There is no configured build, lint, or test suite. For notebook changes, execute the affected notebook
from top to bottom in a fresh kernel when the required data and dependencies are available. Check outputs,
array shapes, numerical ranges, and saved plots; report missing inputs or unexecuted work honestly.
Use fixed random seeds for sampling, noise generation, and splitting. Keep interpretations alongside
results; do not invent observations before running code.

## Assignment 1: Data preprocessing

### Dataset and existing implementation

The input is the UCI "Predict Students' Dropout and Academic Success" dataset: 4,424 rows and 35 columns.
The target is `Target`, with classes `Dropout`, `Enrolled`, and `Graduate`. Predictors are numeric,
but several represent nominal integer codes (for example, marital status, course, and occupation).
Do not treat those codes as continuous measurements.

The notebook explores the data, implements median/mode missing-value handling, retains plausible outliers,
applies `log1p` to strongly skewed curricular-unit counts, one-hot encodes nominal predictors, and uses
a stratified 80/20 split with `random_state=42`. Continuous predictors are standardized using training
statistics. The optional PCA section compares variance thresholds and selects 95%.

For future modeling, split before fitting data-dependent preprocessing, including imputation, category
discovery, skewness-based transform selection, scaling, and PCA. The existing notebook fits some earlier
preprocessing steps on the full dataset; do not assume the entire workflow is leakage-free because its
scaler and PCA use training data. Fit preprocessing inside training folds when cross-validating.
Preserve the previous assignment unless changes to it are part of the requested task.

### Rubric

1. **Data exploration (10):** first rows, summary statistics, data types, missing values, outliers,
   and unique categorical values.
2. **Data cleaning (20):** choose and implement missing-value handling and justify the choices.
3. **Handling outliers (20):** detect outliers (for example, IQR or Z-score), decide whether to remove,
   cap, or transform them, and justify the decisions.
4. **Data transformation (30):** encode categorical data and scale features; justify the encoding
   and explain why scaling is necessary and how it affects models.
5. **Data splitting (10):** create training/test sets and explain their role in evaluating
   generalization and detecting overfitting.
6. **Optional bonus (10):** apply dimensionality reduction such as PCA and discuss its effects.

### Submission

The brief requires a **PDF report**, not only a notebook. Include results and code only where necessary
for the explanation. The word limit is **3,000**, excluding code and references but including table and
figure captions. Be precise and concise. Missing a requested justification incurs a 10-mark deduction
from that task. Assignment 1 datasets must come from those uploaded to Blackboard.

## Assignment 2: Image processing

Passing requires **65 points**. Each task is assessed on **clarity of explanation (30%)**, **quality of
results (30%)**, and **insight in discussion (40%)**. Include implementation, visual results, and discussion;
code alone is insufficient. The supplied Assignment 2 PDF specifies neither a report word limit nor a
submission file format; do not automatically carry over Assignment 1's submission rules.

### Dataset requirements

The general brief allows Blackboard image datasets or outside datasets with a basic description.
However, blob detection specifically requires a provided Blackboard image dataset, and contour detection
must use the **same dataset**. Follow this more specific requirement for those tasks.
Document sources, selected examples, grayscale conversion, resizing, and normalization. If required
Blackboard images are missing, request their location rather than silently substituting another dataset.
Use multiple equally sized grayscale images for PCA.

The selected local detection dataset contains 1,310 JPG images: hatchback (181), motorcycle (122),
pickup (478), sedan (400), and suv (129). The user confirmed that vehicle-type-detection was obtained
from Canvas; use Canvas as its course download source even though the PDF names Blackboard.
Class labels are vehicle types, not ground-truth blob or contour annotations; do not equate
detected regions with vehicles or claim detection accuracy from these labels alone.
Save detection figures under `figures/assignment2/blob_detection/` and
`figures/assignment2/contour_detection/` respectively.

### Fourier transform (20 points)

1. Apply the 2D DFT to a grayscale image. Show the original and magnitude spectrum with an explanation.
2. Implement a frequency-domain low-pass filter to remove high-frequency noise. Compare original
   and filtered images and analyze the results.
3. Implement a high-pass filter to enhance edges. Show the filtered image and discuss its effects.
4. Compress by retaining selected percentages of Fourier coefficients. Reconstruct at multiple
   percentages and discuss image quality and compression ratio.

Use an appropriate spectrum visualization (for example, centered log magnitude). Explain coefficient
selection and filter masks, handle conjugate symmetry for real reconstructions, and distinguish
retained-coefficient fractions from actual stored-file compression ratios.

### PCA (25 points)

Normalize grayscale pixels to `[0, 1]`. Write a Python PCA function implementing the requested steps:

1. Form a matrix with one image per row and one pixel per column.
2. Center the data and compute the covariance matrix.
3. Calculate its eigenvalues and eigenvectors.
4. Sort eigenvectors by descending eigenvalue.
5. Select the top `k` eigenvectors.
6. Project images into the lower-dimensional space.

Reconstruct images (restoring the mean), compare originals and reconstructions for multiple `k` values,
plot explained variance, and justify a component count balancing compression and quality. Compute MSE
and discuss reconstruction error, visual information loss, and compression trade-offs.
Do not replace the required manual steps with only a call to `sklearn.decomposition.PCA`.
Choose a manageable image resolution: the pixel covariance matrix grows quadratically with pixel count.
Document any resizing.

### Image processing (55 points)

- **HOG (12):** compute descriptors using a library such as OpenCV or scikit-image. Apply to at least
  three images spanning simple and complex scenes. Show originals, gradient images, and HOG images.
  Compare descriptors and discuss changes to cell size, block size, and number of orientation bins.
- **LBP (13):** write a function for basic 8-neighbor grayscale LBP and a function for its histogram.
  Produce LBP images and histograms for at least three different grayscale images (for example, a
  natural scene, texture, and face). Explain texture information and compare histogram differences.
  State neighbor order, threshold convention, and image-border handling.
- **Blob detection (15):** implement a blob-detection algorithm on a Blackboard dataset. Mark detections
  with circles or bounding boxes. Report per-image counts, sizes, and positions, and evaluate how
  parameter choices affect detection.
- **Contour detection (15):** implement contour detection on the same dataset. Mark contours with
  different colors and report per-image counts, areas, and perimeters. Compare blob and contour results,
  advantages and limitations, effects of parameters such as thresholds and filter sizes, and examples
  where each method is more suitable.

## Assignment 3: Machine learning

Source of truth: `IT3212-2026-Assignment 3.pdf`. Total 200 points, passing mark 130.
The supplied one-page brief specifies neither submission format nor word limit. The point allocations
shown below follow the PDF; point 2 has no separate displayed allocation.

1. Develop a real-world and machine-learning problem statement, connecting the selected dataset
   and learning task to a real scenario (10).
2. Implement and justify preprocessing; Assignment 1 preprocessing may be reused.
3. Extract features and justify the methods (20; 10 without justification).
4. Select features or reduce dimensionality and justify the method, or justify using neither
   (20; 10 without justification).
5. Implement and justify three of KNN, linear/polynomial regression, additive model, decision trees,
   Naive Bayes, kernel SVM, or neural network (30; 10 without justification).
6. Compare and explain the three algorithms' performance in relation to the problem
   (20; 10 without explanation).
7. Implement boosting and bagging, explain the steps, and use different base algorithms with
   justification (20).
8. Compare basic, boosting, bagging and transfer approaches in relation to the learning problem
   (20; 10 without explanation). The brief does not define the transfer source/task; state the
   chosen interpretation explicitly rather than silently omitting it or claiming external transfer.
9. Implement a CNN or deep neural network with justified hyperparameter tuning (30). The brief
   explicitly permits using data directly after preprocessing for this point.
10. Compare complexity, computation time and classification performance, relating the results
    to both the machine-learning and real-world problems (30).

The selected problem concerns advisor prioritization using background and first-year results.
The combined workflow uses resolved binary outcomes and explicitly limits inference to that
population; it must not treat unresolved students as known non-dropouts. The earlier notebook
remains a three-class registered-status analysis. Maintain the distinction between recorded
status and future dropout after an intervention date.
No evidence that support reduces dropout can be inferred from classification metrics alone.
Retain Assignment 1 deliverables; fix preprocessing leakage in the new workflow rather than
assuming its earlier full-dataset transformations are safe for model evaluation.

## Assignment 4

Source of truth: `IT3212 - Assignment 4.pdf` (six pages). Compare off-policy
Q-learning with on-policy actor-critic PPO. Listed options are MsPacMan, Breakout
and SpaceInvaders. The brief also refers to maps, terminal success and compatible
state/action representations; clarify how these apply to the team's option before
choosing an environment or altering the required settings.

1. **Predictions before training (30):** mandatory timestamped prediction sheet,
   stated due date Friday 13 November. Main submission is not considered without
   it. Predict budget efficiency, seed variation, stochastic behaviour, step-penalty
   effects, training/evaluation differences and the initial recommendation. Include
   three states with expected actions, reasons and falsifying observations. Preserve
   original predictions; never reconstruct them after seeing results.
2. **Learning settings (20):** A deterministic transitions/terminal success reward;
   B stochastic transitions with other conditions fixed; C add a step penalty to B;
   D evaluate B agents on an unseen map without further training. Justify a common
   interaction budget and equal tuning allowance. Use common training seeds where
   practical, separate evaluation seeds and at least 200 evaluation episodes per
   trained agent. Disable evaluation updates, specify the evaluation policy and keep
   held-out evaluation out of tuning. Record success/failure rates, mean return,
   successful/failed episode lengths, interactions, runtime/hardware and variation
   across trained seeds. Plot learning against interactions and document smoothing
   and evaluation frequency. Do not treat pooled episodes as independent training
   replications or compare returns directly across different reward definitions.
3. **Mechanisms (20):** vary one main factor at a time: Q-learning epsilon schedule,
   discount factor, PPO entropy, learning rate and PPO update epochs. Use at least
   five seeds per tested configuration. Revisit predictions, inspect a contrary
   case, propose an alternative explanation and run a diagnostic that distinguishes
   explanations rather than merely increasing the budget.
4. **Episode inspection (20):** select ten recorded episodes: three typical successes,
   three typical failures, two surprising successes and two surprising failures.
   Justify selection using outcome distributions. Annotate three decisions per
   episode with state/actions, chosen action and Q-values or policy probabilities,
   observed transition/outcome and an alternative action with its uncertainty.
   Distinguish bad actions from unlucky outcomes and explain an aggregate limitation.
5. **Human review (30):** team review identifies strongest evidence, an overreaching
   claim and an alternative explanation, and compares these with original predictions.
6. **Recommendations (30):** choose an approach/configuration for a rapid explainable
   demonstration, repeated uncertain operation where failures matter, and changed
   environments. Each needs two experimental supports, an observed weakness, a
   reversal condition and another experiment before use beyond simulation. A conclusion
   that neither approach is supported is allowed; avoid universal superiority claims.

Submit executable notebooks/scripts with a documented reproduction command; experiment
configurations, raw logs, seeds, software versions and episode traces; original and
annotated prediction sheets; an evidence brief of at most 2,000 words excluding tables
and figure captions; and a review record of at most 1,000 words. Every figure must
identify source runs. A short record of consequential AI assistance, accepted/rejected
suggestions and verification is mandatory but ungraded. Do not fabricate runs,
predictions, peer feedback or observations. Keep Assignment 4's implementation,
datasets, figures and submission records distinct from prior assignments.
