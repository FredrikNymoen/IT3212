# Approved three-hour improvement study — 22 November 2026

The report now uses the [subsequent frozen-transfer evaluation](../transfer_evaluation/README.md)
for D. It retains all ten B checkpoints, makes map exits explicit censored
outcomes and adds no training. The interrupted D study documented below is
preserved as historical provenance, rather than silently rewritten.

Work window: 19:58–22:58 UTC (20:58–23:58 Europe/Oslo). Run commands from
the repository root. Preserve all earlier sources, model files and run IDs.
Original predictions are unchanged; no report PDF or old ZIP is rebuilt.

## Verified starting point

`verify_ppo.py` checks original and revised B PPO models on 200 fresh episodes
per model (1100000–1100199), without training. Revised `stable` PPO achieved
98.5%/98.0%, versus original 64.5%/78.5% for training seeds 42/7. Every score
matches native BCD RAM at every decision; frame-level stickiness is 0.25,
NumPy/Torch actors agree, and model hashes/parameters remain frozen. This
supports the historical B-only 94.5% confirmation; it is not an A/C/D result.

## Fixed candidate screen

All tasks use the same native Gymnasium game, five actions, score rewards,
2,000-point goal and 1,000-decision limit. A has sticky probability zero;
B/C/D use 0.25; only C adds a 0.1-point decision penalty. D freezes B models
and changes the native maze selector. No teacher policy, action mask, search,
reward shaping, Double Q, replay or eligibility traces enters this study.

The common observation has 61 values: the previously audited 49 plus the
observed offsets and activity flags of all four ghosts. Compact Q deliberately
discards information. Original PPO variants also retain their previous 49-input
critic and 23-input/action scorer; only `allghost` consumes the extra values.
Thus representation differences are explicit parts of each workflow.

Each algorithm gets four initial candidates and three separately registered
additions, A/B separately, training seed 42,
262,144 decisions per run, 20 development episodes at 0/65,536/131,072/
196,608/262,144 (1110000–1110019). No D outcomes are used for selection.

| Algorithm | Candidate | Change from its current reference |
|---|---|---|
| Q | baseline | Original compact key, alpha .15, gamma .99, epsilon 1→.05 over 80% |
| Q | motion | Previously tested conditional relative-closing bit |
| Q | gamma995 | Only discount changes to .995 |
| Q | bearing | Replace directional danger mask with eight-sector nearest-ghost bearing, two distance bands, absent/far bin |
| PPO | stable | Previously successful scheduled PPO package |
| PPO | long | Only discount changes from .995 to .999 |
| PPO | allghost | Add all four ghosts' directional position/motion/activity to the learned scorer, and full 61-input critic |
| PPO | careful | LR .0002→.00005, four epochs, clip .1, KL .01; other stable settings fixed |
| Q (addition) | alignment | Replace one alignment bit with signed food-relative 3×3 corridor-offset bins |
| PPO (addition) | powerclock | Stable PPO with a low-six-bit power proxy instead of capping the full RAM byte; not a validated exact timer |
| Q (sixth) | gamma999 | Only discount changes from .99 to .999 |
| PPO (sixth) | explore | Only entropy schedule changes from .02→.002 to .04→.01 |
| Q (seventh) | compact | Keep food/corridor/danger/power bins, collapse movement to stopped/moving and omit alignment |
| PPO (seventh) | small | Only network widths change to 16/16 actor and 32/32 critic |

The additions preserve the running screen's source files. Both passed 600
identical native-transition checks. Existing A/B traces contained no active
ghost rows with nonzero RAM116 but zero low six bits, so no binary Q power
status bug was established. The low-bit PPO candidate tests a different
continuous proxy; it is not presented as a proven timer decoder. Alignment
instead addresses the Q key's omission of signed junction position. The
extra study uses one worker alongside the original three (four total).
The sixth-candidate study adds one worker while these finish (five total,
out of eight available CPU cores). It follows the first Q gamma=.995 A
result, before B selection or any new D outcome. Equal additional candidate
counts/budgets are retained for the two algorithms. This remains exploratory
development, not an independent five-seed mechanism comparison.
The final two smaller-representation candidates follow the failed richer Q
keys, before B selection or D testing. They use one further worker (six total
at peak), with the same 262,144-per-setting allowance. No more candidates are
added after this seventh pair. Compact Q still has exactly one ordinary table;
small PPO passed actor parity, critic dimensions and save/load checks.

The v1 runner had a duplicate `checkpoints` keyword in its final metadata
dictionary, after successful model/curve saving. Frozen v1 sources are retained.
`recover_metadata.py` verifies every checkpoint/hash and development episode
before reconstructing metadata, without retraining or outcome edits.
`run_study_v2.py` changes only that dictionary construction; `cohort.py` uses
v2 explicitly. A separate 2,048-decision engineering cohort verified the full
Q/PPO training, checkpoint, metadata and completion path. It never enters
model selection or reported success comparisons.

Q always inherits the original ordinary Bellman update with one table and one
update per observed transition. PPO uses four 512-step environments, batch 256,
GAE .97, entropy .02→.002; stable/long/allghost use up to six epochs, clip .15,
KL .015 and LR .0003→.00005. Scorer has two 32-unit tanh layers, critic two
64-unit layers. `allghost` has 43 actor inputs per action rather than 23.

The shortlist is the top two variants per algorithm, ranked by their best
common-checkpoint mean A/B development success, then mean raw score, then
earlier checkpoint. Replicate both shortlisted candidates on seed 7 with
20 different development episodes (1120000–1120019). The next lock uses mean
A/B success across both training seeds, then score, then earlier checkpoint.
This is an equal new search allowance; unequal historical exploration remains
disclosed. These development studies do not satisfy Part 3's five-seed rule.

## Confirmation plan

Before main training, record selected configurations, common budget, seeds and
all hashes in a separate protocol. The main cohort was locked at 21:40 UTC
with 262,144 decisions (128 PPO rollouts), fresh common training seeds
17/29/101/314/2718, A/B/C and eight single-threaded workers. The earlier target
of 393,216 was reduced before main training after a long interval without
process progress consumed wall time. A temporary system-sleep prevention
request lasts only until completion or the approved deadline. None of these
main training seeds enters the 42/7 variant search. Ten source-condition
development episodes every 65,536 interactions conserve evaluation time.
Variant/hyperparameter selection remains A/B only.
Before main training, the stopping rule is fixed as follows: each A/B/C
trained model selects a checkpoint using only its source-condition development
success, then raw game score, then earlier point. The identical rule and
checkpoint allowance apply to both algorithms and every condition. Variants
and hyperparameters remain locked from A/B; C does not tune those settings.
This responds to source A pilot regressions before main training/final
evaluation. Every training seed remains in the comparison, every run consumes
the full common cap, and weights selected earlier contain fewer transitions.
D reuses the source B checkpoint, and no final/D outcomes select anything.
The final evaluator also supports a common-checkpoint control, labelled
separately if requested; do not silently mix the rules.
Use 200 separate final episodes per model/condition (1150000–1150199), Q greedy
with seeded uniform ties and PPO argmax. D never trains. New D seeds do not
make the previously inspected map an independent domain.

Budget/seeds are fixed before main training and final evaluation, based on
measured runtime rather than final outcomes. If runtime requires a smaller
main scope, disclose that explicitly. Main records include model hashes,
native score checks, frozen-parameter checks, conditional lengths and full
decision traces for all final episodes.

```powershell
python notebooks/assignment4/three_hour/check.py
python notebooks/assignment4/three_hour/verify_ppo.py
python notebooks/assignment4/three_hour/reproduce.py --run-id assignment4_reproduction_NEW --workers 8 --main-budget 262144 --update-report
```

The runner refuses changed-source resumes. Registered source files must stay
unchanged while workers run or checkpoints are reused. New implementations
need new files/run IDs. `check.py` verified 900 identical A/B/C native
transitions, 400 ordinary Q updates, and all four PPO actor/save/load paths.

This is further development after earlier results were known, not a newly
independent algorithm study. Report successful and unsuccessful changes and
variation between trained agents. Never count evaluation episodes as training
replications or fill in human review on the team's behalf.

The current round ran the stages separately, preserving the v1 metadata
failure and source-equivalent rescheduled queued jobs. The reproducer uses the
corrected engine throughout and the same candidate/training/evaluation rules;
it has not been redundantly executed end-to-end during this time-limited run.
The actual stage commands and source/selection locks are retained with results.
`main_cohort.py` schedules the longer PPO jobs first to reduce serial tails,
with at most eight single-threaded workers on the eight available CPU cores.
The final native replayer bypasses all custom task/feature wrappers and
checks representative logged actions directly in Gymnasium's Atari engine.

The frozen source A/B search selected Q `gamma995` and PPO `stable` after
replication on seed 7. Q A success at the final 262,144 checkpoint was
100%/90% for seeds 42/7; the old representation/update with gamma .99 had
0% on A. These are development results, not the new five-seed final table.
PPO's chosen package retained the original capped power proxy; neither the
all-ghost nor low-bit-proxy candidate is silently described as adopted.

`confirm_q_gamma.py --lock-only` prelocks an optional paired confirmation on
fresh seeds1170000–1170199, for source training seeds42/7, A/B/D, gamma .99
versus .995 at exactly262,144 updates. It never selects models or changes
the fresh-seed main cohort. Only report its numbers if `completed.json`
exists; it remains a two-seed check, not the required Part3 five-seed study.

## Frozen final evaluation and report

The main training cohort completed all 30 jobs. Final evaluation completed
39 of 40 cells: **7,800 episodes and 5,195,714 native decisions**. All A/B/C
models and all PPO/D models have 200 episodes. Q/D training seed 314 hit the
map guard on episode seed 1150094, decision 975: native maze 1 changed to 0
at 1,760 points, with two lives and no native termination. The exact event
was independently reproduced with native flags. It is an unresolved protocol
case, not a measured 0% success policy. Its partial directory is preserved;
`interrupted.json` records the missing cell. No `completed.json` is fabricated.
The five-seed Q/D aggregate is suppressed, rather than averaging only the four
complete agents or changing the task after seeing this event.

| Setting | Q success, five-seed mean | PPO success, five-seed mean |
|---|---:|---:|
| A | 32.9% | 80.0% |
| B | 32.2% | 67.4% |
| C | 26.9% | 52.7% |
| D | Unresolved | 13.9% |

These fresh-seed results are not a paired causal comparison against the old
two-seed benchmark. The B operating policies chosen using only development
data achieved Q 66.5% and PPO 92.5%, each on 200 final episodes. Their frozen
B-to-D rates were only 2.5% and 2.0%. Keeping the full training-seed average
visible prevents high fixed-model success from becoming a reliability claim.
The user approved up to fifteen additional minutes for guard diagnosis and
report verification after the training/search window ended; no further
training or model adjustment was made.

The executed `stream_final.py` schedules each completed main model while other
independent training jobs finish. Its plan fixes all test seeds and runtime
hashes before any test episode. Each model/checkpoint is chosen only from its
source-development CSV, with a per-case hash lock before evaluation; D inherits
that B lock. Scheduling changes no policy or stopping rule. Main training and
final evaluation each use at most eight single-thread workers; their overlap
briefly reaches sixteen processes. Reported runtimes reflect shared compute.

`fast_evaluation.py` removes an unused intermediate 15-value feature calculation.
`fast_evaluation_v2.py` also caches unchanged geometry when the relevant RAM
and preceding motion permit it, while refreshing score, lives and time.
Engineering comparisons checked bitwise-identical 61-value observations,
full native RAM, rewards, flags and info over 701 decisions each in A/B/C/D.
Neither optimization skips native game steps or changes the environment.

After a final run is complete or explicitly marked interrupted, these commands verify and
refresh the report from the frozen records:

```powershell
python notebooks/assignment4/three_hour/choose_main_policy.py
python notebooks/assignment4/three_hour/verify_native_replay.py --final-run three_hour_20261122/final_v1
python notebooks/assignment4/three_hour/render_results.py --training-run three_hour_20261122/main_v1 --final-run three_hour_20261122/final_v1
python notebooks/assignment4/three_hour/refresh_discussion.py
python notebooks/assignment4/three_hour/validate_delivery.py --final-run three_hour_20261122/final_v1
```

`choose_main_policy.py` has a separately recorded source-only rule. It chooses
one of the five fresh main policies per source condition using development
success, score, checkpoint age and seed. A small operating-policy table reports
the corresponding already-frozen 200-episode result, with n=1 policy per cell.
It never replaces the primary five-seed table. D always inherits the B choice,
even if that choice transfers poorly. The separate `operating_agents.py`
uniform-50-episode study and paired gamma confirmation are optional plans;
their existence is not evidence that those additional episodes were run.

For a native Gymnasium evaluation demonstration, use `play.py --help` and supply
the condition, algorithm and source-selected seed explicitly. That command
loads a frozen checkpoint and never trains it. The default lowest seed is not
necessarily the source-selected operating model.

The locked source-development rule chose PPO seed 314 for B (checkpoint
131,072), Q seed 2718 for B (262,144), PPO seed 2718 for A (196,608), and Q
seed 29 for A (196,608). The complete individual cells can be loaded now:

```powershell
python notebooks/assignment4/three_hour/play.py --algorithm PPO --setting B --seed 314 --episodes 5
python notebooks/assignment4/three_hour/play.py --algorithm Q-learning --setting A --seed 29 --episodes 5
```

These commands print native episode outcomes without opening a render window.
Their demonstration seeds are separate from the reported test seeds. A high
success rate for a chosen fixed policy does not guarantee the same result
when retraining, or when its B policy is transferred to D.

`finish_partial_report.py` records this specific interruption and preserves
the original 40-case lock. The original `finish_report.py` is only for a fully
completed evaluation. `update_guides.py` was a one-time documentation migration;
the final interruption disclosure and numerical summary were reviewed afterward.
Do not continue this interrupted evaluation under altered task semantics.
A future D boundary design needs a new protocol/run ID and explicit scope.
The reproducer recognises only this exact known interruption, verifies all
39 other cells, reproduces the native failure flags and exports the same
suppressed Q/D row. It refuses other missing cases. This faithfully reproduces
the incomplete study; it does not repair the task definition. The final
exception-handling sequence was reviewed and compiled, while the expensive
whole search was not repeated solely to exercise that recovery branch.
