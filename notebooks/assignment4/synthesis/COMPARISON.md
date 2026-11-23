# Four-branch review for Assignment 4, Part 2

Reviewed 23 November 2026 against all six pages of the root assignment brief.
Scope confirmed by the user: **Part 2 only**, ordinary tabular Q-learning
versus PPO; the user explicitly excludes DQN. A grade cannot be guaranteed.
The brief rewards defensible evidence even when agents perform poorly.

## Fixed source revisions

| Contributor | Reviewed ref | Commit |
|---|---|---|
| Johan | `origin/johan` | `09c262cfffc84d79a01d09c2ab9e4193de4be824` |
| Elling | `origin/elling-assignment4` | `1261b3ad170a0f7b7781f0c829b5ae091bdd8da2` |
| Mats | `origin/mats-øving-4` | `c59c25ca62d0fa562ae119c394894fdd142e0b2c` |
| Fredrik/current | `assignment4` before synthesis | `1a7fb642c08866aa9acffffac0df31e1c9fd68de` |

Sources were read from Git objects into an ignored research directory, without
checking out collaborators' trees or executing their notebooks. The provenance
record identifies exact source blobs. The common branch starts at Fredrik's
revision and imports selected design strengths, not a wholesale merge that
would delete other assignments or mix incompatible result directories.

## Findings by branch

### Johan

`notebooks/assignment4/RL-API-final.ipynb` contains a useful reusable session,
training/evaluation and plotting structure. Its final Part 2 cells propose
image-based DQN versus PPO, 100,000 interactions, five seeds, 200 evaluation
episodes and separately named step penalties. This does not implement the
ordinary Q-table comparison promised in our unchanged prediction sheet.
The Part 2 launch flags default to false; configured experiments must not be
described as completed results merely because generic demonstrations have outputs.

Its D uses ALE `mode=1/2` as an unseen-maze switch. An independent local
Gymnasium/ALE check found available modes 0–3, but RAM maze selector 0 was
still zero immediately after reset in **all four modes**. Thus mode selection
alone does not establish the claimed held-out starting maze. This observation
does not establish that modes have identical subsequent dynamics.

Adopt the single notebook/read-only interface idea. Reject the DQN branch,
the unverified map substitution, and all unmatched performance claims.
The seven-cell policy notebook's code is identical to Elling's policy notebook;
it is shared material, not independent replication.

### Elling

The Part 2 notebook family separates A/B/C/D configurations, records per-agent
outcomes and offers an especially useful local dashboard with recorded-action
replay and RAM checks. The Q-table notebook uses a compact RAM key; the pellet
follow-up adds food direction from RGB corridor geometry and the RAM pellet
bitmap. The pellet utility correctly distinguishes intervening walls and
explicit horizontal wrap edges, useful for understanding our approximate graph.

The initial terminal-reward study uses a 16,384-decision cap, three training
seeds, a 1,024-decision horizon, nine actions, image PPO/DQN and a separately
run tabular-Q notebook. Later B-only studies add a score bonus and extend to
32,768/65,536 decisions. The final improvement notebook controls Q pellet
direction versus the original key and PPO learning rate, with 200 final
episodes and separate monitoring. These are informative follow-ups, not a
complete matched replacement for our ordinary-Q/PPO A–D comparison.

Important differences include terminal reward/score-bonus scale, shorter
budgets, nine rather than five actions, image versus RAM observations, and
sampled versus argmax PPO. Some continuation cells run after earlier final
evaluation on reused test seeds; those inspected checkpoints cannot become
fresh independent test evidence. A native map exit below the target is
treated as a failure in the initial wrapper, whereas our repaired protocol
records unresolved outcomes separately.

Adopt the per-agent overview and native recorded-trace inspection. Do not
install a different pellet/graph representation beneath existing checkpoints
or mix those outcome percentages into our report. No DQN code/results are imported.

### Mats

`øv4/code/` provides a strong explicit experiment contract: five common training
seeds, equal four-candidate searches each repeated across five seeds, 200
primary episodes per model, separately reported sampled/greedy PPO, source
hash checks and observation regression tests. The v8 completion record and
40 primary episode CSVs cover 30 trained A/B/C models plus ten frozen-D cells
(8,000 episodes). A separate 500,000-interaction run also has all 40 cells.

The protocols differ materially from ours: 17 features, nine actions, a
1,500-decision horizon, a +100 success bonus, C penalty 0.5, startup accounting
and sampled primary PPO. The v8 100k B means recomputed from raw outcome
fields are Q 0.2% and PPO 2.9%; 500k B means are 0.3% and 2.8%. This says
nothing about isolated algorithm quality relative to our different workflow.

Later `code_wall_fix/` studies include navigation features, warm-start PPO,
temperature changes, imitation/DAgger, transfer-specific packages and longer
training. They are distinct studies with different cumulative search and
selection. High-scoring individual demo policies must not be presented as
five-seed performance on all A–D settings. The viewer explicitly labels its
highest-test-score selection as retrospective demonstration selection; the
common viewer instead requires an explicitly named recorded model/episode.

The later frontier study is genuinely promising: `cold_30k` averages 98.3%
on B across three existing training seeds, with 300 episodes/model and
sampling temperature 0.2. This is a warm-start continuation with a different
representation, horizon, reward and search history, not a matched ordinary-Q
comparison. Later transfer continuations train on mazes 0/2/3; the averaged
policy's 48.8% on held-out maze 1 is a three-seed, 400-episode/model result.
It does not train on maze 1, but its training-map distribution differs from
our B condition, and it averages multiple continuation networks. We retain
these as explicit next-study leads, rather than replacing a frozen B-to-D
cell with a differently trained policy. A future adoption requires the same
new allowance for Q and PPO and a complete newly locked A/B/C-to-D study.

Adopt strict source/model compatibility, portable archived paths, inspection
of action values/probabilities and a complete per-training-seed table. Retain
our verified RAM pellet bitmap rather than replacing it with a screen estimate
that can become stale. Do not claim the capped power RAM proxy is a precise timer.

### Fredrik/current

The strongest coherent basis for the requested ordinary-Q/PPO Part 2 is the
existing five-seed study: 262,144 interactions per trained A/B/C model, equal
new search allowance, common configurations across settings, source-only
checkpoint selection, 200 frozen episodes per model/condition, native-score
and model-hash checks, complete decision traces and independently replayed
outcomes. All ten frozen B models are included in the repaired D evaluation.

Limitations remain: unequal earlier cumulative development, different Q/PPO
representations, only ten main development episodes/checkpoint, approximate
graph features, an imprecise capped power proxy, large retraining variation
and weak transfer. The D repair follows inspection of a native map change;
new episode seeds do not make that previously inspected map an independent
new domain. The one censored episode is not silently counted as failure.

Retain these runtime sources/models and the original prediction sheet byte
for byte. Keep the existing outcome/cost numbers. Add the all-seed table and
make the representation limitation explicit. The new tools reorganise and
verify existing evidence; they do not improve measured agent performance.

## Requirement-to-evidence map

| Part 2 requirement | Common branch evidence |
|---|---|
| A deterministic; compare learning efficiency | Sticky probability 0, interaction-axis development curve, full-cap and selected-checkpoint counts |
| B stochastic; other conditions fixed | Sticky probability 0.25 only; paired-seed A/B analysis; same chosen hyperparameters |
| C changes incentives | B minus 0.1 points/decision; success/failure and raw score comparison; conditional-length caveat |
| D unseen during training, no further learning | Identical B checkpoint hashes, RAM maze intervention, compatible 61-value/five-action format; explicit censoring |
| Meaningful generalisation | Relative Q key/shared PPO action scorer, plus aliasing, graph and power limitations |
| Common justified interaction cap | 262,144 = 128 × 2,048 PPO rollouts; resource cap, not convergence; D zero new training |
| Common training seeds | 17/29/101/314/2718; identifiers fixed before main training; shared seeds need not produce shared trajectories |
| Frozen separate evaluation | No update calls; fingerprints; development/final ranges separate; 200 episodes for every one of 40 cells |
| Declared policy | Greedy Q with seeded uniform ties; argmax PPO; no sampled-mode figures mixed into the primary table |
| Equal tuning allowance | Seven A/B packages per method at seed 42, top two at seed 7, 4,718,592 interactions each; prior unequal exploration disclosed |
| No final-outcome tuning | Locked configurations/checkpoints; D repair changes measurement only, not learned models |
| Runtime/hardware | Per-agent train/evaluation seconds and common CPU/thread/concurrency description |
| All requested metrics | Outcome/cost tables, failure/censoring prose, mean return, conditional lengths, interaction cap and across-agent sample SD |
| Curves and variation | Unsmoothed development checks every 65,536 interactions, ten episodes/model; all five seeds displayed individually |
| Executable evidence | Common notebook, hash-checked export, overview and native replay; original expensive reproduction commands retained |

This is a Part 2 synthesis. It does not claim to complete the five-seed
one-factor Part 3 studies, the ten annotated Part 4 episodes, a real human
review, or the three Part 6 recommendations. The whole evidence brief still
has a 2,000-word limit excluding tables/captions; Part 2 remains concise.
