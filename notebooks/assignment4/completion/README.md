# Complete evidence study: Parts 3/4/6 and clarified D

Read both root PDFs: `IT3212 - Assignment 4.pdf` and
`clarification-assingment4 (1).pdf`. The clarification defines the 2,000-word
evidence brief as Parts 2/3/4/6; original predictions/annotations and the
1,000-word human review are separate. Part5 now incorporates the group's
reported individual rankings, final priorities, rationale and broad support.
Do not invent further discussion details. Preserve Part1.

## Registered immutable runs

**Completion status, 23 November:** all 45 models, 9,000 Part 3 primary
episodes, 4,000 clarified-D episodes and 1,000 diagnostic episodes are finished.
The current report covers Parts 2/3/4/6 and separately annotated predictions.
Part5 is completed using the four members' rankings, group priorities,
rationale and broad endorsement supplied by the user. Those inputs are
preserved separately. Registered sources below remain
immutable for reproducibility after completion as well.
Report prose/captions omit specific calendar dates at the user's request;
the raw records retain timestamps. A/B/C/D have parallel analytical subsections
and a shared four-panel figure. D additionally reports clean/obstructed pairs,
per-root success changes and both directions of episode outcome changes.

The following commands have been launched on the original Part 2 host/software.
Do not modify their worker sources or `q_learning.py`/`ppo.py` while they run:

```powershell
python notebooks/assignment4/completion/study.py run --run-id completion_part3_v1 --workers 6
python notebooks/assignment4/completion/robustness.py check --run-id clarified_D_v1
python notebooks/assignment4/completion/robustness.py run --run-id clarified_D_v1 --workers 1
```

Use a NEW run ID when reproducing; the recorded IDs preserve originals.
Protocol/model hashes are checked before work and completed jobs are reused
only under the exact same protocol. Partial jobs are never silently overwritten.
Each run stores source snapshots. Records remain under ignored `results/assignment4/`.

## Part 3: prespecified one-factor arms

All arms use native Setting B, 262,144 real interactions, the same five roots
17/29/101/314/2718, the same observation/action/reward/horizon, and a **fixed final
checkpoint**. Both baselines are trained afresh, rather than comparing a newly
modified model against a different computer/library's historical baseline.
The old source-selected Part 2 checkpoints/results are retained separately.

| Algorithm | Baseline | Single-factor alternatives |
|---|---|---|
| Ordinary Q | alpha .15, gamma .995, epsilon 1→.05 over 80% cap | epsilon reaches .05 at 20%; gamma .99; alpha .05 |
| PPO | gamma .995, entropy .02→.002, LR .0003→.00005, six epochs | gamma .99; zero entropy; LR .0001→.00005; two epochs |

This is nine configurations × five training seeds =45 freshly trained models,
11,796,480 training interactions. It is mechanism investigation, **not a search
that promotes a final-test winner**. Every arm is retained. Changing both ends
of the entropy trajectory changes one factor, the entropy schedule.

Development seeds1480000–1480009 supply ten episodes at0 and every65536
interactions; no development checkpoint selection is performed. Final seeds
1500000–1500199 supply200 frozen episodes for each model (9,000 total).
Primary policy is greedy Q with seeded uniform ties or PPO argmax. Full final
traces, raw training episode metrics, checkpoint diagnostics and runtimes are
saved. Fast feature computation was rechecked on701 native transitions in
each A/B/C/D condition before use, with bitwise observation/RAM/reward parity.
One numeric thread/worker is enforced; concurrency/shared CPU costs are explicit.

## Clarified Setting D

The original map-selector experiment is historical supplementary evidence,
not the primary D requested by the clarification. D now compares all ten
original source-selected B models under clean versus obstructed observations,
200 paired seeds1600000–1600199 per model/mode, 4,000 episodes and zero training.
The same native B game/maze, rewards, actions and task limits are retained.

Agents consume object features, not RGB. Painting a rectangle only on their
video would leave input unchanged. `observation_shift.py` implements the
explicit **object-feature analogue**: centres in a fixed32×32 rectangle
`x=[16,48), y=[24,56)` are hidden before recomputing object-derived features.
It is3.05% of the160×210 frame and does not cover score/lives alone. Hidden
food/ghosts cannot leak via nearest-object, density, motion or threat features.
An obscured player retains its last visible pose. Static maze knowledge and
non-spatial game-status channels stay available. This is not a claim of
pixel-CNN robustness, complete perceptual occlusion or unseen-map transfer.

The region/rules and original B model hashes lock before occluded test outcomes;
there is no mask-location/size search. Engineering matched1600 native action
steps, proved no underlying game change, bitwise clean-feature parity, actual
input changes, and noninterference from hidden ghost/player/food variables.

## Part 4 selection and local alternatives

```powershell
python notebooks/assignment4/completion/inspect_episodes.py --run-id completion_inspection_NEW
```

Ten original B/C held-out recorded episodes are selected from outcome
distributions. Typical episodes minimise conditional distance to median length,
then score. Surprises come from the model with the rarest positive relevant
outcome, then that model's conditional median. Exact duplicates are excluded.
This use of held-out records is episode analysis, not parameter/model tuning.

The stricter original ten-episode requirement is retained despite the
clarification mentioning traces for six. Each episode has three separated
trace-driven choices: a score gain, a controllable approach before life loss,
and an ambiguous close-ghost choice. All30 decisions record state, five action
values/probabilities, observed transition and a documented alternative.

Alternatives replay the complete recorded prefix in another identically seeded
native Gymnasium game before taking one different action. Each prefix RAM/reward
is verified, preserving controller and RNG history. RAM snapshot restoration
alone failed the next sticky-action transition in engineering v2, so it is not
used to claim a valid alternative. The baseline episode is never modified.
One coupled step cannot establish expected return or eventual escape/success.
Earlier engineering selections/replays are preserved under separate run IDs.

## Measured findings and what they do not establish

Primary clarified D success averages Q23.2% and PPO48.5%, versus paired clean
29.5% and65.9%. Some roots improve; the higher retained PPO rate does not
mean a smaller degradation. This is one fixed object-feature obstruction,
not pixel perception, native maze transfer or general reliability.

At the full fixed cap, Part3 baseline Q/PPO success is29.2%/37.6%.
Faster Q epsilon decay gives10.6%, gamma.99 gives31.6%, alpha.05 gives13.3%.
PPO gamma.99 gives33.2%, zero entropy65.2%, lower initial LR8.9%, two epochs5.8%.
All configurations have five training roots and200 primary episodes/model.
Zero entropy loses29points at root2718. In the separate50-episode/model/mode
diagnostic its advantage is25.6points under argmax but0.8under sampling;
execution mode mediates the apparent benefit. Do not promote a test winner.

Ten fresh full-budget baseline weights are bitwise identical to their
historical counterparts. Part2's development-selected earlier checkpoints
and different evaluation range explain why its mean cannot be used as a
controlled comparator for Part3 modifications. No causal hardware claim is made.

## Reproduction from the repository root

Use Python3.13.7 and the recorded `requirements-lock.txt` on Windows CPU.
Install the `torch==2.14.1+cpu` wheel from PyTorch's CPU index, then the lockfile;
the root requirements are a broader course environment, not an exact run lock.
Gymnasium/ALE must be able to load MsPacMan before training. No ROM is copied
into the submission. LaTeX compilation requires pdfLaTeX and Poppler `pdftoppm`.

```powershell
# Inspect cost and scope; does not train or create a run.
python notebooks/assignment4/completion/reproduce.py --run-id reproduction_NEW --plan

# Hours of training/evaluation: 30 selected main models +45 mechanism models.
# Fresh directory required. Uses original seed lists, not an independent test.
python notebooks/assignment4/completion/reproduce.py --run-id reproduction_NEW --workers 6
```

This retrains the fixed selected configurations, evaluates A/B/C and the new
clean/obstructed D, then all mechanism arms and the secondary diagnostic.
It never invokes legacy forced-map D or overwrites the canonical report.
The readable algorithms are `../q_learning.py` and `../ppo.py`.
The reproduction coordinator's plan and component implementations were checked;
the complete additional75-model reproduction was not run a second time.

The historical matched search is also reproducible, separately, through
`../three_hour/reproduce.py --run-id search_NEW --workers 6`; that command
includes the old map-D protocol and may stop at its documented map-boundary
guard. Do not use its report-update option for the current report.
All seven candidates/method, failed candidates, top-two replication and
source-development selection records are supplied. Fixed-config reproduction
does not claim to rerun or equalise earlier unequal exploratory tuning.

## Recompute the current report from supplied records

```powershell
python notebooks/assignment4/completion/summarize.py
python notebooks/assignment4/completion/figures.py
python notebooks/assignment4/completion/verify_records.py
python notebooks/assignment4/completion/report.py
python notebooks/assignment4/completion/verify_report.py
python notebooks/assignment4/completion/build_documents.py
```

`verify_records.py` is the **full local archive** audit:95 primary trace
files/19,000 episodes/13,581,678 decisions plus20 secondary trace files/
1,000 episodes/725,743 decisions. It checks episode accounting, native score
increments, greedy decisions/probabilities, model/source hashes and19 direct
wrapper-free native replays. Do not expect a trimmed submission to contain
optional traces for all episodes; the ZIP verifier below audits its manifest
and required ten analyzed full traces instead. The complete local audit receipt
records hashes for all115 original trace files.

`report.py` replaces editorial sections, so preserve manual changes first.
`episode_figures.py` converts the thirty verified decisions into five vector
figures with native state crops, action-value bars and observed/alternative
transitions. Part4 keeps one roster table; per-decision interpretations are
figure captions. This does not remove any required evidence.
`verify_report.py` independently recomputes every primary outcome/cost/root
table and mechanism/diagnostic table and checks all30 plotted vectors/annotations,
original prediction checksum and the **whole** evidence-brief word limit.
`review.py` renders the completed human review from the four initial rankings,
the final group priorities and rationale in `reports/assignment4/review_input.json`.
The user's broad support for the evidence assessments is retained as qualified
support. Do not invent quotations, member-specific arguments or voting details.
The verifier checks all four ranking rows, the final order and the1,000-word
review limit. `build_documents.py` compiles separate
PDFs twice, rejects unresolved references/overflow, and renders every page.

## Record mapping and provenance

| Report evidence | Raw records under `results/assignment4/` |
|---|---|
| Main A/B/C; development-selected weights; matched search | `three_hour_20261122/main_v1`, `final_v1`, `selected.json`, search/replica protocols |
| Clarified clean/obstructed D | `clarified_D_v1/protocol.json`, 20 evaluation directories |
| Five-seed one-factor arms | `completion_part3_v1/protocol.json`, `training`, `evaluation`, `final_lock.json` |
| Frozen argmax/sampled diagnostic | `completion_diagnostic_v1/plan.json`, `locked.json`,20 cell directories |
| Ten episodes/30 native decisions and alternative forks | `completion_inspection_v4/selection.json`, `inspection.json`, `annotations.json`, native PNGs; corresponding full main traces |
| Same-state original prediction probes | `completion_prediction_probes_v1`, original B model hashes |
| Training-vs-evaluation rank check | `completion_training_ranks_v1` |
| Derived95-agent statistics and full verification receipts | `completion_analysis_v1` |

All primary studies use training roots17/29/101/314/2718, with200 final
episodes per model. Seed ranges are phase identifiers, not chosen for good
performance. Main selected weights contain131072--262144 updates; mechanism
weights contain exactly262144. D never adds training. The secondary execution
mode has50 episodes per cell and remains separate.

Host: Intel Core Ultra7 258V, Windows11, CPU-only PyTorch; one numeric thread
per worker. Main historical overlap reached16 workers, mechanisms6, D1 and
diagnostic3. Operational timings include shared-host scheduling, not isolated
CPU benchmarks. A user pause lasted2203.8067seconds; raw times are retained,
and derived active wall costs remove it only in identified phases. The
ledger and checkpoint/publication evidence accompany the records. Development
and search costs are not silently folded into per-main-model training times.

## Submission package

`python notebooks/assignment4/completion/bundle.py` creates one local ZIP in
`results/assignment4/submission/`. It contains PDFs/editable sources, actual
code and environment dependencies, all primary/secondary episode metrics,
training logs/configurations/seeds/software/hardware, models/checkpoints,
search provenance, and full step-level traces for all ten analyzed episodes.
Optional full traces for the other models remain in the local archive and
are represented by the full-audit hash receipt. `--verify` checks every ZIP
entry and all ten required episodes (eight complete model trace archives).
It never packages `.git`, credentials,
virtual environments, unrelated assignments or the ALE ROM.

The compact tracked `evidence.json` exposes all95 primary model-condition
records and19 aggregate groups; it cannot replace the raw experiment package.
The previous report/long AI record and failed engineering attempts are retained
locally; no superseded D result is relabelled as the clarified primary result.
