# Assignment 4 report

## Unified November 26 revision (clarification-compliant)

All primary Part 2 (Settings A, B, C and D) and Part 3/4 experiments in the
current report run in **one consistent computing environment** (AMD Ryzen 5
7500F CPU, Windows 11, Python 3.13.7, Gymnasium 1.4.0, ALE 0.12.1, PyTorch
2.15.0+cpu, Stable-Baselines3 2.9.2), following `clarification-assingment4.pdf`:

- **Setting D** uses paired 32×32 observation obstruction (`x in [16,48), y in [24,56)`)
  on 200 fresh seeds (`1600000--1600199`) with zero further training.
- **Part 3** combines the 7 three-value sweeps (`e8c0c6a8f4071325`, `PPO12 = 90.2 ± 8.0%`)
  with both the early-learning diagnostic (`3142fac3111db314`) and the frozen
  argmax-vs-sampled zero-entropy diagnostic (`results/assignment4/part3_extras/exec_diagnostic`).
- **Part 4** renders complete native RGB frames (`Images/part4_TS1.pdf`--`part4_SF2.pdf`)
  and runs real 1-step and matched 64-decision counterfactual replays in Gymnasium.

### One-sequence reproduction commands (from repository root)

```powershell
# 1. Unified Part 2 (Settings A/C training+eval, Setting D paired obstruction, A/B ghost probe)
python notebooks/assignment4/run_unified_part2.py --stage all --workers 10

# 2. Part 3 execution diagnostic, train-eval rank checks, same-state probes, and figures
python notebooks/assignment4/run_part3_extras.py --stage all --workers 10

# 3. Part 4 native RGB frames and 1-step / 64-step counterfactual replays
python notebooks/assignment4/run_part4_replays_and_frames.py

# 4. Verify word limits (<=2,000 brief / <=1,000 review), LaTeX refs, figures, and source hashes
python notebooks/assignment4/verify_final_report.py
```

---

## Historical notes (November 22--23)

The earlier Part 2 combined the original five-seed A/B/C measurements in
[the November 22 guide](../../notebooks/assignment4/three_hour/README.md) with
[the November 23 frozen-transfer evaluation](../../notebooks/assignment4/transfer_evaluation/README.md).
Ordinary single-table Q-learning (`gamma995`) and PPO (`stable`) use training
seeds 17/29/101/314/2718 and a common 262,144-interaction cap. D retains all
ten source-development-selected B models and adds zero training.

The original D guard aborted one Q test at a native map change. A new locked
evaluation records map exits as censored, stops before any action on the
other maze, and tests every model on 200 fresh episodes (2,000 total).
Observed D success is Q 4.5% and PPO 14.7%. One Q episode is censored, giving
an across-agent mean success bound of 4.5--4.6%; PPO has no censored episodes.
These are measured in-map success rates, with censoring explicitly separate
from failure. The old interrupted run is preserved. The boundary revision
follows inspection of that error and is not an independent new-domain test.

The older complete v3 benchmark and B-only/A/D follow-ups remain archived.
The historical 94.5% PPO result concerned two fixed revised B policies. It
was never the complete A–D benchmark or five-seed training reliability. The
new study independently checks those policies, then tests fresh training
variability separately. Double Q variants remain historical side experiments.

English LaTeX writing project based on all six pages of
`IT3212 - Assignment 4.pdf`. Open `main.tex` locally or upload this directory
to Overleaf and select `main.tex` as the main document. Use pdfLaTeX.
No generated PDF or ZIP is included.

## Original Part 1

The team has pasted its completed answer into
[Sections/01_predictions.tex](Sections/01_predictions.tex). It is preserved
unchanged while implementing Part 2. Keep the original submission timestamp or
reference with the submitted sheet; none is invented by this project.
Keep the original submitted sheet separately as the submission record.
Later annotations belong below the original, dated and linked to actual runs;
do not rewrite the predictions after seeing results.

## Where to write

| Part | File | Content |
|---|---|---|
| 1 | `Sections/01_predictions.tex` and `Sections/01_prediction_annotations.tex` | Original prediction sheet and subsequent evidence annotations |
| 2 | `Sections/02_learning_settings.tex` | Environment/protocol, settings A--D, measured comparison |
| 3 | `Sections/03_mechanisms.tex` | Controlled experiments, prediction check and diagnostic |
| 4 | `Sections/04_episode_inspection.tex` | Ten selected episodes, three decisions per episode and aggregate limitations |
| 5 | `Sections/05_human_review.tex` | Actual team review record |
| 6 | `Sections/06_recommendations.tex` | Three operating scenarios and limits of the conclusions |
| AI record | `ai_assistance.tex` | Consequential suggestions, team decisions and verification |

The complete Part 2, including the A--D interpretation, now lives in
`Sections/02_learning_settings.tex`, included once by `main.tex`. The former
separate discussion source was merged and removed to avoid an incomplete
answer when reading only the Part 2 file. The visible text contains no file paths
or filenames; the figure identifies the 22 November main experiment by its
date, settings, configuration and training seeds. That experiment maps to
`three_hour_20261122/main_v1` for training/curves and `final_v1` for frozen
evaluation. Include the single Part 2 input when extracting the evidence brief.
The current revision has 1,194 prose words, with
outcomes and costs as Tables 1/2 and a compact configuration comparison as
Table 3. Table 4 explains the design choices and their limits; Table 5 shows all five training seeds,
including the exact Q/D censoring cell and its 7.0--7.5% success bound.
Return is retained without a duplicate game-score column; C raw
scores remain in prose. Maze names and arbitrary seed identifiers are
explained. The interpretation answers A--D directly, checks paired-seed
sensitivity and distinguishes lower observed D success from its censoring
bound. All existing outcome/cost table numbers are unchanged.

The justification review explains the target/horizon, RAM features/five actions,
registered four-frame/sticky defaults, 0.1-point penalty, reward scaling, rollout
cap, replication/search/monitoring allowance, Q/PPO designs and exploration,
discount/GAE/update safeguards, exploitation policy and D censoring. It checks
the development-seed ranges against all search/main protocols. The compact Q
key omits time, score and lives while PPO retains them; this is now explicit.
One tested penalty size does not establish a size-dependent effect. These are
design reasons and package-level selection evidence, not proof that every
constant is optimal. Earlier unequal search and the inspected D domain remain
limitations. Keep room for the other parts within the whole 2,000-word limit.

The `assignment4-part2-synthesis` branch compares the fixed Johan, Elling,
Mats and Fredrik revisions in
[the synthesis review](../../notebooks/assignment4/synthesis/COMPARISON.md).
Its [common notebook](../../notebooks/assignment4/02_part2_synthesis.ipynb)
provides the reviewed protocol, portable evidence snapshot, per-seed overview
and native recorded-episode inspection. Only ordinary Q-learning and PPO are
used. Incompatible rewards, representations, budgets and PPO policies are
not pooled into these tables. Registered sources/models and Part 1 stay
unchanged; the synthesis adds no training or new evaluation outcomes.

This is an editorial revision of verified exports. Historical numerical and
discussion generators do not preserve the single-file layout and must not overwrite it.
Use the review records under `results/assignment4/section2_review_20261123/`
for the earlier requirement audit and archived text. The latest D arithmetic,
native replay and report checks are under `results/assignment4/transfer_boundary_v1/`.
The latest prose/claim audit and archived previous version are under
`results/assignment4/part2_quality_review_20261123/`. Its descriptive analysis
does not tune or retrain models. The current validation command is:

```powershell
python notebooks/assignment4/report_review/analyse.py
python notebooks/assignment4/report_review/verify_quality.py
```

Outputs of these maintained checks go to `results/assignment4/report_review_current/`.
The single-file justification revision is archived before editing under
`results/assignment4/part2_justification_review_20261123/before/`.
The older authoring/validation scripts assume the previous two-table layout;
their outputs remain historical and must not overwrite the concise revision.
The original experiment manifests retain their historical report hashes.

Headings follow the brief's Part 1--6 numbering. Comments contain writing
prompts; italic bracketed text is a placeholder, not completed report prose.
Replace the author/date placeholders in `main.tex`. The selected option is
MsPacMan. Part 2's code and reproduction instructions are in
[notebooks/assignment4](../../notebooks/assignment4/README.md).

Part 2 uses score-based rewards in A/B, as the team confirmed was agreed with
the instructor, with a step penalty added in C. The latest curve is `Images/part2_learning_curves_20261122.png`; it shows
unsmoothed development success and variation across five newly trained agents.
No report PDF or ZIP snapshot is rebuilt. Parts 3, 4 and 6 use the frozen-test
records and documented replays below. Part 5 is reserved for the actual team
review. Original predictions are unchanged.
Fresh evaluation seeds do not make the previously inspected D layout an
independent new domain. Prior cumulative search was unequal and is disclosed;
this study gives both methods the same new candidate/interaction allowance.

## Parts 3 and 4 records

The Part 3 notebook and experiment records identify study
`e8c0c6a8f4071325` and frozen test `054b0e6b63381969`. Part 4 reuses
ten of those frozen-test episodes with their original models and seeds.
From the repository root, use the original Part 3 interpreter and
a fresh output directory to reproduce the selection and traces.

```powershell
python notebooks/assignment4/inspect_part4.py --source results/assignment4/part3/study/e8c0c6a8f4071325/held_out_054b0e6b63381969 --output results/assignment4/part4/reproduced
```

The report's replay archive is `results/assignment4/part4/part3_B_054b0e6b63381969`.
Its selection manifest records the distributions, case IDs and original outcomes.
The ten NPZ traces contain observations, RAM, rewards, actions and all five
Q-values or PPO probabilities per decision. `reviewed_decisions.csv` indexes
the 30 decisions in the report.

## Submission components and limits

The brief separately requests an **evidence brief of at most 2,000 words**,
excluding tables and figure captions, and a **review record of at most
1,000 words**, alongside the original/annotated predictions, executable code
with a reproduction command, and experiment records. The AI-assistance record
is mandatory but ungraded.

`main.tex` is a combined working document. This template groups Parts 2, 3, 4
and 6 as the evidence brief, keeps Part 1 as the prediction record, and uses
Part 5 as the review record. This grouping is an editorial choice, not an
additional requirement in the brief. Do not interpret the two word limits as
one shared allowance. For separate documents, retain only the corresponding
`\input` lines in a copy of `main.tex`; every section preserves its original
Part number. Keep the original prediction submission unchanged.

Use the prompts to cover all requirements, especially separate training and
evaluation, at least 200 evaluation episodes per trained agent, at least five
seeds per tested mechanism configuration, and traceable episode evidence.
Record every figure's source configuration/run IDs in its caption. Create
`Images/` when adding actual figures and use literal relative paths such as
`Images/learning_curves.pdf`. Do not invent observations to fill the template.

Assignment 4 code belongs in `notebooks/assignment4/`, numerical logs and traces
in `results/assignment4/`, and generated plots in `figures/assignment4/`.
Part 2 keeps its canonical PNG learning curve in this report's `Images/`
directory. Duplicate generated previews are ignored; older images remain
available in Git checkpoint `714c0cb` and the local cleanup archive.
Results are gitignored, so explicitly include the required experiment records
when preparing the submission. The report folder alone is not the complete
Assignment 4 submission.
