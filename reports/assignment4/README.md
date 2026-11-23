# Assignment 4 report

The current Part 2 combines the original five-seed A/B/C measurements in
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
| 1 | `Sections/01_predictions.tex` | Original prediction sheet and optional later annotations |
| 2 | `Sections/02_learning_settings.tex` | Environment/protocol, settings A--D, measured comparison |
| 3 | `Sections/03_mechanisms.tex` | Controlled experiments, prediction check and diagnostic |
| 4 | `Sections/04_episode_inspection.tex` | Ten selected episodes, three decisions per episode and aggregate limitations |
| 5 | `Sections/05_human_review.tex` | Actual team review record |
| 6 | `Sections/06_recommendations.tex` | Three operating scenarios and limits of the conclusions |
| AI record | `ai_assistance.tex` | Consequential suggestions, team decisions and verification |

Part 2 also imports `Sections/02_discussion.tex` for authored interpretation.
`02_learning_settings.tex` and `02_discussion.tex` were reviewed against all
Part 2 requirements on 23 November. The visible text contains no file paths
or filenames; the figure identifies the 22 November main experiment by its
date, settings, configuration and training seeds. That experiment maps to
`three_hour_20261122/main_v1` for training/curves and `final_v1` for frozen
evaluation. Include both Part 2 input lines when extracting the evidence brief.
The current concise revision has approximately 1,100 prose words, with
outcomes and costs as Tables 1/2 and a compact configuration comparison as
Table 3. Return is retained without a duplicate game-score column; C raw
scores remain in prose. Maze names and arbitrary seed identifiers are
explained. The interpretation answers A--D directly, checks paired-seed
sensitivity and distinguishes lower observed D success from its censoring
bound. All existing outcome/cost table numbers are unchanged.

This is an editorial revision of verified exports. Historical numerical and
discussion generators do not preserve it and must not overwrite these sections.
Use the review records under `results/assignment4/section2_review_20261123/`
for the earlier requirement audit and archived text. The latest D arithmetic,
native replay and report checks are under `results/assignment4/transfer_boundary_v1/`.
The latest prose/claim audit and archived previous version are under
`results/assignment4/part2_quality_review_20261123/`. Its descriptive analysis
does not tune or retrain models. The current validation command is:

```powershell
python results/assignment4/part2_quality_review_20261123/verify_quality.py
```

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
No report PDF or ZIP snapshot is rebuilt. Part 3--6 placeholders remain;
five seeds in Part 2 alone do not complete Part 3's controlled mechanism study.
Part 5 must record an actual team review. Original predictions are unchanged.
Fresh evaluation seeds do not make the previously inspected D layout an
independent new domain. Prior cumulative search was unequal and is disclosed;
this study gives both methods the same new candidate/interaction allowance.

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
Part 2 keeps its PNG learning curve in this report's `Images/` directory and
in `figures/assignment4/three_hour/`.
Results are gitignored, so explicitly include the required experiment records
when preparing the submission. The report folder alone is not the complete
Assignment 4 submission.
