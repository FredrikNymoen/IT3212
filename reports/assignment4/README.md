# Assignment 4 deliverables

Read both root PDFs, including `clarification-assingment4 (1).pdf`.
The combined report includes Parts2,3,4,5 and6, with the human review between
Parts4 and6. Within it, Parts2/3/4/6 form the evidence brief (maximum2,000 words,
excluding tables/captions); Part5 is the review record (maximum1,000 words).
Separate evidence-brief and review exports remain available. Original
predictions with annotations and the AI record are separate documents.

| Document | Editable source | Compiled output |
|---|---|---|
| Combined report, including Part5 | `main.tex`, Sections 02/03/04/05/06 | `../../output/pdf/assignment4_report.pdf` |
| Evidence brief only | `evidence_brief.tex`, Sections 02/03/04/06 | `../../output/pdf/assignment4_evidence_brief.pdf` |
| Original predictions + later annotations | `predictions.tex`, Sections 01 | `../../output/pdf/assignment4_predictions.pdf` |
| Human review | `review.tex`, Section 05, `review_input.json` | `../../output/pdf/assignment4_review_status.pdf` |
| Consequential AI assistance | `ai_record.tex`, `ai_assistance.tex` | `../../output/pdf/assignment4_ai_record.pdf` |

**Part 5 records the completed group review.** The user supplied Fredrik,
Johan, Elling and Mats's initial rankings, the group's final ordering and
rationale, and broad support for the evidence assessments. These inputs are
preserved in `review_input.json`. No additional member-specific arguments or
discussion details are invented. Retain the original Part 1 submission timestamp;
this project preserves the text but does not certify a timestamp.

Use pdfLaTeX. In Overleaf select the appropriate main document above. The
brief contains no visible source filenames/paths. The tables and captions
identify source studies, conditions, training roots and evaluation episodes. Exact
paths and fingerprints belong to the separate experiment records.

## Current evidence

- **Part 2 A/B/C:** learning-settings five-root study,262,144 consumed interactions
  per run; checkpoints selected using source-setting development data only.
- **Primary D:** fixed observation obstruction, all ten frozen B
  models, clean/obstructed paired controls, 4,000 episodes, no learning.
  This explicitly tests a RAM/object-feature analogue of spatial occlusion,
  not unseen-map generalisation or a pixel encoder. Old map-selector D is
  preserved as historical supplementary evidence, not mixed into primary D.
  A/B/C/D each have a dedicated analysis subsection and panel in the same
  figure. D additionally shows every clean/obstructed model pair and both
  directions of episode outcome changes, rather than only its overall mean.
- **Part 3:** 45 fresh models, nine one-factor configurations, five roots each,
  fixed final 262,144-step weights; 9,000 primary episodes. All results retained.
  Ten baseline weights match their historical full-budget counterparts bitwise.
- **Diagnostic:** 1,000 secondary frozen episodes, argmax versus sampled PPO,
  fifty per model/mode, reported separately from 200-episode primary tests.
- **Part 4:** ten distribution-selected B/C episodes, thirty actual decisions
  and thirty full-prefix native one-step alternatives. Local coupled outcomes
  are not claimed as expected risk or complete alternative routes.
  One focused figure illustrates four findings. One grouped decision log
  retains all30 states, full action vectors, observed/alternative outcomes
  and short interpretations, alongside the episode roster. Captions stay short;
  a single notation guide applies to the whole log.

All 19,000 primary episodes and 13,581,678 recorded decisions were independently
checked against episode metrics, native BCD scores, rewards and action values.
The additional 1,000 diagnostic episodes contain 725,743 decisions. Verification
receipts also record direct native replays and original-prediction integrity.
Word count is checked across the entire brief, rather than Part 2 alone.

## Reproduce and audit

See [the completion guide](../../notebooks/assignment4/completion/README.md)
for exact commands, raw-record mapping, seed lists, budget rationale, software,
hardware, timing adjustments, package contents and limitations. Its reproduction
command never overwrites this report or tunes on final tests. Reproduction of
the existing seeds is not a new independent validation study.

Historical `report_review` and `synthesis` utilities describe the earlier
Part 2/map-D snapshot. Use `completion/verify_report.py` for the current brief.
The previous report and long AI history were retained in the ignored
`completion_report_before` records before this replacement.
