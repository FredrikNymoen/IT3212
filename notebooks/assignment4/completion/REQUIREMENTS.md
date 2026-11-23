# Requirement-to-evidence check

Source of truth: the six-page assignment and two-page instructor clarification
in the repository root. This is an audit map, not a grade guarantee.

| Requirement | Evidence and scope |
|---|---|
| Original timestamped predictions | Original text SHA preserved; separate dated annotations. Original submission timestamp must accompany the team's own submitted record; none is reconstructed. |
| A deterministic; B stochastic; C penalty | Native Gymnasium ALE; A sticky0, B sticky.25, C B-minus.1/decision. Instructor-approved score increments replace terminal-only A/B reward. |
| Clarified D | All ten frozen B models, fixed32x32 object-feature obstruction, same game; clean/obstructed paired200 episodes/model/mode. Explicitly not literal pixel-CNN occlusion or unseen-map transfer. |
| Compatible meaningful representation | Shared61-value format, selected49-value prefix, relative Q bins/shared directional actor; Q drops time/score/lives. Power proxy and approximate graph limitations disclosed. |
| Common justified interaction cap | 262144 decisions=128 complete PPO rollouts; Q receives same cap. Source-selected weights can be earlier; costs include full cap. |
| Matched search allowance | Seven packages per method A/B at42, best two repeat7;4718592 decisions/method. Earlier unequal development disclosed, all candidate logs retained. |
| Training/evaluation separation | Roots17/29/101/314/2718; disjoint development/final seed ranges; fingerprints/read-only Q; no final/D selection. Argmax PPO/greedy seeded-tie Q primary. |
| >=200 primary episodes/model | 6000 main A/B/C;4000 clarified-D control/obstruction;9000 mechanisms. Secondary execution mode explicitly50/model/mode, not a substitute primary evaluation. |
| Outcomes and costs | Success, complementary failure, return, success/failure length, full interactions, raw/adjusted runtime, CPU/software, five-root SD. No redundant score column in primary outcome table; raw C score given separately. |
| Learning curves | Native interaction x-axis, unsmoothed, ten development episodes/model at0/every65536; across-root SD. Final tests excluded. |
| One-factor mechanisms, >=5 seeds/config | Nine configurations ×five roots; Q epsilon/gamma/alpha, PPO gamma/entropy/LR/epochs; fresh same-host baselines, fixed final checkpoint, all arms retained. |
| Hardware/checkpoint concern | Ten baseline final weights match historical final checkpoints bitwise. Part2 checkpoint selection differs from Part3 fixed cap; no causal hardware/library attribution. |
| Contrary case, rival explanation, diagnostic | Zero entropy improves argmax mean but loses29points at2718. Frozen all-ten-policy argmax/sampled diagnostic shrinks advantage25.6 to.8points; state-distribution mediation limitation explicit. |
| Original predictions revisited | All six questions plus three state-action probes; training-rank association separately measured, power-timer premise unverified. No retrospective rewriting. |
| Ten distribution-selected episodes | 3 typical successes/3 typical failures/2 rare successes/2 rare failures; conditional median and within-model rarity selection with duplicates excluded. |
| Thirty actual decisions | State, five available controls, full Q/pi vector, chosen action, score/life/position transition; genuine full-prefix seeded alternative, uncertainty and observed episode outcome. Eight full trace archives cover all ten episodes. |
| Bad action vs unlucky outcome | Missed immediate score, productive actions in failures, zero-value absent key, non-unique realised inputs; one coupled fork does not estimate expected survival. |
| Three contextual recommendations | Fixed configuration or neither, two experimental supports each, observed weakness, explicit reversal condition and specific further test before beyond-simulation use. |
| <=2000-word evidence brief | Parts2/3/4/6 only; conservative full-brief counter includes headings/math/numbers, excludes tables/captions. Compiled separately from predictions/review/AI record. |
| Real human review <=1000 words | **Deferred explicitly by user.** Actual participants, rankings and negotiated priorities remain required; no invented review is supplied. |
| Reproducible code/records | Actual Q/PPO in two Python files; documented75-model selected-config command, original search command separately; exact versions/configs, models, all episode CSVs, ten analyzed full traces and fingerprints in one ZIP. |
| Consequential AI assistance | Separate honest record of suggested/implemented work, accepted/rejected choices, checks, limitations and outstanding human ownership. |

`verify_report.py` checks tables and vectors against raw records.
`verify_records.py` validates the complete local archive, including13,581,678
primary and725,743 secondary decisions and19 native wrapper-free replays.
`bundle.py --verify` validates every included file and the ten required episodes;
it explicitly does not claim optional omitted full traces are packaged.
`build_documents.py` checks compilation, references and overflow; rendered
pages are inspected separately for layout.
