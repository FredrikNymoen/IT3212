# Observation and reporting fixes, 2026-11-18

Consequential AI assistance: Codex reviewed and completed the partially applied
observation fixes, separated the changed protocol as v8, updated reproduction
commands, added dependency records and regression checks, and improved tables
and curve presentation. The conservative 10/20-point pellet rule was retained;
it does not establish exact pellet locations and is not a complete bitmap.
No improvement in assessed performance has been claimed for these fixes.

Validation:
- Seven regression tests passed (intervening walls, vertical boundaries, side
  exits, non-dot reward protection, dot counts, empty maps, pellet-distance key
  sensitivity and saved-Q compatibility).
- Existing selfcheck passed, including finite-difference gradient error
  4.17e-09, interaction accounting and both maze layouts.
- Nine random-action emulator episodes across A/B/D: 3,586 steps and 185 dot
  removals consistent with the conservative reward rule. No high rewards were
  encountered in that check; it cannot validate ghost/fruit protection alone.
- Eight diagnostic episodes driven by old PPO B weights on the changed task:
  5,416 steps, 501 dot removals, 12 rewards of 50, seven of 200 and two of 400.
  None of these 21 high-reward steps removed dots. No updates were made and
  these episodes are not part of the assessed results. Fruit-specific cases
  are covered synthetically, not claimed as observed in those episodes.
- End-to-end v8 smoke run completed all six training combinations and both D
  transfers. It used 256 interactions, a 64-decision cap, one seed and two
  evaluation episodes, so it is technical validation only.
- Tables generated successfully for the completed v7 study and the v8 smoke run.

Existing v7 raw logs, models and manifests were retained. Derived v7 tables and
plots were generated from those records. The directory name v7 contains protocol
ID mspacman_maze_pellets_walls_v6; directory names are experiment names, not
protocol identifiers. The full v8 study subsequently completed in results/assignment4_v8: 30 trained
agents, 10 frozen transfers and 8,000 primary evaluation episodes. Its verifier
passed; the tables, curves and v7 comparison are saved in that directory.
