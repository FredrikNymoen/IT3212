# Shared geometry and ordinary Q-learning

The current implementation imports `navigation_features.py`,
`maze_geometry.json` and `tabular_q.py`. Preserve their locations and registered
source hashes. `validate_tabular_q.py` checks the ordinary Bellman target,
terminal handling, native action mapping and read-only evaluation.

Older pilot notebooks/runners are preserved in Git commit `714c0cb`.
Their Double Q/replay results are not the ordinary-Q results in the report.
See the [Assignment4 guide](../README.md) for current entry points.
