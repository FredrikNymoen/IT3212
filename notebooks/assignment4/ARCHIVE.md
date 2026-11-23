# Historical Assignment4 sources

Git commit `714c0cb` preserves the complete pre-cleanup source tree, including
old baselines, rejected candidates, one-off recovery/migration tools and earlier
figures. The cleanup removes these from the current working tree, not history.

Inspect an archived source without changing the working tree:

```powershell
git show 714c0cb:notebooks/assignment4/pilot/FURTHER_EXPLORATION.md
```

The original numerical records/source snapshots remain under the ignored
`results/assignment4/` tree. A checked copy of every removed local file and
its SHA256 is under `results/assignment4/repository_cleanup_20261123/`.
Restore historical sources from the checkpoint or that local archive when
replaying an old study; do not rewrite a frozen source to imitate another run.

Current files preserve the dependency and source-hash closure of the reported
five-seed search, training and frozen transfer evaluation. Some small modules
therefore retain historical directory names. Only the current report's image
is retained in `reports/assignment4/Images/`; duplicate/historical plots remain
in the checkpoint and local archive.
