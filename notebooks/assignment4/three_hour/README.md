# Five-seed main study: 22 November 2026

This directory preserves the registered engines, seven balanced candidates,
source-only selection, five-seed A/B/C training and original frozen evaluation.
Ordinary Q uses one table/update per transition; selected variants are Q
`gamma995` and PPO `stable`. The common cap is262144 decisions per trained
A/B/C model, seeds17/29/101/314/2718. No D result selects parameters.

## Reproduction

From the repository root, with dependencies installed:

```powershell
python notebooks/assignment4/three_hour/check.py
python notebooks/assignment4/three_hour/reproduce.py --run-id assignment4_reproduction_NEW --workers 8 --main-budget 262144
```

Use a new run prefix. This reproduces the original search/main/evaluation
sequence, including diagnosis of the known native D guard when it occurs;
it does not create a new independent test. The expensive whole reproducer
was not rerun merely for repository cleanup. Original executed stages and
hashes remain in the local `results/assignment4/three_hour_20261122/` records.

The original D run completed39/40 cases and aborted one Q case at a native
map change. The [current D evaluation](../transfer_evaluation/README.md) is a
separate measurement repair using the same B checkpoints and explicit censoring.
The report uses its D results rather than silently dropping the interrupted agent.

Do not use `--update-report` to replace the manually reviewed three-table report.
The launcher rejects that overwrite. Current report checks are in
[report_review](../report_review/README.md). The former numerical renderers
remain because original reproduction/recovery uses their verification code.

The development search uses seven packages per method on A/B at seed42, then
the best two repeat at seed7:4718592 interactions per method. Development uses
20 episodes/checkpoint for search,10 for main curves; final evaluation uses200.
Every full run consumes its cap, even if development selects earlier weights.
Original predictions, old sources, failures and unequal earlier search remain
part of the provenance; this is not a five-seed Part3 mechanism study.

One-off migration/recovery helpers and older experiments are preserved in Git
checkpoint `714c0cb`. Shared dependency paths and registered source bytes must
stay unchanged when reusing a checkpoint. See [ARCHIVE.md](../ARCHIVE.md).
