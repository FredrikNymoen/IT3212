# IT3212 - Data-Driven Software Engineering

Course assignments, workshops and lecture examples. Assignment briefs are the
PDFs in the repository root; `IT3212___Assignment_3.pdf` is the supplied group
report, not an assignment brief.

## Work and reports

| Work | Implementation | Report |
|---|---|---|
| Assignment 1: preprocessing | [Notebook](notebooks/assignment1/student_graduation.ipynb) | No separate report stored locally |
| Assignment 2: image processing | `notebooks/assignment2/` | [Overleaf sections](reports/assignment2/README.md) |
| Assignment 3: classification | [Workflow and reproduction](notebooks/assignment3/README.md) | [Current LaTeX sources](report/README.md) |
| Assignment 4: reinforcement learning | [Current workflow and reproduction](notebooks/assignment4/README.md) | [LaTeX writing project](reports/assignment4/README.md) |
| Workshop 1: classification | [Instructions](workshops/workshop1/README.md) | Notebook |
| Lecture examples | `lecture/` | Not assignment deliverables |

Assignment 4 compares ordinary single-table Q-learning with PPO. The five-seed
A/B/C study and frozen D evaluation have separate, documented entry points.
Earlier baselines and exploratory variants are preserved in Git history; see
[the archive guide](notebooks/assignment4/ARCHIVE.md). The reviewed report is
validated against the measured records rather than overwritten by old renderers.

Assignment 3's current report recommends LDA. Reproduce it by running the
combined notebook, `check_lda_vs_adaboost.py`, then `finalize_lda_report.py`.
The original three-class notebook and initial AdaBoost analysis remain as
references. The local full-report project still contains writing prompts in
sections 1--6; it is not a rebuilt version of the supplied group PDF.

## Python and local data

Create or activate a Python environment and install the shared dependencies:

```powershell
python -m pip install -r requirements.txt
```

Select that environment as the Jupyter kernel. Run each notebook from its own
assignment or workshop directory. Assignments 1 and 3 require the original
`data/graduation_dataset.csv`. Assignment 2 uses the Canvas images under
`data/vehicle-type-detection/`. Workshop 1 documents its separate local input.
Datasets are not committed.

`results/` holds local numerical exports and experiment records. `figures/`
holds plot outputs, and report `Images/` directories hold the assets needed
for Overleaf. Keep these results and source files together when preserving an
experiment. Runtime versions and timing provenance are recorded with the
Assignment 3 exports.

## Repository housekeeping

Temporary review copies, package-download caches, LaTeX build intermediates
and obsolete Assignment 3 preview PDFs/ZIPs were removed after Assignment 3.
Current report sources and figure assets are retained. For Overleaf, upload
the current `report/` sources, following its README.

Assignment 3 report figures have one versioned vector PDF each in `report/Images/`.
The figure scripts can regenerate local PNG notebook previews under `figures/`;
these previews are gitignored. Duplicate figure PDFs and report PNGs were removed.

Use the ignored `tmp/` directory for scratch work and LaTeX build output.
Python environments, local datasets, numerical results and notebook checkpoints
are also ignored. The maintained Assignment 3 scripts use the selected Python
environment and have no dependency on packages installed under `tmp/`.
