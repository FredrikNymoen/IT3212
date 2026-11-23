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
| Assignment 4: reinforcement learning | [Current five-seed MsPacMan study](notebooks/assignment4/three_hour/README.md) | [LaTeX writing project](reports/assignment4/README.md) |
| Workshop 1: classification | [Instructions](workshops/workshop1/README.md) | Notebook |
| Lecture examples | `lecture/` | Not assignment deliverables |

Assignment 4's current A--D runner compares ordinary single-table Q-learning
with PPO and refreshes Part 2 only from verified full-run exports. The
[ordinary-Q pilot](notebooks/assignment4/pilot/SINGLE_Q.md) preceded that study.
The [earlier investigation](notebooks/assignment4/pilot/FURTHER_EXPLORATION.md)
also includes archived Double Q/replay variants; those rates must not be
substituted for the ordinary Q-learning described in the original predictions.

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
