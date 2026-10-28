# Workshop 1: Dry Bean classification

`Monday-Workshop (1).pdf` asks participants to start with Dry Bean classification
(page 2, point 1). The other five UCI datasets on page 1 are alternatives for
later experiments.

## Notebook

`workshop1_classification.ipynb` follows all 11 numbered points in the PDF,
including the repeated workflow on Wine. It uses PyTorch neural networks and
scikit-learn random forests, with stratified train/validation/test splits and
controlled hyperparameter comparisons.
The CSV is kept unchanged; modeling removes 68 exact duplicate records before
splitting, leaving 13,543 unique rows and preventing duplicates across partitions.

Install `requirements.txt` from the repository root, select that Python
environment as the notebook kernel, and run all cells from this directory.
The neural networks use the CPU; Wine is bundled with scikit-learn.
A full run of both hyperparameter grids and both datasets takes several minutes.

## Local dataset

- File: `dry_bean.csv` (in this directory).
- Metadata: `dry_bean_metadata.json`.
- Source: [UCI Dry Bean, dataset 602](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
- Direct CSV: <https://archive.ics.uci.edu/static/public/602/data.csv>.
- Downloaded: 2026-10-19, without modifying the CSV.
- DOI: <https://doi.org/10.24432/C50S4B>; license: CC BY 4.0.
- Verified: 13,611 rows, 16 numerical features, and target column `Class` with
  seven bean varieties. Each row contains measurements of one bean.
- CSV SHA-256: `c91a086bb4357e69409d23287ec22be1be9512145cead1d8ca988c918be38fcb`.

| Class | Rows |
| --- | ---: |
| BARBUNYA | 1,322 |
| BOMBAY | 522 |
| CALI | 1,630 |
| DERMASON | 3,546 |
| HOROZ | 1,928 |
| SEKER | 2,027 |
| SIRA | 2,636 |

Load from a notebook running in `workshops/workshop1/`:

```python
import pandas as pd

df = pd.read_csv("dry_bean.csv")
X = df.drop(columns="Class")
y = df["Class"]
```

To download again, run from the repository root (requires internet access):

```python
from pathlib import Path
from urllib.request import urlretrieve

destination = Path("workshops/workshop1/dry_bean.csv")
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise FileExistsError(destination)
urlretrieve("https://archive.ics.uci.edu/static/public/602/data.csv", destination)
```

The dataset and metadata are excluded from Git by explicit rules in `.gitignore`.
