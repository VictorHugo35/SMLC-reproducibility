# Data directory

The raw dataset is **not distributed with this repository** due to its size
(~250 MB compressed / ~700 MB raw). It must be downloaded manually before
running the pipeline.

## Required file

```
data/LD2011_2014.txt
```

## Download instructions

1. Go to the UCI Machine Learning Repository entry for
   **ElectricityLoadDiagrams20112014**:
   https://doi.org/10.24432/C58C86
2. Download the dataset archive and unzip it.
3. Place the raw file `LD2011_2014.txt` in this directory.

For the exact raw file used in the validated experiment, the SHA-256 checksum is:

```text
D51565F2CB5A6B768D06BA1BBD3C084C6E2F3AAB07F00C6F2DCB80E90175124B
```

## Expected format

- `;`-separated values, `,` as decimal separator.
- First column: timestamp (15-minute resolution, 2011-01-01 to 2015-01-01).
- Columns `MT_001` … `MT_370`: per-client readings in **kW per 15-minute
  interval** (370 clients, Portugal). Zeros before a client's start date
  indicate the client was not yet connected.

The pipeline (`scripts/run_pipeline.py`) verifies the file exists before running
and will print these same instructions if it is missing. The file is excluded
from version control via `.gitignore`.
