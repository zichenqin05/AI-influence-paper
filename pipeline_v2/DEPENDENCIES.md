# Dependency record

The command entry point, manifest checks, Signals CSV reader, CPS fixed-width
layout reader, ZIP streaming, and panel aggregation use Python's standard
library. Generating the official Census occupation crosswalk from its XLSX
workbook uses `openpyxl`.

The analysis extra in `pyproject.toml` records optional packages for later
model stages. The bundled environment checked on 2026-09-29 is Python 3.12.14,
with NumPy 2.3.5, pandas 3.0.1, and openpyxl 3.1.5 available. `pyarrow`,
`pyreadstat`, scikit-learn, LightGBM, and pytest are not installed. The current
weighted panels therefore use compressed CSV rather than Parquet. No package
was installed for this work. The declared dependency ranges are not a lockfile.

The Phase 5 inline chart source is `pipeline_v2/phase5_visualization.py`. It uses
Pillow to draw PNGs from the frozen model CSVs and the NCES crosswalk workbook;
the `visualization` optional extra records its NumPy, pandas, Pillow, and
openpyxl requirements. By default, the script writes figures under the system
temporary directory, not into the project tree.
