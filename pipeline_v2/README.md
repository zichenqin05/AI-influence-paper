# Pipeline v2

This is an isolated implementation of the audited study design in
`../Codex-file/14_模型实施规范与论文证据矩阵.md`. It does not import or run the
legacy database, embedding, classification, or risk-score modules.

Run commands from the `项目本体` directory, using the bundled Python environment:

```powershell
& 'C:\path\to\python.exe' -m pipeline_v2.cli audit
& 'C:\path\to\python.exe' -m pipeline_v2.cli fetch --pilot
& 'C:\path\to\python.exe' -m pipeline_v2.cli fetch --history --workers 4
& 'C:\path\to\python.exe' -m pipeline_v2.cli build-panel
& 'C:\path\to\python.exe' -m pipeline_v2.cli validate-panel
& 'C:\path\to\python.exe' -m pipeline_v2.cli build-model-features
& 'C:\path\to\python.exe' -m pipeline_v2.cli backtest --mode retrospective
& 'C:\path\to\python.exe' -m pipeline_v2.cli verify-exposure
& 'C:\path\to\python.exe' -m pipeline_v2.cli ai-short-window
& 'C:\path\to\python.exe' -m pipeline_v2.phase5_visualization --output-dir output
& 'C:\path\to\python.exe' -m unittest discover -s pipeline_v2/tests -v
```

`fetch --pilot` downloads one official CPS month and its layout. `fetch
--acs-pilot` downloads the 2024 ACS national person archive and streams the
seven selected source columns to a compressed pilot cache. `fetch --history`
downloads official monthly CPS archives from 2016 through August 2026 and ACS
1-year PUMS person archives for 2018, 2019, 2021, 2022, and 2023; the already
present 2024 ACS and 2026-08 CPS pilot archives are reused. The download set is
about 4.0 GB before those two existing pilot archives. October 2025 is recorded
as an unavailable CPS month and remains missing.

`build-panel` first checks the annual CPS layouts and ACS dictionaries and
builds the Census occupation crosswalk. It writes:

- `data/processed/reference/occupation_crosswalk.csv` — all 540 codes in the
  2010 Census list and all 570 codes in the 2018 list. A 2010 code is assigned
  to a 2018 SOC major group only when its official destination-code set implies
  one unique group. The official E1 conversion-rate table contributes candidate
  destination codes only; its probabilities are not used to split records.
- `data/processed/reference/reference_schema_inventory.json` — year-by-year
  field positions, occupation-code vintages, weight definitions, and ACS PUMS
  release dates.
- `data/processed/panels/cps_occupation_month.csv.gz` — weighted monthly
  civilian employment, unemployment, and labor-force totals by major group.
- `data/processed/panels/cps_national_month_diagnostic.csv.gz` — separate
  national totals and occupation-identification coverage for reconciliation.
- `data/processed/panels/acs_occupation_year.csv.gz` — weighted employed-civilian
  age and education structure by major group and PUMS year.
- `data/processed/panels/acs_mapping_diagnostic.csv.gz` — annual occupation
  mapping coverage and denominator checks.
- `data/processed/panels/panel_quality_report.json` — key uniqueness, row-grid,
  arithmetic, rate/share ranges, and release-date checks, written by
  `validate-panel`.

Panels are gzip-compressed CSV because `pyarrow` is not provisioned in the
bundled Python runtime; their column headers preserve the panel contract. ACS
rows carry the official PUMS public-release date. Exact CPS public-use archive
publication dates are not verified, so CPS panel rows leave `release_date`
blank and are blocked from strict as-of feature use until a verified date is
available. The panel builder does not fit a model or create predictions.

`data/manifest/source_manifest.json` is the source-of-record index. Its paths
are relative to the workspace root; each retrieved source is hash-pinned, and
unknown release dates remain unknown. `audit` distinguishes verified files,
missing registered files, files without recorded hashes, mismatches, and
planned sources that are not downloaded. Raw data and derived artifacts belong
under this directory's `data/`; the existing project data and legacy code are
kept in place.

The modeling stage now writes fold-specific factor loadings, rolling
out-of-sample predictions and metrics, an occupational exposure coverage audit,
and a seven-month exploratory AI/P comparison under
`data/processed/modeling/`. The seven paper figures are in `output/` and can
be regenerated with the visualization command above after obtaining the
documented public source files. These are current-vintage retrospective
results, not a strict real-time backtest: exact CPS public-use release dates
are still unverified and the historical Signals series is backfilled.
Consequently, the AI/P comparison is predictive association, not a causal
estimate of AI-induced unemployment.

The Git repository keeps code, compact derived results, quality records, and
figures. Re-downloadable CPS/ACS source archives and the large selected-person
intermediate cache are excluded by `.gitignore`; use the fetch commands and
source manifest to rebuild them locally.

The 2010-to-2018 bridge currently maps 504 historical codes to one 2018 SOC
major group. Eight source codes span multiple destination major groups and stay
unmapped; 26 have no verified 2018 destination in the official tables; two are
special no-occupation/military-rank codes. Four unique bridges move between
major groups and are assigned to their official 2018 destination group. The
coverage diagnostics retain unmapped labor-force records separately from the
22 civilian occupation groups.
