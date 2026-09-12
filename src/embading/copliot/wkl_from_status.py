"""
Analyze `merged_ai_impact_jobs.csv`: when any of NWLA (layoff), NWLK (looking for work),
NWRE (recall) equals 1, compute distribution of `WKL 上次工作时间` (categories 0,1,2,3).

Usage:
    python src/embading/copliot/wkl_from_status.py
    python src/embading/copliot/wkl_from_status.py --input src/embading/merged_ai_impact_jobs.csv --output wkl_status_counts.csv

Outputs counts and percentages for WKL categories among filtered rows, plus overall proportions.
"""
import argparse
from pathlib import Path
import pandas as pd


def compute_wkl_for_status(df,
                           wkl_col='WKL 上次工作时间',
                           status_cols=('NWLA 裁员','NWLK 找工作中','NWRE 召回')):
    for c in status_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in dataframe")
    if wkl_col not in df.columns:
        raise KeyError(f"Column '{wkl_col}' not found in dataframe")

    # Filter where any status column == 1
    mask = False
    for c in status_cols:
        mask = mask | (df[c] == 1)

    filtered = df[mask]
    total_filtered = len(filtered)

    # counts for WKL categories (0-3)
    counts = filtered[wkl_col].value_counts(dropna=False).sort_index()
    idx = [0,1,2,3]
    counts = counts.reindex(idx, fill_value=0)

    percent = (counts / total_filtered * 100).round(2) if total_filtered > 0 else pd.Series([0.0]*len(counts), index=counts.index)

    result = pd.DataFrame({'count': counts.astype(int), 'percent': percent.astype(float)})

    # also return share of these filtered rows relative to full data
    share_of_all = total_filtered / len(df) * 100 if len(df) > 0 else 0.0

    return total_filtered, share_of_all, result


def main():
    p = argparse.ArgumentParser(description='Compute WKL distribution when NWLA/NWLK/NWRE==1')
    p.add_argument('--input', '-i', default='src/embading/merged_ai_impact_jobs.csv', help='input CSV path')
    p.add_argument('--wkl-col', default='WKL 上次工作时间', help='WKL column name')
    p.add_argument('--status-cols', nargs='+', default=['NWLA 裁员','NWLK 找工作中','NWRE 召回'], help='status columns to OR together')
    p.add_argument('--output', '-o', default=None, help='optional CSV to save counts')
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    df = pd.read_csv(path)

    total_filtered, share_of_all, result = compute_wkl_for_status(df, wkl_col=args.wkl_col, status_cols=args.status_cols)

    print(f"Total rows where any({', '.join(args.status_cols)}) == 1: {total_filtered}")
    print(f"These rows are {share_of_all:.2f}% of all rows ({len(df)} total rows)")
    print('\nWKL counts (categories 0,1,2,3):')
    print(result.to_string())

    if args.output:
        outp = Path(args.output)
        result.to_csv(outp)
        print(f"Saved counts to {outp}")


if __name__ == '__main__':
    main()
