"""
Correlation of DTR prefix estimates with the full DTR value,
split by correct / incorrect / all answers.

Usage:
    python scripts/prefix_reliability_corr.py --csv /path/to/data.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PREFIX_COLS = ["p50", "p100", "p200", "p500", "p1000"]


def load(path):
    # Header has commas inside column names (e.g. "p,50"), so skip it and
    # assign names manually based on the 9 actual data columns.
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=["sample", "len"] + PREFIX_COLS + ["full", "correct"])
    df["correct"] = df["correct"].str.strip()
    return df


def corr_table(df, label):
    print(f"\n── {label} (n={len(df)}) ──")
    print(f"{'prefix':>6}   {'r':>7}   {'p':>8}")
    for col in PREFIX_COLS:
        r, p = pearsonr(df[col], df["full"])
        sig = "*" if p < 0.05 else " "
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        print(f"{col:>6}   {r:7.4f}{sig}  (p={p_str})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = load(args.csv)

    corr_table(df, "All answers")
    corr_table(df[df["correct"] == "correct"],   "Correct answers")
    corr_table(df[df["correct"] == "incorrect"], "Incorrect answers")


if __name__ == "__main__":
    main()
