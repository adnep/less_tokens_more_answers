#!/usr/bin/env python3
"""
Plot accuracy vs avg_tokens Pareto front for backtracking sweep results.

Each strategy is a point. Pareto-optimal points (best accuracy for a given
token budget, or fewest tokens for a given accuracy) are highlighted and
labelled; all other points are shown as small faded dots.

Usage
-----
    # One sweep directory
    python scripts/plot_pareto.py --sweep-dir outputs/sweep3_gpqa_diamond

    # All sweep3 datasets at once
    python scripts/plot_pareto.py --sweep-dir outputs/sweep3_*

    # Single CSV directly
    python scripts/plot_pareto.py --csv outputs/sweep3_gpqa_diamond/sweep_results.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Dataset display names ─────────────────────────────────────────────────────

DATASET_NAMES = {
    "sweep3_aime24":                "AIME 2024",
    "sweep3_arithmetic_stress_test":"Arithmetic Stress Test",
    "sweep3_char_occur":            "Character Occurrence",
    "sweep3_distinct_char":         "Distinct Characters",
    "sweep3_gpqa_diamond":          "GPQA Diamond",
    "sweep3_hmmt2025":              "HMMT 2025",
    "sweep3_substring_occur":       "Substring Occurrence",
    "sweep3_word_len":              "Word Length",
}


# ── Colour palette (one per strategy family) ─────────────────────────────────

STRATEGY_COLORS = {
    "think_budget":         "#2196F3",  # blue
    "drop_detect":          "#E53935",  # red
    "threshold":            "#FB8C00",  # orange
    "linear_cooling":       "#43A047",  # green
    "confidence_cooling":   "#8E24AA",  # purple
    "confident_region_end": "#00ACC1",  # cyan
    "drop_steering":        "#FF7043",  # deep orange
    "wait_backtrack":       "#6D4C41",  # brown
    "any":                  "#546E7A",  # blue-grey
    "all":                  "#F9A825",  # amber
    "baseline":             "#212121",  # near-black
}
_FALLBACK = "#AAAAAA"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["accuracy"]       = float(row["accuracy"])
                row["avg_tokens"]     = int(row["avg_tokens"])
                row["avg_backtracks"] = float(row.get("avg_backtracks", 0))
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


# ── Pareto ────────────────────────────────────────────────────────────────────

def pareto_indices(rows: list[dict]) -> list[int]:
    """
    Return indices of non-dominated rows.
    We maximise accuracy and minimise avg_tokens simultaneously.
    Row i is dominated if any j has accuracy[j] >= accuracy[i]
    and tokens[j] <= tokens[i] with at least one strict inequality.
    """
    pts = [(r["accuracy"], r["avg_tokens"]) for r in rows]
    dominated = set()
    for i, (ai, ti) in enumerate(pts):
        for j, (aj, tj) in enumerate(pts):
            if i == j:
                continue
            if aj >= ai and tj <= ti and (aj > ai or tj < ti):
                dominated.add(i)
                break
    return [i for i in range(len(rows)) if i not in dominated]


# ── Plot ──────────────────────────────────────────────────────────────────────

def _label(row: dict) -> str:
    strat  = row.get("strategy", "?")
    metric = row.get("metric", "")
    return f"{strat}/{metric}" if metric else strat


def plot_pareto(rows: list[dict], title: str, out_path: Path, legend_loc: str = "lower center") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    pareto_set  = set(pareto_indices(rows))
    seen_strats = set()

    for i, r in enumerate(rows):
        strat       = r.get("strategy", "?")
        color       = STRATEGY_COLORS.get(strat, _FALLBACK)
        is_pareto   = i in pareto_set
        is_baseline = strat == "baseline"

        size       = 90 if is_baseline else (100 if is_pareto else 28)
        alpha      = 1.0  if (is_pareto or is_baseline) else 0.60
        marker     = "D"  if is_baseline else "o"
        edgecolor  = "black" if (is_pareto or is_baseline) else "none"
        lw         = 0.8  if (is_pareto or is_baseline) else 0
        zorder     = 5    if is_baseline else (4 if is_pareto else 2)

        legend_label = strat if strat not in seen_strats else "_nolegend_"
        seen_strats.add(strat)

        ax.scatter(
            r["avg_tokens"], r["accuracy"],
            c=color, marker=marker, s=size,
            alpha=alpha, edgecolors=edgecolor, linewidths=lw,
            zorder=zorder, label=legend_label,
        )

    # Labels on Pareto-optimal points only
    for i in pareto_set:
        r     = rows[i]
        strat = r.get("strategy", "?")
        ax.annotate(
            _label(r),
            xy=(r["avg_tokens"], r["accuracy"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=6.5,
            color=STRATEGY_COLORS.get(strat, _FALLBACK),
            zorder=6,
        )

    ax.set_xlabel("Average token count per problem", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(True, alpha=0.22, linewidth=0.5)

    # Legend: strategy colours + marker-type explanation
    handles, labels = ax.get_legend_handles_labels()
    extra = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markersize=8, markeredgecolor="black", markeredgewidth=0.8,
               label="Pareto-optimal (outlined)"),
    ]
    ax.legend(handles + extra, labels + ["Pareto-optimal (outlined)"],
              fontsize=8, loc=legend_loc, framealpha=0.88,
              ncol=2, borderpad=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pareto plot: accuracy vs avg_tokens for sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv",       nargs="+",
                       help="One or more sweep_results.csv files.")
    group.add_argument("--sweep-dir", nargs="+",
                       help="One or more sweep output directories "
                            "(expects sweep_results.csv inside each).")
    parser.add_argument("--out-dir",  default=None,
                        help="Save PNGs here instead of next to each CSV.")
    args = parser.parse_args()

    csv_paths: list[Path] = []

    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
    else:
        for d in args.sweep_dir:
            p = Path(d) / "sweep_results.csv"
            if p.exists():
                csv_paths.append(p)
            else:
                print(f"WARNING: no sweep_results.csv in {d}", file=sys.stderr)

    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        rows = load_csv(csv_path)
        if not rows:
            print(f"WARNING: no usable rows in {csv_path}", file=sys.stderr)
            continue

        dataset      = csv_path.parent.name
        display_name = DATASET_NAMES.get(dataset, dataset)
        dest         = (out_dir or csv_path.parent) / "pareto_plot.png"
        legend_loc   = "lower right" if "substring_occur" in dataset else "lower center"
        plot_pareto(rows, title=display_name, out_path=dest, legend_loc=legend_loc)


if __name__ == "__main__":
    main()
