#!/usr/bin/env python3
"""
Cross-dataset strategy family analysis.

For each sweep3 dataset, loads sweep_results.csv and computes:
  - rel_acc_gain  = (accuracy - baseline_acc) / baseline_acc * 100
  - rel_tok_delta = (avg_tokens - baseline_tokens) / baseline_tokens * 100

Then produces:
  1. Heatmap     — best rel. acc. gain per strategy family × dataset
  2. Scatter     — rel. token delta vs rel. acc. gain, best config per family per dataset
  3. Summary CSV — mean / std rel. acc. gain per family across datasets
  4. Configs CSV — every config that appears in 2+ datasets, ranked by mean rel. acc. gain
  5. Console     — readable table of cross-dataset configs

Usage
-----
    python scripts/analyze_cross_dataset.py
    python scripts/analyze_cross_dataset.py --out-dir outputs/analysis
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

SWEEP_DIRS = [
    "outputs/sweep3_aime24",
    "outputs/sweep3_arithmetic_stress_test",
    "outputs/sweep3_char_occur",
    "outputs/sweep3_distinct_char",
    "outputs/sweep3_gpqa_diamond",
    "outputs/sweep3_hmmt2025",
    "outputs/sweep3_substring_occur",
    "outputs/sweep3_word_len",
]

DATASET_NAMES = {
    "sweep3_aime24":                 "AIME 2024",
    "sweep3_arithmetic_stress_test": "Arithmetic Stress Test",
    "sweep3_char_occur":             "Character Occurrence",
    "sweep3_distinct_char":          "Distinct Characters",
    "sweep3_gpqa_diamond":           "GPQA Diamond",
    "sweep3_hmmt2025":               "HMMT 2025",
    "sweep3_substring_occur":        "Substring Occurrence",
    "sweep3_word_len":               "Word Length",
}

FAMILY_MAP = {
    "think_budget":         "Think Budget",
    "drop_detect":          "Backtracking",
    "threshold":            "Backtracking",
    "wait_backtrack":       "Backtracking",
    "any":                  "Backtracking",
    "all":                  "Backtracking",
    "linear_cooling":       "Cooling",
    "confidence_cooling":   "Cooling",
    "confident_region_end": "Steered Cooling",
    "drop_steering":        "Steering",
}

FAMILY_COLORS = {
    "Think Budget":  "#2196F3",
    "Backtracking":  "#E53935",
    "Cooling":       "#43A047",
    "Steered Cooling": "#8E24AA",
    "Steering":      "#FF7043",
}

FAMILY_ORDER = ["Think Budget", "Backtracking", "Cooling", "Steered Cooling", "Steering"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(sweep_dir: Path) -> list[dict]:
    csv_path = sweep_dir / "sweep_results.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["accuracy"]   = float(row["accuracy"])
                row["avg_tokens"] = int(row["avg_tokens"])
                row["delta_acc"]  = float(row["delta_acc"]) if row["delta_acc"] else 0.0
            except (ValueError, KeyError):
                continue
            row["dataset"]  = sweep_dir.name
            row["strategy_kwargs"] = row.get("strategy_kwargs", "{}")
            rows.append(row)
    return rows


def get_baseline(rows: list[dict]) -> tuple[float, int]:
    for r in rows:
        if r["strategy"] == "baseline":
            return r["accuracy"], r["avg_tokens"]
    return None, None


def add_relative_metrics(rows: list[dict], baseline_acc: float, baseline_tokens: int):
    for r in rows:
        if baseline_acc:
            r["rel_acc_gain"]  = (r["accuracy"] - baseline_acc) / baseline_acc * 100
        else:
            r["rel_acc_gain"]  = 0.0
        if baseline_tokens:
            r["rel_tok_delta"] = (r["avg_tokens"] - baseline_tokens) / baseline_tokens * 100
        else:
            r["rel_tok_delta"] = 0.0
        r["family"] = FAMILY_MAP.get(r["strategy"], None)


# ── Analysis helpers ──────────────────────────────────────────────────────────

def best_per_family_per_dataset(all_rows: list[dict], by: str = "accuracy") -> dict:
    """
    Returns {(family, dataset): best_row}.
    by="accuracy" : maximise rel_acc_gain
    by="tokens"   : minimise rel_tok_delta (fewest tokens)
    """
    best = {}
    for r in all_rows:
        if r["family"] is None or r["strategy"] == "baseline":
            continue
        key = (r["family"], r["dataset"])
        if key not in best:
            best[key] = r
        elif by == "accuracy" and r["rel_acc_gain"] > best[key]["rel_acc_gain"]:
            best[key] = r
        elif by == "tokens" and r["rel_tok_delta"] < best[key]["rel_tok_delta"]:
            best[key] = r
    return best


def cross_dataset_configs(all_rows: list[dict]) -> list[dict]:
    """
    Groups rows by (strategy, metric, kwargs) config key.
    Returns configs that appear in 2+ datasets, sorted by:
      1. Number of datasets with rel_acc_gain > 0  (descending)
      2. Mean rel_acc_gain across all appearances   (descending)
    """
    groups = defaultdict(list)
    for r in all_rows:
        if r["strategy"] == "baseline":
            continue
        key = (r["strategy"], r.get("metric", ""), r["strategy_kwargs"])
        groups[key].append(r)

    results = []
    for (strategy, metric, kwargs), entries in groups.items():
        if len(entries) < 2:
            continue
        datasets       = [e["dataset"] for e in entries]
        gains          = [e["rel_acc_gain"] for e in entries]
        tok_deltas     = [e["rel_tok_delta"] for e in entries]
        n_positive     = sum(1 for g in gains if g > 0)
        results.append({
            "strategy":      strategy,
            "metric":        metric,
            "kwargs":        kwargs,
            "n_datasets":    len(entries),
            "n_positive":    n_positive,
            "mean_gain":     float(np.mean(gains)),
            "std_gain":      float(np.std(gains)),
            "mean_tok_delta":float(np.mean(tok_deltas)),
            "datasets":      datasets,
            "gains":         gains,
        })

    results.sort(key=lambda x: (-x["n_positive"], -x["mean_gain"]))
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, data, dataset_labels, value_fmt, cmap, cbar_label, title,
                  mean_col=False):
    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(dataset_labels)))
    ax.set_xticklabels(dataset_labels, rotation=25, ha="right", fontsize=11)
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels(FAMILY_ORDER, fontsize=11)

    for fi in range(data.shape[0]):
        for di in range(data.shape[1]):
            v = data[fi, di]
            is_mean = mean_col and di == data.shape[1] - 1
            if not np.isnan(v):
                ax.text(di, fi, value_fmt(v), ha="center", va="center",
                        fontsize=10, fontweight="bold" if is_mean else "normal",
                        color="black" if abs(v) < vmax * 0.7 else "white")
            else:
                ax.text(di, fi, "—", ha="center", va="center", fontsize=11, color="#aaa")

    if mean_col:
        sep = data.shape[1] - 1.5
        ax.axvline(sep, color="black", linewidth=1.5)
        ax.get_xticklabels()[-1].set_fontweight("bold")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")


def plot_heatmaps(best: dict, dataset_order: list[str], out_path: Path):
    dataset_labels = [DATASET_NAMES.get(d, d) for d in dataset_order] + ["Mean"]

    acc_data = np.full((len(FAMILY_ORDER), len(dataset_order)), np.nan)
    tok_data = np.full((len(FAMILY_ORDER), len(dataset_order)), np.nan)

    for fi, family in enumerate(FAMILY_ORDER):
        for di, dataset in enumerate(dataset_order):
            row = best.get((family, dataset))
            if row:
                acc_data[fi, di] = row["rel_acc_gain"]
                tok_data[fi, di] = row["rel_tok_delta"]

    # Append a Mean column: row-wise mean across datasets (ignoring missing cells).
    def _with_mean(data):
        with np.errstate(invalid="ignore"):
            means = np.nanmean(data, axis=1, keepdims=True)
        return np.hstack([data, means])

    acc_data = _with_mean(acc_data)
    tok_data = _with_mean(tok_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    _draw_heatmap(ax1, acc_data, dataset_labels,
                  value_fmt=lambda v: f"{v:+.1f}%",
                  cmap="RdYlGn",
                  cbar_label="Relative accuracy gain (%)",
                  title="Best relative accuracy gain per strategy family and dataset",
                  mean_col=True)

    # For tokens: green = fewer tokens (negative = good), so invert the colormap
    _draw_heatmap(ax2, tok_data, dataset_labels,
                  value_fmt=lambda v: f"{v:+.1f}%",
                  cmap="RdYlGn_r",
                  cbar_label="Relative token count change (%)",
                  title="Token count change per strategy family and dataset",
                  mean_col=True)

    plt.tight_layout(h_pad=3.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


DATASET_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_efficiency_scatter(best: dict, dataset_order: list[str], out_path: Path, title: str,
                            dataset_legend_loc: str = "lower left"):
    from matplotlib.lines import Line2D

    dataset_marker = {d: DATASET_MARKERS[i % len(DATASET_MARKERS)]
                      for i, d in enumerate(dataset_order)}

    fig, ax = plt.subplots(figsize=(11, 7))

    for family in FAMILY_ORDER:
        color = FAMILY_COLORS[family]
        for dataset in dataset_order:
            row = best.get((family, dataset))
            if not row:
                continue
            ax.scatter(row["rel_tok_delta"], row["rel_acc_gain"],
                       c=color, marker=dataset_marker[dataset], s=110, zorder=3,
                       edgecolors="black", linewidths=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Token count change relative to baseline (%)", fontsize=11)
    ax.set_ylabel("Relative accuracy gain (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Legends
    family_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[f],
               markersize=9, markeredgecolor="black", markeredgewidth=0.5, label=f)
        for f in FAMILY_ORDER
    ]
    dataset_handles = [
        Line2D([0], [0], marker=dataset_marker[d], color="w", markerfacecolor="#888",
               markersize=9, markeredgecolor="black", markeredgewidth=0.5,
               label=DATASET_NAMES.get(d, d))
        for d in dataset_order
    ]
    leg1 = ax.legend(handles=family_handles, title="Strategy family",
                     fontsize=8, loc="upper left", framealpha=0.88)
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles, title="Dataset",
              fontsize=8, loc=dataset_legend_loc, framealpha=0.88)

    # Quadrant labels at the centre of each quadrant
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    mid_x_left  = (xl + 0) / 2
    mid_x_right = (0 + xr) / 2
    mid_y_top   = (0 + yt) / 2
    mid_y_bot   = (yb + 0) / 2
    ax.text(mid_x_left,  mid_y_top, "better\n+ cheaper",  ha="center", va="center",
            fontsize=8, color="green",  alpha=0.45, style="italic")
    ax.text(mid_x_right, mid_y_top, "better\n+ costlier", ha="center", va="center",
            fontsize=8, color="orange", alpha=0.45, style="italic")
    ax.text(mid_x_left,  mid_y_bot, "worse\n+ cheaper",   ha="center", va="center",
            fontsize=8, color="gray",   alpha=0.45, style="italic")
    ax.text(mid_x_right, mid_y_bot, "worse\n+ costlier",  ha="center", va="center",
            fontsize=8, color="red",    alpha=0.45, style="italic")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Baseline comparison plot ──────────────────────────────────────────────────

DATASET_TYPES = {
    "sweep3_aime24":                 "Math Olympiad",
    "sweep3_arithmetic_stress_test": "Arithmetic",
    "sweep3_char_occur":             "String Counting",
    "sweep3_distinct_char":          "String Counting",
    "sweep3_gpqa_diamond":           "Science MCQ",
    "sweep3_hmmt2025":               "Math Olympiad",
    "sweep3_substring_occur":        "String Counting",
    "sweep3_word_len":               "String Counting",
}

TYPE_COLORS = {
    "Math Olympiad":  "#E53935",
    "Arithmetic":     "#FB8C00",
    "Science MCQ":    "#8E24AA",
    "String Counting":"#43A047",
}


def plot_baseline_comparison(all_rows: list[dict], dataset_order: list[str], out_path: Path):
    baselines = {}
    for r in all_rows:
        if r["strategy"] == "baseline":
            baselines[r["dataset"]] = r

    labels  = [DATASET_NAMES.get(d, d) for d in dataset_order]
    accs    = [baselines[d]["accuracy"] * 100   for d in dataset_order]
    tokens  = [baselines[d]["avg_tokens"]        for d in dataset_order]
    colors  = [TYPE_COLORS[DATASET_TYPES.get(d, "Arithmetic")] for d in dataset_order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    x = range(len(dataset_order))
    bars1 = ax1.bar(x, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, v in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 112)
    ax1.axhline(100, color="black", linewidth=0.6, linestyle="--", alpha=0.3)
    ax1.set_title("Baseline accuracy and average token count per dataset", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_yticks(range(0, 101, 20))

    bars2 = ax2.bar(x, tokens, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, v in zip(bars2, tokens):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Avg tokens per problem", fontsize=10)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    # Category legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=t)
                      for t, c in TYPE_COLORS.items()]
    ax1.legend(handles=legend_handles, title="Task type", fontsize=8,
               loc="lower right", framealpha=0.88)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary_csv(best: dict, dataset_order: list[str], out_path: Path):
    fieldnames = ["family", "mean_rel_acc_gain", "std_rel_acc_gain",
                  "mean_rel_tok_delta", "n_datasets"] + \
                 [DATASET_NAMES.get(d, d) for d in dataset_order]

    rows_out = []
    for family in FAMILY_ORDER:
        gains = []
        tok_deltas = []
        per_dataset = {}
        for dataset in dataset_order:
            row = best.get((family, dataset))
            if row:
                gains.append(row["rel_acc_gain"])
                tok_deltas.append(row["rel_tok_delta"])
                per_dataset[DATASET_NAMES.get(dataset, dataset)] = f"{row['rel_acc_gain']:+.2f}%"
            else:
                per_dataset[DATASET_NAMES.get(dataset, dataset)] = "—"
        entry = {
            "family":            family,
            "mean_rel_acc_gain": f"{np.mean(gains):+.2f}%" if gains else "—",
            "std_rel_acc_gain":  f"{np.std(gains):.2f}%"  if gains else "—",
            "mean_rel_tok_delta":f"{np.mean(tok_deltas):+.2f}%" if tok_deltas else "—",
            "n_datasets":        len(gains),
        }
        entry.update(per_dataset)
        rows_out.append(entry)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Saved: {out_path}")


def save_configs_csv(configs: list[dict], dataset_order: list[str], out_path: Path):
    fieldnames = ["strategy", "metric", "n_datasets", "n_positive",
                  "mean_gain", "std_gain", "mean_tok_delta", "datasets_gains", "kwargs"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in configs:
            dataset_gains = "  |  ".join(
                f"{DATASET_NAMES.get(d, d)}: {g:+.1f}%"
                for d, g in zip(c["datasets"], c["gains"])
            )
            writer.writerow({
                "strategy":       c["strategy"],
                "metric":         c["metric"],
                "n_datasets":     c["n_datasets"],
                "n_positive":     c["n_positive"],
                "mean_gain":      f"{c['mean_gain']:+.2f}%",
                "std_gain":       f"{c['std_gain']:.2f}%",
                "mean_tok_delta": f"{c['mean_tok_delta']:+.2f}%",
                "datasets_gains": dataset_gains,
                "kwargs":         c["kwargs"],
            })
    print(f"Saved: {out_path}")


def print_configs_table(configs: list[dict], top_n: int = 30):
    print(f"\n{'═'*110}")
    print(f"  Configs appearing in 2+ datasets — ranked by (n_positive, mean_gain)")
    print(f"{'═'*110}")
    print(f"  {'strategy':>20}  {'m':>12}  {'n_ds':>4}  {'n_pos':>5}  "
          f"{'mean_gain':>10}  {'tok_delta':>10}  per-dataset gains")
    print(f"  {'─'*107}")
    for c in configs[:top_n]:
        ds_str = "  ".join(
            f"{DATASET_NAMES.get(d, d)[:12]}: {g:+.1f}%"
            for d, g in zip(c["datasets"], c["gains"])
        )
        print(f"  {c['strategy']:>20}  {c['metric']:>12}  {c['n_datasets']:>4}  "
              f"{c['n_positive']:>5}  {c['mean_gain']:>+9.2f}%  "
              f"{c['mean_tok_delta']:>+9.2f}%  {ds_str}")
    print(f"{'═'*110}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset strategy family analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dirs", nargs="+", default=SWEEP_DIRS,
                        help="Sweep output directories to include.")
    parser.add_argument("--out-dir", default="outputs/analysis",
                        help="Directory for output files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    all_rows = []
    dataset_order = []
    for d in args.sweep_dirs:
        sweep_dir = Path(d)
        rows = load_dataset(sweep_dir)
        if not rows:
            print(f"WARNING: no data in {d}", file=sys.stderr)
            continue
        baseline_acc, baseline_tokens = get_baseline(rows)
        if baseline_acc is None:
            print(f"WARNING: no baseline in {d}", file=sys.stderr)
            continue
        add_relative_metrics(rows, baseline_acc, baseline_tokens)
        all_rows.extend(rows)
        dataset_order.append(sweep_dir.name)

    print(f"\nLoaded {len(all_rows)} rows across {len(dataset_order)} datasets.")

    best_acc = best_per_family_per_dataset(all_rows, by="accuracy")
    best_tok = best_per_family_per_dataset(all_rows, by="tokens")
    configs  = cross_dataset_configs(all_rows)

    # Outputs
    plot_baseline_comparison(all_rows, dataset_order, out_dir / "baseline_comparison.png")
    plot_heatmaps(best_acc, dataset_order, out_dir / "cross_dataset_heatmap.png")
    plot_efficiency_scatter(
        best_acc, dataset_order,
        out_dir / "cross_dataset_efficiency_by_acc.png",
        title="Efficiency profile per strategy family\n(best config selected by accuracy)",
    )
    plot_efficiency_scatter(
        best_tok, dataset_order,
        out_dir / "cross_dataset_efficiency_by_tokens.png",
        title="Efficiency profile per strategy family\n(best config selected by token reduction)",
        dataset_legend_loc="upper right",
    )
    save_summary_csv(best_acc, dataset_order, out_dir / "cross_dataset_summary.csv")
    save_configs_csv(configs, dataset_order, out_dir / "cross_dataset_configs.csv")
    print_configs_table(configs)

    print(f"\nDone. {len(configs)} configs appeared in 2+ datasets.")


if __name__ == "__main__":
    main()
