"""
Analyze results.json: score distributions vs correctness, Pearson correlations.

Supports any score columns saved by run_think_at_n.py:
  score_dtr, score_selfcert, score_logprob

Usage:
    python scripts/analyze_results.py --results outputs/think_at_n/results.json
    python scripts/analyze_results.py --results outputs/think_at_n/results.json --output-dir outputs/analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Loading + classification
# ──────────────────────────────────────────────────────────────────────────────

def load_results(path):
    with open(path) as f:
        return json.load(f)


def _normalize(s):
    """Strip all whitespace — LaTeX spacing variants compare equal."""
    import re
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'\s+', '', s)
    # Strip leading zeros from pure integers
    if s.isdigit():
        s = str(int(s))
    return s


def classify_samples(results):
    """
    Return a flat list of per-sample records.
    Each record has: problem_id, sample_idx, extracted, reference, status,
    and one key per score column found (score_dtr, score_selfcert, score_logprob, …).
    """
    records = []

    # Detect which score columns are present from the first sample_extraction
    score_keys = []
    for prob in results["per_problem"]:
        exts = prob.get("sample_extractions", [])
        if exts:
            score_keys = [k for k in exts[0] if k.startswith("score_")]
            break

    for problem in results["per_problem"]:
        ref = _normalize(problem["reference"])
        for ext in problem.get("sample_extractions", []):
            extracted = _normalize(ext.get("extracted_answer"))
            if extracted is None or extracted == "":
                status = "unanswered"
            elif extracted == ref:
                status = "correct"
            else:
                status = "incorrect"

            record = {
                "problem_id": problem["id"],
                "sample_idx": ext["sample_idx"],
                "extracted":  extracted,
                "reference":  ref,
                "status":     status,
            }
            for k in score_keys:
                record[k] = ext.get(k)

            records.append(record)

    return records, score_keys


COLORS = {"correct": "#2ecc71", "incorrect": "#e74c3c", "unanswered": "#95a5a6"}


# ──────────────────────────────────────────────────────────────────────────────
# Pearson correlation
# ──────────────────────────────────────────────────────────────────────────────

def compute_correlations(records, score_keys):
    """
    For each score column, compute Pearson r between the score and a binary
    correctness label (1=correct, 0=incorrect/unanswered) across all samples.
    Also compute r excluding unanswered samples.
    """
    answered = [r for r in records if r["status"] != "unanswered"]

    print(f"\n{'='*55}")
    print(f"Pearson r: score vs correctness")
    print(f"{'='*55}")
    print(f"  {'metric':<16}  {'r (all)':>10}  {'p':>8}  {'r (answered)':>14}  {'p':>8}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*8}  {'-'*14}  {'-'*8}")

    results_table = {}
    for key in score_keys:
        label = key.replace("score_", "")

        # All samples (unanswered = 0)
        valid_all = [(r[key], 1 if r["status"] == "correct" else 0)
                     for r in records if r[key] is not None]
        if len(valid_all) > 2:
            scores_arr, labels_arr = zip(*valid_all)
            if np.std(scores_arr) < 1e-10:
                r_all, p_all = float("nan"), float("nan")
            else:
                r_all, p_all = stats.pearsonr(scores_arr, labels_arr)
        else:
            r_all, p_all = float("nan"), float("nan")

        # Answered only
        valid_ans = [(r[key], 1 if r["status"] == "correct" else 0)
                     for r in answered if r[key] is not None]
        if len(valid_ans) > 2:
            scores_arr, labels_arr = zip(*valid_ans)
            if np.std(scores_arr) < 1e-10:
                r_ans, p_ans = float("nan"), float("nan")
            else:
                r_ans, p_ans = stats.pearsonr(scores_arr, labels_arr)
        else:
            r_ans, p_ans = float("nan"), float("nan")

        print(f"  {label:<16}  {r_all:>10.4f}  {p_all:>8.4f}  {r_ans:>14.4f}  {p_ans:>8.4f}")
        results_table[label] = {"r_all": r_all, "p_all": p_all,
                                 "r_answered": r_ans, "p_answered": p_ans}

    return results_table


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_score_distribution(records, score_key, output_dir):
    """Histogram of scores coloured by correctness."""
    label = score_key.replace("score_", "")
    fig, ax = plt.subplots(figsize=(10, 5))
    for status in ["correct", "incorrect", "unanswered"]:
        vals = [r[score_key] for r in records
                if r["status"] == status and r[score_key] is not None]
        if vals:
            ax.hist(vals, bins=20, alpha=0.6,
                    label=f"{status} (n={len(vals)})",
                    color=COLORS[status], edgecolor="white")
    ax.set_xlabel(f"{label} score")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} distribution by correctness")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, f"dist_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved dist_{label}.png")


def plot_score_violin(records, score_key, output_dir):
    """Violin plot of score by correctness category."""
    label = score_key.replace("score_", "")
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["correct", "incorrect", "unanswered"]
    data, xlabels, colors = [], [], []
    for cat in categories:
        vals = [r[score_key] for r in records
                if r["status"] == cat and r[score_key] is not None]
        if vals:
            data.append(vals)
            xlabels.append(f"{cat}\n(n={len(vals)})")
            colors.append(COLORS[cat])
    if not data:
        plt.close(fig)
        return
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("blue")
    ax.set_xticks(range(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(f"{label} score")
    ax.set_title(f"{label} by correctness")
    ax.plot([], [], color="black", label="Mean")
    ax.plot([], [], color="blue",  label="Median")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, f"violin_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved violin_{label}.png")


def plot_score_vs_correctness_scatter(records, score_key, output_dir):
    """Scatter: score (x) vs binary correctness (y) with regression line."""
    label = score_key.replace("score_", "")
    answered = [r for r in records
                if r["status"] != "unanswered" and r[score_key] is not None]
    if len(answered) < 3:
        return

    xs = np.array([r[score_key] for r in answered])
    ys = np.array([1 if r["status"] == "correct" else 0 for r in answered])
    colors = [COLORS[r["status"]] for r in answered]

    # Guard: constant scores make correlation undefined
    if np.std(xs) < 1e-10:
        print(f"  Skipping scatter_{label}.png — all scores identical (logprobs not stored?)")
        return

    r, p = stats.pearsonr(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys + np.random.normal(0, 0.02, len(ys)),
               c=colors, alpha=0.5, s=25, edgecolors="none")

    # Regression line
    try:
        m, b = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xline, m * xline + b, "k--", lw=1.5,
                label=f"r={r:.3f}, p={p:.4f}")
    except np.linalg.LinAlgError:
        ax.set_title(f"{label} vs correctness  (regression failed)")
        pass

    ax.set_xlabel(f"{label} score")
    ax.set_ylabel("Correct (1) / Incorrect (0)")
    ax.set_title(f"{label} vs correctness  (answered samples only)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, f"scatter_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved scatter_{label}.png")


def plot_correlation_comparison(corr_table, output_dir):
    """Bar chart comparing Pearson r across all score metrics."""
    if not corr_table:
        return
    labels = list(corr_table.keys())
    r_vals = [corr_table[k]["r_answered"] for k in labels]
    colors = ["#3498db" if r >= 0 else "#e74c3c" for r in r_vals]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    bars = ax.bar(labels, r_vals, color=colors, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Pearson r  (score vs correctness)")
    ax.set_title("Metric comparison: correlation with correctness\n(answered samples only)")
    for bar, val in zip(bars, r_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.03 if val >= 0 else -0.06),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "correlation_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved correlation_comparison.png")


def plot_per_problem_score(records, score_key, output_dir):
    """Per-problem mean score bar chart coloured by majority correctness."""
    label = score_key.replace("score_", "")
    by_problem = defaultdict(list)
    for r in records:
        if r[score_key] is not None:
            by_problem[r["problem_id"]].append(r)

    pids, means, bar_colors = [], [], []
    for pid in sorted(by_problem.keys()):
        samps = by_problem[pid]
        pids.append(pid)
        means.append(np.mean([s[score_key] for s in samps]))
        n_correct = sum(1 for s in samps if s["status"] == "correct")
        n_total   = len(samps)
        if n_correct > n_total / 2:
            bar_colors.append(COLORS["correct"])
        elif n_correct > 0:
            bar_colors.append("#f39c12")
        else:
            bar_colors.append(COLORS["incorrect"])

    fig, ax = plt.subplots(figsize=(max(12, len(pids) * 0.3), 5))
    ax.bar(range(len(pids)), means, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(pids, rotation=90, fontsize=6)
    ax.set_ylabel(f"Mean {label} score")
    ax.set_title(f"Mean {label} per problem")
    patches = [
        mpatches.Patch(color=COLORS["correct"], label="Majority correct"),
        mpatches.Patch(color="#f39c12",          label="Some correct"),
        mpatches.Patch(color=COLORS["incorrect"],label="All incorrect"),
    ]
    ax.legend(handles=patches)
    fig.tight_layout()
    path = os.path.join(output_dir, f"per_problem_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved per_problem_{label}.png")


def print_summary(records, score_keys, results):
    total     = len(records)
    correct   = sum(1 for r in records if r["status"] == "correct")
    incorrect = sum(1 for r in records if r["status"] == "incorrect")
    unanswered= sum(1 for r in records if r["status"] == "unanswered")

    print(f"\n{'='*55}")
    print(f"Sample-level summary  ({total} total samples)")
    print(f"{'='*55}")
    print(f"  Correct:    {correct:4d}  ({correct/total*100:.1f}%)")
    print(f"  Incorrect:  {incorrect:4d}  ({incorrect/total*100:.1f}%)")
    print(f"  Unanswered: {unanswered:4d}  ({unanswered/total*100:.1f}%)")

    for key in score_keys:
        label = key.replace("score_", "")
        print(f"\n  {label} by status:")
        for status in ["correct", "incorrect", "unanswered"]:
            vals = [r[key] for r in records
                    if r["status"] == status and r[key] is not None]
            if vals:
                print(f"    {status:<12}  mean={np.mean(vals):+.4f}  "
                      f"std={np.std(vals):.4f}  "
                      f"range=[{min(vals):+.4f}, {max(vals):+.4f}]")

    print(f"\n  Benchmark accuracies:")
    print(f"    maj@n:    {results.get('maj_accuracy', 'N/A')}")
    print(f"    pass@n:   {results.get('pass_at_n',    'N/A')}")
    think = results.get("think_at_n", {})
    for mode, acc in think.items():
        print(f"    think@n [{mode}]:  {acc}")
    print(f"{'='*55}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze think@n results.json")
    parser.add_argument("--results",    required=True, help="Path to results.json")
    parser.add_argument("--output-dir", default=None,  help="Output directory for plots")
    args = parser.parse_args()

    results    = load_results(args.results)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.results), "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing: {args.results}")
    print(f"Output:    {output_dir}")

    records, score_keys = classify_samples(results)

    if not score_keys:
        print("\nWARNING: No score columns found in sample_extractions.")
        print("Re-run run_think_at_n.py with the updated code to get per-sample scores.")
        score_keys = ["score_dtr"] if any("dtr_scores" in p for p in results["per_problem"]) else []

    print_summary(records, score_keys, results)
    corr_table = compute_correlations(records, score_keys)

    print(f"\nGenerating plots...")
    for key in score_keys:
        plot_score_distribution(records, key, output_dir)
        plot_score_violin(records, key, output_dir)
        plot_score_vs_correctness_scatter(records, key, output_dir)
        plot_per_problem_score(records, key, output_dir)

    plot_correlation_comparison(corr_table, output_dir)

    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
