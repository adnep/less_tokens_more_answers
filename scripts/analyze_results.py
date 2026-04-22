"""
Analyze results.json: DTR scores vs correctness.

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
from collections import Counter


def load_results(path):
    with open(path) as f:
        return json.load(f)


def classify_samples(results):
    """Classify each sample as correct, incorrect, or unanswered."""
    records = []
    for problem in results["per_problem"]:
        ref = str(problem["reference"]).strip()
        for dtr_entry in problem["dtr_scores"]:
            sample_idx, dtr_score = dtr_entry[0], dtr_entry[1]

            # Find the extracted answer for this sample
            extracted = None
            for ext in problem.get("sample_extractions", []):
                if ext["sample_idx"] == sample_idx:
                    extracted = ext["extracted_answer"]
                    break

            if extracted is None:
                status = "unanswered"
            elif str(extracted).strip() == ref:
                status = "correct"
            else:
                status = "incorrect"

            records.append({
                "problem_id": problem["id"],
                "sample_idx": sample_idx,
                "dtr": dtr_score,
                "extracted": extracted,
                "reference": ref,
                "status": status,
            })
    return records


COLORS = {"correct": "#2ecc71", "incorrect": "#e74c3c", "unanswered": "#95a5a6"}


def plot_dtr_distribution(records, output_dir):
    """Histogram of DTR scores colored by correctness."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for status in ["correct", "incorrect", "unanswered"]:
        dtrs = [r["dtr"] for r in records if r["status"] == status]
        if dtrs:
            ax.hist(dtrs, bins=20, alpha=0.6, label=f"{status} (n={len(dtrs)})",
                    color=COLORS[status], edgecolor="white")

    ax.set_xlabel("Prefix DTR Score")
    ax.set_ylabel("Count")
    ax.set_title("DTR Score Distribution by Correctness")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dtr_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved dtr_distribution.png")


def plot_dtr_violin(records, output_dir):
    """Violin/box plot of DTR by correctness category."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["correct", "incorrect", "unanswered"]
    data = []
    labels = []
    colors = []
    for cat in categories:
        dtrs = [r["dtr"] for r in records if r["status"] == cat]
        if dtrs:
            data.append(dtrs)
            labels.append(f"{cat}\n(n={len(dtrs)})")
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

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prefix DTR Score")
    ax.set_title("DTR Score Distribution by Correctness")
    ax.set_ylim(0, 1)

    # Add legend for mean/median
    ax.plot([], [], color="black", label="Mean")
    ax.plot([], [], color="blue", label="Median")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dtr_violin.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved dtr_violin.png")


def plot_dtr_strip(records, output_dir):
    """Strip/swarm plot showing individual samples."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["correct", "incorrect", "unanswered"]
    for i, cat in enumerate(categories):
        dtrs = [r["dtr"] for r in records if r["status"] == cat]
        if dtrs:
            # Add jitter
            jitter = np.random.normal(0, 0.08, len(dtrs))
            ax.scatter(
                [i + 1] * len(dtrs) + jitter, dtrs,
                alpha=0.5, s=30, color=COLORS[cat],
                label=f"{cat} (n={len(dtrs)})", edgecolors="white", linewidth=0.3
            )
            # Add mean marker
            ax.scatter([i + 1], [np.mean(dtrs)], color="black", s=100,
                       marker="D", zorder=5)

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Prefix DTR Score")
    ax.set_title("Individual Sample DTR Scores")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "dtr_strip.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved dtr_strip.png")


def plot_accuracy_by_dtr_threshold(records, output_dir):
    """Accuracy curve: if we only keep samples above DTR threshold, what accuracy?"""
    answered = [r for r in records if r["status"] != "unanswered"]
    if not answered:
        return

    dtrs = sorted(set(r["dtr"] for r in answered))
    thresholds = np.linspace(0, max(dtrs), 50)

    accuracies = []
    counts = []
    for thresh in thresholds:
        above = [r for r in answered if r["dtr"] >= thresh]
        if above:
            acc = sum(1 for r in above if r["status"] == "correct") / len(above)
            accuracies.append(acc)
            counts.append(len(above))
        else:
            accuracies.append(None)
            counts.append(0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Accuracy line
    valid = [(t, a) for t, a in zip(thresholds, accuracies) if a is not None]
    if valid:
        ts, accs = zip(*valid)
        ax1.plot(ts, accs, "b-o", markersize=3, label="Accuracy")
        ax1.set_xlabel("DTR Threshold (keep samples >= threshold)")
        ax1.set_ylabel("Accuracy", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_ylim(0, 1.05)

    # Sample count on secondary axis
    ax2 = ax1.twinx()
    ax2.fill_between(thresholds, counts, alpha=0.15, color="gray")
    ax2.set_ylabel("Samples Remaining", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title("Accuracy vs DTR Threshold (among answered samples)")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_by_dtr_threshold.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_by_dtr_threshold.png")


def plot_per_problem_dtr(records, output_dir):
    """Per-problem bar chart: mean DTR colored by majority correctness."""
    from collections import defaultdict

    by_problem = defaultdict(list)
    for r in records:
        by_problem[r["problem_id"]].append(r)

    problem_ids = []
    mean_dtrs = []
    bar_colors = []

    for pid in sorted(by_problem.keys()):
        samples = by_problem[pid]
        problem_ids.append(pid)
        mean_dtrs.append(np.mean([s["dtr"] for s in samples]))

        # Color by whether majority of samples are correct
        n_correct = sum(1 for s in samples if s["status"] == "correct")
        n_total = len(samples)
        if n_correct > n_total / 2:
            bar_colors.append(COLORS["correct"])
        elif any(s["status"] == "correct" for s in samples):
            bar_colors.append("#f39c12")  # orange: some correct
        else:
            bar_colors.append(COLORS["incorrect"])

    fig, ax = plt.subplots(figsize=(max(12, len(problem_ids) * 0.3), 5))
    bars = ax.bar(range(len(problem_ids)), mean_dtrs, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(problem_ids)))
    ax.set_xticklabels(problem_ids, rotation=90, fontsize=6)
    ax.set_ylabel("Mean DTR Score")
    ax.set_title("Mean DTR per Problem")
    ax.set_ylim(0, 1)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS["correct"], label="Majority correct"),
        mpatches.Patch(color="#f39c12", label="Some correct"),
        mpatches.Patch(color=COLORS["incorrect"], label="All incorrect"),
    ]
    ax.legend(handles=patches, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_problem_dtr.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved per_problem_dtr.png")


def plot_think_vs_maj(results, output_dir):
    """Compare think@n selection vs majority vote per problem."""
    problems = results["per_problem"]

    think_correct = []
    maj_correct = []
    dtrs_mean = []
    labels = []

    for p in problems:
        ref = str(p["reference"]).strip()
        tc = str(p.get("think_answer", "")).strip() == ref
        mc = str(p.get("maj_answer", "")).strip() == ref
        think_correct.append(tc)
        maj_correct.append(mc)
        labels.append(p["id"])

        # Mean DTR for this problem
        if p["dtr_scores"]:
            dtrs_mean.append(np.mean([d[1] for d in p["dtr_scores"]]))
        else:
            dtrs_mean.append(0)

    # Categorize: both correct, only think, only maj, neither
    categories = []
    for tc, mc in zip(think_correct, maj_correct):
        if tc and mc:
            categories.append("both")
        elif tc:
            categories.append("think_only")
        elif mc:
            categories.append("maj_only")
        else:
            categories.append("neither")

    cat_colors = {
        "both": "#2ecc71",
        "think_only": "#3498db",
        "maj_only": "#e67e22",
        "neither": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.3), 5))
    colors = [cat_colors[c] for c in categories]
    ax.bar(range(len(labels)), dtrs_mean, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Mean DTR Score")
    ax.set_title("think@n vs maj@n: Which Problems Does DTR Selection Help?")
    ax.set_ylim(0, 1)

    patches = [
        mpatches.Patch(color=cat_colors["both"], label="Both correct"),
        mpatches.Patch(color=cat_colors["think_only"], label="Only think@n correct"),
        mpatches.Patch(color=cat_colors["maj_only"], label="Only maj@n correct"),
        mpatches.Patch(color=cat_colors["neither"], label="Neither correct"),
    ]
    ax.legend(handles=patches, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "think_vs_maj.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved think_vs_maj.png")


def print_summary(records, results):
    """Print summary statistics."""
    total = len(records)
    correct = sum(1 for r in records if r["status"] == "correct")
    incorrect = sum(1 for r in records if r["status"] == "incorrect")
    unanswered = sum(1 for r in records if r["status"] == "unanswered")

    print(f"\n{'='*50}")
    print(f"Summary Statistics")
    print(f"{'='*50}")
    print(f"Total samples:  {total}")
    print(f"  Correct:      {correct} ({correct/total*100:.1f}%)")
    print(f"  Incorrect:    {incorrect} ({incorrect/total*100:.1f}%)")
    print(f"  Unanswered:   {unanswered} ({unanswered/total*100:.1f}%)")

    for status in ["correct", "incorrect", "unanswered"]:
        dtrs = [r["dtr"] for r in records if r["status"] == status]
        if dtrs:
            print(f"\n  DTR ({status}):")
            print(f"    Mean:   {np.mean(dtrs):.4f}")
            print(f"    Median: {np.median(dtrs):.4f}")
            print(f"    Std:    {np.std(dtrs):.4f}")
            print(f"    Range:  [{min(dtrs):.4f}, {max(dtrs):.4f}]")

    print(f"\n  Benchmark accuracies:")
    print(f"    maj@n:   {results.get('maj_accuracy', 'N/A')}")
    print(f"    think@n: {results.get('think_accuracy', 'N/A')}")
    print(f"    pass@n:  {results.get('pass_at_n', 'N/A')}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DTR results")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()

    results = load_results(args.results)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.results), "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing: {args.results}")
    print(f"Plots will be saved to: {output_dir}")

    records = classify_samples(results)
    print_summary(records, results)

    print(f"\nGenerating plots...")
    plot_dtr_distribution(records, output_dir)
    plot_dtr_violin(records, output_dir)
    plot_dtr_strip(records, output_dir)
    plot_accuracy_by_dtr_threshold(records, output_dir)
    plot_per_problem_dtr(records, output_dir)
    plot_think_vs_maj(results, output_dir)

    print(f"\nDone! All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
