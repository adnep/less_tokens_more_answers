"""
Binned DTR-accuracy correlation, replicating the analysis from Section 3.1
of the paper: sort samples by DTR into quantile bins, compute avg accuracy
per bin, then Pearson r on the bin averages.

Also computes "flat" Pearson r (score vs binary label, per sample) for comparison.

Usage:
    python scripts/dtr_binned_correlation.py --results outputs/aime_full_fixed/results.json
    python scripts/dtr_binned_correlation.py --results outputs/aime_full_fixed/results.json --bins 5
"""

import re
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def _normalize(s):
    if s is None:
        return None
    s = re.sub(r'\s+', '', str(s).strip()).lower()
    return str(int(s)) if s.isdigit() else s


def load_samples(path, exclude_unanswered=False):
    with open(path) as f:
        data = json.load(f)
    samples = []
    n_unanswered = 0
    for prob in data["per_problem"]:
        ref = _normalize(prob["reference"])
        for ext in prob.get("sample_extractions", []):
            dtr = ext.get("score_dtr")
            if dtr is None:
                continue
            extracted = _normalize(ext.get("extracted_answer"))
            answer_text = ext.get("answer_text", "")
            unanswered = answer_text == "" or extracted is None or extracted == ""
            if unanswered:
                n_unanswered += 1
                if exclude_unanswered:
                    continue
            correct = int(not unanswered and extracted == ref)
            samples.append({"dtr": dtr, "correct": correct})
    return samples, n_unanswered


def binned_correlation(samples, n_bins=5):
    samples_sorted = sorted(samples, key=lambda s: s["dtr"])
    bins = np.array_split(samples_sorted, n_bins)

    bin_dtr_means = []
    bin_accuracies = []
    for i, b in enumerate(bins):
        mean_dtr = np.mean([s["dtr"] for s in b])
        mean_acc = np.mean([s["correct"] for s in b])
        bin_dtr_means.append(mean_dtr)
        bin_accuracies.append(mean_acc)
        print(f"  Bin {i+1}: avg DTR={mean_dtr:.4f},  avg accuracy={mean_acc:.4f}  (n={len(b)})")

    r, p = pearsonr(bin_dtr_means, bin_accuracies)
    return bin_dtr_means, bin_accuracies, r, p


def flat_correlation(samples):
    dtrs = [s["dtr"] for s in samples]
    labels = [s["correct"] for s in samples]
    r, p = pearsonr(dtrs, labels)
    return r, p


def plot(bin_dtr_means, bin_accuracies, r_binned, r_flat, output_path):
    xs = np.array(bin_dtr_means)
    ys = np.array(bin_accuracies)
    m, b = np.polyfit(xs, ys, 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, color="steelblue", s=60, zorder=3)
    x_line = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(x_line, m * x_line + b, color="firebrick", linewidth=1.5, zorder=2,
            label=f"r = {r_binned:.3f}")
    ax.legend(fontsize=9)
    ax.set_xlabel("Mean DTR in bin")
    ax.set_ylabel("Average accuracy")
    ax.set_title("DTR vs accuracy using quantile bins", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--exclude-unanswered", action="store_true",
                        help="Drop samples with no extracted answer instead of counting them as incorrect")
    args = parser.parse_args()

    samples, n_unanswered = load_samples(args.results, exclude_unanswered=args.exclude_unanswered)
    label = "excluded" if args.exclude_unanswered else "counted as incorrect"
    print(f"Loaded {len(samples)} samples with DTR scores  "
          f"({n_unanswered} unanswered, {label})\n")

    print(f"── Binned analysis ({args.bins} quantile bins) ──")
    bin_means, bin_accs, r_binned, p_binned = binned_correlation(samples, args.bins)
    print(f"\n  Pearson r (binned): {r_binned:.4f}  (p={p_binned:.4f})")

    print(f"\n── Flat per-sample correlation ──")
    r_flat, p_flat = flat_correlation(samples)
    print(f"  Pearson r (flat):   {r_flat:.4f}  (p={p_flat:.4f})")

    out = args.output or args.results.replace("results.json", f"dtr_binned_corr_{args.bins}bins.png")
    plot(bin_means, bin_accs, r_binned, r_flat, out)


if __name__ == "__main__":
    main()
