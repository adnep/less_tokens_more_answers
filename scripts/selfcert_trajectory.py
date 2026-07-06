"""
Self-certainty trajectory analysis.

For each sample computes a per-token signal, smoothed with a rolling window,
and plots how it evolves over the course of the reasoning trace.

Four metric modes (select at most one flag):
  (default)             SC = KL(u‖p_t)  — Self-Certainty (Kang et al. 2025)
                                          ≥ 0, higher = more certain
  --use-neg-entropy     Neg-entropy      — -H(p_t) ∈ (-∞, 0], closer to 0 = more certain
  --use-lp              Log-prob         — stored per-token log P(y_t) from vLLM
                                          ≤ 0, closer to 0 = more confident
                                          NO model pass needed — uses stored logprobs
  --use-sc-variance     SC variance      — rolling std of SC instead of rolling mean
                                          measures oscillation / thrashing
                                          ≥ 0, lower = model is settling steadily

Output directories are tagged to avoid overwriting:
  sc_trajectories/                default SC level
  sc_trajectories_neg_entropy/    --use-neg-entropy
  sc_trajectories_lp/             --use-lp
  sc_trajectories_var/            --use-sc-variance

Plots produced per problem + one aggregate across all problems:
  1. Absolute x-axis: token position, faint individual lines + mean±std band
                      (mean band shown only where ≥80% samples have data)
  2. Normalized x-axis: [0,1] fraction of sequence, same styling
  3. Late-gain boxplot: metric(last 20%) − metric(first 20%)
     For SC/LP: positive = more certain/confident at end
     For SC-var:  negative = less oscillating at end (settling)

Usage:
    python scripts/selfcert_trajectory.py \
        --samples-file outputs/hmmt_samples16new_full/generated_samples.jsonl \
        --benchmark hmmt2025 \
        --model /path/to/Qwen3-4B-Thinking-2507 \
        [--problem-ids hmmt_1 hmmt_3 hmmt_7]   # omit = all problems
        [--window 100]                           # rolling window size
        [--use-lp | --use-sc-variance]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

from src.inference.sampler import load_samples
from src.evaluation.voting import extract_answer, normalize_numeric_answer
from src.evaluation.metrics import _normalize_for_comparison

SLICE_SIZE = 128   # vocab-slice size for entropy — keep low to avoid OOM
COLORS = {"correct": "#2ecc71", "incorrect": "#e74c3c", "unknown": "#95a5a6"}

COVERAGE_THRESHOLD = 0.50   # mean band shown where ≥50% of samples have data


# ──────────────────────────────────────────────────────────────────────────────
# Per-token metric computation
# ──────────────────────────────────────────────────────────────────────────────

def token_neg_entropy(helper, sample) -> np.ndarray:
    """
    Per-token negative entropy: -H(p_t) = ∑_v p_t(v) log p_t(v).
    Range: (-∞, 0].  Closer to 0 = more certain.
    Single forward pass; computed in vocab-slices to avoid OOM.
    Shape: [T_gen].
    """
    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    T_gen      = gen_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)

    with torch.no_grad():
        all_logits = helper.model(full_ids).logits   # [1, T_full, V], bfloat16

    entropy_arr = np.empty(T_gen, dtype=np.float32)
    for start in range(0, T_gen, SLICE_SIZE):
        end = min(start + SLICE_SIZE, T_gen)
        sl  = all_logits[0, prompt_len + start : prompt_len + end].float()
        lp  = torch.log_softmax(sl, dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()
        entropy_arr[start:end] = ent
        del sl, lp, ent

    del all_logits
    torch.cuda.empty_cache()
    return -entropy_arr   # negative entropy


def token_selfcert(helper, sample) -> np.ndarray:
    """
    Per-token Self-Certainty (Kang et al., 2025):
        SC(t) = KL(u ‖ p_t) = -log|V| - (1/|V|) ∑_v log p_t(v)
    Range: [0, ∞).  0 = uniform (maximum uncertainty).  Higher = more certain.
    Single forward pass; computed in vocab-slices to avoid OOM.
    Shape: [T_gen].
    """
    import math

    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    T_gen      = gen_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)

    with torch.no_grad():
        all_logits = helper.model(full_ids).logits   # [1, T_full, V], bfloat16

    V = all_logits.shape[-1]
    log_v = math.log(V)

    # Accumulate (1/V) ∑_v log p_t(v) across vocab slices
    sum_log_p = np.zeros(T_gen, dtype=np.float64)
    for start in range(0, T_gen, SLICE_SIZE):
        end = min(start + SLICE_SIZE, T_gen)
        sl  = all_logits[0, prompt_len + start : prompt_len + end].float()
        lp  = torch.log_softmax(sl, dim=-1)
        sum_log_p[start:end] += lp.sum(dim=-1).cpu().numpy()
        del sl, lp

    del all_logits
    torch.cuda.empty_cache()

    # KL(u ‖ p_t) = -log|V| - mean_v(log p_t(v))
    return (-log_v - sum_log_p / V).astype(np.float32)


def get_raw_metric(helper, sample, metric: str) -> np.ndarray | None:
    """
    Return the raw per-token metric array for a sample.

    metric="sc"          → KL(u ‖ p_t) via forward pass
    metric="neg_entropy" → -H(p_t) via forward pass
    metric="lp"          → stored token log-probs (no model pass)
    """
    if metric == "lp":
        lps = sample.token_logprobs
        if not lps:
            return None
        return np.array(lps, dtype=np.float32)
    elif metric == "neg_entropy":
        return token_neg_entropy(helper, sample)
    else:
        return token_selfcert(helper, sample)

# ──────────────────────────────────────────────────────────────────────────────
# Smoothing helpers
# ──────────────────────────────────────────────────────────────────────────────

def zscore_causal(arr: np.ndarray, min_samples: int = 30, eps: float = 1e-6) -> np.ndarray:
    """
    Causal z-score via Welford's online algorithm.
    At each position t, normalises using only mean/std of arr[:t] —
    no future information, matching what the online engine metric would see.
    Returns 0 for the first `min_samples` positions (insufficient history).
    """
    out = np.zeros(len(arr), dtype=np.float32)
    n, mean, M2 = 0, 0.0, 0.0
    for t, x in enumerate(arr):
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)
        if n >= min_samples:
            std = np.sqrt(M2 / (n - 1))
            out[t] = (x - mean) / (std + eps)
    return out


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean; edges padded with edge values."""
    if len(arr) <= window:
        return np.full_like(arr, arr.mean())
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling std via stride_tricks; edges padded with edge values."""
    if len(arr) <= window:
        return np.full_like(arr, arr.std())
    half   = window // 2
    padded = np.pad(arr, half, mode="edge")
    shape   = (len(arr), window)
    strides = (padded.strides[0], padded.strides[0])
    wins    = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return wins.std(axis=1).astype(np.float32)


def interpolate_to_grid(arr: np.ndarray, n: int = 200) -> np.ndarray:
    """Linearly interpolate array to a fixed grid of n points in [0, 1]."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, arr)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers — all functions receive a `cfg` dict with display strings
# ──────────────────────────────────────────────────────────────────────────────

def _make_cfg(metric: str, use_sc_variance: bool, use_sc_zscore: bool = False) -> dict:
    """Build display-string config based on selected metric mode."""
    if use_sc_zscore:
        return dict(
            metric_name  = "sc_zscore",
            file_prefix  = "sc_zscore",
            ylabel       = "Z-scored SC  (σ units,  0 = mean,  higher = more certain)",
            late_ylabel  = "Z-SC(last 20%) − Z-SC(first 20%)",
            late_title   = "Late-gain: does z-scored certainty rise toward the end?",
            suptitle_tag = "Z-scored SC trajectories",
        )
    if metric == "lp":
        return dict(
            metric_name   = "lp",
            file_prefix   = "lp",
            ylabel        = "Log-prob  (≤ 0,  closer to 0 = more confident)",
            late_ylabel   = "Log-prob(last 20%) − Log-prob(first 20%)",
            late_title    = "Late-gain: does log-prob rise toward the end?",
            suptitle_tag  = "Log-prob trajectories",
        )
    elif metric == "neg_entropy":
        return dict(
            metric_name   = "neg_entropy",
            file_prefix   = "neg_entropy",
            ylabel        = "Negative entropy  (≤ 0,  closer to 0 = more certain)",
            late_ylabel   = "NegEnt(last 20%) − NegEnt(first 20%)",
            late_title    = "Late-gain: does certainty rise toward the end?",
            suptitle_tag  = "Negative-entropy trajectories",
        )
    elif use_sc_variance:
        return dict(
            metric_name   = "sc_var",
            file_prefix   = "sc_var",
            ylabel        = "SC rolling std  (≥ 0,  lower = steadier reasoning)",
            late_ylabel   = "SC-var(last 20%) − SC-var(first 20%)",
            late_title    = "Late-gain: does oscillation decrease toward the end?\n"
                            "(negative = settling — good sign)",
            suptitle_tag  = "SC-variance trajectories",
        )
    else:
        return dict(
            metric_name   = "sc",
            file_prefix   = "sc",
            ylabel        = "Self-certainty  (≥ 0,  KL(u‖p_t),  higher = more certain)",
            late_ylabel   = "SC(last 20%) − SC(first 20%)",
            late_title    = "Late-gain: does certainty rise toward the end?",
            suptitle_tag  = "Self-certainty trajectories",
        )


def _smooth(raw: np.ndarray, window: int, use_sc_variance: bool) -> np.ndarray:
    """Apply the right smoothing function."""
    if use_sc_variance:
        return rolling_std(raw, window)
    return rolling_mean(raw, window)


def plot_problem(traces_by_status, problem_id, window, output_dir,
                 cfg, use_sc_variance):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # ── Panel 1: absolute token position ─────────────────────────────────────
    ax  = axes[0]
    ax2 = ax.twinx()
    ax2.set_ylabel("# samples", color="gray", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    for status, traces in traces_by_status.items():
        if not traces:
            continue
        color   = COLORS[status]
        max_len = max(len(t) for t in traces)

        for t in traces:
            ax.plot(np.arange(len(t)), _smooth(t, window, use_sc_variance),
                    color=color, alpha=0.15, lw=0.8)

        smoothed  = [_smooth(t, window, use_sc_variance) for t in traces]
        coverage  = np.array([sum(1 for s in smoothed if i < len(s))
                               for i in range(max_len)])
        threshold = max(1, int(len(traces) * COVERAGE_THRESHOLD))

        xs_valid = np.where(coverage >= threshold)[0]
        if len(xs_valid) > 1:
            vals_at_x = [
                [smoothed[s][i] for s in range(len(traces)) if i < len(smoothed[s])]
                for i in xs_valid
            ]
            mean_v = np.array([np.mean(v) for v in vals_at_x])
            std_v  = np.array([np.std(v)  for v in vals_at_x])
            ax.plot(xs_valid, mean_v, color=color, lw=2,
                    label=f"{status} (n={len(traces)})")
            ax.fill_between(xs_valid, mean_v - std_v, mean_v + std_v,
                            color=color, alpha=0.18)

        ax2.plot(np.arange(max_len), coverage, color="lightgray", lw=0.8, ls=":")

    ax.set_xlabel("Token position")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"Absolute token position\n"
                 f"(mean band shown where ≥{int(COVERAGE_THRESHOLD*100)}% samples have data)")
    ax.legend(fontsize=8)

    # ── Panel 2: normalised [0, 1] ────────────────────────────────────────────
    ax = axes[1]
    for status, traces in traces_by_status.items():
        if not traces:
            continue
        color = COLORS[status]
        for t in traces:
            xs = np.linspace(0, 1, len(t))
            ax.plot(xs, _smooth(t, window, use_sc_variance),
                    color=color, alpha=0.15, lw=0.8)
        grids = np.stack([interpolate_to_grid(_smooth(t, window, use_sc_variance))
                          for t in traces])
        xs    = np.linspace(0, 1, grids.shape[1])
        mean, std = grids.mean(0), grids.std(0)
        ax.plot(xs, mean, color=color, lw=2, label=f"{status} (n={len(traces)})")
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel("Fraction of sequence")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title("Normalised (shape comparison)")
    ax.legend(fontsize=8)

    # ── Panel 3: late-gain boxplot ────────────────────────────────────────────
    # Computed on RAW (unsmoothed) values so the 20% segments are not distorted
    # by window padding at the edges.
    ax = axes[2]
    box_data, box_labels, box_colors = [], [], []
    for status in ["correct", "incorrect", "unknown"]:
        traces = traces_by_status.get(status, [])
        if not traces:
            continue
        gains = []
        for t in traces:
            if use_sc_variance:
                # variance of SC in each segment
                n20  = max(1, len(t) // 5)
                gains.append(float(np.std(t[-n20:]) - np.std(t[:n20])))
            else:
                n20  = max(1, len(t) // 5)
                gains.append(float(np.mean(t[-n20:]) - np.mean(t[:n20])))
        box_data.append(gains)
        box_labels.append(f"{status}\n(n={len(traces)})")
        box_colors.append(COLORS[status])

    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(range(1, len(box_labels) + 1))
        ax.set_xticklabels(box_labels, fontsize=9)
        ax.set_ylabel(cfg["late_ylabel"])
        ax.set_title(cfg["late_title"])

    fig.suptitle(f"{cfg['suptitle_tag']} — {problem_id}  (window={window})",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir,
                        f"{cfg['file_prefix']}_trajectory_{problem_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_aggregate(all_traces_by_status, window, output_dir, cfg, use_sc_variance):
    """Same two panels aggregated across all problems (normalised only)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: normalised mean ± std
    ax = axes[0]
    for status in ["correct", "incorrect", "unknown"]:
        traces = all_traces_by_status.get(status, [])
        if not traces:
            continue
        color = COLORS[status]
        grids = np.stack([interpolate_to_grid(_smooth(t, window, use_sc_variance))
                          for t in traces])
        xs    = np.linspace(0, 1, grids.shape[1])
        mean, std = grids.mean(0), grids.std(0)
        ax.plot(xs, mean, color=color, lw=2, label=f"{status} (n={len(traces)})")
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel("Fraction of sequence")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title("All problems — normalised mean ± std")
    ax.legend(fontsize=8)

    # Panel 2: late-gain boxplot
    ax = axes[1]
    box_data, box_labels, box_colors = [], [], []
    for status in ["correct", "incorrect", "unknown"]:
        traces = all_traces_by_status.get(status, [])
        if not traces:
            continue
        gains = []
        for t in traces:
            if use_sc_variance:
                n20 = max(1, len(t) // 5)
                gains.append(float(np.std(t[-n20:]) - np.std(t[:n20])))
            else:
                n20 = max(1, len(t) // 5)
                gains.append(float(np.mean(t[-n20:]) - np.mean(t[:n20])))
        box_data.append(gains)
        box_labels.append(f"{status}\n(n={len(traces)})")
        box_colors.append(COLORS[status])

    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(range(1, len(box_labels) + 1))
        ax.set_xticklabels(box_labels, fontsize=9)
        ax.set_ylabel(cfg["late_ylabel"])
        ax.set_title(cfg["late_title"])

    fig.suptitle(f"{cfg['suptitle_tag']} — AGGREGATE  (window={window})",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir,
                        f"{cfg['file_prefix']}_trajectory_aggregate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file",    required=True)
    parser.add_argument("--model",           default=None,
                        help="HF model path. Not needed with --use-lp.")
    parser.add_argument("--benchmark",       default=None,
                        help="Benchmark name for reference answers (optional)")
    parser.add_argument("--problem-ids",     nargs="*", default=None,
                        help="Specific problem IDs to analyse. Omit = all problems.")
    parser.add_argument("--window",          type=int, default=100,
                        help="Rolling window size for smoothing (default 100)")
    parser.add_argument("--output-dir",      default=None,
                        help="Base output directory (metric suffix appended automatically)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--use-neg-entropy", action="store_true",
                      help="Plot -H(p_t) (negative entropy) instead of SC. "
                           "Requires model.")
    mode.add_argument("--use-lp",          action="store_true",
                      help="Use stored token log-probs instead of SC. "
                           "No model pass needed.")
    mode.add_argument("--use-sc-variance", action="store_true",
                      help="Plot rolling std of SC (oscillation) instead of "
                           "rolling mean.")
    mode.add_argument("--use-sc-zscore", action="store_true",
                      help="Z-score each SC trace (mean=0, std=1) before plotting. "
                           "Makes drop patterns comparable across models.")

    args = parser.parse_args()

    # Resolve which metric is active
    if args.use_lp:
        metric = "lp"
    elif args.use_neg_entropy:
        metric = "neg_entropy"
    else:
        metric = "sc"  # sc, sc_variance, and sc_zscore all compute SC underneath

    # Validate: model is required for SC and neg_entropy
    if metric != "lp" and args.model is None:
        parser.error("--model is required unless --use-lp is set")

    # ── Output directory — tagged by metric ───────────────────────────────────
    base_dir = args.output_dir or os.path.join(
        os.path.dirname(args.samples_file), "sc_trajectories"
    )
    if metric == "lp":
        output_dir = base_dir + "_lp"
    elif metric == "neg_entropy":
        output_dir = base_dir + "_neg_entropy"
    elif args.use_sc_variance:
        output_dir = base_dir + "_var"
    elif args.use_sc_zscore:
        output_dir = base_dir + "_zscore"
    else:
        output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = _make_cfg(metric, args.use_sc_variance, args.use_sc_zscore)
    print(f"Metric: {cfg['metric_name']}   Output: {output_dir}/")

    # ── Load samples ──────────────────────────────────────────────────────────
    print(f"Loading samples ...")
    all_samples = load_samples(args.samples_file)
    by_problem  = defaultdict(list)
    for s in all_samples:
        by_problem[s.problem_id].append(s)

    problem_ids = args.problem_ids or sorted(by_problem.keys())
    print(f"  {len(problem_ids)} problems, "
          f"{sum(len(by_problem[p]) for p in problem_ids)} samples")

    # ── Load reference answers ────────────────────────────────────────────────
    ref_answers  = {}
    answer_types = {}
    if args.benchmark:
        from src.evaluation.benchmarks import load_benchmark
        problems = load_benchmark(args.benchmark)
        for p in problems:
            ref_answers[p.id]  = p.answer
            answer_types[p.id] = p.answer_type
        print(f"  Loaded {len(ref_answers)} reference answers")

    # ── Load model (skipped for LP mode) ─────────────────────────────────────
    helper = None
    if metric != "lp":
        from src.model.qwen3_helper import Qwen3Helper
        print(f"\nLoading model: {args.model}")
        helper = Qwen3Helper(model_name=args.model)
    else:
        print("\nSkipping model load — using stored token log-probs.")

    # ── Process problems ──────────────────────────────────────────────────────
    all_traces_by_status = defaultdict(list)

    for pid in problem_ids:
        samples = by_problem.get(pid, [])
        if not samples:
            print(f"  Skipping {pid} — no samples found")
            continue
        print(f"\nProblem {pid}  ({len(samples)} samples)")

        traces_by_status = defaultdict(list)

        for sample in samples:
            print(f"  Sample {sample.sample_idx:2d} "
                  f"({len(sample.generated_token_ids)} tokens) ...", end=" ", flush=True)

            raw = get_raw_metric(helper, sample, metric)
            if raw is None:
                print("SKIP — no log-probs stored (regenerate samples with updated code)")
                continue
            if args.use_sc_zscore:
                raw = zscore_causal(raw)

            # Correctness label
            if pid in ref_answers:
                atype     = answer_types.get(pid, "integer")
                extracted = extract_answer(sample.generated_text, atype)
                correct   = (_normalize_for_comparison(normalize_numeric_answer(str(extracted))) ==
                             _normalize_for_comparison(normalize_numeric_answer(str(ref_answers[pid]))))
                status = "correct" if correct else "incorrect"
            else:
                status = "unknown"

            metric_label = cfg["metric_name"]
            print(f"status={status}  mean_{metric_label}={raw.mean():.4f}")
            traces_by_status[status].append(raw)
            all_traces_by_status[status].append(raw)

        plot_problem(traces_by_status, pid, args.window, output_dir,
                     cfg, args.use_sc_variance)

    # ── Aggregate plot ────────────────────────────────────────────────────────
    if len(problem_ids) > 1:
        print("\nGenerating aggregate plot ...")
        plot_aggregate(dict(all_traces_by_status), args.window, output_dir,
                       cfg, args.use_sc_variance)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
