"""
DTR prefix reliability analysis.

For all samples of one problem, computes DTR at multiple prefix lengths
(50, 100, 200, 1000 tokens) and compares each to the full-sequence DTR.
Answers: "how reliable is the 50-token prefix estimate?"

Usage:
    python scripts/dtr_prefix_reliability.py \
        --samples-file outputs/aime24_samples_16_full/generated_samples.jsonl \
        --problem-id aime2024_0 \
        --benchmark aime24 \
        --model /path/to/Qwen3-4B-Thinking-2507
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import torch

from src.inference.sampler import load_samples
from src.model.qwen3_helper import Qwen3Helper
from src.dtr.dtr_scorer import DTRScorer
from src.evaluation.voting import extract_answer
from src.evaluation.metrics import _normalize_for_comparison


PREFIX_LENGTHS = [50, 100, 200, 500, 1000]
CHUNK_SIZE     = 150   # tokens per forward pass — reduce if OOM


def compute_all_dtrs(scorer, helper, sample, prefix_lengths):
    """
    ONE chunked forward pass over the full sequence.
    Records DTR at each prefix checkpoint along the way — no repeated computation.

    Strategy:
      - Process generated tokens in chunks of CHUNK_SIZE
      - Accumulate is_deep flags in a list
      - After each chunk, check if any checkpoint has been crossed;
        if so, compute DTR from flags collected so far and store it
      - Full-sequence DTR comes free at the end

    Returns dict: {prefix_len: dtr | None, ..., 'full': dtr, 'actual_length': int}
    """
    from src.dtr.logit_lens import compute_jsd_per_layer
    from src.dtr.settling_depth import compute_settling_depth

    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    total_gen  = gen_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)

    # Sort checkpoints; mark which ones the sequence is long enough to reach
    checkpoints = sorted(prefix_lengths)
    results = {"actual_length": total_gen}
    for p in checkpoints:
        results[p] = None   # will fill as we pass each checkpoint

    is_deep_all  = []   # accumulated bool flags, one per generated token
    next_cp_idx  = 0    # index into checkpoints we haven't recorded yet

    for chunk_start in range(0, total_gen, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_gen)

        # Forward pass on prompt + tokens up to chunk_end
        input_ids_chunk = full_ids[:, : prompt_len + chunk_end]
        with torch.no_grad():
            hidden_states = helper.get_layer_hidden_states(input_ids_chunk)

        token_slice = slice(prompt_len + chunk_start, prompt_len + chunk_end)
        jsd = compute_jsd_per_layer(
            hidden_states,
            helper.norm,
            helper.lm_head,
            token_positions=token_slice,
            batch_layers=(chunk_end - chunk_start) <= 100,
            tuned_lens=scorer.tuned_lens,
        )
        del hidden_states

        depths  = compute_settling_depth(jsd, gamma=scorer.gamma)
        is_deep = (depths >= scorer.deep_threshold).tolist()
        is_deep_all.extend(is_deep)
        del jsd, depths

        # Record DTR for any checkpoint we've now passed
        tokens_done = len(is_deep_all)
        while next_cp_idx < len(checkpoints) and checkpoints[next_cp_idx] <= tokens_done:
            cp = checkpoints[next_cp_idx]
            results[cp] = float(np.mean(is_deep_all[:cp]))
            next_cp_idx += 1

    results["full"] = float(np.mean(is_deep_all)) if is_deep_all else 0.0
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--problem-id",   required=True,
                        help="Problem ID to analyse, e.g. aime2024_0 or hmmt_3")
    parser.add_argument("--benchmark",    default=None,
                        help="Benchmark name (aime24/gpqa_diamond/hmmt2025) for "
                             "reference answers. Optional — skip for unknown.")
    parser.add_argument("--model",        required=True,
                        help="HF model name or local path")
    parser.add_argument("--output-dir",   default=None,
                        help="Where to save plots. Defaults to same dir as samples file.")
    parser.add_argument("--gamma",        type=float, default=0.5)
    parser.add_argument("--rho",          type=float, default=0.85)
    parser.add_argument("--sample-ids",   default=None,
                        help="Comma-separated sample indices to include, e.g. 0,3,7. "
                             "Defaults to all samples for the problem.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.samples_file), "dtr_reliability"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── Load samples for the requested problem ────────────────────────────────
    print(f"Loading samples from {args.samples_file} ...")
    all_samples = load_samples(args.samples_file)
    samples = [s for s in all_samples if s.problem_id == args.problem_id]
    if not samples:
        print(f"ERROR: problem_id '{args.problem_id}' not found.")
        print(f"Available IDs: {sorted(set(s.problem_id for s in all_samples))[:10]} ...")
        return
    print(f"  Found {len(samples)} samples for '{args.problem_id}'")
    if args.sample_ids is not None:
        keep = set(int(x) for x in args.sample_ids.split(","))
        samples = [s for s in samples if s.sample_idx in keep]
        print(f"  Filtered to {len(samples)} samples: {sorted(keep)}")

    # ── Optional: load reference answer for correctness colouring ────────────
    ref_answer = None
    answer_type = "integer"
    if args.benchmark:
        from src.evaluation.benchmarks import load_benchmark
        problems = load_benchmark(args.benchmark)
        prob = next((p for p in problems if p.id == args.problem_id), None)
        if prob:
            ref_answer  = prob.answer
            answer_type = prob.answer_type
            print(f"  Reference answer: {ref_answer}  (type={answer_type})")
        else:
            print(f"  WARNING: problem_id not found in benchmark '{args.benchmark}'")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    helper = Qwen3Helper(model_name=args.model)
    scorer = DTRScorer(helper, gamma=args.gamma, rho=args.rho)

    # ── Compute DTR at all prefix lengths for each sample ────────────────────
    print(f"\nComputing DTR at prefixes {PREFIX_LENGTHS} + full sequence ...")
    all_results = []
    for i, sample in enumerate(samples):
        print(f"  Sample {sample.sample_idx:2d}  "
              f"(len={len(sample.generated_token_ids)} tokens) ...", end=" ", flush=True)
        r = compute_all_dtrs(scorer, helper, sample, PREFIX_LENGTHS)

        # Correctness label
        if ref_answer is not None:
            extracted = extract_answer(sample.generated_text, answer_type)
            correct = (_normalize_for_comparison(str(extracted)) ==
                       _normalize_for_comparison(str(ref_answer)))
        else:
            correct = None

        r["sample_idx"] = sample.sample_idx
        r["correct"]    = correct
        all_results.append(r)
        print(f"full={r['full']:.3f}  "
              f"p50={r.get(50,'N/A') if r.get(50) is not None else 'short'}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Sample':>8}  {'len':>6}  "
          + "  ".join(f"p{p:>4}" for p in PREFIX_LENGTHS)
          + "    full   correct")
    print(f"{'-'*70}")
    for r in all_results:
        row = f"{r['sample_idx']:>8}  {r['actual_length']:>6}  "
        for p in PREFIX_LENGTHS:
            val = r.get(p)
            row += f"  {val:.3f}" if val is not None else "   n/a"
        row += f"  {r['full']:.3f}"
        if r["correct"] is not None:
            row += f"  {'✓' if r['correct'] else '✗'}"
        print(row)

    # ── Pearson r: prefix vs full ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Pearson r  (prefix DTR vs full-sequence DTR)")
    print(f"{'='*50}")
    full_dtrs = np.array([r["full"] for r in all_results])
    for p in PREFIX_LENGTHS:
        vals = [(r[p], r["full"]) for r in all_results if r.get(p) is not None]
        if len(vals) < 3:
            print(f"  prefix {p:>5}: n/a (too few samples with length >= {p})")
            continue
        pref, full = zip(*vals)
        pref, full = np.array(pref), np.array(full)
        if np.std(pref) < 1e-9:
            print(f"  prefix {p:>5}: constant — all samples shorter than {p} tokens?")
            continue
        r_val, p_val = stats.pearsonr(pref, full)
        mae = np.mean(np.abs(pref - full))
        print(f"  prefix {p:>5}:  r={r_val:+.4f}  p={p_val:.4f}  MAE={mae:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    colors_map = {True: "#2ecc71", False: "#e74c3c", None: "#95a5a6"}

    # Plot 1: DTR vs prefix length, one line per sample
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = [str(p) for p in PREFIX_LENGTHS] + ["full"]
    x_pos    = list(range(len(x_labels)))

    for r in all_results:
        y = []
        xs_used = []
        for xi, p in enumerate(PREFIX_LENGTHS):
            if r.get(p) is not None:
                y.append(r[p])
                xs_used.append(xi)
        y.append(r["full"])
        xs_used.append(len(PREFIX_LENGTHS))

        color = colors_map[r["correct"]]
        ax.plot(xs_used, y, "o-", color=color, alpha=0.6, lw=1.2, ms=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Prefix length (tokens)")
    ax.set_ylabel("DTR")
    ax.set_ylim(0, 1)
    ax.set_title(f"DTR vs prefix length — {args.problem_id}  "
                 f"(n={len(samples)}  γ={args.gamma} ρ={args.rho})")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#2ecc71", lw=2, label="correct"),
        Line2D([0], [0], color="#e74c3c", lw=2, label="incorrect"),
        Line2D([0], [0], color="#95a5a6", lw=2, label="unknown"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, f"dtr_prefix_lines_{args.problem_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Plot 2: scatter grid — prefix-N DTR vs full DTR, one subplot per prefix
    valid_prefixes = [p for p in PREFIX_LENGTHS
                      if any(r.get(p) is not None for r in all_results)]
    ncols = min(3, len(valid_prefixes))
    nrows = (len(valid_prefixes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, p in enumerate(valid_prefixes):
        ax = axes[idx // ncols][idx % ncols]
        vals = [(r[p], r["full"], r["correct"])
                for r in all_results if r.get(p) is not None]
        if not vals:
            ax.axis("off")
            continue
        pref_v, full_v, correct_v = zip(*vals)
        pref_v, full_v = np.array(pref_v), np.array(full_v)
        cs = [colors_map[c] for c in correct_v]

        ax.scatter(pref_v, full_v, c=cs, s=60, alpha=0.8, edgecolors="white", lw=0.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="ideal")

        if np.std(pref_v) > 1e-9:
            r_val, _ = stats.pearsonr(pref_v, full_v)
            mae = np.mean(np.abs(pref_v - full_v))
            ax.set_title(f"prefix={p}  r={r_val:+.3f}  MAE={mae:.3f}")
        else:
            ax.set_title(f"prefix={p}  (constant)")

        ax.set_xlabel(f"DTR @ prefix {p}")
        ax.set_ylabel("DTR @ full sequence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused subplots
    for idx in range(len(valid_prefixes), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"Prefix DTR vs Full-sequence DTR — {args.problem_id}", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f"dtr_prefix_scatter_{args.problem_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 3: heatmap — samples × prefix lengths (colour = DTR value)
    fig, ax = plt.subplots(figsize=(8, max(4, len(samples) * 0.3)))
    col_labels = [str(p) for p in PREFIX_LENGTHS] + ["full"]
    matrix = []
    sample_labels = []
    for r in all_results:
        row = [r.get(p) for p in PREFIX_LENGTHS] + [r["full"]]
        # Fill None (too short) with the full-sequence DTR as best estimate
        row = [v if v is not None else r["full"] for v in row]
        matrix.append(row)
        tick = f"S{r['sample_idx']}"
        if r["correct"] is not None:
            tick += " ✓" if r["correct"] else " ✗"
        sample_labels.append(tick)

    M = np.array(matrix)
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(sample_labels)))
    ax.set_yticklabels(sample_labels, fontsize=8)
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Sample")
    ax.set_title(f"DTR heatmap — {args.problem_id}")
    plt.colorbar(im, ax=ax, label="DTR")
    fig.tight_layout()
    path = os.path.join(output_dir, f"dtr_prefix_heatmap_{args.problem_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nAll outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
