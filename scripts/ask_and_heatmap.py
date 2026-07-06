"""
Ask the model a question and save a DTR/JSD heatmap for each consecutive
window of N tokens in the generated answer.

Output files:  <output_dir>/heatmap_{i}_{j}.png
  where i..j is the token range (0-indexed over generated tokens).

Usage:
    python scripts/ask_and_heatmap.py --prompt "What is 17 * 23?" --window 150
    python scripts/ask_and_heatmap.py                          # interactive prompt
    python scripts/ask_and_heatmap.py --window 100 --max-tokens 1024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

from src.model.qwen3_helper import Qwen3Helper
from src.dtr.dtr_scorer import DTRScorer


# ── Heatmap plot (tokens on y-axis, layers on x-axis) ────────────────────────

def plot_window_heatmap(jsd, settling, is_deep, deep_thresh, labels, dtr, title=""):
    """Renders a single window heatmap matching the dtr_explorer.py style."""
    T, L = jsd.shape
    fig_h = max(6, T * 0.18)
    fig_w = max(10, L * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        jsd, aspect="auto", origin="upper",
        cmap="YlOrRd", vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="JSD", fraction=0.02, pad=0.01)

    ax.axvline(deep_thresh - 0.5, color="royalblue", lw=2, ls="--")

    for t, (sd, deep) in enumerate(zip(settling, is_deep)):
        ax.plot(sd, t, "o", color="lime" if deep else "red", ms=3, alpha=0.85)

    def _safe(s):
        s = s.replace("\n", "↵").replace("\t", "→")
        s = repr(s)[:20]
        return s.replace("$", r"\$")

    ax.set_yticks(range(T))
    ax.set_yticklabels([f"{i} {_safe(lbl)}" for i, lbl in enumerate(labels)], fontsize=6)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Token position (within window)", fontsize=9)
    ax.set_title(f"{title}  |  DTR={dtr:.3f}", fontsize=10)
    ax.set_xlim(-0.5, L - 0.5)

    thresh_line = Line2D([0], [0], color="royalblue", lw=2, ls="--",
                         label=f"deep threshold (L{deep_thresh})")
    deep_patch  = mpatches.Patch(color="lime",  label="deep token")
    shallow_patch = mpatches.Patch(color="red", label="shallow token")
    ax.legend(handles=[thresh_line, deep_patch, shallow_patch],
              loc="upper left", fontsize=7)

    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ask the model and save per-window DTR heatmaps")
    parser.add_argument("--model",      default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--local-path", default=None,
                        help="Local path to model weights (overrides --model for loading)")
    parser.add_argument("--prompt",     default=None,
                        help="Question to ask. If omitted, prompts interactively.")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max generated tokens (default 512; increase for longer answers)")
    parser.add_argument("--window",     type=int, default=150,
                        help="Tokens per heatmap window (default 150)")
    parser.add_argument("--gamma",      type=float, default=0.5)
    parser.add_argument("--rho",        type=float, default=0.85)
    parser.add_argument("--output-dir", default="outputs/ask_heatmaps")
    args = parser.parse_args()

    question = args.prompt or input("Ask the model: ").strip()

    print("Loading model...")
    helper = Qwen3Helper(model_name=args.model, local_path=args.local_path)
    scorer = DTRScorer(helper, gamma=args.gamma, rho=args.rho)

    print(f"\nGenerating response for: {question}")
    input_ids = helper.tokenize_chat(question)
    prompt_len = input_ids.shape[1]
    print(f"Prompt length: {prompt_len} tokens")

    with torch.no_grad():
        output = helper.model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            use_cache=False,
        )

    generated_ids = output[:, prompt_len:]
    total_gen = generated_ids.shape[1]
    print(f"Generated {total_gen} tokens")

    response = helper.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(f"\nResponse (first 500 chars):\n{response[:500]}\n{'='*60}")

    # Decode all token labels up front
    all_labels = [
        helper.tokenizer.decode([tid], skip_special_tokens=False)
        for tid in generated_ids[0].tolist()
    ]

    # Save heatmaps
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        for f in out_dir.glob("*.png"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    starts = list(range(0, total_gen, args.window))
    print(f"\nSaving {len(starts)} heatmap(s) to {out_dir}/  (window={args.window} tokens)")

    for w_start in starts:
        w_end = min(w_start + args.window, total_gen)
        abs_start = prompt_len + w_start
        abs_end   = prompt_len + w_end

        result = scorer.compute_dtr(output, abs_start, abs_end)

        jsd      = result["jsd_matrix"].cpu().float().detach().numpy()
        settling = result["settling_depths"].cpu().detach().numpy()
        is_deep  = result["is_deep"].cpu().detach().numpy()
        labels   = all_labels[w_start:w_end]
        dtr      = result["dtr"]

        fig = plot_window_heatmap(
            jsd, settling, is_deep,
            result["deep_threshold"],
            labels, dtr,
            title=f"tokens {w_start}–{w_end - 1}",
        )

        fname = out_dir / f"heatmap_{w_start}_{w_end - 1}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{w_start:>4}–{w_end - 1:<4}]  DTR={dtr:.3f}  → {fname.name}")

    print(f"\nDone. {len(starts)} heatmap(s) in {out_dir}/")


if __name__ == "__main__":
    main()
