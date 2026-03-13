"""
Compute DTR for a single model generation.
Useful for verifying the DTR pipeline end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import argparse
import time

from src.model.qwen3_helper import Qwen3Helper
from src.dtr.dtr_scorer import DTRScorer


def main():
    parser = argparse.ArgumentParser(description="Compute DTR for a single generation")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--prompt", default="Find all integers n such that n^2 + 3n + 5 is divisible by 121.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--prefix-length", type=int, default=50)
    args = parser.parse_args()

    print("Loading model...")
    helper = Qwen3Helper(model_name=args.model, local_path=args.local_path)
    scorer = DTRScorer(helper, gamma=args.gamma, rho=args.rho)

    print(f"\nDeep-thinking threshold: layer >= {scorer.deep_threshold} (of {helper.num_layers})")

    # Generate a response
    print(f"\nGenerating response for: {args.prompt}")
    input_ids = helper.tokenize_chat(args.prompt)
    prompt_len = input_ids.shape[1]
    print(f"Prompt length: {prompt_len} tokens")

    with torch.no_grad():
        output = helper.model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )

    generated_ids = output[:, prompt_len:]
    total_gen = generated_ids.shape[1]
    print(f"Generated {total_gen} tokens")

    # Decode and show response
    response = helper.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(f"\nResponse (first 500 chars):\n{response[:500]}...")

    # Compute full DTR
    print(f"\n{'='*60}")
    print("Computing DTR on full generation...")
    t0 = time.time()
    result = scorer.compute_dtr(output, generated_token_start=prompt_len)
    print(f"  DTR (full): {result['dtr']:.4f}")
    print(f"  Settling depth stats:")
    depths = result["settling_depths"].float()
    print(f"    Mean:   {depths.mean():.1f}")
    print(f"    Median: {depths.median():.1f}")
    print(f"    Min:    {depths.min():.0f}")
    print(f"    Max:    {depths.max():.0f}")
    print(f"  Deep-thinking tokens: {result['is_deep'].sum().item()}/{total_gen}")
    print(f"  DTR computation took {time.time() - t0:.2f}s")

    # Compute prefix DTR
    print(f"\nComputing prefix DTR ({args.prefix_length} tokens)...")
    prefix_dtr = scorer.compute_prefix_dtr(
        input_ids, generated_ids, prefix_length=args.prefix_length
    )
    print(f"  DTR (prefix-{args.prefix_length}): {prefix_dtr:.4f}")

    # Save settling depth distribution
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Settling depth histogram
        axes[0].hist(
            depths.cpu().numpy(), bins=helper.num_layers, range=(0, helper.num_layers),
            edgecolor="black", alpha=0.7
        )
        axes[0].axvline(scorer.deep_threshold, color="red", linestyle="--",
                       label=f"Deep threshold (layer {scorer.deep_threshold})")
        axes[0].set_xlabel("Settling depth (layer)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Settling Depth Distribution")
        axes[0].legend()

        # JSD matrix heatmap (first 100 tokens)
        jsd_mat = result["jsd_matrix"][:min(100, total_gen), :].cpu().numpy()
        im = axes[1].imshow(jsd_mat.T, aspect="auto", cmap="viridis", origin="lower")
        axes[1].set_xlabel("Generated token position")
        axes[1].set_ylabel("Layer")
        axes[1].set_title("JSD(final || layer)")
        plt.colorbar(im, ax=axes[1])

        os.makedirs("outputs/dtr_single", exist_ok=True)
        fig.savefig("outputs/dtr_single/dtr_analysis.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to outputs/dtr_single/dtr_analysis.png")
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")


if __name__ == "__main__":
    main()
