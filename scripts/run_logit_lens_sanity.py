"""
Sanity check: verify logit lens works correctly on Qwen3-4B-Thinking.

Checks:
1. Model loads and generates correctly
2. Layer wrappers capture hidden states of correct shape
3. Final layer logit lens matches model output
4. JSD heatmap looks reasonable (decreasing divergence toward later layers)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import argparse

from src.model.qwen3_helper import Qwen3Helper
from src.dtr.logit_lens import compute_jsd_per_layer


def main():
    parser = argparse.ArgumentParser(description="Logit lens sanity check")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--prompt", default="What is 2 + 3?")
    args = parser.parse_args()

    # Step 1: Load model
    print("=" * 60)
    print("Step 1: Loading model...")
    helper = Qwen3Helper(model_name=args.model, local_path=args.local_path)
    print(f"  Num layers: {helper.num_layers}")
    print(f"  Device: {helper.device}")

    # Step 2: Tokenize and check shapes
    print("\n" + "=" * 60)
    print("Step 2: Checking hidden state capture...")
    input_ids = helper.tokenize_chat(args.prompt)
    print(f"  Input shape: {input_ids.shape}")

    layer_hidden_states = helper.get_layer_hidden_states(input_ids)
    print(f"  Captured {len(layer_hidden_states)} layer hidden states")

    for layer_idx in [0, helper.num_layers // 2, helper.num_layers - 1]:
        hs = layer_hidden_states[layer_idx]
        print(f"  Layer {layer_idx}: shape={hs.shape}, dtype={hs.dtype}")

    # Step 3: Verify final layer logit lens matches model output
    print("\n" + "=" * 60)
    print("Step 3: Verifying final layer logit lens...")
    final_hs = layer_hidden_states[helper.num_layers - 1]
    lens_probs = helper.project_to_vocab(final_hs)

    # Direct model output
    with torch.no_grad():
        model_output = helper.model(input_ids)
        model_probs = torch.softmax(model_output.logits.float(), dim=-1)

    # Compare last token probabilities
    lens_last = lens_probs[0, -1, :]
    model_last = model_probs[0, -1, :]
    max_diff = (lens_last - model_last).abs().max().item()
    print(f"  Max prob difference (final layer lens vs model): {max_diff:.6e}")
    assert max_diff < 1e-3, f"Final layer lens doesn't match model output! Diff: {max_diff}"
    print("  PASSED: Final layer lens matches model output.")

    # Top-5 predictions comparison
    _, lens_top5 = lens_last.topk(5)
    _, model_top5 = model_last.topk(5)
    lens_tokens = [helper.tokenizer.decode(t) for t in lens_top5]
    model_tokens = [helper.tokenizer.decode(t) for t in model_top5]
    print(f"  Lens top-5:  {lens_tokens}")
    print(f"  Model top-5: {model_tokens}")

    # Step 4: Compute JSD heatmap
    print("\n" + "=" * 60)
    print("Step 4: Computing JSD matrix...")
    jsd_tensor = compute_jsd_per_layer(
        layer_hidden_states, helper.norm, helper.lm_head
    )
    print(f"  JSD matrix shape: {jsd_tensor.shape} (tokens x layers)")
    print(f"  JSD range: [{jsd_tensor.min():.4f}, {jsd_tensor.max():.4f}]")

    # Check that JSD decreases toward later layers (on average)
    mean_jsd_per_layer = jsd_tensor.mean(dim=0)
    print(f"  Mean JSD early layers (0-5):   {mean_jsd_per_layer[:6].mean():.4f}")
    print(f"  Mean JSD middle layers (15-20): {mean_jsd_per_layer[15:21].mean():.4f}")
    print(f"  Mean JSD late layers (-5:):     {mean_jsd_per_layer[-5:].mean():.4f}")
    print(f"  Final layer JSD (should be ~0): {mean_jsd_per_layer[-1]:.6f}")

    # Save JSD heatmap
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            jsd_tensor.cpu().detach().numpy().T,
            ax=ax,
            cmap="viridis",
            xticklabels=5,
            yticklabels=5,
        )
        ax.set_xlabel("Token position")
        ax.set_ylabel("Layer")
        ax.set_title(f"JSD(final layer || layer l) for: {args.prompt}")
        os.makedirs("outputs/sanity", exist_ok=True)
        fig.savefig("outputs/sanity/jsd_heatmap.png", dpi=150, bbox_inches="tight")
        print("\n  Saved JSD heatmap to outputs/sanity/jsd_heatmap.png")
    except ImportError:
        print("\n  matplotlib/seaborn not available, skipping heatmap plot.")

    print("\n" + "=" * 60)
    print("All sanity checks PASSED.")


if __name__ == "__main__":
    main()
