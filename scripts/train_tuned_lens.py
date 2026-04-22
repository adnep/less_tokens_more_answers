"""
Train tuned lens translators for Qwen3-4B-Thinking.

Downloads a small calibration corpus (WikiText-2), runs the model on it,
and trains one affine translator per layer to map each layer's hidden state
toward the final layer's representation.

Usage:
    python scripts/train_tuned_lens.py \
        --model /path/to/model \
        --output outputs/tuned_lens/weights.pt \
        --n-texts 200 \
        --epochs 5000

The resulting weights.pt file can then be used in DTRScorer:
    tuned_lens = TunedLens.load("outputs/tuned_lens/weights.pt", device="cuda")
    scorer = DTRScorer(model_helper, tuned_lens=tuned_lens)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch

from src.model.qwen3_helper import Qwen3Helper
from src.model.tuned_lens import TunedLens


def load_calibration_texts(n_texts: int = 200, min_length: int = 100) -> list:
    """Load calibration texts from WikiText-2 (small, freely available)."""
    from datasets import load_dataset
    print(f"Loading WikiText-2 calibration corpus...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_length:
            texts.append(text)
        if len(texts) >= n_texts:
            break

    print(f"  Loaded {len(texts)} calibration texts")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Train tuned lens for Qwen3-4B")
    parser.add_argument("--model", required=True,
                        help="HF model name or local path to Qwen3-4B-Thinking")
    parser.add_argument("--output", default="outputs/tuned_lens/weights.pt",
                        help="Output path for trained weights")
    parser.add_argument("--n-texts", type=int, default=200,
                        help="Number of calibration texts from WikiText-2")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Max sequence length per calibration text")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model_helper = Qwen3Helper(args.model)

    # Get model dimensions
    hidden_size = model_helper.model.config.hidden_size
    num_layers = model_helper.num_layers
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load calibration texts
    texts = load_calibration_texts(n_texts=args.n_texts)

    # Create and train tuned lens
    print(f"\nInitializing TunedLens ({num_layers - 1} translators)...")
    tuned_lens = TunedLens(num_layers=num_layers, hidden_size=hidden_size)

    print(f"\nTraining for {args.epochs} epochs (lr={args.lr})...")
    losses = tuned_lens.train_on_model(
        model_helper=model_helper,
        texts=texts,
        tokenizer=tokenizer,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    print(f"\nTraining complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {losses[0]:.6f} → {losses[-1]:.6f}")

    # Save weights
    tuned_lens.save(args.output)
    print(f"\nRun DTR analysis with tuned lens:")
    print(f"  python scripts/run_think_at_n.py --tuned-lens {args.output} ...")


if __name__ == "__main__":
    main()
