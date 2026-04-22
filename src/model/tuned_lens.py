"""
Tuned Lens: learned affine translators per layer for better hidden state projection.

Instead of logit lens (directly applying norm+lm_head to each layer's hidden state),
tuned lens first applies a learned per-layer affine map T_ell that maps the layer's
hidden state to be closer to what the final layer would produce.

Reference: "Eliciting Latent Predictions from Transformers with the Tuned Lens"
           (Belrose et al., 2023)

Architecture:
  - One affine translator T_ell per intermediate layer: h_ell → T_ell(h_ell)
  - T_ell is initialized as identity (so it starts equivalent to logit lens)
  - T_ell maps pre-norm hidden state → post-norm space (final layer)
  - Then lm_head is applied (without norm) to get logits

Training loss (per layer):
  MSE(T_ell(h_ell), h_final) where h_final is the model's final post-norm hidden state

Usage:
    tuned_lens = TunedLens(num_layers=36, hidden_size=2560)
    tuned_lens.train_on_model(model_helper, calibration_texts, tokenizer)
    tuned_lens.save("outputs/tuned_lens_weights.pt")

    # Later:
    tuned_lens = TunedLens.load("outputs/tuned_lens_weights.pt", device)
"""

import os
import torch
import torch.nn as nn
from typing import List, Optional


class TunedLens(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        """
        Args:
            num_layers: total number of decoder layers (e.g. 36 for Qwen3-4B)
            hidden_size: hidden state dimension (e.g. 2560 for Qwen3-4B)
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # One translator per intermediate layer (not the final layer, which is reference)
        # Each is an affine map h -> W*h + b, initialized as identity
        self.translators = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(num_layers - 1)
        ])

        # Initialize as identity transform (equivalent to logit lens at start)
        for translator in self.translators:
            nn.init.eye_(translator.weight)
            nn.init.zeros_(translator.bias)

    def translate(self, layer_idx: int, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply the tuned lens translator for a given layer.

        Args:
            layer_idx: layer index (0 to num_layers-2)
            hidden_state: [batch, T, D] or [T, D] hidden state from that layer

        Returns:
            translated hidden state in post-norm space, same shape as input
        """
        return self.translators[layer_idx](hidden_state)

    def train_on_model(
        self,
        model_helper,
        texts: List[str],
        tokenizer,
        epochs: int = 3,
        lr: float = 1e-3,
        max_seq_len: int = 512,
        device: Optional[str] = None,
    ) -> List[float]:
        """
        Train the tuned lens translators on a calibration corpus.

        Strategy:
        1. Run model on each text to collect (h_layer, h_final) pairs
        2. Minimize MSE(T_ell(h_layer), h_final) for each layer

        Hidden states are cached to avoid re-running the model each epoch.

        Args:
            model_helper: initialized Qwen3Helper
            texts: list of calibration text strings
            tokenizer: HuggingFace tokenizer for the model
            epochs: training epochs over cached hidden states
            lr: learning rate for Adam
            max_seq_len: truncate texts to this many tokens
            device: torch device ('cuda' or 'cpu')

        Returns:
            list of average losses per epoch
        """
        if device is None:
            device = next(model_helper.model.parameters()).device

        # Match dtype of the model (typically bfloat16) to avoid dtype mismatch
        # when the hidden states are passed through the linear translators.
        model_dtype = next(model_helper.model.parameters()).dtype
        self.to(device=device, dtype=model_dtype)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print(f"Collecting hidden states from {len(texts)} calibration texts...")

        # Phase 1: collect (h_layers, h_final) pairs, no grad needed
        all_layer_states = [[] for _ in range(self.num_layers - 1)]  # one list per layer
        all_final_states = []

        with torch.no_grad():
            for i, text in enumerate(texts):
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_seq_len,
                    truncation=True,
                ).input_ids.to(device)

                hidden_states = model_helper.get_layer_hidden_states(tokens)

                # h_final: post-norm, shape [1, T, D] → [T, D]
                h_final = hidden_states[-1][0].cpu()  # move to CPU to save GPU mem
                all_final_states.append(h_final)

                # h_layer: pre-norm for layers 0..N-2
                for layer_idx in range(self.num_layers - 1):
                    h_layer = hidden_states[layer_idx][0].cpu()
                    all_layer_states[layer_idx].append(h_layer)

                del hidden_states
                torch.cuda.empty_cache()

                if (i + 1) % 10 == 0:
                    print(f"  Collected {i + 1}/{len(texts)} texts")

        # Concatenate across all texts: [T_total, D]
        all_final_states = torch.cat(all_final_states, dim=0).to(device)
        for layer_idx in range(self.num_layers - 1):
            all_layer_states[layer_idx] = torch.cat(
                all_layer_states[layer_idx], dim=0
            ).to(device)

        T_total = all_final_states.shape[0]
        print(f"Training on {T_total} token positions across {len(texts)} texts")

        # Phase 2: train translators
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            optimizer.zero_grad()

            for layer_idx in range(self.num_layers - 1):
                h_layer = all_layer_states[layer_idx]
                h_final = all_final_states

                h_translated = self.translators[layer_idx](h_layer)
                loss = criterion(h_translated, h_final)
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            avg_loss = total_loss / (self.num_layers - 1)
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch + 1}/{epochs} — avg MSE loss: {avg_loss:.6f}")

        # Move back to eval mode
        self.eval()
        return epoch_losses

    def save(self, path: str):
        """Save tuned lens weights and metadata."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "state_dict": self.state_dict(),
        }, path)
        print(f"Saved tuned lens weights to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None,
             dtype: torch.dtype = torch.bfloat16) -> "TunedLens":
        """Load tuned lens weights from disk.

        Args:
            path: path to .pt checkpoint
            device: torch device string (e.g. 'cuda', 'cpu')
            dtype: dtype to cast translators to (default bfloat16 to match Qwen3)
        """
        checkpoint = torch.load(path, map_location=device or "cpu")
        lens = cls(
            num_layers=checkpoint["num_layers"],
            hidden_size=checkpoint["hidden_size"],
        )
        lens.load_state_dict(checkpoint["state_dict"])
        lens.eval()
        lens.to(dtype=dtype)
        if device:
            lens.to(device=device)
        print(f"Loaded tuned lens weights from {path} (dtype={dtype})")
        return lens
