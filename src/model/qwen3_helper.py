"""
Model helper for Qwen3-4B-Thinking-2507.
Loads the model, wraps decoder layers for hidden state capture,
and provides methods for logit lens analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional

from src.model.wrappers import ResidualStreamCapture


class Qwen3Helper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        local_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        load_path = local_path or model_name
        print(f"Loading tokenizer from {load_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)

        print(f"Loading model from {load_path} ({dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        self.num_layers = len(self.model.model.layers)
        self.norm = self.model.model.norm
        self.lm_head = self.model.lm_head

        # Wrap each decoder layer
        for i in range(self.num_layers):
            self.model.model.layers[i] = ResidualStreamCapture(
                self.model.model.layers[i]
            )
        print(f"Wrapped {self.num_layers} layers for hidden state capture.")

    def get_layer_hidden_states(
        self, input_ids: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Run a single forward pass and return hidden states from every layer.

        Args:
            input_ids: shape [1, T] on self.device

        Returns:
            dict mapping layer index -> hidden state tensor [1, T, hidden_dim]
        """
        self.reset_all()
        with torch.no_grad():
            self.model(input_ids)

        hidden_states = {}
        for i, layer in enumerate(self.model.model.layers):
            hidden_states[i] = layer.captured_hidden_state
        return hidden_states

    def project_to_vocab(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply final norm + lm_head + softmax to get probability distribution.

        Args:
            hidden_state: [batch, seq_len, hidden_dim]

        Returns:
            probs: [batch, seq_len, vocab_size] in float32
        """
        normed = self.norm(hidden_state)
        logits = self.lm_head(normed).float()  # float32 for numerical stability
        return torch.softmax(logits, dim=-1)

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids on device."""
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs.input_ids.to(self.device)

    def tokenize_chat(self, user_message: str) -> torch.Tensor:
        """Apply chat template and return input_ids on device."""
        messages = [{"role": "user", "content": user_message}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs.input_ids.to(self.device)

    def reset_all(self):
        """Clear all captured hidden states."""
        for layer in self.model.model.layers:
            layer.reset()
