"""
Model helper for Qwen3-4B-Thinking-2507.
Uses the native `output_hidden_states=True` API instead of layer wrappers,
which is both faster and avoids compatibility issues with model internals.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple


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
        print(f"Model loaded: {self.num_layers} layers, device={self.device}")

    def get_layer_hidden_states(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Run a single forward pass and return per-layer hidden states.
        Uses the native output_hidden_states=True API (no wrappers needed).

        Args:
            input_ids: shape [B, T] on self.device

        Returns:
            Tuple of num_layers tensors, each [B, T, hidden_dim].
            Index 0 = output of decoder layer 0 (pre-norm),
            ...
            Index N-2 = output of decoder layer N-2 (pre-norm),
            Index N-1 = output of decoder layer N-1 (POST-norm, already normed).

        Note:
            HF transformers returns (num_layers + 1) hidden states:
              [0]       = embedding output (input to layer 0)
              [1]       = output of layer 0 = input to layer 1
              ...
              [N-1]     = output of layer N-2 = input to layer N-1
              [N]       = output of layer N-1 AFTER final RMSNorm
            We strip the embedding [0] so indices align with layer numbers.
            The LAST entry is post-norm — callers must handle this.
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        all_hs = outputs.hidden_states
        # Strip embedding [0], keep layer outputs + post-norm final
        return all_hs[1:]  # tuple of num_layers tensors

    def project_to_vocab(
        self, hidden_state: torch.Tensor, already_normed: bool = False
    ) -> torch.Tensor:
        """
        Project hidden state to vocabulary probability distribution.

        Args:
            hidden_state: [batch, seq_len, hidden_dim]
            already_normed: if True, skip the final RMSNorm (for post-norm states)

        Returns:
            probs: [batch, seq_len, vocab_size] in float32
        """
        if already_normed:
            logits = self.lm_head(hidden_state).float()
        else:
            normed = self.norm(hidden_state)
            logits = self.lm_head(normed).float()
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
