"""
DTR (Deep Thinking Ratio) scorer.
Orchestrates the full pipeline: logit lens -> JSD -> settling depth -> DTR.
"""

import math
import torch
from typing import Optional

from src.model.qwen3_helper import Qwen3Helper
from src.dtr.logit_lens import compute_jsd_per_layer
from src.dtr.settling_depth import compute_settling_depth


class DTRScorer:
    def __init__(
        self,
        model_helper: Qwen3Helper,
        gamma: float = 0.5,
        rho: float = 0.85,
    ):
        """
        Args:
            model_helper: initialized Qwen3Helper
            gamma: settling threshold for JSD (paper default: 0.5)
            rho: depth fraction for deep-thinking regime (paper default: 0.85)
        """
        self.helper = model_helper
        self.gamma = gamma
        self.rho = rho
        self.num_layers = model_helper.num_layers
        self.deep_threshold = math.ceil(rho * self.num_layers)

    def compute_dtr(
        self,
        input_ids: torch.Tensor,
        generated_token_start: int,
        generated_token_end: Optional[int] = None,
    ) -> dict:
        """
        Compute DTR for a sequence of generated tokens.

        Args:
            input_ids: [1, T_total] full sequence (prompt + generated tokens)
            generated_token_start: index where generated tokens begin
            generated_token_end: index where generated tokens end (exclusive).
                If None, uses all tokens from start to end of sequence.

        Returns:
            dict with:
                - dtr: float, the deep thinking ratio
                - settling_depths: [T_gen] settling depth per generated token
                - jsd_matrix: [T_gen, L] JSD values
                - is_deep: [T_gen] boolean mask of deep-thinking tokens
                - deep_threshold: int, layer index threshold for deep thinking
        """
        if generated_token_end is None:
            generated_token_end = input_ids.shape[1]

        token_positions = slice(generated_token_start, generated_token_end)
        T_gen = generated_token_end - generated_token_start

        # Use batch mode for short sequences, sequential for long
        batch_layers = T_gen <= 100

        # Single forward pass with output_hidden_states=True
        hidden_states = self.helper.get_layer_hidden_states(input_ids)

        # Compute JSD between each layer and final layer
        jsd_tensor = compute_jsd_per_layer(
            hidden_states,
            self.helper.norm,
            self.helper.lm_head,
            token_positions=token_positions,
            batch_layers=batch_layers,
        )

        del hidden_states

        # Compute settling depth per token
        settling_depths = compute_settling_depth(jsd_tensor, gamma=self.gamma)

        # Classify deep-thinking tokens
        is_deep = settling_depths >= self.deep_threshold

        # Compute DTR
        dtr = is_deep.float().mean().item()

        return {
            "dtr": dtr,
            "settling_depths": settling_depths,
            "jsd_matrix": jsd_tensor,
            "is_deep": is_deep,
            "deep_threshold": self.deep_threshold,
        }

    def compute_prefix_dtr(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        prefix_length: int = 50,
    ) -> float:
        """
        Compute DTR for a prefix of generated tokens.
        Used by think@n for early rejection.

        Args:
            prompt_ids: [1, T_prompt] the prompt token ids
            generated_ids: [1, T_gen] the generated token ids
            prefix_length: number of generated tokens to use for DTR

        Returns:
            dtr: float
        """
        actual_prefix = min(prefix_length, generated_ids.shape[1])
        prefix_ids = generated_ids[:, :actual_prefix]

        # Concatenate prompt + prefix
        full_ids = torch.cat([prompt_ids, prefix_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        result = self.compute_dtr(
            full_ids,
            generated_token_start=prompt_len,
            generated_token_end=prompt_len + actual_prefix,
        )
        return result["dtr"]
