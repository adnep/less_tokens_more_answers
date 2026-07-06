"""
LP metric: LP(t) = log P(y_t | y_{<t}, x)

Range: (−∞, 0].  Closer to 0 = model was very confident about THIS token.

This is the per-token log-probability of the SAMPLED token — one entry
from the same softmax distribution that SC uses.  Cheaper signal (no
sum over vocabulary) but noisier (depends on which token was drawn).
"""

import torch
import torch.nn.functional as F
from . import MetricComputer


class LPMetric(MetricComputer):
    """
    LP = log p(y_t) = log_softmax(logits_t)[token_id].

    Higher (closer to 0) means the model assigned high probability to
    the token it actually generated — it was confident in this choice.
    """

    @property
    def name(self) -> str:
        return "lp"

    @property
    def higher_is_better(self) -> bool:
        return True  # closer to 0 = more confident

    def compute(self, logits: torch.Tensor, token_id: int) -> float:
        """
        Args:
            logits:   [vocab_size] float32 raw logits.
            token_id: The id of the token that was sampled at this step.
        Returns:
            log P(token_id | context) in (−∞, 0].
        """
        log_probs = F.log_softmax(logits.float(), dim=-1)  # [V]
        return log_probs[token_id].item()
