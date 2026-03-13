"""
Lightweight layer wrappers for capturing residual stream hidden states.
Adapted from LogitLens4LLMs (github.com/zhenyu-02/LogitLens4LLMs).

Unlike the original BlockOutputWrapper which unembeds inside each wrapper
(storing full vocab-size tensors per layer), this only captures the raw
hidden state output. Unembedding is done in bulk afterwards, one layer
at a time, to save memory.
"""

import torch
import torch.nn as nn


class ResidualStreamCapture(nn.Module):
    """Wraps a transformer decoder layer to capture its output hidden state."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.captured_hidden_state = None

    def __getattr__(self, name: str):
        """Proxy attribute access to the wrapped block for attributes like attention_type."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            self.captured_hidden_state = output[0]
        else:
            self.captured_hidden_state = output
        return output

    def reset(self):
        self.captured_hidden_state = None
