"""
SelfPromptedClassifier — re-prompt the SAME generation model to extract the
answer from the partial reasoning trace.

Idea
----
Given the partial trace so far, build a new prompt:

    [original problem]
    [reasoning trace so far, truncated]
    Based on the reasoning above, the final answer is: \\boxed{

and run a SHORT (e.g. 10–20 tokens) greedy generation.  If the model
confidently produces a valid answer (extractable via voting.extract_answer
with high-probability tokens), report can_answer=True.

This re-uses the engine's own model — no second checkpoint loaded.

Cost
----
One short forward pass (≤ 30 tokens, no KV cache shared) per call.  Use
`check_every` in the strategy to bound cost.
"""

import torch
import torch.nn.functional as F

from . import AnswerClassifier, ClassifierResult


_PROBE_PROMPT = "\n\nBased on the reasoning above, the final answer is: \\boxed{"


class SelfPromptedClassifier(AnswerClassifier):
    """
    Args:
        helper:              A Qwen3Helper (or compatible) instance — model + tokenizer.
        answer_type:         For type-specific extraction.
        max_probe_tokens:    Max tokens to sample for the probe (small).
        confidence_threshold: Mean log-prob of the probe tokens above which we
                              consider the answer extractable.  Closer to 0 = more
                              confident.  Default -1.0 (≈ avg P=0.37).
    """

    def __init__(
        self,
        helper,
        answer_type: str = "integer",
        max_probe_tokens: int = 20,
        confidence_threshold: float = -1.0,
    ):
        self.helper = helper
        self.answer_type = answer_type
        self.max_probe_tokens = max_probe_tokens
        self.confidence_threshold = confidence_threshold

        # Pre-tokenise the probe suffix once
        self._probe_ids: torch.Tensor = torch.tensor(
            [self.helper.tokenizer.encode(_PROBE_PROMPT, add_special_tokens=False)],
            device=self.helper.device,
        )
        # Tokens that close a \boxed{...} expression — used as stop tokens
        self._close_brace_id: int = self.helper.tokenizer.encode(
            "}", add_special_tokens=False
        )[0]

    @property
    def name(self) -> str:
        return "self_prompted"

    @torch.no_grad()
    def can_answer(self, generated_text: str) -> ClassifierResult:
        # Build the probe input: trace_so_far + probe_suffix
        # We tokenise the trace fresh to avoid stale ids; this is cheap.
        trace_ids = self.helper.tokenize(generated_text)
        full_ids  = torch.cat([trace_ids, self._probe_ids], dim=1)

        # Prefill
        out     = self.helper.model(full_ids, use_cache=True)
        past_kv = out.past_key_values
        logits  = out.logits[:, -1, :].squeeze(0)

        # Greedy short generation, accumulate log-probs
        produced_ids = []
        logprobs     = []
        for _ in range(self.max_probe_tokens):
            log_p   = F.log_softmax(logits.float(), dim=-1)
            next_id = int(log_p.argmax().item())
            produced_ids.append(next_id)
            logprobs.append(float(log_p[next_id].item()))

            if next_id == self._close_brace_id:
                break

            out = self.helper.model(
                torch.tensor([[next_id]], device=self.helper.device),
                past_key_values=past_kv, use_cache=True,
            )
            past_kv = out.past_key_values
            logits  = out.logits[:, -1, :].squeeze(0)

        decoded = self.helper.tokenizer.decode(produced_ids, skip_special_tokens=False)
        # Try to extract a typed answer from "boxed{<decoded>}" — i.e. wrap and parse
        from src.evaluation.voting import extract_answer
        wrapped = "\\boxed{" + decoded.split("}")[0] + "}"
        ans = extract_answer(wrapped, self.answer_type)

        mean_lp = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
        confident = mean_lp >= self.confidence_threshold
        valid     = ans is not None

        return ClassifierResult(
            can_answer=(valid and confident),
            predicted_answer=str(ans) if ans is not None else None,
            confidence=float(min(1.0, max(0.0, (mean_lp + 5.0) / 5.0))),
            reason=(
                f"probe answer={ans!r} mean_log_p={mean_lp:.3f} "
                f"(threshold={self.confidence_threshold}) "
                f"{'OK' if (valid and confident) else 'reject'}"
            ),
        )
