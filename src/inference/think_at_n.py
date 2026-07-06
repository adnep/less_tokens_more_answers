"""
think@n: test-time scaling with pluggable sample ranking.

Ranking modes
-------------
dtr        : prefix Deep-Thinking Ratio  (needs HF model + intermediate layers)
logprob    : mean log P(y_t) over prefix (free — stored by vLLM during generation)
selfcert   : mean KL(uniform ‖ p_t) over prefix — Kang et al. (2025) Eq. 10
             (needs HF model, final layer only — much cheaper than dtr)
neg_entropy: mean negative entropy -H(p_t) over prefix (old selfcert definition)
             (needs HF model, final layer only)

All modes produce a scalar per sample; top-eta samples are kept and
majority-voted to produce a final answer.
"""

import torch
import math
from typing import List, Optional
from collections import Counter

from src.model.qwen3_helper import Qwen3Helper
from src.inference.sampler import GeneratedSample
from src.evaluation.voting import normalize_numeric_answer

RANKING_MODES = ("dtr", "logprob", "selfcert", "neg_entropy")


# ──────────────────────────────────────────────────────────────────────────────
# Per-mode scoring functions
# ──────────────────────────────────────────────────────────────────────────────

def _score_logprob(sample: GeneratedSample, prefix_length: int) -> float:
    """Mean log P(y_t) over the first prefix_length generated tokens.
    Returns 0.0 if log probs were not stored (old samples file)."""
    lps = sample.token_logprobs
    if not lps:
        return 0.0
    lps = lps[:prefix_length]
    return sum(lps) / len(lps)


def _forward_pass(helper: Qwen3Helper, sample: GeneratedSample, prefix_length: Optional[int]):
    """Run one forward pass and return (all_logits, prompt_len, actual_gen_len)."""
    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    actual     = gen_ids.shape[1] if prefix_length is None else min(prefix_length, gen_ids.shape[1])
    full_ids   = torch.cat([prompt_ids, gen_ids[:, :actual]], dim=1)
    with torch.no_grad():
        all_logits = helper.model(full_ids).logits  # [1, T_full, V], bfloat16
    return all_logits, prompt_ids.shape[1], actual


def _score_selfcert(
    helper: Qwen3Helper,
    sample: GeneratedSample,
    prefix_length: Optional[int],
) -> float:
    """Mean SC = KL(uniform ‖ p_t) over the generated prefix.

    SC(t) = -log|V| - (1/|V|) Σ_v log p_t(v)   (Kang et al. 2025, Eq. 10)

    Range: [0, ∞).  0 = uniform (maximum uncertainty).  Higher = more confident.
    Full sequence is used when prefix_length is None for a stable offline estimate.
    """
    SLICE = 128
    all_logits, prompt_len, actual = _forward_pass(helper, sample, prefix_length)

    sc_sum   = 0.0
    n_tokens = 0
    for start in range(0, actual, SLICE):
        end   = min(start + SLICE, actual)
        sl    = all_logits[0, prompt_len + start : prompt_len + end].float()  # [s, V]
        log_p = torch.log_softmax(sl, dim=-1)
        sc    = -math.log(log_p.shape[-1]) - log_p.mean(dim=-1)              # [s]
        sc_sum   += sc.sum().item()
        n_tokens += sc.shape[0]
        del sl, log_p, sc

    del all_logits
    torch.cuda.empty_cache()
    return sc_sum / n_tokens if n_tokens > 0 else 0.0


def _score_neg_entropy(
    helper: Qwen3Helper,
    sample: GeneratedSample,
    prefix_length: Optional[int],
) -> float:
    """Mean negative entropy -H(p_t) over the generated prefix.

    Range: (-∞, 0].  Closer to 0 = more confident.
    This was the original 'selfcert' definition before the KL formulation.
    """
    SLICE = 128
    all_logits, prompt_len, actual = _forward_pass(helper, sample, prefix_length)

    entropy_sum = 0.0
    n_tokens    = 0
    for start in range(0, actual, SLICE):
        end   = min(start + SLICE, actual)
        sl    = all_logits[0, prompt_len + start : prompt_len + end].float()  # [s, V]
        log_p = torch.log_softmax(sl, dim=-1)
        ent   = -(log_p.exp() * log_p).sum(dim=-1)                           # [s]
        entropy_sum += ent.sum().item()
        n_tokens    += ent.shape[0]
        del sl, log_p, ent

    del all_logits
    torch.cuda.empty_cache()
    mean_entropy = entropy_sum / n_tokens if n_tokens > 0 else 0.0
    return -mean_entropy


def _score_dtr(
    scorer,
    helper: Qwen3Helper,
    sample: GeneratedSample,
    prefix_length: int,
) -> float:
    """Prefix DTR score (existing logic, factored out)."""
    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    return scorer.compute_prefix_dtr(prompt_ids, gen_ids, prefix_length)


# ──────────────────────────────────────────────────────────────────────────────
# Main think@n function
# ──────────────────────────────────────────────────────────────────────────────

def think_at_n(
    samples: List[GeneratedSample],
    model_helper: Optional[Qwen3Helper] = None,
    gamma: float = 0.5,
    rho: float = 0.85,
    eta: float = 0.5,
    prefix_length: int = 50,
    answer_extractor=None,
    tuned_lens=None,
    ranking_mode: str = "dtr",
) -> dict:
    """
    Apply think@n selection to samples for one problem.

    Args:
        samples:        list of GeneratedSample for one problem
        model_helper:   Qwen3Helper (required for dtr / selfcert, not for logprob)
        gamma:          DTR settling threshold (used only when ranking_mode='dtr')
        rho:            DTR depth fraction   (used only when ranking_mode='dtr')
        eta:            fraction of samples to keep (top by chosen metric)
        prefix_length:  tokens used for scoring
        answer_extractor: callable(str) -> Optional[str]
        tuned_lens:     optional TunedLens (used only when ranking_mode='dtr')
        ranking_mode:   one of 'dtr', 'logprob', 'selfcert', 'neg_entropy'

    Returns:
        dict with keys:
            selected_answer  – majority-voted answer from top-eta samples
            scores           – list of (sample_idx, score) sorted descending
            all_answers      – extracted answers from selected samples
            n_selected       – how many samples were kept
            ranking_mode     – which mode was used
    """
    if ranking_mode not in RANKING_MODES:
        raise ValueError(f"ranking_mode must be one of {RANKING_MODES}, got {ranking_mode!r}")

    # Build DTR scorer only if needed
    scorer = None
    if ranking_mode == "dtr":
        from src.dtr.dtr_scorer import DTRScorer
        scorer = DTRScorer(model_helper, gamma=gamma, rho=rho, tuned_lens=tuned_lens)

    # Score every sample
    scores = []
    for sample in samples:
        if ranking_mode == "dtr":
            s = _score_dtr(scorer, model_helper, sample, prefix_length)
        elif ranking_mode == "logprob":
            s = _score_logprob(sample, prefix_length)
        elif ranking_mode == "selfcert":
            # KL(uniform ‖ p_t) — use full sequence for a stable offline estimate
            s = _score_selfcert(model_helper, sample, prefix_length=None)
        else:  # neg_entropy — use full sequence for a stable offline estimate
            s = _score_neg_entropy(model_helper, sample, prefix_length=None)
        scores.append((sample.sample_idx, s))

    # Sort descending, keep top eta fraction
    scores.sort(key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(len(samples) * eta))
    selected_indices = {idx for idx, _ in scores[:n_keep]}

    # Extract and normalise answers from selected samples
    answers = []
    for sample in samples:
        if sample.sample_idx not in selected_indices:
            continue
        ans = answer_extractor(sample.generated_text) if answer_extractor else sample.answer_text
        if ans:
            ans = normalize_numeric_answer(ans)
            answers.append(ans)

    # Majority vote
    selected_answer = Counter(answers).most_common(1)[0][0] if answers else ""

    return {
        "selected_answer": selected_answer,
        "scores":          scores,
        # keep dtr_scores as alias so existing downstream code doesn't break
        "dtr_scores":      scores,
        "all_answers":     answers,
        "n_selected":      n_keep,
        "ranking_mode":    ranking_mode,
    }
