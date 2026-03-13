"""
think@n: test-time scaling strategy that selects high-DTR samples.

Two-phase approach:
  Phase 1: Generate n samples per problem using vLLM (fast)
  Phase 2: Compute prefix DTR using HF model with layer wrappers, select top-eta
"""

import torch
from typing import List, Optional
from collections import Counter

from src.model.qwen3_helper import Qwen3Helper
from src.dtr.dtr_scorer import DTRScorer
from src.inference.sampler import GeneratedSample


def think_at_n(
    samples: List[GeneratedSample],
    model_helper: Qwen3Helper,
    gamma: float = 0.5,
    rho: float = 0.85,
    eta: float = 0.5,
    prefix_length: int = 50,
    answer_extractor=None,
) -> dict:
    """
    Apply think@n selection to a set of samples for one problem.

    Args:
        samples: list of GeneratedSample for one problem
        model_helper: initialized Qwen3Helper
        gamma: settling threshold
        rho: depth fraction
        eta: fraction of samples to keep (top DTR)
        prefix_length: number of generated tokens for DTR estimation
        answer_extractor: callable(str) -> str to extract answer from text.
            If None, uses the raw answer_text field.

    Returns:
        dict with:
            - selected_answer: str, majority-voted answer
            - dtr_scores: list of (sample_idx, dtr) tuples sorted desc
            - all_answers: list of extracted answers from selected samples
            - n_selected: number of samples kept
    """
    scorer = DTRScorer(model_helper, gamma=gamma, rho=rho)

    # Compute prefix DTR for each sample
    dtr_scores = []
    for sample in samples:
        prompt_ids = torch.tensor(
            [sample.prompt_token_ids], device=model_helper.device
        )
        gen_ids = torch.tensor(
            [sample.generated_token_ids], device=model_helper.device
        )

        dtr = scorer.compute_prefix_dtr(prompt_ids, gen_ids, prefix_length)
        dtr_scores.append((sample.sample_idx, dtr))

    # Sort by DTR descending
    dtr_scores.sort(key=lambda x: x[1], reverse=True)

    # Keep top eta fraction
    n_keep = max(1, int(len(samples) * eta))
    selected_indices = {idx for idx, _ in dtr_scores[:n_keep]}

    # Extract answers from selected samples
    answers = []
    for sample in samples:
        if sample.sample_idx in selected_indices:
            if answer_extractor:
                ans = answer_extractor(sample.generated_text)
            else:
                ans = sample.answer_text
            if ans:
                answers.append(ans)

    # Majority vote
    if answers:
        counter = Counter(answers)
        selected_answer = counter.most_common(1)[0][0]
    else:
        selected_answer = ""

    return {
        "selected_answer": selected_answer,
        "dtr_scores": dtr_scores,
        "all_answers": answers,
        "n_selected": n_keep,
    }
