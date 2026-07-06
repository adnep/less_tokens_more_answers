"""
Evaluation metrics: accuracy, pass@n.
"""

import re
from typing import List, Optional
from src.evaluation.voting import normalize_numeric_answer


def _normalize_for_comparison(s: str | None) -> str:
    """Normalize answer for comparison.

    - Lowercases (so "Yes"/"yes"/"YES" all compare equal for yes_no datasets;
      safe for integers and A/B/C/D choices which are normalised upstream)
    - Strips leading/trailing whitespace
    - Collapses all internal whitespace (LaTeX spaces are semantically meaningless:
      '\\frac{9 \\sqrt{23}}{23}' == '\\frac{9\\sqrt{23}}{23}')
    - Strips leading zeros from pure integers
    """
    if s is None:
        return s
    s = str(s).strip().lower()
    # Remove all internal whitespace so LaTeX spacing variants compare equal
    s = re.sub(r'\s+', '', s)
    normalized = normalize_numeric_answer(s).replace("dfrac", "frac")
    return normalized if normalized else s


def accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute accuracy as fraction of exact matches."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p is not None and _normalize_for_comparison(p) == _normalize_for_comparison(r)
    )
    return correct / len(references)


def pass_at_n(
    all_sample_answers: List[List[Optional[str]]],
    references: List[str],
) -> float:
    """
    Compute pass@n: fraction of problems where at least one sample is correct.

    Args:
        all_sample_answers: outer list per problem, inner list of extracted answers
        references: ground truth answers
    """
    if not references:
        return 0.0
    correct = 0
    for answers, ref in zip(all_sample_answers, references):
        ref_normalized = _normalize_for_comparison(ref)
        if any(_normalize_for_comparison(a) == ref_normalized for a in answers if a is not None):
            correct += 1
    return correct / len(references)
