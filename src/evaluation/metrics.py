"""
Evaluation metrics: accuracy, pass@n.
"""

from typing import List, Optional


def accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute accuracy as fraction of exact matches."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p is not None and str(p).strip() == str(r).strip()
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
        ref_str = str(ref).strip()
        if any(str(a).strip() == ref_str for a in answers if a is not None):
            correct += 1
    return correct / len(references)
