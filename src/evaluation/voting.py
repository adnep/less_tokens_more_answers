"""
Answer extraction and majority voting.
"""

import re
from typing import List, Optional
from collections import Counter


def extract_answer_math(text: str) -> Optional[str]:
    """
    Extract numeric answer from model output.
    Looks for \\boxed{}, "final answer is X", or last standalone number.
    """
    # Try \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Try "answer is X" pattern
    answer_pattern = re.search(
        r"(?:final\s+)?answer\s+is\s*[:\s]*(\d+)", text, re.IGNORECASE
    )
    if answer_pattern:
        return answer_pattern.group(1)

    # Try "= X" at end of reasoning
    equals = re.findall(r"=\s*(\d+)\s*$", text, re.MULTILINE)
    if equals:
        return equals[-1]

    # Last standalone integer
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return numbers[-1]

    return None


def extract_answer_choice(text: str) -> Optional[str]:
    """Extract multiple choice answer (A/B/C/D) from model output."""
    # Look for explicit answer patterns
    patterns = [
        r"(?:answer|choice)\s+is\s*[:\s]*([A-D])\b",
        r"\b([A-D])\s*[\.\)]\s*$",
        r"\\boxed\{([A-D])\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last standalone letter A-D
    letters = re.findall(r"\b([A-D])\b", text)
    if letters:
        return letters[-1].upper()

    return None


def extract_answer(text: str, answer_type: str) -> Optional[str]:
    """Extract answer based on type."""
    # Split off thinking part if present
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    if answer_type == "integer":
        return extract_answer_math(text)
    elif answer_type == "choice":
        return extract_answer_choice(text)
    return text.strip()


def majority_vote(answers: List[str]) -> Optional[str]:
    """Return the most common answer."""
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]
