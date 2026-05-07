"""
Answer extraction and majority voting.
"""

import re
from typing import List, Optional
from collections import Counter


def _extract_boxed_contents(text: str) -> list:
    """
    Extract all \\boxed{...} contents, correctly handling nested braces.
    e.g. \\boxed{\\frac{1}{576}} returns ['\\frac{1}{576}']
    """
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        # Start after the opening brace
        start = idx + len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j - 1])
        i = j
    return results


def extract_answer_math(text: str) -> Optional[str]:
    """
    Extract answer from a region of model output (typically post-</think> section).
    Accepts: last \\boxed{...} found, or "answer is X" pattern.
    Returns None if no clear answer marker is found.
    """
    # Try \boxed{...} with proper nested brace handling
    boxed = _extract_boxed_contents(text)
    if boxed:
        return boxed[-1].strip()

    # Try "answer is X" pattern
    answer_pattern = re.search(
        r"(?:final\s+)?answer\s+is\s*[:\s]*(\d+)", text, re.IGNORECASE
    )
    if answer_pattern:
        return answer_pattern.group(1)

    # No clear answer found — don't guess from random numbers
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


def extract_answer_yesno(text: str) -> Optional[str]:
    """Extract a Yes/No answer from model output.

    Priority:
    1. Last \\boxed{Yes} or \\boxed{No}
    2. Explicit "answer is Yes/No" pattern
    3. Last standalone Yes/No in the text (case-insensitive)
    Returns capitalised "Yes" or "No", or None if not found.
    """
    # 1. Check \boxed{Yes} / \boxed{No}
    boxed = _extract_boxed_contents(text)
    if boxed:
        last = boxed[-1].strip().lower()
        if last == "yes":
            return "Yes"
        if last == "no":
            return "No"

    # 2. Explicit answer pattern
    match = re.search(
        r"(?:final\s+)?(?:answer|result)\s+is\s*[:\s]*(yes|no)\b",
        text, re.IGNORECASE
    )
    if match:
        return match.group(1).capitalize()

    # 3. Last standalone yes/no in the text
    matches = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()

    return None


def extract_answer(text: str, answer_type: str) -> Optional[str]:
    """Extract answer based on type.

    Strategy:
    - If the model produced </think>, ONLY look at the text after </think>.
      Boxes inside <think> are intermediate steps, not the final answer.
      If nothing is found after </think> → return None (unanswered).
    - If the model did NOT produce </think> (truncated or no think block),
      search the full text as a fallback.

    This prevents mid-reasoning \\boxed{} from being mistaken for the answer.
    """
    has_think_close = "</think>" in text
    if has_think_close:
        answer_region = text.split("</think>", 1)[1]
    else:
        # No </think> found: model may have been truncated or skipped thinking.
        # Fall back to full text.
        answer_region = text

    if answer_type == "integer":
        return extract_answer_math(answer_region)
    elif answer_type == "choice":
        return extract_answer_choice(answer_region)
    elif answer_type == "expression":
        # For HMMT-style: extract boxed content as-is (may be LaTeX expression)
        return extract_answer_math(answer_region)
    elif answer_type == "yes_no":
        return extract_answer_yesno(answer_region)
    return answer_region.strip()


def expressions_equal(pred: Optional[str], ref: Optional[str]) -> bool:
    """
    Compare two mathematical expressions for equality.
    Used for HMMT-style answers (fractions, surds, expressions).

    Strategy:
    1. Exact string match (handles most integer/simple cases)
    2. Numeric float comparison via sympy parsing
    """
    if pred is None or ref is None:
        return False

    # Normalize both: strip whitespace and leading zeros for pure integers
    pred_norm = normalize_numeric_answer(pred.strip())
    ref_norm = normalize_numeric_answer(ref.strip())

    if pred_norm == ref_norm:
        return True

    # Try symbolic comparison via sympy
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N

        pred_expr = parse_latex(pred_norm)
        ref_expr = parse_latex(ref_norm)

        # Check symbolic equality
        if simplify(pred_expr - ref_expr) == 0:
            return True

        # Check numeric equality (handles cases like sqrt(2) vs 1.41...)
        pred_val = float(N(pred_expr))
        ref_val = float(N(ref_expr))
        return abs(pred_val - ref_val) < 1e-4

    except Exception:
        # sympy couldn't parse — fall back to string match only
        return False


def normalize_numeric_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize numeric answers by removing leading zeros.

    "033" and "33" should be treated as the same answer.
    Only strips leading zeros from pure numeric answers.
    """
    if answer is None or not answer:
        return answer

    # If it's a pure number, strip leading zeros
    if answer.strip().isdigit():
        return str(int(answer.strip()))

    # Otherwise keep as-is (e.g., "A", "B", "pi", etc.)
    return answer


def majority_vote(answers: List[str]) -> Optional[str]:
    """Return the most common answer."""
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]
