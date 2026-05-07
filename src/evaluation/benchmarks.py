"""
Benchmark dataset loaders.
Provides a common interface for loading AIME 2024, GPQA-Diamond, HMMT 2025,
and local str / math datasets.
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional


DATA_DIR = "/storage/praha1/home/adnep/deep-thinking-replication/datasets"


@dataclass
class BenchmarkProblem:
    id: str
    question: str
    answer: str
    answer_type: str  # "integer" | "float" | "choice" | "expression" | "yes_no"
    source: str


# ── HuggingFace loaders ───────────────────────────────────────────────────────

def load_aime_2024() -> List[BenchmarkProblem]:
    """
    Load AIME 2024 problems from HuggingFace datasets.
    AIME answers are integers 0-999.
    """
    from datasets import load_dataset

    # Try multiple possible dataset sources
    try:
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    except Exception:
        try:
            ds = load_dataset("qq8933/AIME_2024", split="train")
        except Exception:
            print("Could not load AIME 2024 from HuggingFace.")
            print("Please provide a local JSONL file with 'question' and 'answer' fields.")
            return []

    problems = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        answer = str(row.get("answer", row.get("solution", "")))
        # Extract numeric answer if embedded in text
        numeric = re.search(r"\b(\d+)\b", answer)
        if numeric:
            answer = numeric.group(1)

        problems.append(BenchmarkProblem(
            id=f"aime2024_{i}",
            question=question,
            answer=answer,
            answer_type="integer",
            source="AIME 2024",
        ))
    return problems


def load_gpqa_diamond(seed: int = 42) -> List[BenchmarkProblem]:
    """
    Load GPQA-Diamond multiple choice problems.

    The dataset stores answers as raw text (not A/B/C/D). This function
    shuffles the four options, assigns letters A-D, includes the choices
    in the question text, and stores the correct letter as the answer.

    seed: controls shuffle so results are reproducible.
    """
    import random
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    problems = []
    for i, row in enumerate(ds):
        question = row["Question"]
        correct_text = row["Correct Answer"]
        wrong_texts = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]

        # Shuffle all four options and track which letter is correct
        options = [correct_text] + wrong_texts
        rng.shuffle(options)
        correct_letter = "ABCD"[options.index(correct_text)]

        # Build question with choices embedded
        choices_text = "\n".join(
            f"{letter}. {text}"
            for letter, text in zip("ABCD", options)
        )
        question_with_choices = f"{question}\n\n{choices_text}"

        problems.append(BenchmarkProblem(
            id=f"gpqa_{i}",
            question=question_with_choices,
            answer=correct_letter,
            answer_type="choice",
            source="GPQA-Diamond",
        ))
    return problems


def load_hmmt_2025() -> List[BenchmarkProblem]:
    """
    Load HMMT February 2025 problems from HuggingFace (FlagEval/HMMT_2025).
    Answers are free-form mathematical expressions (integers, fractions, surds, etc.)
    """
    from datasets import load_dataset

    ds = load_dataset("FlagEval/HMMT_2025", split="train")
    problems = []
    for row in ds:
        problems.append(BenchmarkProblem(
            id=f"hmmt_{row['id']}",
            question=row["question"],
            answer=str(row["answer"]),
            answer_type="expression",
            source="HMMT 2025",
        ))
    return problems


# ── Local JSONL loaders ───────────────────────────────────────────────────────

def _read_jsonl(path: str) -> List[dict]:
    """Read a local JSONL file, return list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Set REASONING_DATA_DIR env var if the reasoning-analysis repo is elsewhere."
        )
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_char_occur() -> List[BenchmarkProblem]:
    """
    Local: how many times does character X appear in word Y?
    Answer is a non-negative integer (stored as string in the file).
    """
    rows = _read_jsonl(os.path.join(DATA_DIR, "str", "char_occur.jsonl"))
    return [
        BenchmarkProblem(
            id=f"char_occur_{row['question_id']}",
            question=row["question"],
            answer=str(row["answer"]),
            answer_type="integer",
            source="char_occur",
        )
        for row in rows
    ]


def load_distinct_char() -> List[BenchmarkProblem]:
    """
    Local: how many distinct characters are in word Y?
    Answer is a non-negative integer.
    """
    rows = _read_jsonl(os.path.join(DATA_DIR, "str", "distinct_char.jsonl"))
    return [
        BenchmarkProblem(
            id=f"distinct_char_{row['question_id']}",
            question=row["question"],
            answer=str(row["answer"]),
            answer_type="integer",
            source="distinct_char",
        )
        for row in rows
    ]


def load_word_len() -> List[BenchmarkProblem]:
    """
    Local: how many characters are in word Y?
    Answer is a non-negative integer.
    """
    rows = _read_jsonl(os.path.join(DATA_DIR, "str", "word_len.jsonl"))
    return [
        BenchmarkProblem(
            id=f"word_len_{row['question_id']}",
            question=row["question"],
            answer=str(row["answer"]),
            answer_type="integer",
            source="word_len",
        )
        for row in rows
    ]


def load_substring_occur() -> List[BenchmarkProblem]:
    """
    Local: is substring X part of word Y?
    Answer is 'Yes' or 'No'.
    Note: question_id is a string like '0_true', not an integer.
    """
    rows = _read_jsonl(os.path.join(DATA_DIR, "str", "substring_occur.jsonl"))
    return [
        BenchmarkProblem(
            id=f"substring_occur_{row['question_id']}",
            question=row["question"],
            answer=str(row["answer"]),   # "Yes" or "No"
            answer_type="yes_no",
            source="substring_occur",
        )
        for row in rows
    ]


def load_arithmetic_stress_test() -> List[BenchmarkProblem]:
    """
    Local: multi-step arithmetic sequences.
    Uses 'problem' field (not 'question'). Answer is a raw integer.
    Has 'category' and 'difficulty' metadata fields (not used for eval).
    """
    rows = _read_jsonl(os.path.join(DATA_DIR, "math", "arithmetic_stress_test.jsonl"))
    return [
        BenchmarkProblem(
            id=f"arith_{i}",
            question=row["problem"],        # note: 'problem', not 'question'
            answer=str(row["answer"]),      # cast int → str for uniform handling
            answer_type="integer",
            source="arithmetic_stress_test",
        )
        for i, row in enumerate(rows)
    ]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_benchmark(name: str) -> List[BenchmarkProblem]:
    """Load a benchmark by name."""
    loaders = {
        # HuggingFace benchmarks
        "aime24":       load_aime_2024,
        "gpqa_diamond": load_gpqa_diamond,
        "hmmt2025":     load_hmmt_2025,
        # Local str_based datasets
        "char_occur":         load_char_occur,
        "distinct_char":      load_distinct_char,
        "word_len":           load_word_len,
        "substring_occur":    load_substring_occur,
        # Local math_based datasets
        "arithmetic_stress_test": load_arithmetic_stress_test,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown benchmark: '{name}'.\n"
            f"Available: {list(loaders.keys())}"
        )
    return loaders[name]()


# ── Prompt formatting ─────────────────────────────────────────────────────────

_TERSE_SUFFIX = (
    " Think concisely: no filler, no restating the problem, direct logical steps only."
)


def format_prompt(problem: BenchmarkProblem, model_name: str = "", terse: bool = False) -> str:
    """Format a problem into a model prompt.

    Args:
        problem:    the benchmark problem
        model_name: unused, kept for backwards compatibility
        terse:      if True, append an instruction to reason concisely.
    """
    suffix = _TERSE_SUFFIX if terse else ""

    if problem.answer_type == "integer":
        return (
            f"Solve the following problem. "
            f"Give your final answer as a single integer in Answer: \\boxed{{...}}.{suffix}\n\n"
            f"Problem: {problem.question}\n"
        )
    elif problem.answer_type == "choice":
        return (
            f"Answer the following question. "
            f"Give your final answer as a single letter (A, B, C, or D) in Answer: \\boxed{{...}}.{suffix}\n\n"
            f"Question: {problem.question}\n"
        )
    elif problem.answer_type == "expression":
        return (
            f"Solve the following math competition problem. "
            f"Give your final answer in \\boxed{{...}}. "
            f"Your answer may be a number, fraction, or expression — write it in simplified exact form.{suffix}\n\n"
            f"Problem: {problem.question}\n"
        )
    elif problem.answer_type == "yes_no":
        return (
            f"Answer the following question with exactly Yes or No. "
            f"Give your final answer in \\boxed{{Yes}} or \\boxed{{No}}.{suffix}\n\n"
            f"Question: {problem.question}\n"
        )
    return problem.question
