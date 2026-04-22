"""
Benchmark dataset loaders.
Provides a common interface for loading AIME 2024/2025 and GPQA-Diamond problems.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkProblem:
    id: str
    question: str
    answer: str
    answer_type: str  # "integer", "float", "choice"
    source: str


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


def load_benchmark(name: str) -> List[BenchmarkProblem]:
    """Load a benchmark by name."""
    loaders = {
        "aime24": load_aime_2024,
        "gpqa_diamond": load_gpqa_diamond,
        "hmmt2025": load_hmmt_2025,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(loaders.keys())}")
    return loaders[name]()


def format_prompt(problem: BenchmarkProblem, model_name: str = "") -> str:
    """Format a problem into a model prompt."""
    if problem.answer_type == "integer":
        return (
            f"Solve the following math competition problem. "
            f"Give your final answer as a single integer in Answer: \\boxed{{...}}.\n\n"
            f"Problem: {problem.question}\n"
            f"Now reason and give me the final answer in \\boxed{{...}}."
        )
    elif problem.answer_type == "choice":
        return (
            f"Answer the following question. "
            f"Give your final answer as a single letter (A, B, C, or D) in Answer: \\boxed{{...}}.\n\n"
            f"Question: {problem.question}\n"
            f"Now reason and give me the final answer in \\boxed{{...}}."
        )
    elif problem.answer_type == "expression":
        return (
            f"Solve the following math competition problem. "
            f"Give your final answer in \\boxed{{...}}. "
            f"Your answer may be a number, fraction, or expression — write it in simplified exact form.\n\n"
            f"Problem: {problem.question}\n"
            f"Now reason and give me the final answer in \\boxed{{...}}."
        )
    return problem.question
