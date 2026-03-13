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


def load_gpqa_diamond() -> List[BenchmarkProblem]:
    """Load GPQA-Diamond multiple choice problems."""
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    problems = []
    for i, row in enumerate(ds):
        question = row["Question"]
        answer = row["Correct Answer"]
        problems.append(BenchmarkProblem(
            id=f"gpqa_{i}",
            question=question,
            answer=answer,
            answer_type="choice",
            source="GPQA-Diamond",
        ))
    return problems


def load_benchmark(name: str) -> List[BenchmarkProblem]:
    """Load a benchmark by name."""
    loaders = {
        "aime24": load_aime_2024,
        "gpqa_diamond": load_gpqa_diamond,
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
        )
    elif problem.answer_type == "choice":
        return (
            f"Answer the following question. "
            f"Give your final answer as a single letter (A, B, C, or D) in Answer: \\boxed{{...}}.\n\n"
            f"Question: {problem.question}\n"
        )
    return problem.question
