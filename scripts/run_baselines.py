"""
Run baseline evaluations: maj@n and pass@n without DTR selection.
Works on pre-generated samples (from sampler.py).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
from collections import defaultdict

from src.inference.sampler import load_samples
from src.evaluation.benchmarks import load_benchmark
from src.evaluation.voting import extract_answer, majority_vote
from src.evaluation.metrics import accuracy, pass_at_n


def main():
    parser = argparse.ArgumentParser(description="Compute baselines on generated samples")
    parser.add_argument("--samples-file", required=True, help="Path to generated_samples.jsonl")
    parser.add_argument("--benchmark", default="aime24")
    args = parser.parse_args()

    problems = load_benchmark(args.benchmark)
    problem_map = {p.id: p for p in problems}

    samples = load_samples(args.samples_file)
    grouped = defaultdict(list)
    for s in samples:
        grouped[s.problem_id].append(s)

    print(f"Benchmark: {args.benchmark} ({len(problems)} problems)")
    print(f"Samples loaded: {len(samples)} total")

    references = []
    all_extracted = []
    maj_predictions = []

    for problem in problems:
        if problem.id not in grouped:
            print(f"  WARNING: No samples for {problem.id}")
            continue

        problem_samples = grouped[problem.id]
        answers = [
            extract_answer(s.generated_text, problem.answer_type) for s in problem_samples
        ]
        valid = [a for a in answers if a is not None]

        references.append(problem.answer)
        all_extracted.append(answers)
        maj_predictions.append(majority_vote(valid))

        n = len(problem_samples)

    print(f"\nResults ({n} samples per problem):")
    maj_acc = accuracy(maj_predictions, references)
    pass_n_score = pass_at_n(all_extracted, references)

    print(f"  maj@{n}:  {maj_acc:.4f} ({int(maj_acc * len(references))}/{len(references)})")
    print(f"  pass@{n}: {pass_n_score:.4f} ({int(pass_n_score * len(references))}/{len(references)})")


if __name__ == "__main__":
    main()
