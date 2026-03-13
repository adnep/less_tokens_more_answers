"""
Run think@n evaluation on a benchmark.

Two-phase approach:
  1. Generate n samples per problem using vLLM (or load pre-generated samples)
  2. Compute prefix DTR and select top-eta samples, majority vote
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import yaml

from src.evaluation.benchmarks import load_benchmark, format_prompt
from src.evaluation.voting import extract_answer, majority_vote
from src.evaluation.metrics import accuracy, pass_at_n
from src.inference.sampler import (
    generate_samples_vllm, save_samples, load_samples, GeneratedSample,
)


def main():
    parser = argparse.ArgumentParser(description="Run think@n on a benchmark")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples-file", default=None,
                       help="Load pre-generated samples instead of generating")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip generation, only run DTR selection on existing samples")
    parser.add_argument("--output-dir", default="outputs/think_at_n")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    benchmark_name = config["benchmark"]["name"]
    n = config["think_at_n"]["n"]
    eta = config["think_at_n"]["eta"]
    prefix_length = config["think_at_n"]["ell_prefix"]
    gamma = config["dtr"]["gamma"]
    rho = config["dtr"]["rho"]

    # Load benchmark
    print(f"Loading benchmark: {benchmark_name}")
    problems = load_benchmark(benchmark_name)
    print(f"  {len(problems)} problems loaded")

    # Phase 1: Generate samples
    if args.samples_file:
        print(f"\nLoading pre-generated samples from {args.samples_file}")
        all_samples_flat = load_samples(args.samples_file)
        # Group by problem_id
        from collections import defaultdict
        grouped = defaultdict(list)
        for s in all_samples_flat:
            grouped[s.problem_id].append(s)
        all_samples = [grouped[p.id] for p in problems if p.id in grouped]
    elif not args.skip_generation:
        print(f"\nPhase 1: Generating {n} samples per problem using vLLM...")
        prompts = [
            {"id": p.id, "text": format_prompt(p)}
            for p in problems
        ]
        all_samples = generate_samples_vllm(
            model_name=model_name,
            prompts=prompts,
            n_samples=n,
            temperature=config["generation"]["temperature"],
            top_p=config["generation"]["top_p"],
            max_tokens=config["generation"]["max_new_tokens"],
        )
        os.makedirs(args.output_dir, exist_ok=True)
        save_samples(all_samples, args.output_dir)
    else:
        print("ERROR: --skip-generation requires --samples-file")
        return

    # Compute baselines: maj@n and pass@n (no DTR needed)
    print(f"\n{'='*60}")
    print("Computing baselines (no DTR, no HF model needed)...")

    all_extracted = []  # per-problem list of all answers
    for problem, samples in zip(problems, all_samples):
        answers = [
            extract_answer(s.generated_text, problem.answer_type) for s in samples
        ]
        all_extracted.append(answers)

    references = [p.answer for p in problems]

    # maj@n: majority vote over all n samples
    maj_predictions = []
    for answers in all_extracted:
        valid = [a for a in answers if a is not None]
        maj_predictions.append(majority_vote(valid))
    maj_acc = accuracy(maj_predictions, references)
    print(f"  maj@{n} accuracy: {maj_acc:.4f} ({int(maj_acc * len(references))}/{len(references)})")

    # pass@n: any correct
    pass_n = pass_at_n(all_extracted, references)
    print(f"  pass@{n}: {pass_n:.4f} ({int(pass_n * len(references))}/{len(references)})")

    # Phase 2: think@n with DTR selection
    print(f"\n{'='*60}")
    print(f"Phase 2: think@{n} with DTR selection (prefix={prefix_length}, eta={eta})...")
    print("Loading HF model for DTR analysis...")

    from src.model.qwen3_helper import Qwen3Helper
    from src.inference.think_at_n import think_at_n

    helper = Qwen3Helper(model_name=model_name)

    think_predictions = []
    think_results = []
    for i, (problem, samples) in enumerate(zip(problems, all_samples)):
        print(f"  Problem {i+1}/{len(problems)}: {problem.id}")

        def answer_extractor(text):
            return extract_answer(text, problem.answer_type)

        result = think_at_n(
            samples=samples,
            model_helper=helper,
            gamma=gamma,
            rho=rho,
            eta=eta,
            prefix_length=prefix_length,
            answer_extractor=answer_extractor,
        )
        think_predictions.append(result["selected_answer"])
        think_results.append(result)

        correct = str(result["selected_answer"]).strip() == str(problem.answer).strip()
        top_dtr = result["dtr_scores"][0][1] if result["dtr_scores"] else 0
        print(f"    Answer: {result['selected_answer']} (ref: {problem.answer}) "
              f"{'CORRECT' if correct else 'WRONG'} | Top DTR: {top_dtr:.4f}")

    think_acc = accuracy(think_predictions, references)
    print(f"\n{'='*60}")
    print(f"Results on {benchmark_name}:")
    print(f"  maj@{n}:   {maj_acc:.4f}")
    print(f"  think@{n}: {think_acc:.4f}")
    print(f"  pass@{n}:  {pass_n:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "benchmark": benchmark_name,
        "model": model_name,
        "n": n,
        "eta": eta,
        "prefix_length": prefix_length,
        "gamma": gamma,
        "rho": rho,
        "maj_accuracy": maj_acc,
        "think_accuracy": think_acc,
        "pass_at_n": pass_n,
        "per_problem": [
            {
                "id": p.id,
                "reference": p.answer,
                "maj_answer": maj_predictions[i],
                "think_answer": think_predictions[i],
                "dtr_scores": think_results[i]["dtr_scores"],
            }
            for i, p in enumerate(problems)
        ],
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
