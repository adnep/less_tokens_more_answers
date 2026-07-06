"""
Run think@n evaluation on a benchmark.

Two-phase approach:
  1. Generate n samples per problem using vLLM (or load pre-generated samples)
  2. Rank samples by a prefix score (DTR / log-prob / self-certainty), majority vote

Ranking modes
-------------
  dtr         Best signal from the paper, needs HF model + intermediate layers
  logprob     Free — stored during vLLM generation, no extra model pass needed
  selfcert    KL(uniform ‖ p_t) of final-layer distribution (needs HF, final layer only)
  neg_entropy Negative entropy -H(p_t) of final-layer distribution (old selfcert definition)
  all         Run all modes and print a comparison table (default)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import yaml

from src.evaluation.benchmarks import load_benchmark, format_prompt
from src.evaluation.voting import extract_answer, majority_vote, normalize_numeric_answer
from src.evaluation.metrics import accuracy, pass_at_n, _normalize_for_comparison
from src.inference.sampler import (
    generate_samples_vllm, save_samples, load_samples,
)
from src.inference.think_at_n import think_at_n, RANKING_MODES


def get_answer_source(generated_text: str, extracted_answer) -> str:
    """
    Returns:
      'terminated' — </think> present and answer found after it
      'fallback'   — no </think>, answer found by searching full text
      'unanswered' — no answer found either way
    """
    if extracted_answer is None:
        return "unanswered"
    return "terminated" if "</think>" in generated_text else "fallback"


def main():
    parser = argparse.ArgumentParser(description="Run think@n on a benchmark")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples-file", default=None,
                        help="Load pre-generated samples instead of generating")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--output-dir", default="outputs/aime_full_fixed_sc")
    parser.add_argument(
        "--ranking-mode",
        default="all",
        choices=list(RANKING_MODES) + ["all"],
        help="Ranking metric for think@n. 'all' runs every mode and compares.",
    )
    parser.add_argument(
        "--terse",
        action="store_true",
        help="Use terse prompt when generating new samples (caveman ablation).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name     = config["model"]["name"]
    benchmark_name = config["benchmark"]["name"]
    n              = config["think_at_n"]["n"]
    eta            = config["think_at_n"]["eta"]
    prefix_length  = config["think_at_n"]["ell_prefix"]
    gamma          = config["dtr"]["gamma"]
    rho            = config["dtr"]["rho"]

    modes_to_run = list(RANKING_MODES) if args.ranking_mode == "all" else [args.ranking_mode]

    # ── Load benchmark ────────────────────────────────────────────────────────
    print(f"Loading benchmark: {benchmark_name}")
    problems = load_benchmark(benchmark_name)
    print(f"  {len(problems)} problems loaded")

    # ── Phase 1: samples ──────────────────────────────────────────────────────
    if args.samples_file:
        print(f"\nLoading pre-generated samples from {args.samples_file}")
        all_samples_flat = load_samples(args.samples_file)
        from collections import defaultdict
        grouped = defaultdict(list)
        for s in all_samples_flat:
            grouped[s.problem_id].append(s)
        all_samples = [grouped[p.id] for p in problems if p.id in grouped]
        # Keep problems list in sync with loaded samples
        problems = [p for p in problems if p.id in grouped]
    elif not args.skip_generation:
        print(f"\nPhase 1: Generating {n} samples per problem (terse={args.terse})...")
        prompts = [
            {"id": p.id, "text": format_prompt(p, terse=args.terse)}
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

    # ── Baselines: maj@n and pass@n (no model needed) ─────────────────────────
    print(f"\n{'='*60}")
    print("Computing baselines...")

    all_extracted = []
    all_sources   = []
    for problem, samples in zip(problems, all_samples):
        answers = [_normalize_for_comparison(extract_answer(s.generated_text, problem.answer_type)) for s in samples]
        sources = [get_answer_source(s.generated_text, a) for s, a in zip(samples, answers)]
        all_extracted.append(answers)
        all_sources.append(sources)

    references = [_normalize_for_comparison(normalize_numeric_answer(p.answer)) for p in problems]

    maj_predictions = [
        majority_vote([a for a in answers if a is not None])
        for answers in all_extracted
    ]
    maj_acc = accuracy(maj_predictions, references)
    pass_n  = pass_at_n(all_extracted, references)
    print(f"  maj@{n}:    {maj_acc:.4f}  ({int(maj_acc*len(references))}/{len(references)})")
    print(f"  pass@{n}:   {pass_n:.4f}  ({int(pass_n*len(references))}/{len(references)})")

    # ── Phase 2: think@n with each ranking mode ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: think@{n}  (prefix={prefix_length}, eta={eta})")
    print(f"Modes: {modes_to_run}")

    # Only load HF model when at least one mode requires it
    needs_model = any(m in ("dtr", "selfcert", "neg_entropy") for m in modes_to_run)
    helper = tuned_lens = None
    if needs_model:
        print("\nLoading HF model...")
        from src.model.qwen3_helper import Qwen3Helper
        helper = Qwen3Helper(model_name=model_name)

    think_results_by_mode = {}

    for mode in modes_to_run:
        print(f"\n  --- ranking_mode={mode} ---")
        predictions = []
        per_problem_results = []

        for i, (problem, samples) in enumerate(zip(problems, all_samples)):
            def answer_extractor(text, _p=problem):
                return extract_answer(text, _p.answer_type)

            result = think_at_n(
                samples=samples,
                model_helper=helper,
                gamma=gamma,
                rho=rho,
                eta=eta,
                prefix_length=prefix_length,
                answer_extractor=answer_extractor,
                tuned_lens=tuned_lens,
                ranking_mode=mode,
            )
            predictions.append(result["selected_answer"])
            per_problem_results.append(result)

            correct   = _normalize_for_comparison(normalize_numeric_answer(str(result["selected_answer"]))) == _normalize_for_comparison(normalize_numeric_answer(str(problem.answer)))
            top_score = result["scores"][0][1] if result["scores"] else 0.0
            print(f"    [{i+1:3d}] {problem.id}  "
                  f"ans={result['selected_answer']}  ref={problem.answer}  "
                  f"{'✓' if correct else '✗'}  top_score={top_score:.4f}")

        acc = accuracy(predictions, references)
        think_results_by_mode[mode] = {
            "accuracy":     acc,
            "predictions":  predictions,
            "per_problem":  per_problem_results,
        }
        print(f"  think@{n} [{mode}]:  {acc:.4f}  "
              f"({int(acc*len(references))}/{len(references)})")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS — {benchmark_name}  "
          f"(n={n}, eta={eta}, prefix={prefix_length}, terse={args.terse})")
    print(f"{'='*60}")
    print(f"  maj@{n}:                  {maj_acc:.4f}")
    print(f"  pass@{n}:                 {pass_n:.4f}")
    for mode in modes_to_run:
        acc = think_results_by_mode[mode]["accuracy"]
        print(f"  think@{n} [{mode:8s}]:      {acc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "benchmark":     benchmark_name,
        "model":         model_name,
        "n":             n,
        "eta":           eta,
        "prefix_length": prefix_length,
        "gamma":         gamma,
        "rho":           rho,
        "terse":         args.terse,
        "maj_accuracy":  maj_acc,
        "pass_at_n":     pass_n,
        "think_at_n":    {m: v["accuracy"] for m, v in think_results_by_mode.items()},
        "per_problem": [
            {
                "id":        p.id,
                "reference": p.answer,
                "maj_answer": maj_predictions[i],
                "properly_terminated_fraction": (
                    sum(1 for src in all_sources[i] if src == "terminated") / len(all_sources[i])
                    if all_sources[i] else 0.0
                ),
                **{f"think_answer_{m}": think_results_by_mode[m]["predictions"][i]
                   for m in modes_to_run},
                "sample_extractions": [
                    {
                        "sample_idx":       s.sample_idx,
                        "extracted_answer": all_extracted[i][j],
                        "answer_source":    all_sources[i][j],
                        "answer_text":      s.answer_text[:500] if s.answer_text else "",
                        **{
                            f"score_{m}": dict(think_results_by_mode[m]["per_problem"][i]["scores"]).get(s.sample_idx)
                            for m in modes_to_run
                        },
                    }
                    for j, s in enumerate(all_samples[i])
                ],
            }
            for i, p in enumerate(problems)
        ],
    }
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
