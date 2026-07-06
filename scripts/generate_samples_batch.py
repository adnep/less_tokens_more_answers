"""
Batch sample generation script for Metacentrum.
Generates n samples per problem using vLLM and saves to disk incrementally.
Can be run as a standalone batch job — no DTR computation, just generation.

Samples are written to disk as they're generated (streaming), so you can:
  - Monitor progress in real-time: tail -f outputs/aime24_samples/generated_samples.jsonl
  - Recover partial results if the job crashes
  - Start DTR analysis early on partial results

Usage:
  python scripts/generate_samples_batch.py \
    --benchmark aime24 \
    --n-samples 48 \
    --output-dir outputs/aime24_samples
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from src.evaluation.benchmarks import load_benchmark, format_prompt
from src.inference.sampler import generate_samples_vllm, GeneratedSample


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples batch job for Metacentrum (streaming writes)"
    )
    parser.add_argument(
        "--benchmark",
        default="aime24",
        help="Benchmark name (aime24, gpqa_diamond)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=48,
        help="Number of samples per problem",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/generated_samples",
        help="Output directory for samples",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Resume from this problem ID (skip all problems before it). "
             "If not set, auto-detects completed problems from existing output file.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Skip chat template wrapping. By default prompts are wrapped so "
             "the model starts in thinking mode (produces <think>...</think>). "
             "Pass this flag to revert to raw-text prompts (legacy behaviour).",
    )
    parser.add_argument(
        "--problem-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Only generate for these specific problem IDs (e.g. arith_0 arith_4). "
             "Default: all problems in the benchmark.",
    )
    parser.add_argument(
        "--terse",
        action="store_true",
        help="Append terse-reasoning instruction to each prompt (caveman ablation).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n{'='*60}")
    print(f"Batch Sample Generation Job (Streaming Writes)")
    print(f"Started: {timestamp}")
    print(f"{'='*60}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"Model:       {args.model}")
    print(f"Samples:     {args.n_samples} per problem")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p:       {args.top_p}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Terse:       {args.terse}")
    if args.problem_ids:
        print(f"Problems:    {args.problem_ids} (filtered)")
    print(f"{'='*60}")
    print(f"Samples will be written to disk as they're generated.")
    print(f"Monitor with: tail -f {args.output_dir}/generated_samples.jsonl")
    print(f"{'='*60}\n")

    # Load benchmark
    print(f"Loading benchmark: {args.benchmark}")
    try:
        problems = load_benchmark(args.benchmark)
        print(f"  ✓ Loaded {len(problems)} problems")
    except Exception as e:
        print(f"✗ ERROR loading benchmark: {e}")
        return 1

    if not problems:
        print("✗ ERROR: No problems loaded!")
        return 1

    # Format prompts
    print(f"\nFormatting prompts...")
    prompts = [
        {"id": p.id, "text": format_prompt(p, terse=args.terse)}
        for p in problems
    ]
    print(f"  ✓ {len(prompts)} prompts ready")

    # Filter to specific problem IDs if requested
    if args.problem_ids:
        id_set = set(args.problem_ids)
        prompts = [p for p in prompts if p["id"] in id_set]
        missing = id_set - {p["id"] for p in prompts}
        if missing:
            print(f"  ⚠ Problem IDs not found in benchmark: {sorted(missing)}")
        print(f"  → Filtered to {len(prompts)} problem(s): {[p['id'] for p in prompts]}")
        if not prompts:
            print("ERROR: No matching problems found.")
            return 1

    # Setup output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "generated_samples.jsonl")

    # Find already-completed problems from existing output file
    completed_ids = set()
    samples_written = 0
    if os.path.exists(output_file):
        from collections import Counter
        id_counts = Counter()
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    id_counts[d["problem_id"]] += 1
                    samples_written += 1
                except json.JSONDecodeError:
                    pass
        # A problem is "complete" if it already has n_samples samples
        completed_ids = {pid for pid, cnt in id_counts.items() if cnt >= args.n_samples}
        print(f"\n  Found existing file with {samples_written} samples "
              f"({len(completed_ids)} fully completed problems). Appending new samples.")

    # Filter prompts: skip completed problems, optionally start from a given ID
    if args.start_from:
        problem_ids = [p["id"] for p in prompts]
        if args.start_from not in problem_ids:
            print(f"✗ ERROR: --start-from '{args.start_from}' not found in benchmark.")
            print(f"  Available IDs: {problem_ids[:5]} ...")
            return 1
        start_idx = problem_ids.index(args.start_from)
        skipped = start_idx
        prompts = prompts[start_idx:]
        print(f"\n  Resuming from '{args.start_from}' (skipping {skipped} problems).")
    else:
        # Auto-skip problems already fully completed
        before = len(prompts)
        prompts = [p for p in prompts if p["id"] not in completed_ids]
        skipped = before - len(prompts)
        if skipped > 0:
            print(f"\n  Auto-skipping {skipped} already-completed problems.")

    if not prompts:
        print("\n✓ All problems already completed! Nothing to generate.")
        return 0

    print(f"  Problems remaining: {len(prompts)}")

    # Generate samples (streaming to disk)
    print(f"\nGenerating {args.n_samples} samples per problem...")
    print(f"Writing samples to {output_file}")
    print(f"(You can monitor progress: tail -f {output_file})\n")

    try:
        # Generate samples with streaming directly to disk
        # (samples are written immediately as vLLM processes each problem)
        all_samples = generate_samples_vllm(
            model_name=args.model,
            prompts=prompts,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            output_file=output_file,  # Enable streaming to disk
            use_chat_template=not args.no_chat_template,
        )

        # Count samples written
        samples_written = sum(len(problem_samples) for problem_samples in all_samples)

        print(f"\n✓ Generation and streaming complete!")
        print(f"  Total samples written: {samples_written}")

    except Exception as e:
        print(f"\n✗ ERROR during generation: {e}")
        print(f"  Partial results saved to {output_file}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary stats
    print(f"\n{'='*60}")
    print(f"Job Summary")
    print(f"{'='*60}")
    print(f"Benchmark:        {args.benchmark}")
    print(f"Problems:         {len(problems)}")
    print(f"Samples/problem:  {args.n_samples}")
    print(f"Total samples:    {samples_written}")
    print(f"Output location:  {output_file}")
    print(f"File size:        {os.path.getsize(output_file) / 1e6:.1f} MB")
    print(f"Finished:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    print("Next steps:")
    print(f"  1. Verify the samples file:")
    print(f"     head {output_file}")
    print(f"     wc -l {output_file}")
    print(f"  2. Run DTR computation:")
    print(f"     python scripts/run_think_at_n.py \\")
    print(f"       --samples-file {output_file} \\")
    print(f"       --skip-generation")
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
