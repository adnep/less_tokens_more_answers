"""
Intervention generation experiment.

Runs InterventionEngine on benchmark problems with a chosen strategy and
metric, then reports accuracy vs. the baseline (no intervention).

Both the baseline (NoInterventionStrategy) and the experiment strategy are run on
the same problems so results are directly comparable.  Results are saved
to a JSONL file for later analysis.

Usage
-----
    python scripts/run_intervention.py \\
        --benchmark hmmt2025 \\
        --model /path/to/Qwen3-4B-Thinking-2507 \\
        --strategy drop_detect \\
        --metric sc \\
        --max-backtracks 5 \\
        --n-problems 20 \\
        --output-dir outputs/backtracking_hmmt

Strategy kwargs are passed with --strategy-kwargs as a JSON string:
    --strategy-kwargs '{"drop_threshold": 0.5, "window": 30}'

Available strategies: no_intervention, threshold, drop_detect
Available metrics:    sc, lp

The script runs BOTH conditions (no_intervention + chosen strategy) unless
--no-baseline is given.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime

from src.evaluation.benchmarks import load_benchmark, format_prompt
from src.evaluation.metrics import _normalize_for_comparison
from src.model.qwen3_helper import Qwen3Helper
from src.interventions.metrics import get_metric
from src.interventions.strategies import get_strategy
from src.interventions.strategies.composite import NoInterventionStrategy
from src.interventions.engine import InterventionEngine
from src.interventions.result import GenerationResult


# ── helpers ────────────────────────────────────────────────────────────────────

def is_correct(result: GenerationResult, reference: str) -> bool:
    if result.extracted_answer is None:
        return False
    return (
        _normalize_for_comparison(result.extracted_answer)
        == _normalize_for_comparison(reference)
    )


def run_condition(
    label: str,
    engine: InterventionEngine,
    problems,
    output_file: str,
) -> dict:
    """Run engine on all problems, write JSONL, return summary stats."""
    correct = 0
    total_backtracks = 0
    total_tokens = 0
    skipped = 0
    results = []

    for i, prob in enumerate(problems):
        prompt = format_prompt(prob)
        t0 = time.time()
        try:
            result = engine.generate(prompt, answer_type=prob.answer_type)
        except Exception as exc:
            print(f"  [{label}] Problem {prob.id}: ERROR — {exc}")
            skipped += 1
            continue

        elapsed = time.time() - t0
        ok = is_correct(result, prob.answer)
        correct += ok
        total_backtracks += result.n_backtracks
        total_tokens += result.total_tokens_generated

        row = {
            "condition": label,
            "problem_id": prob.id,
            "reference": prob.answer,
            "extracted": result.extracted_answer,
            "correct": ok,
            "n_backtracks": result.n_backtracks,
            "total_tokens_generated": result.total_tokens_generated,
            "finish_reason": result.finish_reason,
            "elapsed_s": round(elapsed, 2),
            "backtrack_events": [
                {
                    "trigger_pos": e.trigger_position,
                    "backtrack_to": e.backtrack_to,
                    "trigger_metric": round(e.trigger_metric, 4),
                    "restore_metric": round(e.restore_metric, 4),
                    "reason": e.reason,
                }
                for e in result.backtrack_events
            ],
            "interventions": [
                {
                    "type": ev.type, "position": ev.position,
                    "detail": ev.detail, "reason": ev.reason,
                }
                for ev in result.interventions
            ],
        }
        results.append(row)

        status = "✓" if ok else "✗"
        bt_str = f" [{result.n_backtracks} BT]" if result.n_backtracks else ""
        print(
            f"  [{label}] {i+1:3d}/{len(problems)} {prob.id}  "
            f"{status}  ans={result.extracted_answer!r}"
            f"  ref={prob.answer!r}"
            f"{bt_str}  {elapsed:.1f}s"
        )

    # Write to JSONL (append so baseline + experiment share one file)
    with open(output_file, "a") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")

    n_evaluated = len(problems) - skipped
    accuracy = correct / n_evaluated if n_evaluated else 0.0
    avg_bt = total_backtracks / n_evaluated if n_evaluated else 0.0

    return {
        "label": label,
        "n_problems": n_evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "avg_backtracks": avg_bt,
        "total_tokens": total_tokens,
        "skipped": skipped,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intervention generation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── benchmark ──────────────────────────────────────────────────────────
    parser.add_argument("--benchmark", default="hmmt2025",
                        help="Benchmark name (aime24, hmmt2025, gpqa_diamond, …)")
    parser.add_argument("--n-problems", type=int, default=None,
                        help="Limit to first N problems (default: all)")
    parser.add_argument("--problem-ids", nargs="+", default=None,
                        help="Run only these specific problem IDs")

    # ── model ──────────────────────────────────────────────────────────────
    parser.add_argument("--model", required=True,
                        help="Model name or local path")

    # ── metric ─────────────────────────────────────────────────────────────
    parser.add_argument("--metric", default="sc",
                        choices=["sc", "neg_entropy", "lp", "z_score_sc"],
                        help="Per-token metric to monitor during generation. "
                             "sc = KL(u‖p_t) ∈ [0,∞) (Kang et al. 2025); "
                             "neg_entropy = -H(p_t) ∈ (-∞,0]; "
                             "lp = log p(y_t) ∈ (-∞,0]; "
                             "z_score_sc = online z-score of SC ∈ (−∞,+∞), "
                             "typically (−2,+2) — thresholds in σ units.")

    # ── strategy ───────────────────────────────────────────────────────────
    parser.add_argument("--strategy", default="drop_detect",
                        choices=[
                            "no_intervention", "threshold", "drop_detect",
                            "linear_cooling", "confidence_cooling",
                            "drop_steering", "confident_region_end",
                            "early_stop", "think_budget", "wait_backtrack",
                            "any", "all",
                        ],
                        help="Intervention strategy to apply.  "
                             "BACKTRACKING:   no_intervention, threshold, drop_detect.  "
                             "TEMPERATURE:    linear_cooling, confidence_cooling.  "
                             "STEERING:       drop_steering, confident_region_end, think_budget.  "
                             "EARLY-STOP:     early_stop (needs --classifier).  "
                             "think_budget injects </think> to truncate reasoning early.  "
                             "COMPOSITE:      any, all — combine multiple "
                             "strategies with OR / AND logic. Pass sub-strategies via "
                             "--strategy-kwargs '{\"sub_strategies\": [{\"name\": \"drop_detect\", "
                             "\"kwargs\": {...}}, {\"name\": \"threshold\", \"kwargs\": {...}}]}'. "
                             "All sub-strategies must operate on the same metric (the one "
                             "specified with --metric).")
    parser.add_argument("--strategy-kwargs", default="{}", type=json.loads,
                        help='JSON dict of strategy constructor kwargs, '
                             'e.g. \'{"drop_threshold": 0.5, "window": 30}\'')
    parser.add_argument("--classifier", default="self_prompted",
                        choices=["self_prompted"],
                        help="Classifier for --strategy=early_stop.")
    parser.add_argument("--classifier-kwargs", default="{}", type=json.loads,
                        help='JSON kwargs for the classifier, '
                             'e.g. \'{"answer_type": "expression"}\'')
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the NoInterventionStrategy baseline run")
    parser.add_argument("--verbose", action="store_true",
                        help="Stream tokens to stdout as they are generated. "
                             "Reasoning tokens are grey, answer tokens green, "
                             "backtrack/stop events yellow. No extra memory cost.")

    # ── generation ─────────────────────────────────────────────────────────
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-backtracks", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature-boost", type=float, default=0.2,
                        help="Extra temperature for first --boost-tokens steps "
                             "after each backtrack (encourages diverse continuation)")
    parser.add_argument("--boost-tokens", type=int, default=15)
    parser.add_argument("--checkpoint-interval", type=int, default=128,
                        help="Save a KV cache checkpoint every N tokens. "
                             "On backtrack restores nearest checkpoint and replays "
                             "only the short gap (≤ interval tokens). 0 = disable "
                             "(falls back to full prefix recomputation).")
    parser.add_argument("--max-checkpoints", type=int, default=0,
                        help="Keep at most N checkpoints per generation "
                             "(0 = unlimited). Oldest is evicted first.")
    parser.add_argument("--no-offload", action="store_true",
                        help="Keep checkpoint tensors on GPU instead of CPU RAM. "
                             "Faster restore but uses more VRAM.")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Skip chat template wrapping. By default prompts are "
                             "wrapped so the model starts in thinking mode. Pass this "
                             "flag only if you intentionally want raw-text completion.")
    parser.add_argument("--seed", type=int, default=None)

    # ── output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", default="outputs/backtracking_results")

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"{args.benchmark}_{args.strategy}_{args.metric}"
    output_dir = os.path.join(args.output_dir, f"{tag}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.jsonl")

    print(f"\n{'='*65}")
    print("Intervention Generation Experiment")
    print(f"{'='*65}")
    print(f"Benchmark:    {args.benchmark}")
    print(f"Model:        {args.model}")
    print(f"Metric:       {args.metric}")
    print(f"Strategy:     {args.strategy}  {args.strategy_kwargs}")
    print(f"Max BT:       {args.max_backtracks}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Temperature:  {args.temperature} (boost +{args.temperature_boost} for {args.boost_tokens} tokens)")
    ckpt_str = f"every {args.checkpoint_interval} tokens" if args.checkpoint_interval else "disabled (full recompute)"
    print(f"Checkpoints:  {ckpt_str}  offload_cpu={not args.no_offload}")
    print(f"Output:       {output_file}")
    print(f"{'='*65}\n")

    # ── Load benchmark ─────────────────────────────────────────────────────
    print(f"Loading benchmark: {args.benchmark}")
    problems = load_benchmark(args.benchmark)
    print(f"  ✓ {len(problems)} problems loaded")

    if args.problem_ids:
        id_set = set(args.problem_ids)
        problems = [p for p in problems if p.id in id_set]
        print(f"  → Filtered to {len(problems)} problem(s): {args.problem_ids}")

    if args.n_problems and args.n_problems < len(problems):
        problems = problems[:args.n_problems]
        print(f"  → Limited to first {len(problems)} problem(s)")

    if not problems:
        print("ERROR: No problems to evaluate.")
        return 1

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    helper = Qwen3Helper(local_path=args.model)

    # ── Save run config ────────────────────────────────────────────────────
    config = {
        "timestamp": timestamp,
        "benchmark": args.benchmark,
        "n_problems": len(problems),
        "model": args.model,
        "metric": args.metric,
        "strategy": args.strategy,
        "strategy_kwargs": args.strategy_kwargs,
        "max_tokens": args.max_tokens,
        "max_backtracks": args.max_backtracks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "temperature_boost": args.temperature_boost,
        "boost_tokens": args.boost_tokens,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Metric ─────────────────────────────────────────────────────────────
    metric = get_metric(args.metric)

    # ── Baseline run (NoInterventionStrategy) ──────────────────────────────
    summaries = []

    if not args.no_baseline:
        print(f"\n{'─'*65}")
        print("Condition 1/2: BASELINE (no intervention)")
        print(f"{'─'*65}")
        baseline_engine = InterventionEngine(
            helper=helper,
            metric=metric,
            strategy=NoInterventionStrategy(),
            max_tokens=args.max_tokens,
            max_backtracks=0,
            temperature=args.temperature,
            top_p=args.top_p,
            checkpoint_interval=0,   # no checkpointing needed for baseline
            use_chat_template=not args.no_chat_template,
            verbose=args.verbose,
            seed=args.seed,
        )
        summary = run_condition("baseline", baseline_engine, problems, output_file)
        summaries.append(summary)
        print(f"\n  Baseline accuracy: {summary['correct']}/{summary['n_problems']} "
              f"= {summary['accuracy']:.1%}")

    # ── Experiment run ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Condition {2 if not args.no_baseline else 1}/{2 if not args.no_baseline else 1}: "
          f"EXPERIMENT ({args.strategy} + {args.metric})")
    print(f"{'─'*65}")

    # Strategies that never backtrack don't need (or benefit from) KV checkpointing.
    # Auto-disable it to avoid wasting CPU RAM — checkpoints cost ~37 GB at 8192 tokens.
    _NO_BACKTRACK_STRATEGIES = {
        "no_intervention", "early_stop", "think_budget",
        "linear_cooling", "confidence_cooling",
        "drop_steering", "confident_region_end",
    }
    effective_checkpoint_interval = args.checkpoint_interval
    if args.strategy in _NO_BACKTRACK_STRATEGIES and args.checkpoint_interval != 0:
        effective_checkpoint_interval = 0
        print(f"  ℹ  checkpoint_interval forced to 0 — '{args.strategy}' never "
              f"backtracks so checkpointing would only waste CPU RAM.")

    # Strategies that need the tokenizer injected at construction time
    if args.strategy == "early_stop":
        from src.interventions.classifiers import get_classifier
        clf_kwargs = dict(args.classifier_kwargs)
        if args.classifier == "self_prompted":
            clf_kwargs.setdefault("helper", helper)
        classifier = get_classifier(args.classifier, **clf_kwargs)
        sk = dict(args.strategy_kwargs)
        sk["classifier"] = classifier
        sk["tokenizer"]  = helper.tokenizer
        experiment_strategy = get_strategy("early_stop", **sk)
    elif args.strategy in ("think_budget", "wait_backtrack"):
        sk = dict(args.strategy_kwargs)
        sk["tokenizer"] = helper.tokenizer
        experiment_strategy = get_strategy(args.strategy, **sk)
    elif args.strategy in ("any", "all"):
        from src.interventions.strategies.composite import AnyStrategy, AllStrategy
        sub_strats = []
        for sub in args.strategy_kwargs.get("sub_strategies", []):
            sname = sub["name"]
            skw   = dict(sub.get("kwargs", {}))
            if sname in ("think_budget", "wait_backtrack"):
                skw["tokenizer"] = helper.tokenizer
            sub_strats.append(get_strategy(sname, **skw))
        if not sub_strats:
            raise ValueError("--strategy any/all requires 'sub_strategies' in --strategy-kwargs")
        CompositeClass = AnyStrategy if args.strategy == "any" else AllStrategy
        experiment_strategy = CompositeClass(sub_strats)
    else:
        experiment_strategy = get_strategy(args.strategy, **args.strategy_kwargs)
    experiment_engine = InterventionEngine(
        helper=helper,
        metric=metric,
        strategy=experiment_strategy,
        max_tokens=args.max_tokens,
        max_backtracks=args.max_backtracks,
        temperature=args.temperature,
        top_p=args.top_p,
        temperature_boost=args.temperature_boost,
        boost_tokens=args.boost_tokens,
        checkpoint_interval=effective_checkpoint_interval,
        max_checkpoints=args.max_checkpoints,
        offload_to_cpu=not args.no_offload,
        use_chat_template=not args.no_chat_template,
        verbose=args.verbose,
        seed=args.seed,
    )
    summary = run_condition(
        f"{args.strategy}_{args.metric}", experiment_engine, problems, output_file
    )
    summaries.append(summary)

    # ── Final report ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Results Summary")
    print(f"{'='*65}")
    for s in summaries:
        print(
            f"  {s['label']:30s}  "
            f"acc={s['correct']}/{s['n_problems']} ({s['accuracy']:.1%})  "
            f"avg_BT={s['avg_backtracks']:.2f}"
        )

    if len(summaries) == 2:
        delta = summaries[1]["accuracy"] - summaries[0]["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"\n  Δ accuracy (experiment − baseline): {sign}{delta:.1%}")

    print(f"\n  Results saved to: {output_file}")
    print(f"  Config saved to:  {os.path.join(output_dir, 'config.json')}")
    print(f"{'='*65}\n")

    # Save summary JSON
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
