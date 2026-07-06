#!/usr/bin/env python3
"""
Collect and rank all completed backtracking sweep results.

Usage
-----
    python scripts/collect_sweep_results.py \\
        --sweep-dir outputs/sweep_arithmetic_stress_test \\
        --baseline-dir outputs/sweep_arithmetic_stress_test/baseline

    # Without baseline (shows raw accuracy only, no delta)
    python scripts/collect_sweep_results.py \\
        --sweep-dir outputs/sweep_char_occur

Output
------
  - Ranked table printed to stdout (sorted by accuracy).
  - Diagnostic warnings for jobs that never triggered or hit max_backtracks.
  - sweep_results.csv saved into --sweep-dir.

Expected directory layout (produced by submit_backtracking_sweep.py)
---------------------------------------------------------------------
    sweep_dir/
        baseline/
            {benchmark}_no_intervention_sc_{timestamp}/
                summary.json
                results.jsonl
                config.json
        dd_sc_dt0p4_w30_lm_mp80/
            {benchmark}_drop_detect_sc_{timestamp}/
                summary.json
                results.jsonl
                config.json
        thr_sc_n0p50_cw3_mp200/
            ...
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Strategies that actually rewind the KV cache (backtrack_to is set).
# All others (think_budget, early_stop, cooling, steering, no_intervention, …) use
# inject/stop/temperature and will always report avg_bt=0 legitimately.
_BACKTRACKING_STRATEGIES = {"drop_detect", "threshold"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _find_result_subdirs(parent: Path):
    """
    Yield all timestamped result subdirs under parent.
    run_intervention.py creates: parent/{benchmark}_{strategy}_{metric}_{timestamp}/
    """
    for d in sorted(parent.iterdir()):
        if d.is_dir() and (d / "summary.json").exists():
            yield d


def load_run(run_dir: Path):
    """
    Load a single completed run from a timestamped result subdir.
    Returns dict or None if files are missing / run incomplete.
    """
    config_path  = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"

    if not config_path.exists() or not summary_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)
    with open(summary_path) as f:
        summaries = json.load(f)

    if not summaries:
        return None

    # With --no-baseline there is exactly one summary entry (the experiment).
    exp = summaries[-1]

    # finish_reason breakdown from raw results
    finish_counts = {}
    n_max_bt = 0
    if results_path.exists():
        rows = [json.loads(l) for l in open(results_path) if l.strip()]
        exp_rows = [r for r in rows if r.get("condition") != "baseline"]
        for r in exp_rows:
            fr = r.get("finish_reason", "unknown")
            finish_counts[fr] = finish_counts.get(fr, 0) + 1
        n_max_bt = finish_counts.get("max_backtracks", 0)

    n_problems   = exp.get("n_problems", 0)
    total_tokens = exp.get("total_tokens", 0)

    return {
        "strategy":        config.get("strategy", "?"),
        "metric":          config.get("metric",   "?"),
        "strategy_kwargs": config.get("strategy_kwargs", {}),
        "n_problems":      n_problems,
        "correct":         exp.get("correct",        0),
        "accuracy":        exp.get("accuracy",        0.0),
        "avg_backtracks":  exp.get("avg_backtracks",  0.0),
        "avg_tokens":      total_tokens // max(n_problems, 1),
        "n_max_bt":        n_max_bt,
        "finish_counts":   finish_counts,
        "run_dir":         str(run_dir),
    }


def _detect_benchmark(sweep_dir: Path) -> str | None:
    """Read benchmark name from the first config.json found in the sweep."""
    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir() or config_dir.name in ("baseline", "sweep_logs"):
            continue
        for subdir in _find_result_subdirs(config_dir):
            config_path = subdir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                benchmark = cfg.get("benchmark")
                if benchmark:
                    return benchmark
    return None


def load_baseline(baseline_parent: Path, benchmark_hint: str = None):
    """
    Load baseline accuracy from the baseline run directory.
    Searches one level deep for the timestamped subdir.
    If benchmark_hint is given, only subdirs whose name starts with it are considered.
    Returns (accuracy, correct, n_problems, avg_tokens) or (None, 0, 0, 0).
    """
    for subdir in _find_result_subdirs(baseline_parent):
        if benchmark_hint and not subdir.name.startswith(benchmark_hint):
            continue
        summary_path = subdir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries = json.load(f)
            if summaries:
                s = summaries[0]
                n = s.get("n_problems", 0)
                avg_tok = s.get("total_tokens", 0) // max(n, 1)
                return s.get("accuracy", None), s.get("correct", 0), n, avg_tok
    return None, 0, 0, 0


# ── Formatting helpers ────────────────────────────────────────────────────────

def _kw_short(kw: dict) -> str:
    """Format strategy_kwargs as a compact string for the table."""
    return "  ".join(f"{k}={v}" for k, v in kw.items())


def _col(val, width, fmt=""):
    return format(val, fmt).rjust(width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect and rank backtracking sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dir",    required=True,
                        help="Root sweep output directory (e.g. outputs/sweep_arithmetic_stress_test).")
    parser.add_argument("--baseline-dir", default=None,
                        help="Baseline run directory for Δacc computation. "
                             "Defaults to {sweep-dir}/baseline if it exists.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: sweep-dir not found: {sweep_dir}")
        sys.exit(1)

    # ── Load baseline ─────────────────────────────────────────────────────────
    bl_dir = Path(args.baseline_dir) if args.baseline_dir else sweep_dir / "baseline"
    benchmark_hint = _detect_benchmark(sweep_dir)
    baseline_acc, bl_correct, bl_n, bl_avg_tok = (None, 0, 0, 0)
    if bl_dir.exists():
        baseline_acc, bl_correct, bl_n, bl_avg_tok = load_baseline(bl_dir, benchmark_hint=benchmark_hint)

    if baseline_acc is not None:
        print(f"\nBaseline (no_intervention strategy): {bl_correct}/{bl_n} = {baseline_acc:.1%}")
    else:
        print("\nBaseline not loaded — Δacc will be omitted.")
        print(f"  (Looked in: {bl_dir})")

    # ── Scan for completed runs ───────────────────────────────────────────────
    runs = []
    missing = []

    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        if config_dir.name == "baseline":
            continue
        if config_dir.name == "sweep_logs":
            continue

        # Each config_dir contains one timestamped result subdir
        subdirs = list(_find_result_subdirs(config_dir))
        if not subdirs:
            missing.append(config_dir.name)
            continue

        run = load_run(subdirs[-1])  # most recent if somehow multiple exist
        if run is not None:
            runs.append(run)
        else:
            missing.append(config_dir.name)

    if missing:
        print(f"\nNot yet complete ({len(missing)} jobs):")
        for m in missing:
            print(f"  {m}")

    if not runs:
        print("\nNo completed runs found yet. Check job status with: qstat -u $USER")
        return

    # ── Compute Δacc and sort ─────────────────────────────────────────────────
    for r in runs:
        r["delta_acc"] = (r["accuracy"] - baseline_acc) if baseline_acc is not None else None

    runs.sort(key=lambda r: r["accuracy"], reverse=True)

    # ── Print ranked table ────────────────────────────────────────────────────
    n_done = len(runs)
    n_total = n_done + len(missing)

    print(f"\n{'═'*100}")
    print(f"  Completed: {n_done}/{n_total}   Benchmark: {sweep_dir.name}")
    print(f"{'═'*100}")
    header = (
        f"  {'#':>3}  {'strategy':>12}  {'m':>3}  {'acc':>6}  {'Δacc':>6}  "
        f"{'avg_bt':>6}  {'avg_tok':>8}  strategy_kwargs"
    )
    print(header)
    print(f"  {'─'*97}")

    for i, r in enumerate(runs, 1):
        da_str  = f"{r['delta_acc']:+.1%}" if r["delta_acc"] is not None else "  n/a"
        kw_str  = _kw_short(r["strategy_kwargs"])
        # Highlight the top 3
        marker = " ★" if i <= 3 else "  "
        print(
            f"{marker}{i:>3}  {r['strategy']:>12}  {r['metric']:>3}  "
            f"{r['accuracy']:>6.1%}  {da_str:>6}  "
            f"{r['avg_backtracks']:>6.2f}  {r['avg_tokens']:>8,}  "
            f"{kw_str}"
        )

    print(f"{'═'*100}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    warnings = []
    for r in runs:
        name = f"{r['strategy']}/{r['metric']} {r['strategy_kwargs']}"
        bt = r["avg_backtracks"]
        n  = r["n_problems"]

        if bt < 0.1 and r["strategy"] in _BACKTRACKING_STRATEGIES:
            warnings.append(
                f"  ⚠  avg_bt=0.00 — strategy never triggered.  "
                f"Try lower drop_threshold or threshold.  [{name}]"
            )
        elif bt > 4.5:
            warnings.append(
                f"  ⚠  avg_bt={bt:.2f} — hitting max_backtracks constantly. "
                f"Parameters too aggressive.  [{name}]"
            )

        max_bt_frac = r["n_max_bt"] / max(n, 1)
        if max_bt_frac > 0.3:
            warnings.append(
                f"  ⚠  {max_bt_frac:.0%} of problems hit max_backtracks cap. "
                f"Strategy is thrashing.  [{name}]"
            )

    if warnings:
        print(f"\nDiagnostics:")
        for w in warnings:
            print(w)

    # ── Best config summary ───────────────────────────────────────────────────
    best = runs[0]
    print(f"\nBest config:")
    print(f"  strategy:        {best['strategy']}")
    print(f"  metric:          {best['metric']}")
    print(f"  strategy_kwargs: {json.dumps(best['strategy_kwargs'])}")
    print(f"  accuracy:        {best['accuracy']:.1%}  (Δ={best['delta_acc']:+.1%})" if best["delta_acc"] is not None else f"  accuracy: {best['accuracy']:.1%}")
    print(f"  avg_backtracks:  {best['avg_backtracks']:.2f}")
    print(f"  avg_tokens:      {best['avg_tokens']:,}")
    print(f"\n  Re-run command:")
    print(f"    python scripts/run_intervention.py \\")
    print(f"      --benchmark <bench> --model <model> \\")
    print(f"      --strategy {best['strategy']} --metric {best['metric']} \\")
    print(f"      --strategy-kwargs '{json.dumps(best['strategy_kwargs'])}' \\")
    print(f"      --n-problems <N> --seed 42")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = sweep_dir / "sweep_results.csv"
    fieldnames = [
        "rank", "strategy", "metric", "accuracy", "delta_acc",
        "avg_backtracks", "avg_tokens", "n_problems",
        "n_max_bt_problems", "strategy_kwargs",
    ]
    all_rows = list(runs)
    if baseline_acc is not None:
        all_rows.append({
            "strategy":        "baseline",
            "metric":          "",
            "strategy_kwargs": {},
            "n_problems":      bl_n,
            "correct":         bl_correct,
            "accuracy":        baseline_acc,
            "avg_backtracks":  0.0,
            "avg_tokens":      bl_avg_tok,
            "n_max_bt":        0,
            "finish_counts":   {},
            "run_dir":         str(bl_dir),
            "delta_acc":       0.0,
        })
    rows_by_tokens = sorted(all_rows, key=lambda r: r["avg_tokens"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows_by_tokens, 1):
            writer.writerow({
                "rank":               i,
                "strategy":           r["strategy"],
                "metric":             r["metric"],
                "accuracy":           round(r["accuracy"], 4),
                "delta_acc":          round(r["delta_acc"], 4) if r["delta_acc"] is not None else "",
                "avg_backtracks":     round(r["avg_backtracks"], 3),
                "avg_tokens":         r["avg_tokens"],
                "n_problems":         r["n_problems"],
                "n_max_bt_problems":  r["n_max_bt"],
                "strategy_kwargs":    json.dumps(r["strategy_kwargs"]),
            })

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
