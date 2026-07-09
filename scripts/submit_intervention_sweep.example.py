#!/usr/bin/env python3
"""
Submit backtracking sweep3 — all 8 benchmarks, incl. z_score_sc metric.

EXAMPLE / TEMPLATE. The cluster-specific constants below are read from
environment variables (with sensible fallbacks) so no personal paths or email
are committed. Copy to submit_intervention_sweep3.py and either export the vars
or hard-code your own values:
    export CLUSTER_HOME=/storage/<CLUSTER>/home/<USERNAME>
    export PBS_EMAIL=you@example.com
    export MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507   # or a local snapshot path
    set YOUR_ENV properly too!!

Usage
-----
    python scripts/submit_intervention_sweep3.py --benchmark distinct_char --dry-run
    python scripts/submit_intervention_sweep3.py --benchmark word_len
    python scripts/submit_intervention_sweep3.py --benchmark hmmt2025 --no-baseline
    python scripts/submit_intervention_sweep3.py --benchmark gpqa_diamond --baseline-only

Output layout
-------------
    outputs/sweep3_{benchmark}/
        baseline/
        dd_sc_dt5_w30_lm_mp400/
        thr_sc_28_cw3_mp400/
        tb_fixed_m1024/
        zs_dd_dt0p5_w30_lm_mp400/   # z_score_sc configs
        ...
    outputs/sweep_logs/
        submitted_{bench}_{ts}.json
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Cluster constants ─────────────────────────────────────────────────────────

# Fill these in via environment variables, or replace the fallbacks with your own.
HOME       = os.environ.get("CLUSTER_HOME", "/storage/<CLUSTER>/home/<USERNAME>")
WORK_DIR   = f"{HOME}/less_tokens_more_answers"
# HF model id, or a local snapshot path under {HOME}/hf_cache for offline nodes.
MODEL      = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Thinking-2507")
PYTHON_ENV = f"{HOME}/<YOUR_ENV>/bin/activate"
EMAIL      = os.environ.get("PBS_EMAIL", "you@example.com")

# ── Fixed generation settings ─────────────────────────────────────────────────

GEN = dict(
    max_backtracks=5,
    temperature=0.6,
    top_p=0.95,
    temperature_boost=0.2,
    boost_tokens=15,
    checkpoint_interval=256,
    seed=42,
)

# ── Benchmark-specific settings ───────────────────────────────────────────────
# n_problems=200 for easy sets so that errors actually appear (2–20 per run).
# max_tokens=8192 for word_len (p90=1366 tok; no point capping at 16384).

BENCH = {
    "arithmetic_stress_test": dict(n_problems=320, walltime="18:00:00", max_tokens=16384, output_prefix="sweep"),
    "char_occur":             dict(n_problems=200, walltime="15:00:00", max_tokens=16384, output_prefix="sweep"),
    "distinct_char":          dict(n_problems=200, walltime="15:00:00", max_tokens=16384, output_prefix="sweep"),
    "word_len":               dict(n_problems=200, walltime="15:00:00",  max_tokens=16384,  output_prefix="sweep"),
    "substring_occur":        dict(n_problems=200, walltime="15:00:00", max_tokens=16384, output_prefix="sweep"),
    "hmmt2025":               dict(n_problems=50,  walltime="12:00:00", max_tokens=16384, output_prefix="sweep"),
    "aime24":                 dict(n_problems=50,  walltime="14:00:00", max_tokens=16384, output_prefix="sweep"),
    "gpqa_diamond":           dict(n_problems=50,  walltime="12:00:00", max_tokens=16384, output_prefix="sweep"),
}


# ── Grid helper functions ─────────────────────────────────────────────────────

def _dd(metric, drop_threshold, window, backtrack_mode, min_position=80):
    """drop_detect strategy config."""
    return {
        "strategy": "drop_detect",
        "metric":   metric,
        "strategy_kwargs": {
            "drop_threshold": drop_threshold,
            "window":         window,
            "backtrack_mode": backtrack_mode,
            "min_position":   min_position,
        },
    }


def _thr(metric, threshold, confirm_window, min_position=80, backtrack_n_tokens=40):
    """threshold strategy config."""
    return {
        "strategy": "threshold",
        "metric":   metric,
        "strategy_kwargs": {
            "threshold":          threshold,
            "confirm_window":     confirm_window,
            "min_position":       min_position,
            "backtrack_n_tokens": backtrack_n_tokens,
        },
    }


def _tb(trigger, max_think_tokens, metric="sc",
        drop_threshold=8.0, window=30, min_position=100,
        wait_window=20, wait_threshold=32.0, wait_exit_on="high"):
    """think_budget strategy config (injects </think> — does NOT backtrack)."""
    kw = {
        "trigger":          trigger,
        "max_think_tokens": max_think_tokens,
        "min_position":     min_position,
    }
    if trigger in ("drop", "drop_or_fixed", "wait_or_drop", "any"):
        kw["drop_threshold"] = drop_threshold
        kw["window"]         = window
    if "wait" in trigger or trigger == "any":
        kw["wait_window"]    = wait_window
        kw["wait_threshold"] = wait_threshold
        kw["wait_exit_on"]   = wait_exit_on
    return {
        "strategy":        "think_budget",
        "metric":          metric,
        "strategy_kwargs": kw,
    }


def _wb(exit_on, threshold=0.5, window=20, metric="sc",
        backtrack_mode="local_max", backtrack_n_tokens=40,
        search_window=60, min_position=80, cooldown=60):
    """wait_backtrack strategy config."""
    return {
        "strategy": "wait_backtrack",
        "metric":   metric,
        "strategy_kwargs": {
            "exit_on":            exit_on,
            "threshold":          threshold,
            "window":             window,
            "backtrack_mode":     backtrack_mode,
            "backtrack_n_tokens": backtrack_n_tokens,
            "search_window":      search_window,
            "min_position":       min_position,
            "cooldown":           cooldown,
        },
    }


def _lc(start_temp=0.6, end_temp=0.2, start_token=0, end_token=4000, metric="sc"):
    """linear_cooling: deterministic temperature schedule (no backtracking)."""
    return {
        "strategy": "linear_cooling",
        "metric":   metric,
        "strategy_kwargs": {
            "start_temp":  start_temp,
            "end_temp":    end_temp,
            "start_token": start_token,
            "end_token":   end_token,
        },
    }


def _cc(metric, confidence_threshold, sustained_window=30, dropout_window=15,
        warm_temp=0.6, cool_temp=0.3, min_position=80):
    """confidence_cooling: cool when smoothed metric ≥ threshold (no backtracking)."""
    return {
        "strategy": "confidence_cooling",
        "metric":   metric,
        "strategy_kwargs": {
            "confidence_threshold": confidence_threshold,
            "sustained_window":     sustained_window,
            "dropout_window":       dropout_window,
            "warm_temp":            warm_temp,
            "cool_temp":            cool_temp,
            "min_position":         min_position,
        },
    }


def _ds(metric, steering_text, drop_threshold, window, min_position=80,
        cooldown=80, also_lower_temp=None, max_injections=3):
    """drop_steering: inject text on metric drop (no backtracking)."""
    return {
        "strategy": "drop_steering",
        "metric":   metric,
        "strategy_kwargs": {
            "steering_text":   steering_text,
            "drop_threshold":  drop_threshold,
            "window":          window,
            "min_position":    min_position,
            "cooldown":        cooldown,
            "also_lower_temp": also_lower_temp,
            "max_injections":  max_injections,
        },
    }


def _cre(metric, confidence_threshold, sustained_window=50, drop_threshold=0.5,
         steering_text=None, cool_to_temp=None, min_position=0, max_position=100000):
    """confident_region_end: act at first drop after sustained confident region."""
    return {
        "strategy": "confident_region_end",
        "metric":   metric,
        "strategy_kwargs": {
            "confidence_threshold": confidence_threshold,
            "sustained_window":     sustained_window,
            "drop_threshold":       drop_threshold,
            "steering_text":        steering_text,
            "cool_to_temp":         cool_to_temp,
            "min_position":         min_position,
            "max_position":         max_position,
        },
    }


def _composite(mode, sub_cfgs):
    """Composite (any=OR / all=AND) of same-metric backtracking sub-strategies.

    All sub-strategies must use the same metric as the engine (specified by
    the top-level 'metric' key).  Cooling/steering sub-strategies are ignored
    by Any/AllStrategy because they never set should_backtrack=True.
    """
    metric = sub_cfgs[0]["metric"]
    return {
        "strategy": mode,   # "any" or "all"
        "metric":   metric,
        "strategy_kwargs": {
            "sub_strategies": [
                {"name": c["strategy"], "kwargs": c["strategy_kwargs"]}
                for c in sub_cfgs
            ],
        },
    }


# ── Parameter grids ───────────────────────────────────────────────────────────

GRIDS = {

    # ── arithmetic_stress_test ─────────────────────────────────────────────────
    "arithmetic_stress_test": [

        # ── drop_detect SC ─────────────────────────────────────────────────────
        # dt=3: sensitive (fires on 3 KL drop); dt=5: moderate; dt=8: conservative.
        # local_max rewinds to peak of preceding search_window.
        _dd("sc", 3, 20, "local_max"),
        _dd("sc", 3, 40, "local_max"),
        _dd("sc", 5, 20, "local_max"),
        _dd("sc", 5, 40, "local_max"),
        _dd("sc", 8, 20, "local_max"),
        _dd("sc", 8, 40, "local_max"),
        _dd("sc", 5, 20, "fixed_n"),

        # ── threshold SC ───────────────────────────────────────────────────────
        # cw=3: confirm 3 consecutive tokens; cw=5: more conservative.
        _thr("sc", 28, 3),
        _thr("sc", 28, 5),
        _thr("sc", 28, 10),
        _thr("sc", 31, 3),
        _thr("sc", 31, 5),
        _thr("sc", 34, 3),
        _thr("sc", 34, 5),

        # ── think_budget: inject </think> on SC drop or hard cap ───────────────
        _tb("drop",  2048, drop_threshold=5, window=20),
        _tb("drop",  2048, drop_threshold=5, window=40),
        _tb("drop",  2048, drop_threshold=8, window=20),
        _tb("drop",  1024, drop_threshold=5, window=20),
        _tb("drop",  1024, drop_threshold=3, window=20),
        _tb("drop",  1024, drop_threshold=3, window=40),
        _tb("fixed",  512),
        _tb("fixed", 1024),
        _tb("fixed", 2048),

        # ── think_budget: wait trigger (inject </think> at Wait token) ──────────
        _tb("wait",         2048, wait_threshold=32, wait_exit_on="high"),
        _tb("wait",         2048, wait_threshold=34, wait_exit_on="high"),
        _tb("wait_or_drop", 2048, drop_threshold=5, window=20,
            wait_threshold=32, wait_exit_on="high"),

        # ── wait_backtrack SC ──────────────────────────────────────────────────
        # Rewind to local max when model says Wait AND was confident pre-wait.
        _wb("high", threshold=32, window=20, min_position=80),
        _wb("high", threshold=34, window=20, min_position=80),
        _wb("high", threshold=34, window=20, backtrack_mode="fixed_n", min_position=80),

        # ── drop_detect z_score_sc ─────────────────────────────────────────────
        # dt in σ units: 0.4 σ is a moderate drop below the running average.
        _dd("z_score_sc", 0.4, 20, "local_max"),
        _dd("z_score_sc", 0.5, 30, "local_max"),
        _dd("z_score_sc", 0.6, 30, "local_max"),

        # ── threshold z_score_sc ───────────────────────────────────────────────
        # Fire when z < −0.25 (model is 0.25σ below its own running average).
        _thr("z_score_sc", -0.25, 3),
        _thr("z_score_sc", -0.30, 3),
        _thr("z_score_sc", -0.30, 5),
        _thr("z_score_sc", -0.30, 10),
        _thr("z_score_sc", -0.30, 20),

        # ── think_budget z_score_sc drop ───────────────────────────────────────
        _tb("drop", 2048, metric="z_score_sc", drop_threshold=0.4, window=20),
        _tb("drop", 2048, metric="z_score_sc", drop_threshold=0.5, window=30),

        # ── linear_cooling ─────────────────────────────────────────────────────
        # Correct 35→42, incorrect flat 26–30 from token ~200 onwards.
        # Cool from 0.6 → 0.2 once the model enters the stable phase.
        _lc(0.6, 0.2, start_token=200, end_token=3000),
        _lc(0.8, 0.2, start_token=200, end_token=3000),
        _lc(0.6, 0.15, start_token=400, end_token=3000),
        _lc(0.8, 0.15, start_token=400, end_token=3000),

        # ── confidence_cooling SC ──────────────────────────────────────────────
        # Cool when smoothed SC ≥ 30 (above incorrect range) for 30 consecutive
        # tokens.  Re-warm if it drops below.
        _cc("sc", 30, sustained_window=30, dropout_window=15,
            cool_temp=0.25, min_position=80),
        _cc("sc", 34, sustained_window=20, dropout_window=15,
            cool_temp=0.2, min_position=80),

        # ── drop_steering SC ───────────────────────────────────────────────────
        # Inject a recalculation prompt when SC drops by ≥5 within window 20.
        # The text nudges the model to redo its calculation rather than wander.
        _ds("sc",
            "\nWait, let me redo this calculation more carefully.\n",
            drop_threshold=5, window=20, min_position=80, cooldown=80),
        _ds("sc",
            "\nWait, let me redo this calculation more carefully.\n",
            drop_threshold=8, window=30, min_position=80, cooldown=100,
            also_lower_temp=0.3),

        # ── confident_region_end SC ────────────────────────────────────────────
        # Wait for SC ≥ 34 (correct's range 35–42) for 40 tokens, then on
        # first drop ≥ 5, inject + lower temperature to lock in the answer.
        _cre("sc", confidence_threshold=34, sustained_window=40, drop_threshold=5,
             steering_text="\nHmm, I should double-check my arithmetic here.\n",
             cool_to_temp=0.2, min_position=80),

        # ── composite: any (OR) — broad backtrack net ──────────────────────────
        # Fires on SC drop ≥ 5 OR SC below 28.  Catches both abrupt falls from
        # high confidence and traces that drift down into the incorrect range.
        _composite("any", [
            _dd("sc", 5, 20, "local_max"),
            _thr("sc", 28, 5),
        ]),
        # ── composite: all (AND) — high-precision backtrack ────────────────────
        # Requires BOTH a drop of ≥ 5 AND the resulting value < 28.  Avoids
        # backtracking on drops that stay in the correct zone (35–42).
        _composite("all", [
            _dd("sc", 5, 20, "local_max"),
            _thr("sc", 28, 3),
        ]),
    ],

    # ── char_occur ─────────────────────────────────────────────────────────────

    "char_occur": [

        # ── drop_detect SC: post-hump descent of incorrect traces ──────────────
        _dd("sc",  5, 30, "local_max", min_position=500),
        _dd("sc",  8, 30, "local_max", min_position=500),
        _dd("sc",  8, 50, "local_max", min_position=500),
        _dd("sc", 12, 30, "local_max", min_position=500),
        _dd("sc",  8, 30, "fixed_n",   min_position=500),
        _dd("sc",  5, 30, "local_max", min_position=700),
        _dd("sc",  8, 30, "local_max", min_position=700),
        _dd("sc",  8, 50, "local_max", min_position=700),
        _dd("sc", 12, 30, "local_max", min_position=700),

        # ── threshold SC: post-hump incorrect range 27–30 ─────────────────────
        _thr("sc", 28, 3, min_position=500),
        _thr("sc", 28, 5, min_position=500),
        _thr("sc", 33, 3, min_position=500),
        _thr("sc", 33, 5, min_position=500),
        _thr("sc", 30, 3, min_position=700),
        _thr("sc", 28, 3, min_position=700),
        _thr("sc", 28, 5, min_position=700),

        # ── think_budget SC: inject </think> at onset of post-hump drop ────────
        # min_position=400 is just past the ascending phase; drop detects descent.
        _tb("drop",  2048, drop_threshold=5,  window=20, min_position=400),
        _tb("drop",  2048, drop_threshold=6,  window=30, min_position=400),
        _tb("drop",  2048, drop_threshold=6,  window=50, min_position=400),
        _tb("drop",  1024, drop_threshold=6,  window=30, min_position=400),
        _tb("drop_or_fixed",  1024, drop_threshold=6,  window=30, min_position=400),
        _tb("fixed", 1024, min_position=400),
        _tb("fixed", 2048, min_position=400),

        # ── z_score_sc: drop & threshold (exploratory — z-score less clean here)
        _dd("z_score_sc", 0.5, 30, "local_max", min_position=400),
        _dd("z_score_sc", 0.6, 50, "local_max", min_position=400),
        _thr("z_score_sc", -0.20, 3, min_position=500),
        _thr("z_score_sc", -0.30, 5, min_position=500),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.2, start_token=500, end_token=1132),
        _lc(0.8, 0.2, start_token=500, end_token=1132),
        _lc(0.6, 0.15, start_token=700, end_token=1132),
        _lc(0.8, 0.15, start_token=700, end_token=1132),

        # ── confidence_cooling SC ──────────────────────────────────────────────
        _cc("sc", 33, sustained_window=30, dropout_window=15,
            cool_temp=0.25, min_position=500),
        _cc("sc", 35, sustained_window=20, dropout_window=10,
            cool_temp=0.2, min_position=700),

        # ── drop_steering SC ───────────────────────────────────────────────────
        _ds("sc",
            "\nWait, let me carefully recount the occurrences of each character.\n",
            drop_threshold=6, window=30, min_position=500, cooldown=100),
        _ds("sc",
            "\nWait, let me carefully recount the occurrences of each character.\n",
            drop_threshold=8, window=50, min_position=700, cooldown=120,
            also_lower_temp=0.3),

        # ── confident_region_end SC ────────────────────────────────────────────
        _cre("sc", confidence_threshold=34, sustained_window=40, drop_threshold=5,
             steering_text="\nI need to verify my character count more carefully.\n",
             cool_to_temp=0.2, min_position=500),

        # ── composite: any — broad post-hump backtrack net ─────────────────────
        _composite("any", [
            _dd("sc",  5, 30, "local_max", min_position=500),
            _thr("sc", 28, 5, min_position=500),
        ]),
    ],

    # ── distinct_char ──────────────────────────────────────────────────────────
    "distinct_char": [

        # ── drop_detect SC ─────────────────────────────────────────────────────
        _dd("sc", 3, 20, "local_max", min_position=400),
        _dd("sc", 5, 20, "local_max", min_position=400),
        _dd("sc", 5, 30, "local_max", min_position=400),
        _dd("sc", 8, 30, "local_max", min_position=400),
        _dd("sc", 8, 50, "local_max", min_position=400),
        _dd("sc", 5, 30, "local_max", min_position=600),
        _dd("sc", 8, 30, "local_max", min_position=600),
        _dd("sc", 8, 50, "local_max", min_position=600),
        _dd("sc", 5, 30, "fixed_n",   min_position=400),

        # ── threshold SC: incorrect 26–30, correct 32–36 ──────────────────────
        _thr("sc", 28, 3, min_position=400),
        _thr("sc", 28, 5, min_position=400),
        _thr("sc", 31, 3, min_position=400),
        _thr("sc", 31, 5, min_position=400),
        _thr("sc", 28, 3, min_position=600),
        _thr("sc", 31, 3, min_position=600),
        _thr("sc", 31, 5, min_position=600),

        # ── think_budget SC ─────────────────────────────────────────────────────
        _tb("drop",  2048, drop_threshold=5, window=30, min_position=400),
        _tb("drop",  2048, drop_threshold=8, window=30, min_position=400),
        _tb("drop",  1024, drop_threshold=5, window=30, min_position=400),
        _tb("drop_or_fixed",  1024, drop_threshold=5, window=30, min_position=400),
        _tb("fixed", 1024),
        _tb("fixed", 2048),

        # ── z_score_sc drop & threshold ────────────────────────────────────────
        _dd("z_score_sc", 0.3, 20, "local_max", min_position=400),
        _dd("z_score_sc", 0.5, 30, "local_max", min_position=400),
        _dd("z_score_sc", 0.5, 30, "local_max", min_position=600),
        _thr("z_score_sc", -0.20, 3, min_position=400),
        _thr("z_score_sc", -0.30, 3, min_position=400),
        _thr("z_score_sc", -0.30, 5, min_position=400),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.25, start_token=400, end_token=1372),
        _lc(0.8, 0.25, start_token=400, end_token=1372),
        _lc(0.6, 0.2,  start_token=600, end_token=1372),
        _lc(0.8, 0.2,  start_token=600, end_token=1372),

        # ── confidence_cooling SC ──────────────────────────────────────────────
        _cc("sc", 30, sustained_window=30, dropout_window=15,
            cool_temp=0.25, min_position=400),
        _cc("sc", 32, sustained_window=25, dropout_window=15,
            cool_temp=0.2, min_position=600),

        # ── drop_steering SC ───────────────────────────────────────────────────
        _ds("sc",
            "\nWait, let me re-examine which characters are distinct in this string.\n",
            drop_threshold=4, window=20, min_position=400, cooldown=80,
            max_injections=2),
        _ds("sc",
            "\nWait, let me re-examine which characters are distinct in this string.\n",
            drop_threshold=5, window=30, min_position=600, cooldown=100,
            also_lower_temp=0.3, max_injections=2),

        # ── confident_region_end SC ────────────────────────────────────────────
        _cre("sc", confidence_threshold=31, sustained_window=35, drop_threshold=4,
             steering_text="\nLet me systematically list the distinct characters again.\n",
             cool_to_temp=0.25, min_position=400),

        # ── composite: any (z_score_sc) — relative-drop OR absolute floor ───────
        _composite("any", [
            _dd("z_score_sc", 0.4, 20, "local_max", min_position=400),
            _thr("z_score_sc", -0.25, 3, min_position=400),
        ]),
        # ── composite: all — conservative AND filter ───────────────────────────
        _composite("all", [
            _dd("sc", 4, 20, "local_max", min_position=400),
            _thr("sc", 28, 3, min_position=400),
        ]),
    ],

    # ── word_len ───────────────────────────────────────────────────────────────
    "word_len": [

        # ── drop_detect SC: sensitive settings for the small 2–4 KL gap ────────
        _dd("sc", 2, 15, "local_max", min_position=200),
        _dd("sc", 3, 15, "local_max", min_position=200),
        _dd("sc", 3, 25, "local_max", min_position=200),
        _dd("sc", 5, 15, "local_max", min_position=200),
        _dd("sc", 5, 25, "local_max", min_position=200),
        _dd("sc", 3, 15, "local_max", min_position=300),
        _dd("sc", 5, 20, "local_max", min_position=300),
        _dd("sc", 3, 15, "fixed_n",   min_position=200),

        # ── threshold SC: tight thresholds just below incorrect range 28–32 ────
        _thr("sc", 27, 3, min_position=200),
        _thr("sc", 27, 5, min_position=200),
        _thr("sc", 29, 3, min_position=200),
        _thr("sc", 29, 5, min_position=200),
        _thr("sc", 27, 3, min_position=300),
        _thr("sc", 29, 3, min_position=300),

        # ── think_budget fixed: token-efficiency sweep (main focus here) ────────
        _tb("fixed",  256),
        _tb("fixed",  512),
        _tb("fixed", 1024),

        # ── think_budget drop: SC-triggered early exit ─────────────────────────
        _tb("drop",  512, drop_threshold=3, window=15, min_position=150),
        _tb("drop", 1024, drop_threshold=3, window=15, min_position=150),
        _tb("drop", 1024, drop_threshold=5, window=20, min_position=200),
        _tb("drop_or_fixed",  512, drop_threshold=3, window=15, min_position=150),

        # ── think_budget wait ──────────────────────────────────────────────────
        _tb("wait", 1024, wait_threshold=31, wait_exit_on="high"),
        _tb("wait", 2048, wait_threshold=31, wait_exit_on="high"),

        # ── wait_backtrack SC ──────────────────────────────────────────────────
        _wb("low", threshold=29, window=20, min_position=200, cooldown=60),
        _wb("any", window=20, min_position=200, cooldown=80),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.15, start_token=100, end_token=600),
        _lc(0.8, 0.15, start_token=100, end_token=600),
        _lc(0.6, 0.1,  start_token=150, end_token=500),   # fastest commitment
        _lc(0.8, 0.1,  start_token=150, end_token=500),   # fastest commitment
        _lc(0.6, 0.2,  start_token=200, end_token=777),   # gentler, full trace

        # ── confidence_cooling SC ──────────────────────────────────────────────
        _cc("sc", 31, sustained_window=20, dropout_window=10,
            cool_temp=0.2, min_position=200),

        # ── drop_steering SC ───────────────────────────────────────────────────
        _ds("sc",
            "\nWait, let me recount the number of characters in each word.\n",
            drop_threshold=3, window=15, min_position=200, cooldown=100,
            max_injections=1),

        # ── confident_region_end SC ────────────────────────────────────────────
        _cre("sc", confidence_threshold=31, sustained_window=25, drop_threshold=3,
             steering_text="\nI should verify the word length one character at a time.\n",
             cool_to_temp=0.2, min_position=200, max_position=600),

        # ── composite: all — require BOTH drop AND floor (noise filter) ─────────
        _composite("all", [
            _dd("sc", 3, 15, "local_max", min_position=200),
            _thr("sc", 27, 3, min_position=200),
        ]),
    ],

    # ── substring_occur ────────────────────────────────────────────────────────
    "substring_occur": [

        # ── drop_detect SC: post-dip region, sensitive to incorrect plateau ─────
        _dd("sc",  5, 30, "local_max", min_position=400),
        _dd("sc",  5, 50, "local_max", min_position=400),
        _dd("sc",  8, 30, "local_max", min_position=400),
        _dd("sc",  8, 50, "local_max", min_position=400),
        _dd("sc", 10, 30, "local_max", min_position=400),
        _dd("sc",  5, 30, "local_max", min_position=600),
        _dd("sc",  8, 30, "local_max", min_position=600),
        _dd("sc",  8, 50, "local_max", min_position=600),
        _dd("sc",  8, 30, "fixed_n",   min_position=400),

        # ── threshold SC: incorrect post-dip 24–28, correct 32–37 ─────────────
        _thr("sc", 27, 3, min_position=400),
        _thr("sc", 27, 5, min_position=400),
        _thr("sc", 30, 3, min_position=400),
        _thr("sc", 30, 5, min_position=400),
        _thr("sc", 27, 3, min_position=600),
        _thr("sc", 30, 3, min_position=600),
        _thr("sc", 30, 5, min_position=600),

        # ── think_budget fixed:
        _tb("fixed", 1024),
        _tb("fixed", 2048),
        _tb("fixed", 3072),

        # ── think_budget drop: inject </think> on first significant SC drop ─────
        _tb("drop", 2048, drop_threshold=5, window=30, min_position=300),
        _tb("drop", 2048, drop_threshold=8, window=30, min_position=300),
        _tb("drop", 2048, drop_threshold=8, window=50, min_position=300),
        _tb("drop", 1024, drop_threshold=5, window=30, min_position=300),
        _tb("drop_or_fixed", 2048, drop_threshold=8, window=30, min_position=300),

        # ── think_budget wait ──────────────────────────────────────────────────
        _tb("wait", 3072, wait_threshold=33, wait_exit_on="high"),
        _tb("wait_or_drop", 2048, drop_threshold=8, window=30,
            wait_threshold=33, wait_exit_on="high", min_position=300),

        # ── z_score_sc drop & threshold: normalises out the structural dip ──────
        _dd("z_score_sc", 0.4, 30, "local_max", min_position=400),
        _dd("z_score_sc", 0.6, 30, "local_max", min_position=400),
        _dd("z_score_sc", 0.6, 50, "local_max", min_position=400),
        _thr("z_score_sc", -0.30, 3, min_position=400),
        _thr("z_score_sc", -0.40, 3, min_position=400),
        _thr("z_score_sc", -0.40, 5, min_position=400),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.1,  start_token=400, end_token=2072),   # most aggressive
        _lc(0.8, 0.1,  start_token=400, end_token=2072),   
        _lc(0.6, 0.15, start_token=400, end_token=1500),   # commit earlier
        _lc(0.8, 0.15, start_token=400, end_token=1500),   # commit earlier
        _lc(0.6, 0.2,  start_token=600, end_token=2072),   # post-dip, gentler

        # ── confidence_cooling SC ──────────────────────────────────────────────
        _cc("sc", 33, sustained_window=30, dropout_window=15,
            cool_temp=0.15, min_position=400),
        _cc("sc", 30, sustained_window=40, dropout_window=20,
            cool_temp=0.2, min_position=600),

        # ── drop_steering SC ───────────────────────────────────────────────────
        _ds("sc",
            "\nWait, let me search for the substring more systematically.\n",
            drop_threshold=6, window=30, min_position=400, cooldown=120,
            also_lower_temp=0.2),

        # ── confident_region_end SC ────────────────────────────────────────────
        _cre("sc", confidence_threshold=33, sustained_window=40, drop_threshold=5,
             steering_text="\nI should use a more careful scanning approach to find all occurrences.\n",
             cool_to_temp=0.15, min_position=400),
        _cre("sc", confidence_threshold=30, sustained_window=50, drop_threshold=6,
             cool_to_temp=0.1, min_position=600),

        # ── composite: any — broad trigger to catch the unstable late phase ─────
        _composite("any", [
            _dd("sc",  6, 30, "local_max", min_position=400),
            _thr("sc", 27, 5, min_position=400),
        ]),
    ],

    # ── hmmt2025 ───────────────────────────────────────────────────────────────
    "hmmt2025": [

        # ── drop_detect SC ─────────────────────────────────────────────────────
        _dd("sc",  5, 30, "local_max", min_position=400),
        _dd("sc",  5, 50, "local_max", min_position=400),
        _dd("sc",  8, 30, "local_max", min_position=400),
        _dd("sc",  8, 50, "local_max", min_position=400),
        _dd("sc", 12, 30, "local_max", min_position=400),
        _dd("sc",  5, 50, "local_max", min_position=800),
        _dd("sc",  8, 30, "local_max", min_position=800),
        _dd("sc",  8, 50, "local_max", min_position=800),
        _dd("sc",  8, 30, "fixed_n",   min_position=400),

        # ── threshold SC ───────────
        _thr("sc", 24, 3, min_position=400),
        _thr("sc", 24, 5, min_position=400),
        _thr("sc", 26, 3, min_position=400),
        _thr("sc", 26, 5, min_position=400),
        _thr("sc", 29, 5, min_position=400),
        _thr("sc", 24, 3, min_position=800),
        _thr("sc", 26, 3, min_position=800),

        # ── think_budget SC ─────────────────────────────────────────────────────
        _tb("fixed", 1024),
        _tb("fixed", 2048),
        _tb("fixed", 4096),
        # Drop-triggered budget cap using SC; late-falling incorrect traces fire first.
        _tb("drop", 3000, drop_threshold=5, window=50, min_position=400),
        _tb("drop", 3000, drop_threshold=8, window=50, min_position=400),
        _tb("drop", 2000, drop_threshold=8, window=50, min_position=400),

        # ── wait_backtrack SC ──────────────────────────────────────────────────
        # "high": backtrack when Wait emitted but model WAS confident (trust it).
        # "low":  backtrack when Wait emitted AND model was already uncertain.
        _wb("high", threshold=32, window=30, min_position=400, cooldown=80),
        _wb("high", threshold=35, window=30, min_position=400, cooldown=80),
        _wb("low",  threshold=25, window=30, min_position=800, cooldown=80),

        # ── z_score_sc drop & threshold (BEST metric for hmmt) ─────────────────
        _dd("z_score_sc", 0.3, 30, "local_max", min_position=400),
        _dd("z_score_sc", 0.5, 30, "local_max", min_position=400),
        _dd("z_score_sc", 0.5, 50, "local_max", min_position=400),
        _dd("z_score_sc", 0.5, 30, "local_max", min_position=800),
        _dd("z_score_sc", 0.7, 50, "local_max", min_position=800),
        _thr("z_score_sc", -0.20, 3, min_position=400),
        _thr("z_score_sc", -0.30, 3, min_position=400),
        _thr("z_score_sc", -0.30, 5, min_position=400),
        _thr("z_score_sc", -0.30, 3, min_position=800),

        # ── think_budget z_score_sc drop ───────────────────────────────────────
        _tb("drop", 3000, metric="z_score_sc", drop_threshold=0.4, window=50, min_position=400),
        _tb("drop", 3000, metric="z_score_sc", drop_threshold=0.6, window=50, min_position=400),

        # ── neg_entropy exploratory (correct ~−0.35, incorrect ~−0.50) ─────────
        _thr("neg_entropy", -0.45, 3, min_position=800),
        _thr("neg_entropy", -0.45, 5, min_position=800),
        _thr("neg_entropy", -0.50, 3, min_position=800),
        _dd("neg_entropy", 0.10, 30, "local_max", min_position=800),
        _dd("neg_entropy", 0.15, 30, "local_max", min_position=800),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.15, start_token=1000, end_token=8000),
        _lc(0.8, 0.15, start_token=1000, end_token=8000),
        _lc(0.6, 0.1,  start_token=2000, end_token=8000),   # very late onset
        _lc(0.8, 0.1,  start_token=2000, end_token=8000),   # very late onset

        # ── confidence_cooling z_score_sc ──────────────────────────────────────
        _cc("z_score_sc", 0.3, sustained_window=40, dropout_window=20,
            cool_temp=0.2, min_position=400),
        _cc("z_score_sc", 0.4, sustained_window=30, dropout_window=15,
            cool_temp=0.15, min_position=800),

        # ── confidence_cooling SC ──────────────────────────────────────────────
        _cc("sc", 30, sustained_window=40, dropout_window=20,
            cool_temp=0.2, min_position=800),

        # ── drop_steering z_score_sc ───────────────────────────────────────────
        _ds("z_score_sc",
            "\nWait, I think I may have made an error. Let me reconsider my approach.\n",
            drop_threshold=0.4, window=30, min_position=800, cooldown=120,
            max_injections=2),

        # ── confident_region_end z_score_sc ────────────────────────────────────
        _cre("z_score_sc", confidence_threshold=0.2, sustained_window=50,
             drop_threshold=0.4,
             steering_text="\nHmm, let me re-examine this part of my solution more carefully.\n",
             cool_to_temp=0.2, min_position=400),

        # ── composite: any (z_score_sc) — broad late trigger ───────────────────
        _composite("any", [
            _dd("z_score_sc", 0.4, 30, "local_max", min_position=800),
            _thr("z_score_sc", -0.3, 3, min_position=800),
        ]),
        # ── composite: all (SC) — conservative double-confirm ──────────────────
        _composite("all", [
            _dd("sc", 5, 30, "local_max", min_position=800),
            _thr("sc", 25, 3, min_position=800),
        ]),
    ],

    # ── aime24 ─────────────────────────────────────────────────────────────────
    "aime24": [

        # ── drop_detect SC ─────────────────────────────────────────────────────
        _dd("sc",  5, 30, "local_max", min_position=1000),
        _dd("sc",  8, 30, "local_max", min_position=1000),
        _dd("sc",  8, 50, "local_max", min_position=1000),
        _dd("sc", 12, 30, "local_max", min_position=1000),
        _dd("sc", 12, 50, "local_max", min_position=1000),
        _dd("sc",  5, 50, "local_max", min_position=2000),
        _dd("sc",  8, 30, "local_max", min_position=2000),
        _dd("sc",  8, 50, "local_max", min_position=2000),
        _dd("sc", 12, 50, "local_max", min_position=2000),
        _dd("sc",  8, 30, "fixed_n",   min_position=1000),

        # ── threshold SC: incorrect 22–26, correct 28–36 ──────────────────────
        _thr("sc", 24, 3, min_position=1000),
        _thr("sc", 24, 5, min_position=1000),
        _thr("sc", 27, 3, min_position=1000),
        _thr("sc", 27, 5, min_position=1000),
        # _thr("sc", 30, 3, min_position=1000),
        _thr("sc", 24, 3, min_position=2000),
        _thr("sc", 27, 3, min_position=2000),
        _thr("sc", 27, 5, min_position=2000),
        # _thr("sc", 30, 5, min_position=2000),

        # ── think_budget SC ─────────────────────────────────────────────────────
        _tb("fixed", 2048),
        _tb("fixed", 4096),
        _tb("drop", 4096, drop_threshold=5, window=50, min_position=500),
        _tb("drop", 4096, drop_threshold=8, window=50, min_position=500),
        _tb("drop", 4096, drop_threshold=8, window=50, min_position=1000),
        _tb("drop_or_fixed", 4096, drop_threshold=8, window=50, min_position=500),

        # ── think_budget wait SC ───────────────────────────────────────────────
        _tb("wait", 6000, wait_window=50, wait_threshold=30, wait_exit_on="high",
            min_position=1000),
        _tb("wait", 6000, wait_window=50, wait_threshold=32, wait_exit_on="high",
            min_position=1000),
        _tb("wait_or_drop", 4096, drop_threshold=8, window=50,
            wait_window=50, wait_threshold=30, wait_exit_on="high",
            min_position=1000),

        # ── wait_backtrack SC ──────────────────────────────────────────────────
        _wb("high", threshold=32, window=50, min_position=1000, cooldown=150),
        _wb("high", threshold=35, window=50, min_position=1000, cooldown=150),

        # ── z_score_sc drop & threshold ────────────────────────────────────────
        _dd("z_score_sc", 0.3, 30, "local_max", min_position=1000),
        _dd("z_score_sc", 0.5, 50, "local_max", min_position=1000),
        _dd("z_score_sc", 0.5, 50, "local_max", min_position=2000),
        _dd("z_score_sc", 0.7, 50, "local_max", min_position=2000),
        _thr("z_score_sc", -0.20, 3, min_position=1000),
        _thr("z_score_sc", -0.30, 3, min_position=1000),
        _thr("z_score_sc", -0.30, 5, min_position=1000),
        _thr("z_score_sc", -0.30, 3, min_position=2000),

        # ── think_budget z_score_sc ────────────────────────────────────────────
        _tb("drop", 4096, metric="z_score_sc", drop_threshold=0.4, window=50,
            min_position=500),
        _tb("drop", 4096, metric="z_score_sc", drop_threshold=0.6, window=50,
            min_position=1000),
        _tb("wait", 6000, metric="z_score_sc", wait_window=50,
            wait_threshold=0.3, wait_exit_on="high", min_position=1000),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.15, start_token=2000, end_token=12000),
        _lc(0.8, 0.15, start_token=2000, end_token=12000),
        _lc(0.6, 0.1,  start_token=4000, end_token=12000),   # very late, commit
        _lc(0.8, 0.1,  start_token=4000, end_token=12000),   # very late, commit

        # ── confidence_cooling z_score_sc ──────────────────────────────────────
        _cc("z_score_sc", 0.3, sustained_window=50, dropout_window=25,
            cool_temp=0.2, min_position=2000),
        _cc("z_score_sc", 0.5, sustained_window=40, dropout_window=20,
            cool_temp=0.15, min_position=2000),

        # ── drop_steering z_score_sc ───────────────────────────────────────────
        _ds("z_score_sc",
            "\nWait, let me re-examine this step. I may have made an error.\n",
            drop_threshold=0.5, window=50, min_position=2000, cooldown=200,
            max_injections=2),

        # ── confident_region_end z_score_sc ────────────────────────────────────
        _cre("z_score_sc", confidence_threshold=0.3, sustained_window=80,
             drop_threshold=0.5,
             steering_text="\nLet me reconsider my solution approach from this point.\n",
             cool_to_temp=0.2, min_position=2000),

        # ── composite: any (z_score_sc) — catches both drop and floor ──────────
        _composite("any", [
            _dd("z_score_sc", 0.4, 50, "local_max", min_position=2000),
            _thr("z_score_sc", -0.3, 3, min_position=2000),
        ]),
        # ── composite: all (SC) — conservative filter past the early spike ──────
        _composite("all", [
            _dd("sc",  8, 50, "local_max", min_position=2000),
            _thr("sc", 24, 3, min_position=2000),
        ]),
    ],

    # ── gpqa_diamond ───────────────────────────────────────────────────────────
    "gpqa_diamond": [

        # ── drop_detect SC: post-dip, min_position ≥ 2000 ─────────────────────
        _dd("sc",  5, 30, "local_max", min_position=2000),
        _dd("sc",  7, 30, "local_max", min_position=2000),
        _dd("sc",  7, 50, "local_max", min_position=2000),
        _dd("sc", 9, 30, "local_max", min_position=2000),
        _dd("sc", 9, 50, "local_max", min_position=2000),
        _dd("sc",  5, 50, "local_max", min_position=3000),
        _dd("sc",  7, 30, "local_max", min_position=3000),
        _dd("sc",  7, 50, "local_max", min_position=3000),
        _dd("sc", 9, 50, "local_max", min_position=3000),
        _dd("sc",  7, 30, "fixed_n",   min_position=2000),

        # ── threshold SC: incorrect post-dip 24–28, correct 38–42 ─────────────
        _thr("sc", 26, 3, min_position=2000),
        _thr("sc", 26, 5, min_position=2000),
        _thr("sc", 29, 3, min_position=2000),
        _thr("sc", 29, 5, min_position=2000),
        _thr("sc", 32, 3, min_position=2000),
        _thr("sc", 26, 3, min_position=3000),
        _thr("sc", 29, 3, min_position=3000),
        _thr("sc", 32, 5, min_position=3000),

        # ── think_budget SC ─────────────────────────────────────────────────────
        _tb("fixed", 2048),
        _tb("fixed", 4096),
        _tb("drop", 4096, drop_threshold=5, window=30, min_position=1000),
        _tb("drop", 4096, drop_threshold=8, window=30, min_position=1000),
        _tb("drop", 4096, drop_threshold=8, window=50, min_position=2000),
        _tb("drop_or_fixed", 4096, drop_threshold=8, window=30, min_position=1000),

        # ── think_budget wait SC: min_position=2000 skips U-curve ─────────────
        _tb("wait", 6000, wait_window=50, wait_threshold=34, wait_exit_on="high",
            min_position=2000),
        _tb("wait", 6000, wait_window=50, wait_threshold=38, wait_exit_on="high",
            min_position=2000),
        _tb("wait_or_drop", 4096, drop_threshold=8, window=30,
            wait_window=50, wait_threshold=34, wait_exit_on="high",
            min_position=2000),

        # ── wait_backtrack SC ──────────────────────────────────────────────────
        _wb("high", threshold=34, window=50, min_position=2000, cooldown=100),
        _wb("high", threshold=38, window=50, min_position=2000, cooldown=100),
        _wb("high", threshold=34, window=50, backtrack_mode="fixed_n",
            min_position=2000, cooldown=100),

        # ── z_score_sc drop & threshold ────────────────────────────────────────
        _dd("z_score_sc", 0.4, 30, "local_max", min_position=2000),
        _dd("z_score_sc", 0.5, 50, "local_max", min_position=2000),
        _dd("z_score_sc", 0.6, 50, "local_max", min_position=2000),
        _dd("z_score_sc", 0.6, 50, "local_max", min_position=3000),
        _thr("z_score_sc", -0.20, 3, min_position=2000),
        _thr("z_score_sc", -0.30, 3, min_position=2000),
        _thr("z_score_sc", -0.30, 5, min_position=2000),
        _thr("z_score_sc", -0.30, 3, min_position=3000),

        # ── think_budget z_score_sc ────────────────────────────────────────────
        _tb("drop", 4096, metric="z_score_sc", drop_threshold=0.4, window=50,
            min_position=2000),
        _tb("drop", 4096, metric="z_score_sc", drop_threshold=0.6, window=50,
            min_position=2000),
        _tb("wait", 6000, metric="z_score_sc", wait_window=50,
            wait_threshold=0.3, wait_exit_on="high", min_position=2000),

        # ── wait_backtrack z_score_sc ──────────────────────────────────────────
        _wb("high", threshold=0.3, window=50, metric="z_score_sc",
            min_position=2000, cooldown=100),

        # ── neg_entropy exploratory (correct ~−0.25, incorrect ~−0.40) ─────────
        _thr("neg_entropy", -0.35, 3, min_position=2000),
        _thr("neg_entropy", -0.40, 3, min_position=2000),
        _thr("neg_entropy", -0.40, 5, min_position=2000),
        _dd("neg_entropy", 0.10, 30, "local_max", min_position=2000),
        _dd("neg_entropy", 0.15, 30, "local_max", min_position=2000),

        # ── linear_cooling ─────────────────────────────────────────────────────
        _lc(0.6, 0.1,  start_token=2000, end_token=8000),   # most aggressive
        _lc(0.8, 0.1,  start_token=2000, end_token=8000),
        _lc(0.6, 0.15, start_token=3000, end_token=8000),   # later onset
        _lc(0.8, 0.15, start_token=3000, end_token=8000),

        # ── confidence_cooling SC ──────────────────────────────────────────────
        _cc("sc", 36, sustained_window=50, dropout_window=20,
            cool_temp=0.1, min_position=2000),
        _cc("sc", 33, sustained_window=60, dropout_window=25,
            cool_temp=0.15, min_position=3000),

        # ── confidence_cooling z_score_sc ──────────────────────────────────────
        _cc("z_score_sc", 0.4, sustained_window=50, dropout_window=20,
            cool_temp=0.15, min_position=2000),

        # ── drop_steering SC ───────────────────────────────────────────────────
        _ds("sc",
            "\nWait, I should reconsider which option is more scientifically accurate.\n",
            drop_threshold=7, window=30, min_position=2000, cooldown=150,
            also_lower_temp=0.2, max_injections=2),
        _ds("sc",
            "\nWait, I should reconsider which option is more scientifically accurate.\n",
            drop_threshold=9, window=50, min_position=3000, cooldown=150,
            max_injections=2),

        # ── confident_region_end SC ────────────────────────────────────────────
        _cre("sc", confidence_threshold=36, sustained_window=50, drop_threshold=6,
             steering_text="\nLet me re-evaluate the scientific reasoning for each option.\n",
             cool_to_temp=0.1, min_position=2000),
        _cre("sc", confidence_threshold=33, sustained_window=60, drop_threshold=7,
             cool_to_temp=0.15, min_position=3000),

        # ── composite: any (SC) — broad trigger post U-curve ───────────────────
        _composite("any", [
            _dd("sc",  7, 30, "local_max", min_position=2000),
            _thr("sc", 28, 5, min_position=2000),
        ]),
        # ── composite: all (SC) — precision filter for large-gap dataset ───────
        _composite("all", [
            _dd("sc",  7, 30, "local_max", min_position=2000),
            _thr("sc", 28, 3, min_position=2000),
        ]),
        # ── composite: any (z_score_sc) — relative signal ──────────────────────
        _composite("any", [
            _dd("z_score_sc", 0.5, 50, "local_max", min_position=2000),
            _thr("z_score_sc", -0.3, 3, min_position=2000),
        ]),
    ],
}


# ── Job name generation ───────────────────────────────────────────────────────

def _job_name(cfg):
    s  = cfg["strategy"]
    m  = cfg["metric"]
    kw = cfg["strategy_kwargs"]

    # Short metric prefix for z_score_sc and neg_entropy to keep names concise.
    m_sfx = {"sc": "sc", "z_score_sc": "zs", "neg_entropy": "ne", "nll": "nll"}.get(m, m)

    if s == "linear_cooling":
        st = str(kw["start_temp"]).replace(".", "p")
        et = str(kw["end_temp"]).replace(".", "p")
        s0 = kw["start_token"]
        e0 = kw["end_token"]
        return f"lc_st{st}_et{et}_s{s0}_e{e0}"

    if s == "confidence_cooling":
        thr  = str(abs(kw["confidence_threshold"])).replace(".", "p")
        sign = "n" if kw["confidence_threshold"] < 0 else ""
        sw   = kw["sustained_window"]
        cool = str(kw["cool_temp"]).replace(".", "p")
        mp   = kw.get("min_position", 80)
        return f"cc_{m_sfx}_{sign}{thr}_sw{sw}_cool{cool}_mp{mp}"

    if s == "drop_steering":
        dt   = str(kw["drop_threshold"]).replace(".", "p")
        w    = kw["window"]
        mp   = kw.get("min_position", 80)
        tsf  = "_cool" if kw.get("also_lower_temp") is not None else ""
        return f"ds_{m_sfx}_dt{dt}_w{w}_mp{mp}{tsf}"

    if s == "confident_region_end":
        thr  = str(abs(kw["confidence_threshold"])).replace(".", "p")
        sign = "n" if kw["confidence_threshold"] < 0 else ""
        sw   = kw["sustained_window"]
        dt   = str(kw["drop_threshold"]).replace(".", "p")
        mp   = kw.get("min_position", 0)
        act  = ("steer" if kw.get("steering_text") else "") + \
               ("cool"  if kw.get("cool_to_temp")  else "")
        act_sfx = f"_{act}" if act else ""
        return f"cre_{m_sfx}_{sign}{thr}_sw{sw}_dt{dt}_mp{mp}{act_sfx}"

    if s in ("any", "all"):
        def _sub_abbrev(sub):
            sn = sub["name"]
            sk = sub["kwargs"]
            if sn == "drop_detect":
                dt = str(sk.get("drop_threshold", "")).replace(".", "p")
                mp = sk.get("min_position", 80)
                return f"dd{dt}mp{mp}"
            if sn == "threshold":
                thr  = str(abs(sk.get("threshold", ""))).replace(".", "p")
                sign = "n" if float(sk.get("threshold", 0)) < 0 else ""
                return f"thr{sign}{thr}"
            return sn[:4]
        subs = kw.get("sub_strategies", [])
        abbrevs = "_".join(_sub_abbrev(sub) for sub in subs[:2])
        return f"{s}_{m_sfx}_{abbrevs}"

    if s == "drop_detect":
        dt   = str(kw["drop_threshold"]).replace(".", "p")
        w    = kw["window"]
        mode = "lm" if kw["backtrack_mode"] == "local_max" else "fn"
        mp   = kw.get("min_position", 80)
        return f"dd_{m_sfx}_dt{dt}_w{w}_{mode}_mp{mp}"

    if s == "threshold":
        thr  = str(abs(kw["threshold"])).replace(".", "p")
        sign = "n" if kw["threshold"] < 0 else ""
        cw   = kw["confirm_window"]
        mp   = kw.get("min_position", 80)
        return f"thr_{m_sfx}_{sign}{thr}_cw{cw}_mp{mp}"

    if s == "think_budget":
        trigger = kw["trigger"]
        budget  = kw["max_think_tokens"]
        mp      = kw.get("min_position", 100)
        mp_sfx  = f"_mp{mp}" if mp != 100 else ""
        if trigger == "fixed":
            return f"tb_fixed_m{budget}{mp_sfx}"
        if trigger in ("drop", "drop_or_fixed", "any"):
            dt = str(kw.get("drop_threshold", "")).replace(".", "p")
            w  = kw.get("window", "")
            return f"tb_{trigger}_{m_sfx}_dt{dt}_w{w}_m{budget}{mp_sfx}"
        # wait-based triggers
        eo  = kw.get("wait_exit_on", "any")
        thr = str(kw.get("wait_threshold", "")).replace(".", "p")
        eo_sfx = eo if eo == "any" else f"{eo}{thr}"
        short = {"wait": "wait", "wait_or_drop": "wod", "wait_or_fixed": "wof"}[trigger]
        if trigger == "wait_or_drop":
            dt = str(kw.get("drop_threshold", "")).replace(".", "p")
            w  = kw.get("window", "")
            return f"tb_{m_sfx}_{short}_{eo_sfx}_dt{dt}_w{w}_m{budget}{mp_sfx}"
        return f"tb_{m_sfx}_{short}_{eo_sfx}_m{budget}{mp_sfx}"

    if s == "wait_backtrack":
        eo   = kw["exit_on"]
        mp   = kw.get("min_position", 80)
        mode = "lm" if kw.get("backtrack_mode", "local_max") == "local_max" else "fn"
        w    = kw.get("window", 20)
        if eo == "any":
            return f"wb_{m_sfx}_any_w{w}_{mode}_mp{mp}"
        thr = str(kw.get("threshold", "")).replace(".", "p")
        sign = "n" if float(kw.get("threshold", 0)) < 0 else ""
        return f"wb_{m_sfx}_{eo}_{sign}thr{thr}_w{w}_{mode}_mp{mp}"

    return f"{s}_{m}"


# ── PBS script templates ──────────────────────────────────────────────────────

_HEADER = """\
#!/bin/bash
#PBS -N {pbs_name}
#PBS -o {log_dir}/{job_name}.log
#PBS -e {log_dir}/{job_name}.err
#PBS -l select=1:ncpus=8:ngpus=1:mem=50gb:gpu_mem=40gb
#PBS -l walltime={walltime}
#PBS -q gpu
#PBS -j oe
#PBS -m ae
#PBS -M {email}

set -e

export TMPDIR={home}/tmp
export HF_HOME={home}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TORCH_HOME={home}/torch_cache
export XDG_CACHE_HOME={home}/.cache
mkdir -p $TMPDIR $HF_HOME $TORCH_HOME $XDG_CACHE_HOME

echo "============================================================"
echo "Job:     {job_name}"
echo "PBS ID:  $PBS_JOBID   Node: $(hostname)"
echo "============================================================"
nvidia-smi

source {python_env}
export CUDA_VISIBLE_DEVICES=0
cd {work_dir}
"""

_EXPERIMENT_BODY = """\
python scripts/run_intervention.py \\
  --benchmark {benchmark} \\
  --model {model} \\
  --strategy {strategy} \\
  --metric {metric} \\
  --strategy-kwargs '{strategy_kwargs}' \\
  --n-problems {n_problems} \\
  --seed {seed} \\
  --max-backtracks {max_backtracks} \\
  --max-tokens {max_tokens} \\
  --temperature {temperature} \\
  --top-p {top_p} \\
  --temperature-boost {temperature_boost} \\
  --boost-tokens {boost_tokens} \\
  --checkpoint-interval {checkpoint_interval} \\
  --max-checkpoints 5 \\
  --no-baseline \\
  --output-dir {output_dir}
"""

_BASELINE_BODY = """\
python scripts/run_intervention.py \\
  --benchmark {benchmark} \\
  --model {model} \\
  --strategy no_intervention \\
  --metric sc \\
  --n-problems {n_problems} \\
  --seed {seed} \\
  --max-tokens {max_tokens} \\
  --temperature {temperature} \\
  --top-p {top_p} \\
  --checkpoint-interval 0 \\
  --no-baseline \\
  --output-dir {output_dir}
"""

_FOOTER = """\
echo "============================================================"
echo "Done: {job_name}"
echo "============================================================"
"""


def _render_experiment(cfg, benchmark, job_name, output_dir, n_problems, walltime, max_tokens):
    header = _HEADER.format(
        pbs_name=job_name[:15],
        job_name=job_name,
        log_dir=f"{WORK_DIR}/outputs/sweep_logs",
        walltime=walltime,
        email=EMAIL,
        home=HOME,
        python_env=PYTHON_ENV,
        work_dir=WORK_DIR,
    )
    body = _EXPERIMENT_BODY.format(
        benchmark=benchmark,
        model=MODEL,
        strategy=cfg["strategy"],
        metric=cfg["metric"],
        strategy_kwargs=json.dumps(cfg["strategy_kwargs"]),
        n_problems=n_problems,
        output_dir=output_dir,
        max_tokens=max_tokens,
        **GEN,
    )
    footer = _FOOTER.format(job_name=job_name)
    return header + "\n" + body + "\n" + footer


def _render_baseline(benchmark, job_name, output_dir, n_problems, walltime, max_tokens):
    header = _HEADER.format(
        pbs_name=job_name[:15],
        job_name=job_name,
        log_dir=f"{WORK_DIR}/outputs/sweep_logs",
        walltime=walltime,
        email=EMAIL,
        home=HOME,
        python_env=PYTHON_ENV,
        work_dir=WORK_DIR,
    )
    body = _BASELINE_BODY.format(
        benchmark=benchmark,
        model=MODEL,
        n_problems=n_problems,
        output_dir=output_dir,
        max_tokens=max_tokens,
        **{k: v for k, v in GEN.items() if k in ("seed", "temperature", "top_p")},
    )
    footer = _FOOTER.format(job_name=job_name)
    return header + "\n" + body + "\n" + footer


# ── Submission ────────────────────────────────────────────────────────────────

def _qsub(script_str, dry_run):
    if dry_run:
        print(script_str)
        print("─" * 60)
        return "DRY_RUN"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pbs", delete=False) as f:
        f.write(script_str)
        tmp = f.name
    try:
        out = subprocess.run(["qsub", tmp], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    finally:
        os.unlink(tmp)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description="Submit backtracking sweep3 (all 8 benchmarks, incl. z_score_sc).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", required=True, choices=list(GRIDS),
                        help="Which benchmark grid to submit.")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print PBS scripts to stdout without submitting.")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Submit only the baseline (no_intervention) job.")
    parser.add_argument("--no-baseline",   action="store_true",
                        help="Skip the baseline job (already submitted).")
    args = parser.parse_args()

    bench         = args.benchmark
    bs            = BENCH[bench]
    n_problems    = bs["n_problems"]
    walltime      = bs["walltime"]
    max_tokens    = bs["max_tokens"]
    output_prefix = bs["output_prefix"]
    grid          = GRIDS[bench]
    base_out      = f"{WORK_DIR}/outputs/{output_prefix}_{bench}"
    log_dir       = f"{WORK_DIR}/outputs/sweep_logs"
    timestamp     = datetime.now().strftime("%m%d_%H%M")

    if not args.dry_run:
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nBenchmark : {bench}  ({output_prefix})")
        print(f"Problems  : {n_problems} per job  |  Walltime: {walltime}  |  max_tokens: {max_tokens}")
        print(f"Grid size : {len(grid)} experiment configs + 1 baseline")
        print(f"Output    : {base_out}/")
        print(f"Logs      : {log_dir}/\n")

    submitted = []

    # ── Baseline ─────────────────────────────────────────────────────────────
    if not args.no_baseline:
        job_name   = f"baseline_{bench}"
        output_dir = f"{base_out}/baseline"
        script     = _render_baseline(bench, job_name, output_dir, n_problems, walltime, max_tokens)
        job_id     = _qsub(script, args.dry_run)
        submitted.append({"job_name": job_name, "job_id": job_id})
        if not args.dry_run:
            print(f"  BASELINE   {job_id:<30}  {job_name}")

    if args.baseline_only:
        print(f"\nSubmitted baseline job only.")
        return

    # ── Experiment jobs ───────────────────────────────────────────────────────
    for i, cfg in enumerate(grid):
        job_name   = _job_name(cfg)
        output_dir = f"{base_out}/{job_name}"
        script     = _render_experiment(cfg, bench, job_name, output_dir,
                                         n_problems, walltime, max_tokens)
        job_id     = _qsub(script, args.dry_run)
        submitted.append({
            "job_name":        job_name,
            "job_id":          job_id,
            "strategy":        cfg["strategy"],
            "metric":          cfg["metric"],
            "strategy_kwargs": cfg["strategy_kwargs"],
        })
        if not args.dry_run:
            print(f"  {i+1:>3}/{len(grid)}  {job_id:<30}  {job_name}")

    # ── Save submission log ───────────────────────────────────────────────────
    if not args.dry_run:
        log_path = f"{log_dir}/submitted_{bench}_{timestamp}.json"
        with open(log_path, "w") as f:
            json.dump(submitted, f, indent=2)

        print(f"\nSubmitted {len(submitted)} jobs.")
        print(f"Submission log : {log_path}")
        print(f"\nMonitor        : qstat -u $USER")
        print(f"\nWhen done, collect results:")
        print(f"  python scripts/collect_sweep_results.py \\")
        print(f"    --sweep-dir {base_out} \\")
        print(f"    --baseline-dir {base_out}/baseline")


if __name__ == "__main__":
    main()
