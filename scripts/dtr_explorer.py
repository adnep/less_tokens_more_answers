"""
DTR Explorer — interactive Streamlit app for inspecting JSD heatmaps and DTR.

Run with:
    streamlit run scripts/dtr_explorer.py

On Metacentrum (runs on GPU node, port-forward to local):
    streamlit run scripts/dtr_explorer.py --server.port 8501 --server.headless true
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import yaml
import streamlit as st
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import math
from src.inference.sampler import GeneratedSample
from src.evaluation.voting import extract_answer, majority_vote
from src.evaluation.metrics import _normalize_for_comparison

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DTR Explorer", layout="wide")
st.title("DTR Explorer")

# ──────────────────────────────────────────────────────────────────────────────
# Robust JSONL loader (handles concatenated objects from streaming writes)
# ──────────────────────────────────────────────────────────────────────────────
def _load_jsonl_robust(path: str) -> List[dict]:
    decoder = json.JSONDecoder()
    records = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            pos = 0
            while pos < len(line):
                try:
                    obj, end = decoder.raw_decode(line, pos)
                    records.append(obj)
                    pos = end
                    while pos < len(line) and line[pos] in " \t":
                        pos += 1
                except json.JSONDecodeError:
                    break
    return records


@st.cache_data(show_spinner="Loading samples file…")
def load_and_group(path: str) -> Dict[str, List[GeneratedSample]]:
    records = _load_jsonl_robust(path)
    grouped = defaultdict(list)
    for r in records:
        s = GeneratedSample(
            problem_id=r["problem_id"],
            sample_idx=r["sample_idx"],
            prompt=r.get("prompt", ""),
            prompt_token_ids=r["prompt_token_ids"],
            generated_text=r.get("generated_text", ""),
            generated_token_ids=r["generated_token_ids"],
            thinking_text=r.get("thinking_text"),
            answer_text=r.get("answer_text"),
        )
        grouped[s.problem_id].append(s)
    # Sort each group by sample_idx
    for pid in grouped:
        grouped[pid].sort(key=lambda s: s.sample_idx)
    return dict(grouped)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark loader (reference answers + answer_type)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading benchmark…")
def load_benchmark_answers(benchmark_name: str) -> Dict[str, dict]:
    """Returns {problem_id: {"answer": str, "answer_type": str}}."""
    from src.evaluation.benchmarks import load_benchmark
    problems = load_benchmark(benchmark_name)
    return {p.id: {"answer": p.answer, "answer_type": p.answer_type} for p in problems}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached for the whole session — expensive)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model (once per session)…")
def load_model(model_name: str, tuned_lens_path: str):
    from src.model.qwen3_helper import Qwen3Helper
    helper = Qwen3Helper(model_name=model_name)
    tuned_lens = None
    if tuned_lens_path and os.path.exists(tuned_lens_path):
        from src.model.tuned_lens import TunedLens
        device = str(next(helper.model.parameters()).device)
        tuned_lens = TunedLens.load(tuned_lens_path, device=device)
    return helper, tuned_lens


# ──────────────────────────────────────────────────────────────────────────────
# DTR computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_dtr(helper, scorer, sample: GeneratedSample, prefix_len: int) -> dict:
    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids = torch.tensor([sample.generated_token_ids], device=helper.device)

    actual_prefix = min(prefix_len, gen_ids.shape[1])
    full_ids = torch.cat([prompt_ids, gen_ids[:, :actual_prefix]], dim=1)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        result = scorer.compute_dtr(
            full_ids,
            generated_token_start=prompt_len,
            generated_token_end=prompt_len + actual_prefix,
        )

    token_ids = sample.generated_token_ids[:actual_prefix]
    token_labels = [
        helper.tokenizer.decode([tid], skip_special_tokens=False)
        for tid in token_ids
    ]

    return {
        "dtr": result["dtr"],
        "jsd": result["jsd_matrix"].cpu().float().numpy(),     # [T, L]
        "settling": result["settling_depths"].cpu().numpy(),   # [T]
        "is_deep": result["is_deep"].cpu().numpy(),            # [T] bool
        "deep_threshold": result["deep_threshold"],
        "token_labels": token_labels,
        "actual_prefix": actual_prefix,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_heatmap(r: dict, title: str = "") -> plt.Figure:
    jsd = r["jsd"]           # [T, L]
    T, L = jsd.shape
    settling = r["settling"]
    is_deep = r["is_deep"]
    deep_thresh = r["deep_threshold"]
    labels = r["token_labels"]

    # Cap display height so the figure stays readable
    max_display = 80
    if T > max_display:
        jsd = jsd[:max_display]
        settling = settling[:max_display]
        is_deep = is_deep[:max_display]
        labels = labels[:max_display]
        truncated = True
        T_disp = max_display
    else:
        truncated = False
        T_disp = T

    fig_h = max(5, T_disp * 0.22)
    fig_w = max(10, L * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        jsd[:T_disp], aspect="auto", origin="upper",
        cmap="YlOrRd", vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="JSD", fraction=0.02, pad=0.01)

    # Deep threshold line
    ax.axvline(deep_thresh - 0.5, color="royalblue", lw=2, ls="--")

    # Settling depth dots: green = deep, red = shallow
    for t, (sd, deep) in enumerate(zip(settling[:T_disp], is_deep[:T_disp])):
        ax.plot(sd, t, "o", color="lime" if deep else "red", ms=4, alpha=0.85)

    # Y-axis token labels
    ax.set_yticks(range(T_disp))
    def _mpl_safe(s: str) -> str:
        s = s.replace("\n", "↵").replace("\t", "→")
        s = repr(s)[:20]           # quotes make backslashes literal
        s = s.replace("$", r"\$")  # escape any remaining dollar signs
        return s

    clean = [_mpl_safe(lbl) for lbl in labels]
    ax.set_yticklabels([f"{i} {c}" for i, c in enumerate(clean)], fontsize=6.5)

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Token position", fontsize=9)
    dtr_str = f"DTR={r['dtr']:.3f}"
    trunc_str = f"  (showing first {T_disp}/{r['actual_prefix']})" if truncated else ""
    ax.set_title(f"{title}  |  {dtr_str}{trunc_str}", fontsize=10)

    # Legend — use Line2D proxy artists to avoid fake axvline distorting x-axis
    from matplotlib.lines import Line2D
    thresh_line = Line2D([0], [0], color="royalblue", lw=2, ls="--",
                         label=f"deep threshold (L{deep_thresh})")
    deep_patch = mpatches.Patch(color="lime", label="deep token")
    shallow_patch = mpatches.Patch(color="red", label="shallow token")
    ax.legend(handles=[thresh_line, deep_patch, shallow_patch],
              loc="upper left", fontsize=7)
    ax.set_xlim(-0.5, L - 0.5)

    plt.tight_layout()
    return fig


def plot_dtr_bars(dtr_scores: List[float], sample_indices: List[int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6, len(dtr_scores) * 0.7), 3.5))
    colors = ["steelblue"] * len(dtr_scores)
    ax.bar(range(len(dtr_scores)), dtr_scores, color=colors, edgecolor="white")
    ax.set_xticks(range(len(dtr_scores)))
    ax.set_xticklabels([f"S{i}" for i in sample_indices], fontsize=8)
    ax.set_ylabel("Prefix DTR")
    ax.set_ylim(0, 1)
    mean_val = float(np.mean(dtr_scores))
    ax.axhline(mean_val, color="orange", lw=1.5, ls="--", label=f"mean={mean_val:.3f}")
    ax.axhline(max(dtr_scores), color="green", lw=1, ls=":", label=f"top={max(dtr_scores):.3f}")
    ax.axhline(min(dtr_scores), color="red", lw=1, ls=":", label=f"min={min(dtr_scores):.3f}")
    ax.legend(fontsize=8)
    ax.set_title("DTR scores across all samples")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")

    # Discover outputs directory
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        st.error("No `outputs/` directory found. Run from the project root.")
        st.stop()

    sample_files = sorted(outputs_root.glob("**/generated_samples.jsonl"))
    if not sample_files:
        st.error("No generated_samples.jsonl found under outputs/")
        st.stop()

    file_labels = [str(f.relative_to(outputs_root.parent)) for f in sample_files]
    sel_file_label = st.selectbox("Samples file", file_labels)
    sel_file = Path(sel_file_label)

    grouped = load_and_group(str(sel_file))
    problem_ids = list(grouped.keys())
    st.caption(f"{len(problem_ids)} problems, "
               f"{sum(len(v) for v in grouped.values())} total samples")

    # Problem selector — support search by typing
    sel_problem = st.selectbox("Problem ID", problem_ids)
    problem_samples = grouped[sel_problem]

    # Sample selector
    sample_options = ["All samples"] + [f"Sample {s.sample_idx}" for s in problem_samples]
    sel_sample_label = st.selectbox(f"Sample  (n={len(problem_samples)})", sample_options)

    st.divider()
    st.header("Benchmark (for answers)")
    BENCHMARKS = ["none", "aime24", "gpqa_diamond", "hmmt2025", "arithmetic_stress_test", "char_occur", "substring_occur", "distinct_char", "word_len"]
    sel_benchmark = st.selectbox("Benchmark", BENCHMARKS)

    st.divider()
    st.header("DTR Settings")
    prefix_length = st.slider("Prefix length (tokens)", 10, 200, 50, step=5)
    gamma = st.slider("γ  settling threshold", 0.1, 1.0, 0.5, step=0.05)
    rho = st.slider("ρ  deep layer fraction", 0.5, 1.0, 0.85, step=0.05)
    eta = st.slider("η  think@n keep fraction", 0.1, 1.0, 0.5, step=0.05)

    st.divider()
    st.header("Model")
    config_path = st.text_input("Config file", "configs/default.yaml")
    tuned_lens_path = st.text_input("Tuned lens weights (optional)",
                                    placeholder="outputs/tuned_lens/weights.pt")

    compute_btn = st.button("▶  Compute DTR", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark reference answers (optional)
# ──────────────────────────────────────────────────────────────────────────────
bench_data: Dict[str, dict] = {}
if sel_benchmark != "none":
    try:
        bench_data = load_benchmark_answers(sel_benchmark)
    except Exception as e:
        st.warning(f"Could not load benchmark '{sel_benchmark}': {e}")

ref_info = bench_data.get(sel_problem)  # {"answer": ..., "answer_type": ...} or None
answer_type = ref_info["answer_type"] if ref_info else "integer"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: correctness check against any ref answer
# ──────────────────────────────────────────────────────────────────────────────
def _is_correct_ref(ans, ref_answer: str) -> bool:
    if ans is None:
        return False
    return _normalize_for_comparison(str(ans)) == _normalize_for_comparison(str(ref_answer))


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_explorer, tab_dataset = st.tabs(["🔍 Problem Explorer", "📊 Dataset Summary"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Problem Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown(f"### `{sel_problem}`  —  {len(problem_samples)} samples")
    if ref_info:
        st.markdown(f"**Reference answer:** `{ref_info['answer']}`  |  type: `{answer_type}`")

    with st.expander("📄 Show full prompt"):
        prompt_text = problem_samples[0].prompt if problem_samples else "(no samples)"
        st.text(prompt_text)

    if not compute_btn:
        st.info("Configure settings in the sidebar, then click **▶ Compute DTR**.")
    else:
        # ── Load model ─────────────────────────────────────────────────────────
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            model_name = cfg["model"]["name"]
        except Exception as e:
            st.error(f"Cannot read config: {e}")
            model_name = None

        if model_name:
            helper, tuned_lens = load_model(model_name, tuned_lens_path or "")
            from src.dtr.dtr_scorer import DTRScorer
            scorer = DTRScorer(helper, gamma=gamma, rho=rho, tuned_lens=tuned_lens)
            lens_label = f"tuned lens ({tuned_lens_path})" if tuned_lens else "logit lens"
            st.caption(f"Model: `{model_name}` | Lens: {lens_label} | γ={gamma} ρ={rho}")

            # ── All samples ────────────────────────────────────────────────────
            if sel_sample_label == "All samples":
                results = []
                prog = st.progress(0, text="Computing…")
                for i, sample in enumerate(problem_samples):
                    r = compute_dtr(helper, scorer, sample, prefix_length)
                    results.append(r)
                    prog.progress((i + 1) / len(problem_samples),
                                  text=f"Sample {sample.sample_idx}  DTR={r['dtr']:.3f}")
                prog.empty()

                dtr_vals = [r["dtr"] for r in results]
                best_idx = int(np.argmax(dtr_vals))
                worst_idx = int(np.argmin(dtr_vals))

                extracted_answers = [
                    extract_answer(s.generated_text, answer_type) for s in problem_samples
                ]

                def _is_correct(ans):
                    return ref_info is not None and _is_correct_ref(ans, ref_info["answer"])

                maj_answer = majority_vote([a for a in extracted_answers if a is not None])
                maj_correct = _is_correct(maj_answer)
                pass_correct = any(_is_correct(a) for a in extracted_answers)

                k = max(1, math.ceil(eta * len(problem_samples)))
                top_k_idx = sorted(range(len(dtr_vals)), key=lambda i: dtr_vals[i], reverse=True)[:k]
                think_answers = [extracted_answers[i] for i in top_k_idx if extracted_answers[i] is not None]
                think_answer = majority_vote(think_answers) if think_answers else None
                think_correct = _is_correct(think_answer)

                c1, c2, c3 = st.columns(3)
                c1.metric("Top DTR",  f"{max(dtr_vals):.3f}", f"S{problem_samples[best_idx].sample_idx}")
                c2.metric("Mean DTR", f"{np.mean(dtr_vals):.3f}")
                c3.metric("Min DTR",  f"{min(dtr_vals):.3f}", f"S{problem_samples[worst_idx].sample_idx}")

                if ref_info:
                    st.markdown("#### Voting metrics")
                    v1, v2, v3 = st.columns(3)
                    v1.metric(f"maj@{len(problem_samples)}", f"`{maj_answer}`",
                              "✅ correct" if maj_correct else "❌ wrong")
                    v2.metric(f"pass@{len(problem_samples)}",
                              "✅ yes" if pass_correct else "❌ no",
                              f"{sum(_is_correct(a) for a in extracted_answers)}/{len(extracted_answers)} correct")
                    v3.metric(f"think@{len(problem_samples)}  (η={eta}, top-{k})", f"`{think_answer}`",
                              "✅ correct" if think_correct else "❌ wrong")

                st.pyplot(plot_dtr_bars(dtr_vals, [s.sample_idx for s in problem_samples]))

                st.divider()
                st.markdown("#### Per-sample heatmaps")
                for sample, r, extracted in zip(problem_samples, results, extracted_answers):
                    correct = _is_correct(extracted)
                    verdict = "✅" if correct else ("❌" if ref_info else "")
                    with st.expander(
                        f"Sample {sample.sample_idx}  —  DTR={r['dtr']:.3f}"
                        f"  |  extracted: `{extracted}`  {verdict}",
                        expanded=(sample.sample_idx == problem_samples[best_idx].sample_idx),
                    ):
                        ca, cb = st.columns(2)
                        ca.markdown(f"**Extracted:** `{extracted}`")
                        if ref_info:
                            cb.markdown(f"**Reference:** `{ref_info['answer']}`  {verdict}")
                        fig = plot_heatmap(r, title=f"Sample {sample.sample_idx}")
                        st.pyplot(fig)
                        plt.close(fig)
                        with st.expander("Generated text"):
                            st.text(sample.generated_text)

            # ── Single sample ──────────────────────────────────────────────────
            else:
                idx = int(sel_sample_label.split()[-1])
                sample = next(s for s in problem_samples if s.sample_idx == idx)

                with st.spinner(f"Computing DTR for sample {idx}…"):
                    r = compute_dtr(helper, scorer, sample, prefix_length)

                extracted = extract_answer(sample.generated_text, answer_type)
                correct = ref_info is not None and _is_correct_ref(extracted, ref_info["answer"])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("DTR",            f"{r['dtr']:.3f}")
                c2.metric("Deep tokens",    f"{r['is_deep'].sum()} / {r['actual_prefix']}")
                c3.metric("Deep threshold", f"Layer {r['deep_threshold']}")
                c4.metric("Prefix used",    f"{r['actual_prefix']} tokens")

                ac, bc = st.columns(2)
                verdict = "✅ correct" if correct else ("❌ wrong" if ref_info else "")
                ac.markdown(f"**Extracted answer:** `{extracted}`  {verdict}")
                if ref_info:
                    bc.markdown(f"**Reference answer:** `{ref_info['answer']}`")

                fig = plot_heatmap(r, title=f"Sample {idx}")
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("Generated text"):
                    st.text(sample.generated_text)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Summary
# ══════════════════════════════════════════════════════════════════════════════

# ── Pre-computed results helpers ──────────────────────────────────────────────

def _find_results_file(samples_path: Path) -> Optional[Path]:
    """Look for results.jsonl or results.json next to the samples file."""
    parent = samples_path.parent
    for name in ("results.jsonl", "results.json"):
        p = parent / name
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Loading pre-computed results…")
def _load_results_file(path: str) -> List[dict]:
    """
    Load per-problem rows from results.jsonl (one JSON per line) or
    results.json (run_think_at_n.py format with a "per_problem" array).

    Always returns a list of dicts, each containing at minimum:
        problem_id, reference, maj_answer, maj_correct, pass_correct,
        n_correct, n_total
    and optionally:
        think_answer_<mode>, think_correct_<mode>
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        # One record per line — written by this explorer's "Compute & save"
        records = _load_jsonl_robust(str(path))
        return records
    else:
        # results.json written by run_think_at_n.py
        with open(path) as f:
            data = json.load(f)
        rows = []
        for pp in data.get("per_problem", []):
            ref = pp.get("reference", "")
            extractions = [
                se.get("extracted_answer") for se in pp.get("sample_extractions", [])
            ]
            n_total   = len(extractions)
            n_correct = sum(1 for e in extractions if _is_correct_ref(e, ref))
            maj_ans   = pp.get("maj_answer")
            row = {
                "problem_id":  pp.get("id", ""),
                "reference":   ref,
                "maj_answer":  str(maj_ans) if maj_ans is not None else "—",
                "maj_correct": _is_correct_ref(maj_ans, ref),
                "pass_correct": n_correct > 0,
                "n_correct":   n_correct,
                "n_total":     n_total,
            }
            # Preserve any think@n fields
            for k, v in pp.items():
                if k.startswith("think_answer_"):
                    mode = k[len("think_answer_"):]
                    row[f"think_answer_{mode}"] = str(v) if v is not None else "—"
                    row[f"think_correct_{mode}"] = _is_correct_ref(v, ref)
            rows.append(row)
        return rows


def _rows_to_df(rows: List[dict], n_per_problem: int):
    """Convert per-problem rows into a display DataFrame."""
    import pandas as pd
    display_rows = []
    for r in rows:
        d = {
            "problem_id": r["problem_id"],
            "reference":  r["reference"],
            "maj_vote":   r.get("maj_answer", "—"),
            "maj✓":       "✅" if r.get("maj_correct") else "❌",
            "pass✓":      "✅" if r.get("pass_correct") else "❌",
            f"n✓/{r.get('n_total', n_per_problem)}": r.get("n_correct", "?"),
        }
        # Add think@n columns if present
        for k in r:
            if k.startswith("think_answer_"):
                mode = k[len("think_answer_"):]
                d[f"think@n ({mode})"]  = r[k]
                d[f"think✓ ({mode})"]   = "✅" if r.get(f"think_correct_{mode}") else "❌"
        display_rows.append(d)
    return pd.DataFrame(display_rows)


def _render_dataset_summary(rows: List[dict], n_per_problem, source_label: str):
    """Render aggregate metrics + per-problem table from pre-computed rows."""
    import pandas as pd
    df = _rows_to_df(rows, n_per_problem)
    n_probs   = len(df)
    maj_acc   = df["maj✓"].eq("✅").mean()
    pass_acc  = df["pass✓"].eq("✅").mean()

    # Detect think@n modes
    think_modes = [
        k[len("think_answer_"):] for k in rows[0] if k.startswith("think_answer_")
    ] if rows else []

    metric_cols = st.columns(2 + len(think_modes))
    metric_cols[0].metric(f"maj@{n_per_problem} accuracy",  f"{maj_acc*100:.1f}%",
                          f"{df['maj✓'].eq('✅').sum()}/{n_probs} problems")
    metric_cols[1].metric(f"pass@{n_per_problem} accuracy", f"{pass_acc*100:.1f}%",
                          f"{df['pass✓'].eq('✅').sum()}/{n_probs} problems")
    for ci, mode in enumerate(think_modes, 2):
        col_name = f"think✓ ({mode})"
        if col_name in df.columns:
            think_acc = df[col_name].eq("✅").mean()
            metric_cols[ci].metric(f"think@n ({mode})", f"{think_acc*100:.1f}%",
                                   f"{df[col_name].eq('✅').sum()}/{n_probs} problems")

    st.caption(f"Source: `{source_label}`")
    st.divider()
    st.dataframe(df, use_container_width=True, hide_index=True)


with tab_dataset:
    if sel_benchmark == "none":
        st.warning("Select a benchmark in the sidebar to enable dataset-level metrics.")
    else:
        st.markdown(f"### Dataset: `{sel_benchmark}`  ·  {len(problem_ids)} problems")
        n_samples_per_problem = len(list(grouped.values())[0]) if grouped else "?"
        st.caption(f"Samples file: `{sel_file_label}`  |  n={n_samples_per_problem} samples/problem  |  η={eta}")

        # ── Check for pre-computed results next to the samples file ───────────
        results_file = _find_results_file(sel_file)

        if results_file is not None:
            st.success(f"✅ Pre-computed results found: `{results_file.name}`")
            if st.button("↺ Reload results from file"):
                st.cache_data.clear()
            rows = _load_results_file(str(results_file))
            if rows:
                _render_dataset_summary(rows, n_samples_per_problem,
                                        str(results_file.relative_to(sel_file.parent.parent)))
            else:
                st.warning("Results file is empty or could not be parsed.")

        else:
            st.info("No pre-computed results found next to the samples file. "
                    "Compute below — results will be saved as `results.jsonl` "
                    "for instant loading next time.")

            dataset_btn = st.button("Compute maj@n · pass@n (no model needed)",
                                    type="primary")
            if dataset_btn:
                rows = []
                prog2 = st.progress(0, text="Extracting answers…")
                for p_idx, (pid, samples) in enumerate(grouped.items()):
                    bref = bench_data.get(pid)
                    if bref is None:
                        continue
                    atype   = bref["answer_type"]
                    ref_ans = bref["answer"]

                    exts    = [extract_answer(s.generated_text, atype) for s in samples]
                    maj_ans = majority_vote([a for a in exts if a is not None])
                    maj_ok  = _is_correct_ref(maj_ans, ref_ans)
                    pass_ok = any(_is_correct_ref(a, ref_ans) for a in exts)
                    n_ok    = sum(_is_correct_ref(a, ref_ans) for a in exts)

                    rows.append({
                        "problem_id":  pid,
                        "reference":   ref_ans,
                        "maj_answer":  str(maj_ans) if maj_ans is not None else "—",
                        "maj_correct": maj_ok,
                        "pass_correct": pass_ok,
                        "n_correct":   n_ok,
                        "n_total":     len(samples),
                    })
                    prog2.progress((p_idx + 1) / len(grouped),
                                   text=f"{pid}  {'✅' if maj_ok else '❌'}")
                prog2.empty()

                # ── Save results.jsonl next to samples file ────────────────────
                save_path = sel_file.parent / "results.jsonl"
                with open(save_path, "w") as fout:
                    for r in rows:
                        fout.write(json.dumps(r) + "\n")
                st.success(f"Saved `{save_path.name}` — will load automatically next time.")

                _render_dataset_summary(rows, n_samples_per_problem, "just computed")
