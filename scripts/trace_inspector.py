"""
Trace inspector: per-token SC / NLL coloring of reasoning traces as HTML.

For each selected sample generates a self-contained HTML file where every
token is coloured by its metric value (red = low/uncertain, green = high/certain).
Top drop and rise events are annotated with underlines + superscript markers so
you can immediately see what the model was writing at those moments.

Modes:
  --metric sc   self-certainty (needs model forward pass)
  --metric nll  stored token log-probs (no model pass — instant)

Usage:
    # NLL (fast, no GPU needed):
    python scripts/trace_inspector.py \\
        --samples-file outputs/arithmetic_stress_test_full/generated_samples.jsonl \\
        --benchmark arithmetic_stress_test \\
        --model /path/to/Qwen3-4B-Thinking-2507 \\
        --problem-id hmmt_7 \\
        --metric nll

    # SC (needs GPU):
    python scripts/trace_inspector.py \\
        --samples-file outputs/hmmt_samples16/generated_samples.jsonl \\
        --benchmark hmmt2025 \\
        --model /path/to/Qwen3-4B-Thinking-2507 \\
        --problem-id hmmt_7 \\
        --metric sc

    # Inspect specific samples (default: auto first correct + first incorrect):
    python scripts/trace_inspector.py ... --sample-idx 2 7 11

Output: {output_dir}/{metric}_trace_{problem_id}_s{sample_idx}.html
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import base64
import html as html_lib
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from collections import defaultdict

from src.inference.sampler import load_samples
from src.evaluation.voting import extract_answer, normalize_numeric_answer
from src.evaluation.metrics import _normalize_for_comparison

SLICE_SIZE   = 128
STATUS_COLOR = {"correct": "#27ae60", "incorrect": "#e74c3c", "unknown": "#7f8c8d"}


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sc(helper, sample) -> np.ndarray:
    """Per-token self-certainty (negative entropy). Needs model."""
    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    T_gen      = gen_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)

    with torch.no_grad():
        all_logits = helper.model(full_ids).logits

    ent_arr = np.empty(T_gen, dtype=np.float32)
    for s in range(0, T_gen, SLICE_SIZE):
        e  = min(s + SLICE_SIZE, T_gen)
        sl = all_logits[0, prompt_len + s : prompt_len + e].float()
        lp = torch.log_softmax(sl, dim=-1)
        ent_arr[s:e] = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()
        del sl, lp

    del all_logits
    torch.cuda.empty_cache()
    return -ent_arr   # negative entropy = self-certainty


def get_metric(helper, sample, use_nll: bool):
    if use_nll:
        lps = sample.token_logprobs
        return np.array(lps, dtype=np.float32) if lps else None
    return compute_sc(helper, sample)


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) <= window:
        return np.full_like(arr, arr.mean())
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def find_events(smoothed: np.ndarray, top_n: int):
    """
    Return (drop_positions, rise_positions).
    Drops: positions in smoothed where the next-step change is most negative.
    Rises: positions where the next-step change is most positive.
    Positions refer to the token BEFORE the change (i.e. onset token).
    """
    if len(smoothed) < 2:
        return [], []
    diff  = np.diff(smoothed)
    drops = sorted(np.argsort(diff)[:top_n].tolist())
    rises = sorted(np.argsort(diff)[-top_n:][::-1].tolist())
    return drops, rises


# ─────────────────────────────────────────────────────────────────────────────
# Color mapping
# ─────────────────────────────────────────────────────────────────────────────

def val_to_rgba(v: float, vmin: float, vmax: float, alpha: float = 0.55) -> str:
    """Map metric value → CSS rgba string. vmin=worst(red), vmax=best(green)."""
    norm    = float(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1))
    r, g, b, _ = cm.RdYlGn(norm)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"


def colorbar_gradient(vmin: float, vmax: float) -> str:
    """CSS linear-gradient string for the legend colorbar."""
    stops = []
    for i in range(11):
        r, g, b, _ = cm.RdYlGn(i / 10)
        stops.append(f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.7) {i*10}%")
    return ", ".join(stops)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory plot → base64 PNG
# ─────────────────────────────────────────────────────────────────────────────

def trajectory_plot_b64(raw, smoothed, drops, rises, metric_label, title):
    fig, ax = plt.subplots(figsize=(13, 2.6))
    xs = np.arange(len(raw))
    ax.plot(xs, raw,      color="#bdc3c7", lw=0.5, alpha=0.8, label="raw")
    ax.plot(xs, smoothed, color="#2c3e50", lw=1.5, label="smoothed")
    for d in drops:
        ax.axvline(d, color="#e74c3c", lw=1.0, alpha=0.75)
    for r in rises:
        ax.axvline(r, color="#27ae60", lw=1.0, alpha=0.75)
    ax.set_xlabel("Token position", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body {{ font-family: Georgia,serif; max-width: 1150px; margin: 20px auto;
       background: #f4f4f4; color: #2c3e50; }}
.header {{ background:#fff; border:1px solid #ddd; padding:16px 20px;
          border-radius:8px; margin-bottom:14px;
          border-left:6px solid {status_color}; }}
.header h2 {{ margin:0 0 6px 0; font-size:16px; }}
.header p  {{ margin:3px 0; font-size:13px; color:#555; }}
.plot-wrap {{ background:#fff; border:1px solid #ddd; border-radius:8px;
             padding:12px 16px; margin-bottom:14px; }}
.plot-wrap img {{ width:100%; display:block; }}
.legend {{ display:flex; gap:22px; align-items:center;
          font-size:12px; margin-bottom:10px; }}
.colorbar {{ width:220px; height:18px;
            background:linear-gradient(to right,{gradient});
            border:1px solid #aaa; border-radius:3px; }}
.cb-labels {{ display:flex; justify-content:space-between;
             font-size:10px; color:#777; width:220px; }}
.events-key p {{ margin:2px 0; font-size:12px; }}
.trace {{ font-family:"Courier New",monospace; font-size:12.5px;
         line-height:2.1; background:#fff; padding:22px;
         border:1px solid #ddd; border-radius:8px;
         white-space:pre-wrap; word-wrap:break-word; }}
/* event underlines */
.ev-drop {{ text-decoration:underline;
           text-decoration-color:#e74c3c; text-decoration-thickness:2px; }}
.ev-rise  {{ text-decoration:underline;
            text-decoration-color:#27ae60; text-decoration-thickness:2px; }}
/* event superscript labels */
.ev-lbl {{ font-size:8px; font-weight:bold; vertical-align:super;
          margin-left:1px; line-height:0; }}
.drop-lbl {{ color:#c0392b; }}
.rise-lbl {{ color:#27ae60; }}
/* </think> boundary marker */
.think-end-marker {{ display:inline-block; background:#2c3e50; color:#fff;
                    font-size:10px; padding:1px 6px; border-radius:3px;
                    margin:0 4px; vertical-align:middle; }}
"""


def build_html(sample, metric_arr, smoothed, drops, rises,
               token_texts, status, ref_answer, extracted,
               metric_name, metric_label, window, problem_id):

    vmin, vmax = float(metric_arr.min()), float(metric_arr.max())

    HALO = 5   # tokens before+after event that get the underline
    drop_halo = set()
    rise_halo = set()
    for d in drops:
        drop_halo.update(range(max(0, d - HALO), min(len(metric_arr), d + HALO + 1)))
    for r in rises:
        rise_halo.update(range(max(0, r - HALO), min(len(metric_arr), r + HALO + 1)))

    drop_label = {d: i + 1 for i, d in enumerate(drops)}
    rise_label = {r: i + 1 for i, r in enumerate(rises)}

    status_color = STATUS_COLOR.get(status, "#7f8c8d")

    # Find where </think> ends in the token sequence (for visual boundary)
    think_end_token = None
    cumtext = ""
    for i, (tid, txt) in enumerate(zip(sample.generated_token_ids, token_texts)):
        cumtext += txt
        if "</think>" in cumtext and think_end_token is None:
            think_end_token = i

    title_str = (f"{problem_id} · s{sample.sample_idx} · {status.upper()} · "
                 f"mean={metric_arr.mean():.4f}  window={window}")
    plot_b64 = trajectory_plot_b64(metric_arr, smoothed, drops, rises,
                                   metric_label, title_str)

    # ── Build token spans ────────────────────────────────────────────────────
    spans = []
    for i, (txt, val) in enumerate(zip(token_texts, metric_arr)):
        safe_txt = html_lib.escape(txt)
        bg = val_to_rgba(val, vmin, vmax)

        classes = []
        if i in drop_halo: classes.append("ev-drop")
        if i in rise_halo:  classes.append("ev-rise")
        cls = f' class="{" ".join(classes)}"' if classes else ""

        # Show both raw and smoothed so there's no confusion with the event table
        tooltip = f'pos={i} | raw={val:.4f} | smoothed={smoothed[i]:.4f}'
        span = f'<span{cls} style="background:{bg}" data-val="{tooltip}">{safe_txt}</span>'

        # Superscript event marker AT the event onset token
        if i in drop_label:
            span += f'<sup class="ev-lbl drop-lbl">▼{drop_label[i]}</sup>'
        if i in rise_label:
            span += f'<sup class="ev-lbl rise-lbl">▲{rise_label[i]}</sup>'

        # Visual </think> boundary
        if i == think_end_token:
            span += '<span class="think-end-marker">&lt;/think&gt;</span>'

        spans.append(span)

    trace_body = "".join(spans)

    gradient = colorbar_gradient(vmin, vmax)
    css = _CSS.format(status_color=status_color, gradient=gradient)

    # Drop / rise event summary table
    event_rows = []
    for label, positions, etype, color in [
        ("▼ Drop", drops, "drop", "#c0392b"),
        ("▲ Rise", rises, "rise", "#27ae60"),
    ]:
        for rank, pos in enumerate(positions, 1):
            context_start = max(0, pos - 8)
            context_end   = min(len(token_texts), pos + 12)
            ctx = html_lib.escape("".join(token_texts[context_start:context_end])
                                  .replace("\n", " "))
            # Smoothed values used for event detection
            sm_before = float(smoothed[pos])
            sm_after  = float(smoothed[min(pos + 1, len(smoothed) - 1)])
            delta     = sm_after - sm_before
            # Raw value at the onset token
            raw_at    = float(metric_arr[pos])
            event_rows.append(
                f'<tr><td style="color:{color};font-weight:bold">{label} #{rank}</td>'
                f'<td>token {pos}</td>'
                f'<td>'
                f'<span title="Rolling-mean of raw values over the smoothing window">'
                f'smoothed: {sm_before:.4f} → {sm_after:.4f} <em>(Δ={delta:+.4f})</em>'
                f'</span>'
                f'<br><span style="color:#888;font-size:10px">'
                f'raw at token: {raw_at:.4f}</span>'
                f'</td>'
                f'<td style="font-family:monospace;font-size:11px">…{ctx}…</td></tr>'
            )

    event_table = (
        '<table border="0" cellpadding="5" cellspacing="0" style="'
        'width:100%;border-collapse:collapse;font-size:12px;'
        'background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:14px">'
        '<thead><tr style="background:#ecf0f1">'
        '<th align="left">Event</th><th align="left">Position</th>'
        '<th align="left">Value change (smoothed · raw)</th>'
        '<th align="left">Context (±tokens)</th>'
        '</tr></thead><tbody>'
        + "\n".join(event_rows)
        + "</tbody></table>"
    )

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trace: {html_lib.escape(problem_id)} s{sample.sample_idx}</title>
<style>{css}</style>
</head>
<body>

<div class="header">
  <h2>Trace Inspector &mdash; <em>{html_lib.escape(problem_id)}</em>
      &nbsp;·&nbsp; Sample #{sample.sample_idx}
      &nbsp;·&nbsp; <span style="color:{status_color}">{status.upper()}</span>
  </h2>
  <p>Reference: <strong>{html_lib.escape(str(ref_answer))}</strong>
     &nbsp;|&nbsp; Extracted: <strong>{html_lib.escape(str(extracted))}</strong></p>
  <p>Metric: <strong>{metric_name}</strong>
     &nbsp;|&nbsp; Mean: {metric_arr.mean():.4f}
     &nbsp;|&nbsp; Range: [{vmin:.4f}, {vmax:.4f}]
     &nbsp;|&nbsp; Tokens: {len(metric_arr):,}
     &nbsp;|&nbsp; Smoothing window: {window}
  </p>
</div>

<div class="plot-wrap">
  <div class="legend">
    <div>
      <div class="colorbar"></div>
      <div class="cb-labels">
        <span style="color:#c0392b">&#x25A0; {vmin:.3f} low (uncertain)</span>
        <span style="color:#27ae60">high (certain) {vmax:.3f} &#x25A0;</span>
      </div>
    </div>
    <div class="events-key">
      <p><span style="color:#e74c3c">▼ red vertical lines + underline</span>
         = sharpest drops (onset of uncertainty)</p>
      <p><span style="color:#27ae60">▲ green vertical lines + underline</span>
         = sharpest rises (onset of confidence)</p>
    </div>
    <span style="color:#888;font-size:11px">&#x1F4CC; Hover any token to see its exact value</span>
  </div>
  <img src="data:image/png;base64,{plot_b64}" alt="metric trajectory">
</div>

{event_table}

<div class="trace" id="trace-div">{trace_body}</div>

<!-- floating tooltip ─────────────────────────────────────────────────────── -->
<div id="tok-tip" style="
  position:fixed; display:none; z-index:9999;
  background:#1a252f; color:#ecf0f1;
  padding:5px 10px; border-radius:5px;
  font-family:'Courier New',monospace; font-size:12px;
  pointer-events:none; white-space:nowrap;
  box-shadow:0 2px 8px rgba(0,0,0,0.45);
  border-left:3px solid #3498db;
"></div>

<script>
(function() {{
  var tip  = document.getElementById('tok-tip');
  var wrap = document.getElementById('trace-div');
  if (!wrap) return;

  wrap.addEventListener('mousemove', function(e) {{
    var el = e.target;
    // Walk up to find a span that carries a data-val attribute
    while (el && el !== wrap) {{
      if (el.dataset && el.dataset.val) {{
        tip.textContent = el.dataset.val;
        tip.style.display = 'block';
        // Position: 14px right, 36px above cursor so it doesn't block the token
        var x = e.clientX + 14;
        var y = e.clientY - 36;
        // Clamp so tip never goes off the right edge of viewport
        var tipW = tip.offsetWidth;
        if (x + tipW > window.innerWidth - 10) {{
          x = e.clientX - tipW - 14;
        }}
        tip.style.left = x + 'px';
        tip.style.top  = y + 'px';
        return;
      }}
      el = el.parentElement;
    }}
    tip.style.display = 'none';
  }});

  wrap.addEventListener('mouseleave', function() {{
    tip.style.display = 'none';
  }});
}})();
</script>

</body>
</html>"""

    return html_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--problem-id",   required=True)
    parser.add_argument("--model",        required=True,
                        help="HF model path. Used for tokenizer always; "
                             "for model weights only with --metric sc.")
    parser.add_argument("--benchmark",    default=None)
    parser.add_argument("--sample-idx",   nargs="*", type=int, default=None,
                        help="Sample indices to inspect. "
                             "Default: auto first correct + first incorrect.")
    parser.add_argument("--metric",       choices=["sc", "nll"], default="nll",
                        help="sc = self-certainty (GPU needed); "
                             "nll = stored log-probs (no GPU)  [default: nll]")
    parser.add_argument("--window",       type=int, default=50,
                        help="Smoothing window for event detection (default 50)")
    parser.add_argument("--top-events",   type=int, default=50,
                        help="Number of drop/rise events to annotate (default 5)")
    parser.add_argument("--output-dir",   default=None)
    args = parser.parse_args()

    use_nll = (args.metric == "nll")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.samples_file), "trace_inspector"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── Load samples ──────────────────────────────────────────────────────────
    print("Loading samples...")
    all_samples = load_samples(args.samples_file)
    samples = [s for s in all_samples if s.problem_id == args.problem_id]
    if not samples:
        avail = sorted({s.problem_id for s in all_samples})[:10]
        print(f"ERROR: '{args.problem_id}' not found. Available: {avail} ...")
        return
    print(f"  {len(samples)} samples for '{args.problem_id}'")

    # ── Reference answer ──────────────────────────────────────────────────────
    ref_answer  = "?"
    answer_type = "integer"
    if args.benchmark:
        from src.evaluation.benchmarks import load_benchmark
        for p in load_benchmark(args.benchmark):
            if p.id == args.problem_id:
                ref_answer  = p.answer
                answer_type = p.answer_type
                print(f"  Reference answer: {ref_answer}  (type={answer_type})")
                break

    # ── Label all samples ─────────────────────────────────────────────────────
    labeled = []
    for s in samples:
        ext = extract_answer(s.generated_text, answer_type)
        if ref_answer != "?":
            ok     = (_normalize_for_comparison(normalize_numeric_answer(str(ext))) ==
                      _normalize_for_comparison(normalize_numeric_answer(str(ref_answer))))
            status = "correct" if ok else "incorrect"
        else:
            status = "unknown"
        labeled.append((s, status, str(ext) if ext is not None else "—"))

    correct_count   = sum(1 for _, st, _ in labeled if st == "correct")
    incorrect_count = sum(1 for _, st, _ in labeled if st == "incorrect")
    print(f"  correct={correct_count}  incorrect={incorrect_count}  "
          f"unknown={len(labeled)-correct_count-incorrect_count}")

    # ── Auto-select samples ───────────────────────────────────────────────────
    if args.sample_idx is None:
        chosen = set()
        # First correct + first incorrect
        for want in ("correct", "incorrect"):
            for s, st, _ in labeled:
                if st == want and s.sample_idx not in chosen:
                    chosen.add(s.sample_idx)
                    break
        # Fewest-token sample (cheapest / fastest completion)
        if labeled:
            fewest = min(labeled, key=lambda x: len(x[0].generated_token_ids))
            if fewest[0].sample_idx not in chosen:
                chosen.add(fewest[0].sample_idx)
                print(f"  Added fewest-token sample: s{fewest[0].sample_idx} "
                      f"({len(fewest[0].generated_token_ids):,} tokens, {fewest[1]})")
        if not chosen:
            chosen = {labeled[0][0].sample_idx}
        print(f"  Auto-selected sample indices: {sorted(chosen)}")
    else:
        chosen = set(args.sample_idx)

    # ── Load tokenizer (always) and model (SC only) ───────────────────────────
    print(f"\nLoading tokenizer from: {args.model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  Tokenizer vocab size: {tokenizer.vocab_size:,}")

    helper = None
    if not use_nll:
        print(f"Loading model weights (SC mode)...")
        from src.model.qwen3_helper import Qwen3Helper
        helper = Qwen3Helper(model_name=args.model)

    metric_name  = "log-prob" if use_nll else "self-certainty"
    metric_label = "log P(y_t)" if use_nll else "SC = −H(p_t)"

    # ── Process samples ───────────────────────────────────────────────────────
    generated = []
    for s, status, extracted in labeled:
        if s.sample_idx not in chosen:
            continue

        print(f"\n  Sample {s.sample_idx:2d}  [{status}]  "
              f"({len(s.generated_token_ids):,} tokens)  "
              f"extracted={extracted} ...", end=" ", flush=True)

        metric_arr = get_metric(helper, s, use_nll)
        if metric_arr is None:
            print("SKIP — no token_logprobs stored (regenerate samples)")
            continue

        smoothed      = rolling_mean(metric_arr, args.window)
        drops, rises  = find_events(smoothed, args.top_events)
        token_texts   = [tokenizer.decode([tid]) for tid in s.generated_token_ids]

        html_str = build_html(
            sample       = s,
            metric_arr   = metric_arr,
            smoothed     = smoothed,
            drops        = drops,
            rises        = rises,
            token_texts  = token_texts,
            status       = status,
            ref_answer   = ref_answer,
            extracted    = extracted,
            metric_name  = metric_name,
            metric_label = metric_label,
            window       = args.window,
            problem_id   = args.problem_id,
        )

        fname = f"{args.metric}_trace_{args.problem_id}_s{s.sample_idx}_greedy.html"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html_str)
        generated.append((s.sample_idx, status, fpath))
        print(f"→ {fname}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Generated {len(generated)} HTML report(s) in:")
    print(f"  {output_dir}/")
    print()
    for idx, st, path in generated:
        print(f"  s{idx} [{st}]  {os.path.basename(path)}")
    print()
    print("Open in browser:")
    for _, _, path in generated:
        print(f"  open \"{path}\"")


if __name__ == "__main__":
    main()
