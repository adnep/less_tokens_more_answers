"""
Trace inspector: per-token SC / LP / neg-entropy coloring of reasoning traces as HTML.

For each selected sample generates a self-contained HTML file where every
token is coloured by its metric value (red = low/uncertain, green = high/certain).
Top drop and rise events are annotated with underlines + superscript markers so
you can immediately see what the model was writing at those moments.

Modes:
  --metric sc          self-certainty KL(u‖p_t) ∈ [0,∞) (needs model forward pass)
  --metric neg_entropy negative entropy -H(p_t) ∈ (-∞,0] (needs model forward pass)
  --metric lp          stored token log-probs (no model pass — instant)
  --metric both        embed SC + LP in one file with a toggle button

Usage:
    # LP (fast, no GPU needed):
    python scripts/trace_inspector.py \\
        --samples-file outputs/hmmt_samples16/generated_samples.jsonl \\
        --benchmark hmmt2025 \\
        --model /path/to/Qwen3-4B-Thinking-2507 \\
        --problem-id hmmt_7 \\
        --metric lp

    # Negative entropy (needs GPU):
    python scripts/trace_inspector.py ... --metric neg_entropy

    # Both SC + LP metrics in one file (needs GPU for SC):
    python scripts/trace_inspector.py ... --metric both

    # Inspect specific samples (default: auto first correct + first incorrect
    # + fewest-token sample):
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

# Human-readable metadata per metric key
METRIC_META = {
    "lp":         {"name": "log-prob",       "label": "log P(y_t)"},
    "sc":          {"name": "self-certainty", "label": "SC = KL(u‖p_t)"},
    "neg_entropy": {"name": "neg-entropy",    "label": "-H(p_t)"},
}

# Event underline colours per metric
# SC          → solid red/green
# LP          → dashed orange/blue
# neg_entropy → dotted purple/teal
METRIC_COLORS = {
    "sc":          {"drop": "#e74c3c", "rise": "#27ae60", "style": "solid"},
    "lp":         {"drop": "#e67e22", "rise": "#3498db", "style": "dashed"},
    "neg_entropy": {"drop": "#8e44ad", "rise": "#1abc9c", "style": "dotted"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sc(helper, sample) -> np.ndarray:
    """
    Per-token Self-Certainty (Kang et al., 2025):
        SC(t) = KL(u ‖ p_t) = -log|V| - (1/|V|) ∑_v log p_t(v)
    Range: [0, ∞).  Higher = more certain.  Needs model forward pass.
    """
    import math

    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    T_gen      = gen_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)

    with torch.no_grad():
        all_logits = helper.model(full_ids).logits

    V     = all_logits.shape[-1]
    log_v = math.log(V)

    sum_log_p = np.zeros(T_gen, dtype=np.float64)
    for s in range(0, T_gen, SLICE_SIZE):
        e  = min(s + SLICE_SIZE, T_gen)
        sl = all_logits[0, prompt_len + s : prompt_len + e].float()
        lp = torch.log_softmax(sl, dim=-1)
        sum_log_p[s:e] += lp.sum(dim=-1).cpu().numpy()
        del sl, lp

    del all_logits
    torch.cuda.empty_cache()
    return (-log_v - sum_log_p / V).astype(np.float32)


def compute_neg_entropy(helper, sample) -> np.ndarray:
    """
    Per-token negative entropy: -H(p_t) = ∑_v p_t(v) log p_t(v).
    Range: (-∞, 0].  Closer to 0 = more certain.  Needs model forward pass.
    """
    prompt_ids = torch.tensor([sample.prompt_token_ids], device=helper.device)
    gen_ids    = torch.tensor([sample.generated_token_ids], device=helper.device)
    T_gen      = gen_ids.shape[1]
    prompt_len = prompt_ids.shape[1]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)

    with torch.no_grad():
        all_logits = helper.model(full_ids).logits

    entropy_arr = np.empty(T_gen, dtype=np.float32)
    for s in range(0, T_gen, SLICE_SIZE):
        e   = min(s + SLICE_SIZE, T_gen)
        sl  = all_logits[0, prompt_len + s : prompt_len + e].float()
        lp  = torch.log_softmax(sl, dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()
        entropy_arr[s:e] = ent
        del sl, lp, ent

    del all_logits
    torch.cuda.empty_cache()
    return -entropy_arr   # negative entropy


def get_metric_arrays(helper, sample, metric_keys: list) -> dict:
    """
    Compute requested metrics for one sample.
    Returns {metric_key: np.ndarray or None}.
    Each model-based metric is computed at most once.
    """
    result = {}
    if "lp" in metric_keys:
        lps = sample.token_logprobs
        result["lp"] = np.array(lps, dtype=np.float32) if lps else None
    if "sc" in metric_keys:
        result["sc"] = compute_sc(helper, sample)
    if "neg_entropy" in metric_keys:
        result["neg_entropy"] = compute_neg_entropy(helper, sample)
    return result


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
# Per-metric data bundle
# ─────────────────────────────────────────────────────────────────────────────

def build_metric_bundle(metric_key, metric_arr, window, top_events,
                        sample, token_texts, problem_id):
    """Compute everything needed to render one metric in the HTML."""
    HALO = 5
    smoothed     = rolling_mean(metric_arr, window)
    drops, rises = find_events(smoothed, top_events)
    vmin, vmax   = float(metric_arr.min()), float(metric_arr.max())

    drop_halo = set()
    rise_halo = set()
    for d in drops:
        drop_halo.update(range(max(0, d - HALO), min(len(metric_arr), d + HALO + 1)))
    for r in rises:
        rise_halo.update(range(max(0, r - HALO), min(len(metric_arr), r + HALO + 1)))

    drop_label = {d: i + 1 for i, d in enumerate(drops)}
    rise_label = {r: i + 1 for i, r in enumerate(rises)}

    meta  = METRIC_META[metric_key]
    title = (f"{problem_id} · {meta['name']} · "
             f"mean={metric_arr.mean():.4f}  window={window}")
    plot_b64 = trajectory_plot_b64(
        metric_arr, smoothed, drops, rises, meta["label"], title
    )

    # Pre-compute per-token background colour strings
    bg_colors = [val_to_rgba(v, vmin, vmax) for v in metric_arr]

    # Event table rows (HTML)
    mc    = METRIC_COLORS[metric_key]
    rows  = []
    for label, positions, color in [
        (f"▼ Drop", drops, mc["drop"]),
        (f"▲ Rise", rises, mc["rise"]),
    ]:
        for rank, pos in enumerate(positions, 1):
            ctx_start = max(0, pos - 8)
            ctx_end   = min(len(token_texts), pos + 12)
            ctx = html_lib.escape(
                "".join(token_texts[ctx_start:ctx_end]).replace("\n", " ")
            )
            sm_before = float(smoothed[pos])
            sm_after  = float(smoothed[min(pos + 1, len(smoothed) - 1)])
            delta     = sm_after - sm_before
            raw_at    = float(metric_arr[pos])
            rows.append(
                f'<tr>'
                f'<td style="color:{color};font-weight:bold">{label} #{rank}</td>'
                f'<td>token {pos}</td>'
                f'<td>'
                f'<span title="Rolling-mean of raw values over the smoothing window">'
                f'smoothed: {sm_before:.4f} → {sm_after:.4f} <em>(Δ={delta:+.4f})</em>'
                f'</span>'
                f'<br><span style="color:#888;font-size:10px">'
                f'raw at token: {raw_at:.4f}</span>'
                f'</td>'
                f'<td style="font-family:monospace;font-size:11px">…{ctx}…</td>'
                f'</tr>'
            )

    gradient = colorbar_gradient(vmin, vmax)

    return {
        "key":        metric_key,
        "name":       meta["name"],
        "label":      meta["label"],
        "arr":        metric_arr,
        "smoothed":   smoothed,
        "drops":      drops,
        "rises":      rises,
        "drop_halo":  drop_halo,
        "rise_halo":  rise_halo,
        "drop_label": drop_label,
        "rise_label": rise_label,
        "vmin":       vmin,
        "vmax":       vmax,
        "bg_colors":  bg_colors,
        "plot_b64":   plot_b64,
        "event_rows": rows,
        "gradient":   gradient,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def build_html(sample, bundles: dict, token_texts, status,
               ref_answer, extracted, window, problem_id):
    """
    Build a self-contained HTML file.

    bundles: dict mapping metric_key → bundle dict from build_metric_bundle().
             May contain one or two keys ("lp", "sc").
             When two are present a toggle button is shown.
    """
    status_color  = STATUS_COLOR.get(status, "#7f8c8d")
    metric_keys   = list(bundles.keys())   # ordered: first is the default shown
    multi         = len(metric_keys) > 1
    primary       = metric_keys[0]         # shown on page load

    # Find </think> boundary
    think_end_token = None
    cumtext = ""
    for i, (tid, txt) in enumerate(zip(sample.generated_token_ids, token_texts)):
        cumtext += txt
        if "</think>" in cumtext and think_end_token is None:
            think_end_token = i

    # ── Build token spans ────────────────────────────────────────────────────
    spans = []
    for i, txt in enumerate(token_texts):
        safe_txt = html_lib.escape(txt)

        # Classes: event halos for ALL metrics (hidden/shown via CSS)
        classes = []
        for mk, b in bundles.items():
            if i in b["drop_halo"]: classes.append(f"ev-drop-{mk}")
            if i in b["rise_halo"]: classes.append(f"ev-rise-{mk}")
        cls = f' class="{" ".join(classes)}"' if classes else ""

        # Background colour for primary metric (default style)
        primary_bg = bundles[primary]["bg_colors"][i]

        # data-bg-{metric} for JS switching, data-val for tooltip
        data_parts = []
        for mk, b in bundles.items():
            data_parts.append(f'data-bg-{mk}="{b["bg_colors"][i]}"')

        # Tooltip always shows all available metrics
        tip_parts = [f"pos={i}"]
        for mk, b in bundles.items():
            tip_parts.append(
                f'{mk.upper()}  raw={b["arr"][i]:.4f}  sm={b["smoothed"][i]:.4f}'
            )
        tooltip = " | ".join(tip_parts)
        data_parts.append(f'data-val="{tooltip}"')

        data_str = " ".join(data_parts)
        span = (f'<span{cls} style="background:{primary_bg}" {data_str}>'
                f'{safe_txt}</span>')

        # Superscript markers (all metrics, always rendered — visibility via CSS)
        for mk, b in bundles.items():
            mc = METRIC_COLORS[mk]
            if i in b["drop_label"]:
                span += (f'<sup class="ev-lbl drop-lbl-{mk}" '
                         f'style="color:{mc["drop"]}">▼{b["drop_label"][i]}</sup>')
            if i in b["rise_label"]:
                span += (f'<sup class="ev-lbl rise-lbl-{mk}" '
                         f'style="color:{mc["rise"]}">▲{b["rise_label"][i]}</sup>')

        if i == think_end_token:
            span += '<span class="think-end-marker">&lt;/think&gt;</span>'

        spans.append(span)

    trace_body = "".join(spans)

    # ── CSS ──────────────────────────────────────────────────────────────────
    # Build per-metric underline rules
    underline_css = ""
    for mk, mc in METRIC_COLORS.items():
        style = mc["style"]
        underline_css += f"""
.ev-drop-{mk} {{ text-decoration: underline {style};
                 text-decoration-color:{mc['drop']}; text-decoration-thickness:2px; }}
.ev-rise-{mk} {{ text-decoration: underline {style};
                 text-decoration-color:{mc['rise']}; text-decoration-thickness:2px; }}"""

    # Hide inactive metric's underlines and superscripts via wrapper class
    hide_css = ""
    for mk in metric_keys:
        for other in metric_keys:
            if other != mk:
                hide_css += f"""
.active-{mk} .ev-drop-{other},
.active-{mk} .ev-rise-{other} {{ text-decoration: none !important; }}
.active-{mk} .drop-lbl-{other},
.active-{mk} .rise-lbl-{other} {{ display: none; }}"""

    css = f"""
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
            background:linear-gradient(to right,{bundles[primary]["gradient"]});
            border:1px solid #aaa; border-radius:3px; }}
.cb-labels {{ display:flex; justify-content:space-between;
             font-size:10px; color:#777; width:220px; }}
.events-key p {{ margin:2px 0; font-size:12px; }}
.trace {{ font-family:"Courier New",monospace; font-size:12.5px;
         line-height:2.1; background:#fff; padding:22px;
         border:1px solid #ddd; border-radius:8px;
         white-space:pre-wrap; word-wrap:break-word; }}
.ev-lbl {{ font-size:8px; font-weight:bold; vertical-align:super;
          margin-left:1px; line-height:0; }}
.think-end-marker {{ display:inline-block; background:#2c3e50; color:#fff;
                    font-size:10px; padding:1px 6px; border-radius:3px;
                    margin:0 4px; vertical-align:middle; }}
/* metric toggle */
.metric-switcher {{ display:flex; gap:8px; margin-bottom:14px; align-items:center; }}
.metric-switcher span {{ font-size:13px; color:#555; margin-right:4px; }}
.metric-btn {{
  padding:6px 18px; border:2px solid #2c3e50; border-radius:20px;
  background:#fff; color:#2c3e50; font-size:13px; cursor:pointer;
  transition:all 0.15s;
}}
.metric-btn.active {{ background:#2c3e50; color:#fff; }}
.metric-btn:hover:not(.active) {{ background:#ecf0f1; }}
{underline_css}
{hide_css}
"""

    # ── Metric toggle markup (hidden when single metric) ─────────────────────
    switcher_html = ""
    if multi:
        buttons = ""
        for i, mk in enumerate(metric_keys):
            active = "active" if mk == primary else ""
            buttons += (f'<button class="metric-btn {active}" '
                        f'data-metric="{mk}" onclick="switchMetric(\'{mk}\')">'
                        f'{METRIC_META[mk]["name"].upper()}</button>')
        switcher_html = (f'<div class="metric-switcher">'
                         f'<span>Metric:</span>{buttons}</div>')

    # ── Trajectory plots (one per metric, non-primary hidden) ────────────────
    plots_html = ""
    for mk, b in bundles.items():
        display = "" if mk == primary else ' style="display:none"'
        vmin_str = f'{b["vmin"]:.3f}'
        vmax_str = f'{b["vmax"]:.3f}'
        mc = METRIC_COLORS[mk]
        plots_html += f"""
<div class="plot-wrap" id="plot-{mk}"{display}>
  <div class="legend">
    <div>
      <div class="colorbar" style="background:linear-gradient(to right,{b['gradient']})"></div>
      <div class="cb-labels">
        <span style="color:#c0392b">&#x25A0; {vmin_str} low (uncertain)</span>
        <span style="color:#27ae60">high (certain) {vmax_str} &#x25A0;</span>
      </div>
    </div>
    <div class="events-key">
      <p><span style="color:{mc['drop']}">▼ red vertical lines + underline</span>
         = sharpest drops</p>
      <p><span style="color:{mc['rise']}">▲ green vertical lines + underline</span>
         = sharpest rises</p>
    </div>
    <span style="color:#888;font-size:11px">&#x1F4CC; Hover any token to see exact values</span>
  </div>
  <img src="data:image/png;base64,{b['plot_b64']}" alt="{mk} trajectory">
</div>"""

    # ── Event tables (one per metric, non-primary hidden) ────────────────────
    tables_html = ""
    for mk, b in bundles.items():
        display = "" if mk == primary else ' style="display:none"'
        event_table = (
            '<table border="0" cellpadding="5" cellspacing="0" style="'
            'width:100%;border-collapse:collapse;font-size:12px;'
            'background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:14px">'
            '<thead><tr style="background:#ecf0f1">'
            '<th align="left">Event</th><th align="left">Position</th>'
            '<th align="left">Value change (smoothed · raw)</th>'
            '<th align="left">Context (±tokens)</th>'
            '</tr></thead><tbody>'
            + "\n".join(b["event_rows"])
            + "</tbody></table>"
        )
        tables_html += f'<div id="events-{mk}"{display}>{event_table}</div>'

    # ── Header stats ─────────────────────────────────────────────────────────
    stat_parts = []
    for mk, b in bundles.items():
        stat_parts.append(
            f'{METRIC_META[mk]["name"]}: mean={b["arr"].mean():.4f} '
            f'[{b["vmin"]:.4f}, {b["vmax"]:.4f}]'
        )
    stats_str = "  |  ".join(stat_parts)

    # ── JS ───────────────────────────────────────────────────────────────────
    js = """
(function() {
  // ── Metric switcher ──────────────────────────────────────────────────────
  window.switchMetric = function(m) {
    // 1. Swap token background colours
    var spans = document.querySelectorAll('#trace-div span[data-val]');
    spans.forEach(function(s) {
      var bg = s.getAttribute('data-bg-' + m);
      if (bg) s.style.background = bg;
    });

    // 2. Swap event underlines via wrapper class
    var traceDiv = document.getElementById('trace-div');
    traceDiv.className = traceDiv.className.replace(/active-\\S+/, '').trim();
    traceDiv.classList.add('active-' + m);

    // 3. Toggle plots
    document.querySelectorAll('.plot-wrap[id^="plot-"]').forEach(function(el) {
      el.style.display = (el.id === 'plot-' + m) ? 'block' : 'none';
    });

    // 4. Toggle event tables
    document.querySelectorAll('div[id^="events-"]').forEach(function(el) {
      el.style.display = (el.id === 'events-' + m) ? 'block' : 'none';
    });

    // 5. Update button states
    document.querySelectorAll('.metric-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.metric === m);
    });
  };

  // ── Token hover tooltip ──────────────────────────────────────────────────
  var tip  = document.getElementById('tok-tip');
  var wrap = document.getElementById('trace-div');
  if (!wrap) return;

  wrap.addEventListener('mousemove', function(e) {
    var el = e.target;
    while (el && el !== wrap) {
      if (el.dataset && el.dataset.val) {
        tip.textContent = el.dataset.val;
        tip.style.display = 'block';
        var x = e.clientX + 14;
        var y = e.clientY - 36;
        var tipW = tip.offsetWidth;
        if (x + tipW > window.innerWidth - 10) x = e.clientX - tipW - 14;
        tip.style.left = x + 'px';
        tip.style.top  = y + 'px';
        return;
      }
      el = el.parentElement;
    }
    tip.style.display = 'none';
  });

  wrap.addEventListener('mouseleave', function() {
    tip.style.display = 'none';
  });
})();
"""

    primary_b = bundles[primary]
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
  <p>{stats_str}
     &nbsp;|&nbsp; Tokens: {len(primary_b["arr"]):,}
     &nbsp;|&nbsp; Smoothing window: {window}
  </p>
</div>

{switcher_html}

{plots_html}

{tables_html}

<div class="trace active-{primary}" id="trace-div">{trace_body}</div>

<!-- floating tooltip -->
<div id="tok-tip" style="
  position:fixed; display:none; z-index:9999;
  background:#1a252f; color:#ecf0f1;
  padding:5px 10px; border-radius:5px;
  font-family:'Courier New',monospace; font-size:12px;
  pointer-events:none; white-space:nowrap;
  box-shadow:0 2px 8px rgba(0,0,0,0.45);
  border-left:3px solid #3498db;
"></div>

<script>{js}</script>
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
                             "for model weights with --metric sc or both.")
    parser.add_argument("--benchmark",    default=None)
    parser.add_argument("--sample-idx",   nargs="*", type=int, default=None,
                        help="Sample indices to inspect. "
                             "Default: auto first correct + first incorrect "
                             "+ fewest-token sample.")
    parser.add_argument("--metric",
                        choices=["sc", "lp", "neg_entropy", "both"],
                        default="lp",
                        help="sc = self-certainty KL(u‖p_t) (GPU needed); "
                             "neg_entropy = -H(p_t) (GPU needed); "
                             "lp = stored log-probs (no GPU); "
                             "both = embed SC + LP in one file with toggle  "
                             "[default: lp]")
    parser.add_argument("--window",       type=int, default=50,
                        help="Smoothing window for event detection (default 50)")
    parser.add_argument("--top-events",   type=int, default=20,
                        help="Number of drop/rise events to annotate (default 5)")
    parser.add_argument("--output-dir",   default=None)
    args = parser.parse_args()

    metric_keys = ["sc", "lp"] if args.metric == "both" else [args.metric]
    needs_model = any(mk in metric_keys for mk in ("sc", "neg_entropy"))

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
                    print(f"  Added first-{want} sample: s{s.sample_idx} "
                          f"({len(s.generated_token_ids):,} tokens)")
                    break

        # Fewest-token sample (cheapest completion)
        if labeled:
            fewest = min(labeled, key=lambda x: len(x[0].generated_token_ids))
            if fewest[0].sample_idx not in chosen:
                chosen.add(fewest[0].sample_idx)
                print(f"  Added fewest-token sample: s{fewest[0].sample_idx} "
                      f"({len(fewest[0].generated_token_ids):,} tokens, {fewest[1]})")

        # Most-token sample (longest reasoning chain) — only adds diversity
        # when all samples share the same correctness class
        if labeled:
            most = max(labeled, key=lambda x: len(x[0].generated_token_ids))
            if most[0].sample_idx not in chosen:
                chosen.add(most[0].sample_idx)
                print(f"  Added most-token sample:   s{most[0].sample_idx} "
                      f"({len(most[0].generated_token_ids):,} tokens, {most[1]})")

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
    if needs_model:
        print(f"Loading model weights (SC mode)...")
        from src.model.qwen3_helper import Qwen3Helper
        helper = Qwen3Helper(model_name=args.model)

    # ── Process samples ───────────────────────────────────────────────────────
    generated = []
    for s, status, extracted in labeled:
        if s.sample_idx not in chosen:
            continue

        print(f"\n  Sample {s.sample_idx:2d}  [{status}]  "
              f"({len(s.generated_token_ids):,} tokens)  "
              f"extracted={extracted} ...", end=" ", flush=True)

        # Compute all requested metrics
        raw_metrics = get_metric_arrays(helper, s, metric_keys)

        # Skip if any required metric is missing
        skip = False
        for mk in metric_keys:
            if raw_metrics.get(mk) is None:
                print(f"SKIP — no {mk} data (regenerate samples or add --metric lp)")
                skip = True
                break
        if skip:
            continue

        # Decode tokens once (shared across metrics)
        token_texts = [tokenizer.decode([tid]) for tid in s.generated_token_ids]

        # Build per-metric bundles
        bundles = {}
        for mk in metric_keys:
            bundles[mk] = build_metric_bundle(
                mk, raw_metrics[mk], args.window, args.top_events,
                s, token_texts, args.problem_id
            )

        html_str = build_html(
            sample      = s,
            bundles     = bundles,
            token_texts = token_texts,
            status      = status,
            ref_answer  = ref_answer,
            extracted   = extracted,
            window      = args.window,
            problem_id  = args.problem_id,
        )

        fname = f"{args.metric}_trace_{args.problem_id}_s{s.sample_idx}.html"
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
