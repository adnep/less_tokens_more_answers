# Intervention Framework

Token-by-token generation for reasoning LLMs with **mid-stream interventions**,
driven by per-token confidence signals. Built on HuggingFace Transformers (single
GPU) — **not** vLLM, because interventions need access to logits and the KV cache
at every step.

An *intervention* is a per-token policy over a confidence metric. Supported families:
**backtracking**, **temperature cooling**, **text steering**, **early-stop**, **think-budget**.

## Requirements

`torch`, `transformers`, `accelerate`, and a model (Qwen3-4B-Thinking-2507). From the repo root:

```bash
pip install -r requirements.txt
```

Run from the **repo root** so `src` is importable (the scripts add it to `sys.path`).

## Run from the CLI

```bash
python scripts/run_intervention.py \
    --benchmark aime24 \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --strategy drop_detect --metric sc \
    --strategy-kwargs '{"drop_threshold": 0.5, "window": 30}' \
    --n-problems 20 \
    --output-dir outputs/my_run
```

Runs the chosen strategy **and** a no-intervention baseline on the same problems, then
reports both accuracies. Results are written to `<output-dir>/<tag>_<timestamp>/results.jsonl`.

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--benchmark` | `hmmt2025` | dataset (see choices below) |
| `--model` | *(required)* | HF name or local path |
| `--metric` | `sc` | per-token confidence signal |
| `--strategy` | `drop_detect` | intervention to apply |
| `--strategy-kwargs` | `{}` | JSON dict of strategy constructor args |
| `--n-problems` | all | limit to first N problems |
| `--max-backtracks` | `5` | per-problem backtrack budget |
| `--max-tokens` | `16384` | generation cap |
| `--temperature` / `--top-p` | `0.6` / `0.95` | sampling |
| `--checkpoint-interval` | `128` | KV-cache checkpoint every N tokens (`0` = disable) |
| `--classifier` | `self_prompted` | answer-readiness check for `--strategy early_stop` |
| `--no-baseline` | off | skip the baseline run |
| `--verbose` | off | stream tokens to stdout as they generate |

Full list: `python scripts/run_intervention.py --help`.

### Choices

- **Metrics:** `sc` (self-certainty), `neg_entropy`, `lp` (log-prob), `z_score_sc`
- **Strategies:** `no_intervention`, `threshold`, `drop_detect`, `linear_cooling`,
  `confidence_cooling`, `drop_steering`, `confident_region_end`, `think_budget`,
  `wait_backtrack`, `early_stop`, `any`, `all`
- **Benchmarks:** `aime24`, `hmmt2025`, `gpqa_diamond`, `char_occur`, `distinct_char`,
  `word_len`, `substring_occur`, `arithmetic_stress_test`

## Use from Python

```python
from src.model.qwen3_helper import Qwen3Helper
from src.interventions.metrics import get_metric
from src.interventions.strategies import get_strategy
from src.interventions.engine import InterventionEngine

helper   = Qwen3Helper(local_path="/path/to/Qwen3-4B-Thinking-2507")
metric   = get_metric("sc")
strategy = get_strategy("drop_detect", drop_threshold=0.5, window=30)

engine = InterventionEngine(
    helper, metric, strategy,
    max_backtracks=5,
    checkpoint_interval=128,   # KV-cache checkpoint every 128 tokens
    offload_to_cpu=True,       # keep VRAM free
)

result = engine.generate(prompt_text, answer_type="integer")
print(result.extracted_answer, result.n_backtracks)
```

## Architecture

Three pluggable interfaces + one engine:

| Component | Contract | Where |
|-----------|----------|-------|
| **Metric** | `(logits, token_id) → float` confidence | `metrics/` |
| **Strategy** | `on_token(...) → InterventionDecision` | `strategies/` |
| **Classifier** | `can_answer(text) → ClassifierResult` (early-stop only) | `classifiers/` |
| **Engine** | drives generation; the *only* part that touches the model / KV cache | `engine.py` |

Compose strategies with the `any` (OR) and `all` (AND) meta-strategies (`strategies/composite.py`).

## Add your own

- **Metric:** new file in `metrics/`, subclass `MetricComputer`, add to `_REGISTRY` in `metrics/__init__.py`.
- **Strategy:** new file in `strategies/`, subclass `InterventionStrategy`, add to `_REGISTRY` in `strategies/__init__.py`.

The engine and CLI pick it up automatically.

## Sweeps (cluster)

```bash
python scripts/submit_intervention_sweep3.py --benchmark aime24 --dry-run   # preview jobs
python scripts/collect_sweep_results.py --sweep-dir outputs/sweep3_aime24   # rank results
```
