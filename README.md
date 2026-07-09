# Intervention Framework & Experimental Work

Master's thesis code, in two parts:

1. **Intervention framework** — a pluggable engine that steers a reasoning model's generation **mid-stream** (backtracking, temperature cooling, steering, early-stop, think-budget) using per-token confidence signals. *(Thesis chapters 5–6.)*
2. **Scripts: the experiment & analysis pipeline** — a suite of CLI tools under `scripts/` that drive the whole workflow: generate samples, run and sweep experiments, aggregate & visualize results, and reproduce the Deep-Thinking-Ratio (DTR) sample-selection method plus the confidence-signal analysis (self-certainty, negative entropy, z-scored self-certainty, log-probability). *(Thesis appendix B + the tooling behind chapters 5–6.)*

Model used throughout: [**Qwen3-4B-Thinking-2507**](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).

## Layout

```
src/
  dtr/            DTR computation (JSD settling, deep-thinking ratio)
  evaluation/     benchmarks, answer extraction/voting, accuracy metrics
  inference/      think-at-n selection, sampling
  interventions/  mid-generation intervention framework  ← has its own README
  model/          Qwen3 helper (HuggingFace wrapper)
  visualization/  plotting
scripts/          CLI entrypoints + cluster (PBS) job scripts
configs/          default.yaml — DTR / generation hyperparameters
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is required (bf16 model; vLLM needs CUDA). Run all scripts from the repo root.

## Part 1 — Intervention framework (Main Contribution)

Full guide: [`src/interventions/README.md`](src/interventions/README.md). Quick start:

```bash
python scripts/run_intervention.py \
    --benchmark arithmetic_stress_test \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --strategy drop_detect --metric sc \
    --n-problems 1 \
    --verbose
```

To run this at scale (grids of strategies × metrics × benchmarks on the cluster) and turn the results into figures, see the **run/sweep** and **aggregate & visualize** stages in Part 2.

## Part 2 — Scripts: the experiment & analysis pipeline

Everything in `scripts/` is a CLI entrypoint. Run any of them with `--help` for full options. The scripts fall into five pipeline stages: generate, run/sweep, aggregate & visualize, DTR replication, logit lens inspection. Sample generation and DTR are config-driven via `configs/default.yaml` (model, DTR `gamma`/`rho`, think@n `n`/`eta`). Cluster submission targets Metacentrum (PBS / `qsub`).

### 1. Generate samples

```bash
# vLLM, batched, incremental — reads configs/default.yaml
python scripts/generate_samples_batch.py
#    on the cluster instead:  qsub scripts/submit_generate_job.sh
```

| Script | What it does |
|---|---|
| `generate_samples_batch.py` | Generate _n_ samples/problem with vLLM, saved incrementally. |
| `submit_generate_job.sh` | PBS batch jobs for generation on Metacentrum. |

### 2. Run & sweep experiments

```bash
# Baselines (no DTR/intervention selection) on multiple pre-generated samples
python scripts/run_baselines.py  --samples-file <dir>/generated_samples.jsonl --benchmark arithmetic_stress_test

# Think@n / DTR selection + accuracy
python scripts/run_think_at_n.py --output-dir outputs/arithmetic_think_at_n

# Sweep a grid of intervention configs as PBS jobs (all 8 benchmarks)
python scripts/submit_intervention_sweep.py
```

| Script | What it does |
|---|---|
| `run_intervention.py` | Single intervention run (see Part 1 quick-start). |
| `run_baselines.py` | maj@n / pass@n baselines without DTR selection. |
| `run_think_at_n.py` | Think@n / DTR sample selection + accuracy. |
| `submit_intervention_sweep.py` | Submit grids of intervention experiments as PBS jobs — all 8 benchmarks. |

### 3. Aggregate & visualize sweep results

```bash
# Collect + rank every completed run in a sweep directory
python scripts/collect_sweep_results.py --sweep-dir outputs/sweep_arithmetic_stress_test

# Cross-dataset strategy-family analysis → heatmaps + summary CSVs
python scripts/analyze_cross_dataset.py --out-dir outputs/analysis

# Per-dataset accuracy-vs-tokens Pareto plots
python scripts/plot_pareto.py --sweep-dir outputs/sweep_*
```

| Script | What it does |
|---|---|
| `collect_sweep_results.py` | Collect and rank all completed sweep runs (`--sweep-dir`, optional `--baseline-dir`). |
| `analyze_cross_dataset.py` | Cross-dataset per-family analysis; writes `cross_dataset_heatmap.png`, efficiency scatters, and summary CSVs to `--out-dir` (default `outputs/analysis`). |
| `plot_pareto.py` | Accuracy vs. avg-tokens Pareto front per dataset; `--sweep-dir` (dirs) or `--csv` (files). |
| `analyze_results.py` | Score distributions vs. correctness + Pearson correlations from a `results.json`. |

### 4. DTR replication & confidence analysis

Reproduces the Deep-Thinking-Ratio metric from [*"Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"*](https://arxiv.org/abs/2602.13517) and the confidence-signal analyses.

```bash
# Single-problem DTR/JSD heatmap (verifies the pipeline end-to-end)
python scripts/compute_dtr_single.py --model Qwen/Qwen3-4B-Thinking-2507

# Binned DTR–accuracy correlation
python scripts/dtr_binned_correlation.py --results <dir>/results.json --bins 5

# Interactive explorer
streamlit run scripts/dtr_explorer.py
```

| Script | What it does |
|---|---|
| `compute_dtr_single.py` | Compute DTR for a single generation — end-to-end pipeline check. |
| `dtr_binned_correlation.py` | Bin samples by DTR, average accuracy per bin, Pearson _r_. |
| `dtr_prefix_reliability.py` | DTR prefix-reliability analysis. |
| `prefix_reliability_corr.py` | Correlate DTR prefix estimates with the full DTR value, split by correct/incorrect/all. |
| `ask_and_heatmap.py` | Ask the model a question; save a DTR/JSD heatmap per _N_-token window. |
| `selfcert_trajectory.py` | Self-certainty trajectory analysis over a generation. |
| `trace_inspector.py` | Per-token SC / LP / neg-entropy coloring of reasoning traces as HTML. |
| `dtr_explorer.py` | Streamlit app for inspecting JSD heatmaps and DTR interactively. |

### 5. Logit / tuned lens inspection

| Script | What it does |
|---|---|
| `train_tuned_lens.py` | Train tuned-lens (to compare with the logit lens) translators for Qwen3-4B-Thinking. Implemented according to paper [*"Eliciting Latent Predictions from Transformers with the Tuned Lens"*](https://arxiv.org/abs/2303.08112)  |
| `run_logit_lens_sanity.py` | Sanity-check that the logit lens works on Qwen3-4B-Thinking. |



## Notes

- generated experiment results appear in `outputs/` (not included in git because it is too large)
