# Deep-Thinking Replication & Intervention Framework

Master's thesis code, in two parts:

1. **Intervention framework** — a pluggable engine that steers a reasoning model's
   generation **mid-stream** (backtracking, temperature cooling, steering, early-stop,
   think-budget) using per-token confidence signals. *(Thesis chapters 5–6.)*
2. **DTR replication** — reproduces the Deep-Thinking-Ratio (DTR) sample-selection
   method from *"Think Deep, Not Just Long"*, plus the confidence-signal trajectory
   analysis (Self-Certainty, log-probability). *(Thesis appendix B.)*

Model used throughout: [https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507](**Qwen3-4B-Thinking-2507**).

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
    --benchmark aime24 \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --strategy drop_detect --metric sc \
    --n-problems 20
```

## Part 2 — DTR replication & confidence analysis

Config-driven via `configs/default.yaml` (model, DTR `gamma`/`rho`, think@n `n`/`eta`).

```bash
# 1. Generate samples (vLLM, batched)     — reads configs/default.yaml
python scripts/generate_samples_batch.py
#    on the cluster instead:  qsub scripts/submit_generate_job.sh

# 2. Think@n / DTR selection + accuracy
python scripts/run_think_at_n.py --output-dir outputs/aime_think_at_n
python scripts/run_baselines.py  --samples-file <dir>/generated_samples.jsonl --benchmark aime24

# 3. Single-problem DTR heatmap
python scripts/compute_dtr_single.py --model Qwen/Qwen3-4B-Thinking-2507

# 4. Interactive explorer (Streamlit)
streamlit run scripts/dtr_explorer.py
```

Run any script with `--help` for its full options.



## Notes

- generated experiment results appear in `outputs/` (not included in git because it is too large)
