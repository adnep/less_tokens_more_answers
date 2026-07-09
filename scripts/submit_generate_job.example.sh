#!/bin/bash
# ---------------------------------------------------------------------------
# EXAMPLE / TEMPLATE PBS job for batch sample generation on a cluster.
# Copy to submit_generate_job.sh and fill in the placeholders below:
#   <YOUR_EMAIL>   your notification email address
#   <CLUSTER>      storage volume name (e.g. praha1)
#   <USERNAME>     your cluster username
#   <YOUR_ENV>     virtual env name
# ---------------------------------------------------------------------------

#PBS -N qwen3-generate
#PBS -o outputs/logs/generate_cld.log
#PBS -e outputs/logs/generate_cld.err
#PBS -l select=1:ncpus=8:ngpus=1:mem=30gb:gpu_mem=40gb
#PBS -l walltime=40:00:00
#PBS -q gpu
#PBS -j oe
#PBS -m abe
#PBS -M <YOUR_EMAIL>

set -e

export TMPDIR=/storage/<CLUSTER>/home/<USERNAME>/tmp
export HF_HOME=/storage/<CLUSTER>/home/<USERNAME>/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TORCH_HOME=/storage/<CLUSTER>/home/<USERNAME>/torch_cache
export XDG_CACHE_HOME=/storage/<CLUSTER>/home/<USERNAME>/.cache
mkdir -p $TMPDIR $HF_HOME $TORCH_HOME $XDG_CACHE_HOME


# ============ STORAGE SETUP ============
# Point HOME_STORAGE at a volume with enough quota for the HF/torch caches.
HOME_STORAGE=/storage/<CLUSTER>/home/<USERNAME>
WORK_DIR=$HOME_STORAGE/less_tokens_more_answers

echo "=========================================="
echo "Metacentrum Batch Job: Sample Generation"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "HF_HOME: $HF_HOME"
echo "TMPDIR: $TMPDIR"
echo "=========================================="
echo ""

nvidia-smi

echo ""
echo "Disk usage before:"
du -sh $HOME_STORAGE/hf_cache $HOME_STORAGE/torch_cache 2>/dev/null || true
echo ""

source $HOME_STORAGE/<YOUR_ENV>/bin/activate

export CUDA_VISIBLE_DEVICES=0

cd $WORK_DIR
mkdir -p outputs/dataset_results

echo "Starting sample generation..."
echo ""

python scripts/generate_samples_batch.py \
  --benchmark arithmetic_stress_test \
  --n-samples 24 \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --output-dir outputs/dataset_results \
  --max-tokens 16384 \
  --temperature 0.6 \
  --top-p 0.95

echo ""
echo "=========================================="
echo "Job completed successfully!"
echo "Disk usage after:"
du -sh $HOME_STORAGE/hf_cache $HOME_STORAGE/torch_cache $WORK_DIR/outputs
echo "=========================================="
