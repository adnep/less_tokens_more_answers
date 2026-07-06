#!/bin/bash

#PBS -N qwen3-generate
#PBS -o outputs/logs/generate_cld.log
#PBS -e outputs/logs/generate_cld.err
#PBS -l select=1:ncpus=8:ngpus=1:mem=30gb:gpu_mem=40gb
#PBS -l walltime=40:00:00
#PBS -q gpu
#PBS -j oe
#PBS -m abe
#PBS -M pendasmajljaj@gmail.com

set -e

export TMPDIR=/storage/praha1/home/adnep/tmp
export HF_HOME=/storage/praha1/home/adnep/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TORCH_HOME=/storage/praha1/home/adnep/torch_cache
export XDG_CACHE_HOME=/storage/praha1/home/adnep/.cache
mkdir -p $TMPDIR $HF_HOME $TORCH_HOME $XDG_CACHE_HOME


# ============ STORAGE SETUP ============
# vestec1-elixir = praha1 home, has 2.2TB quota (only 19GB used)
HOME_STORAGE=/storage/praha1/home/adnep
WORK_DIR=$HOME_STORAGE/deep-thinking-replication

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

source $HOME_STORAGE/mainenv/bin/activate

export CUDA_VISIBLE_DEVICES=0

cd $WORK_DIR
mkdir -p outputs/dataset_results

echo "Starting sample generation..."
echo ""

python scripts/generate_samples_batch.py \
  --benchmark substring_occur \
  --n-samples 24 \
  --model /storage/praha1/home/adnep/hf_cache/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea \
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
