#!/bin/bash

#PBS -N qwen3-generate
#PBS -o outputs/logs/generate_${PBS_JOBID}.log
#PBS -e outputs/logs/generate_${PBS_JOBID}.err
#PBS -l select=1:ncpus=8:ngpus=1:mem=30gb:gpu_mem=40gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -j oe
#PBS -m abe
#PBS -M pendasmajljaj@gmail.com

set -e  # exit on error

echo "=========================================="
echo "Metacentrum Batch Job: Sample Generation"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "=========================================="
echo ""

nvidia-smi

cd /storage/praha1/home/adnep
source mainenv/bin/activate

export CUDA_VISIBLE_DEVICES=0

export HF_HOME=/storage/praha1/home/adnep/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p $HF_HOME

cd /storage/praha1/home/adnep/deep-thinking-replication

# Create output directories
mkdir -p outputs/logs
mkdir -p outputs/aime24_samples

# Run generation
echo "Starting sample generation..."
echo ""

python scripts/generate_samples_batch.py \
  --benchmark aime24 \
  --n-samples 12 \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --output-dir outputs/aime24_samples \
  --max-tokens 8192 \
  --temperature 0.6 \
  --top-p 0.95

echo ""
echo "=========================================="
echo "Job completed successfully!"
echo "=========================================="
