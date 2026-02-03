#!/bin/bash
#SBATCH -J "SFT-Train"           # işin adı
#SBATCH -A idm001                # account / proje adı
#SBATCH -p gpu2dq                # kuyruk (partition/queue) adı
#SBATCH -n 64                    # çekirdek / işlemci sayısı
#SBATCH -N 1                     # bilgisayar sayısı
#SBATCH --gres=gpu:1             # ilave kaynak (1 GPU gerekli)

# Miniconda'yı başlat
source $HOME/miniconda3/etc/profile.d/conda.sh

# Ortamı aktive et
conda activate turkish-gpt2-training

# ============================================================================
# CONFIGURATION
# ============================================================================

export TRAINING_WORKSPACE="$HOME"
export JOB_ID="${SLURM_JOB_ID}"

# Model and dataset
export MODEL_NAME="final_unwrapped"
export DATASET_NAME="sft_ready_dataset.jsonl"

# Dataset configuration
export TEST_SIZE="4000"
export MAX_SEQ_LENGTH="1024"
export DATASET_SEED="42"

# Training hyperparameters
export NUM_EPOCHS="3"
export LEARNING_RATE="2e-5"
export WEIGHT_DECAY="0.01"

# Batch size and gradient accumulation
export BATCH_SIZE="4"
export GRAD_ACCUM_STEPS="4"
export AUTO_BATCH_SIZE="false"
export DATALOADER_NUM_WORKERS="8"

# Mixed precision
export USE_BF16="true"

# Logging and checkpointing
export LOGGING_STEPS="10"
export EVAL_STEPS="500"
export SAVE_STEPS="500"
export SAVE_TOTAL_LIMIT="2"
export GPU_LOGGING_STEPS="50"

# LoRA configuration
export LORA_R="64"
export LORA_ALPHA="128"
export LORA_DROPOUT="0.1"

# SFT-specific settings
export PACKING="true"

# Resume training
export RESUME_FROM_CHECKPOINT="auto"

# Random seed
export SEED="42"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo "=========================================="
echo "SFT Training Job ${SLURM_JOB_ID}"
echo "=========================================="
echo "Start time: $(date)"
echo "Working directory: ${TRAINING_WORKSPACE}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${TRAINING_WORKSPACE}/jobs/${SLURM_JOB_ID}/output"
echo "=========================================="

python sft_training.py

echo "=========================================="
echo "Training job completed at $(date)"
echo "=========================================="
