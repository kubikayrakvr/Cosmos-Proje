#!/bin/bash
#SBATCH -J "GRPO-Offline"        # İşin adı
#SBATCH -A idm001                # Proje adı
#SBATCH -p gpu2dq                # Kuyruk adı
#SBATCH -n 64                    # İşlemci sayısı
#SBATCH -N 1                     # Bilgisayar sayısı
#SBATCH --gres=gpu:1             # 1 GPU istiyoruz

# Miniconda ve Ortam (Kendi yoluna göre ayarla)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate turkish-gpt2-training

# ============================================================================
# CONFIGURATION 
# ============================================================================

export TRAINING_WORKSPACE="$HOME/Cosmos-Proje"
export JOB_ID="${SLURM_JOB_ID}"

# TODO: Buraya SFT eğitimi bitmiş modelin klasör yolunu ver!!!
export MODEL_NAME="$HOME/Cosmos-Proje/jobs/SFT_IS_ID/output/final_model" 

# TODO: Dataset dosyasının adı!!!
export DATASET_NAME="grpo_offline.jsonl"

# Hiperparametreler
export MAX_SEQ_LENGTH="1024"
export NUM_EPOCHS="1"           
export LEARNING_RATE="1e-5"     
export BATCH_SIZE="4"
export GRAD_ACCUM_STEPS="4"
export WEIGHT_DECAY="0.01"

# Donanım
export USE_BF16="true"           # GPU destekliyorsa true

# Loglama ve Kayıt
export LOGGING_STEPS="10"
export SAVE_STEPS="100"
export SAVE_TOTAL_LIMIT="2"
export GPU_LOGGING_STEPS="50"

# LoRA Ayarları
export LORA_R="64"
export LORA_ALPHA="128"
export LORA_DROPOUT="0.05"

export SEED="42"

# ============================================================================
# RUN (ÇALIŞTIRMA)
# ============================================================================

echo "=========================================="
echo "GRPO Offline Job ${SLURM_JOB_ID}"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output: ${TRAINING_WORKSPACE}/jobs/${SLURM_JOB_ID}/output"
echo "=========================================="

python train_grpo_offline.py

echo "=========================================="
echo "Training job completed at $(date)"
echo "=========================================="