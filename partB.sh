#!/bin/sh
#BSUB -q gpua100
#BSUB -J train_ddpm
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -o ddpm_%J.out
#BSUB -e ddpm_%J.err

module load cuda/13.0.2 && module load python3/3.14.2 
source .venv/bin/activate

DEVICE="cuda"  # or "cpu"
BATCH_SIZE=64
LATENT_DIM=32
LEARNING_RATE=1e-4

VAE_EPOCHS=50
DDPM_EPOCHS=100
LATENT_DDPM_EPOCHS=100

# Filenames (you can suffix with beta to avoid overwriting)
VAE_MODEL="vae_model"
DDPM_MODEL="ddpm_model.pt"
LATENT_DDPM_MODEL="latent_ddpm_model"

VAE_SAMPLES="vae_samples"
DDPM_SAMPLES="ddpm_samples.png"
LATENT_DDPM_SAMPLES="latent_ddpm_samples"

VAE_SAMPLES_DATA="vae_samples.pt"
DDPM_SAMPLES_DATA="ddpm_samples.pt"
LATENT_DDPM_SAMPLES_DATA="latent_ddpm_samples.pt"

POSTERIOR_PLOT="vae_prior_vs_posterior.png"

# 1. Train DDPM on raw MNIST (once)
echo "=== Training DDPM on MNIST ==="
python3 ddpm.py train \
    --data mnist \
    --model $DDPM_MODEL \
    --samples $DDPM_SAMPLES \
    --samples-data $DDPM_SAMPLES_DATA \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $DDPM_EPOCHS \
    --lr $LEARNING_RATE \
    --network unet

echo "=== Time DDPM on MNIST ==="
python3 ddpm.py sample \
    --data mnist \
    --model $DDPM_MODEL \
    --samples $DDPM_SAMPLES \
    --samples-data $DDPM_SAMPLES_DATA \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $DDPM_EPOCHS \
    --lr $LEARNING_RATE \
    --network unet

# List of beta values
BETA_LIST=(0.1 0.5 1.0 2.0 5.0)

for BETA in "${BETA_LIST[@]}"; do
    echo "=== Running experiments with beta=$BETA ==="

    # VAE filenames with beta suffix
    VAE_MODEL_B="${VAE_MODEL}_beta${BETA}.pt"
    VAE_SAMPLES_B="${VAE_SAMPLES}_beta${BETA}.png"
    VAE_SAMPLES_DATA_B="${VAE_SAMPLES_DATA}_beta${BETA}.pt"
    POSTERIOR_PLOT_B="vae_prior_vs_posterior_beta${BETA}.png"

    # 2. Train VAE
    echo "=== Training VAE with beta=$BETA ==="
    python3 vae.py train \
        --model $VAE_MODEL_B \
        --samples $VAE_SAMPLES_B \
        --samples-data $VAE_SAMPLES_DATA_B \
        --posterior $POSTERIOR_PLOT_B \
        --device $DEVICE \
        --batch-size $BATCH_SIZE \
        --epochs $VAE_EPOCHS \
        --latent-dim $LATENT_DIM \
        --prior gaussian \
        --decoder gaussian \
        --beta $BETA

    # 2. Time VAE
    echo "=== Time VAE with beta=$BETA ==="
    python3 vae.py sample \
        --model $VAE_MODEL_B \
        --samples $VAE_SAMPLES_B \
        --samples-data $VAE_SAMPLES_DATA_B \
        --posterior $POSTERIOR_PLOT_B \
        --device $DEVICE \
        --batch-size $BATCH_SIZE \
        --epochs $VAE_EPOCHS \
        --latent-dim $LATENT_DIM \
        --prior gaussian \
        --decoder gaussian \
        --beta $BETA

    # Latent DDPM filenames with beta suffix
    LATENT_DDPM_MODEL_B="${LATENT_DDPM_MODEL}_beta${BETA}.pt"
    LATENT_DDPM_SAMPLES_B="${LATENT_DDPM_SAMPLES}_beta${BETA}.png"
    LATENT_DDPM_SAMPLES_DATA_B="${LATENT_DDPM_SAMPLES_DATA}_beta${BETA}.pt"

    # 3. Train Latent DDPM
    echo "=== Training Latent DDPM with beta=$BETA ==="
    python3 ddpm.py train \
        --data mnist \
        --model $LATENT_DDPM_MODEL_B \
        --samples $LATENT_DDPM_SAMPLES_B \
        --samples-data $LATENT_DDPM_SAMPLES_DATA_B \
        --device $DEVICE \
        --batch-size $BATCH_SIZE \
        --epochs $LATENT_DDPM_EPOCHS \
        --lr $LEARNING_RATE \
        --network fcn \
        --vae-model $VAE_MODEL_B \
        --latent-dim $LATENT_DIM \
        --prior gaussian \
        --decoder gaussian \
        --beta $BETA

    # 3. Time Latent DDPM
    echo "=== Time Latent DDPM with beta=$BETA ==="
    python3 ddpm.py sample \
        --data mnist \
        --model $LATENT_DDPM_MODEL_B \
        --samples $LATENT_DDPM_SAMPLES_B \
        --samples-data $LATENT_DDPM_SAMPLES_DATA_B \
        --device $DEVICE \
        --batch-size $BATCH_SIZE \
        --epochs $LATENT_DDPM_EPOCHS \
        --lr $LEARNING_RATE \
        --network fcn \
        --vae-model $VAE_MODEL_B \
        --latent-dim $LATENT_DIM \
        --prior gaussian \
        --decoder gaussian \
        --beta $BETA

done