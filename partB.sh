#!/bin/sh
#BSUB -q gpuv100
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
BETA="1.0"
BATCH_SIZE=64
LATENT_DIM=32

VAE_EPOCHS=10
DDPM_EPOCHS=50
LATENT_DDPM_EPOCHS=50

# Filenames
VAE_MODEL="vae_model.pt"
DDPM_MODEL="ddpm_model.pt"
LATENT_DDPM_MODEL="latent_ddpm_model.pt"

VAE_SAMPLES="vae_samples.png"
DDPM_SAMPLES="ddpm_samples.png"
LATENT_DDPM_SAMPLES="latent_ddpm_samples.png"

POSTERIOR_PLOT="vae_prior_vs_posterior.png"

# 1. Train VAE
echo "=== Training VAE ==="
python3 vae.py train \
    --model $VAE_MODEL \
    --samples $VAE_SAMPLES \
    --posterior $POSTERIOR_PLOT \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $VAE_EPOCHS \
    --latent-dim $LATENT_DIM \
    --prior gaussian \
    --decoder gaussian \
    --beta $BETA


# 2. Train DDPM on raw MNIST
echo "=== Training DDPM on MNIST ==="
python3 ddpm.py train \
    --data mnist \
    --model $DDPM_MODEL \
    --samples $DDPM_SAMPLES \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $DDPM_EPOCHS \
    --network unet


# 3. Train Latent DDPM (DDPM in VAE latent space)
echo "=== Training Latent DDPM ==="
python3 ddpm.py train \
    --data mnist \
    --model $LATENT_DDPM_MODEL \
    --samples $LATENT_DDPM_SAMPLES \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $LATENT_DDPM_EPOCHS \
    --network fcn \
    --vae_model $VAE_MODEL \
    --latent-dim $LATENT_DIM \
    --prior gaussian \
    --decoder gaussian \
    --beta $BETA

# 4. Sample and record wall-clock times
echo "=== Sampling and timing ==="

# 1. Time VAE
echo "=== Time VAE ==="
time python3 vae.py sample \
    --model $VAE_MODEL \
    --samples $VAE_SAMPLES \
    --posterior $POSTERIOR_PLOT \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $VAE_EPOCHS \
    --latent-dim $LATENT_DIM \
    --prior gaussian \
    --decoder gaussian \
    --beta $BETA


# 2. Time DDPM on raw MNIST
echo "=== Time DDPM on MNIST ==="
time python3 ddpm.py sample \
    --data mnist \
    --model $DDPM_MODEL \
    --samples $DDPM_SAMPLES \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $DDPM_EPOCHS \
    --network unet


# 3. Time Latent DDPM (DDPM in VAE latent space)
echo "=== TIME Latent DDPM ==="
time python3 ddpm.py sample \
    --data mnist \
    --model $LATENT_DDPM_MODEL \
    --samples $LATENT_DDPM_SAMPLES \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $LATENT_DDPM_EPOCHS \
    --network fcn \
    --vae_model $VAE_MODEL \
    --latent-dim $LATENT_DIM \
    --prior gaussian \
    --decoder gaussian \
    --beta $BETA