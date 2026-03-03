#!/bin/bash
# Batch script for Part A: VAE priors comparison on binarized MNIST

DEVICE="cuda"  # or "cpu"
BATCH_SIZE=128
EPOCHS=50
LATENT_DIM=32
NUM_COMPONENTS=10  # for MoG prior

# Filenames
GAUSS_MODEL="vae_gaussian.pt"
MOG_MODEL="vae_mog.pt"
FLOW_MODEL="vae_flow.pt"

GAUSS_SAMPLES="vae_gaussian_samples.png"
MOG_SAMPLES="vae_mog_samples.png"
FLOW_SAMPLES="vae_flow_samples.png"

GAUSS_POSTERIOR="gaussian_prior_vs_posterior.png"
MOG_POSTERIOR="mog_prior_vs_posterior.png"
FLOW_POSTERIOR="flow_prior_vs_posterior.png"

# 1. Train VAE with Gaussian prior
echo "=== Training VAE with Gaussian prior ==="
python3 vae.py train \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --latent-dim $LATENT_DIM \
    --prior gaussian \
    --decoder bernoulli \
    --model $GAUSS_MODEL \
    --samples $GAUSS_SAMPLES \
    --posterior $GAUSS_POSTERIOR

# 2. Train VAE with MoG prior
echo "=== Training VAE with MoG prior ==="
python3 vae.py train \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --latent-dim $LATENT_DIM \
    --prior mog \
    --num-components $NUM_COMPONENTS \
    --decoder bernoulli \
    --model $MOG_MODEL \
    --samples $MOG_SAMPLES \
    --posterior $MOG_POSTERIOR

# 3. Train VAE with Flow prior
echo "=== Training VAE with Flow prior ==="
python3 vae.py train \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --latent-dim $LATENT_DIM \
    --prior flow \
    --decoder bernoulli \
    --mask random \
    --model $FLOW_MODEL \
    --samples $FLOW_SAMPLES \
    --posterior $FLOW_POSTERIOR

# 4. Sample and measure wall-clock times
echo "=== Sampling and timing ==="

sample_time() {
    MODEL_FILE=$1
    OUTPUT_FILE=$2
    PRIOR_NAME=$3

    echo "Sampling $PRIOR_NAME..."
    START=$(date +%s.%N)
    python3 vae.py sample \
        --device $DEVICE \
        --batch-size $BATCH_SIZE \
        --model $MODEL_FILE \
        --samples $OUTPUT_FILE
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "$PRIOR_NAME sampling time (s): $ELAPSED"
}

# Sample from Gaussian prior VAE
sample_time $GAUSS_MODEL $GAUSS_SAMPLES "Gaussian prior"

# Sample from MoG prior VAE
sample_time $MOG_MODEL $MOG_SAMPLES "MoG prior"

# Sample from Flow prior VAE
sample_time $FLOW_MODEL $FLOW_SAMPLES "Flow prior"