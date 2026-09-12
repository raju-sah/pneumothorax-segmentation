#!/usr/bin/env bash
# Download SIIM-ACR inputs via Kaggle CLI (run once; ~3.3 GB).
# Requires: kaggle CLI authenticated (kaggle.json or env vars) + competition rules accepted.
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d jesperdramsch/siim-acr-pneumothorax-segmentation-data -p data/raw --unzip
kaggle datasets download -d rajucode/siim-acr-processed-splits -p data/raw --unzip
kaggle datasets download -d rajucode/pneumothorax-ensemble-weights-m3 -p results/checkpoints --unzip
echo "OK: data/raw + results/checkpoints populated"
