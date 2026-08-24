#!/usr/bin/env bash
# Quick Test Script: Download 10 images and test YOLOv8 model inference
set -e

echo "=================================================================="
echo " [SKYGUARD UAV: QUICK SMOKE TEST]"
echo "=================================================================="

# 1. Download/check 10 sample images via KaggleHub
echo "[1/2] Fetching sample UAV dataset (10 images)..."
python src/data_pipeline/kaggle_download.py --limit 10

# 2. Run inference on those 10 images
echo "[2/2] Running model inference on 10 sample images..."
python src/inference/test_real_data.py --limit 10

echo "=================================================================="
echo " Quick test completed successfully!"
echo " Results and visualizations saved to: results/real_data_test/"
echo "=================================================================="
