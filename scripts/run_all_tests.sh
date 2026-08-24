#!/usr/bin/env bash
# Full Project Test Suite Runner
set -e

echo "=================================================================="
echo " [SKYGUARD UAV: COMPLETE TEST SUITE RUNNER]"
echo "=================================================================="

# 1. Pytest Unit & Integration Tests
echo "[1/4] Running Pytest Unit & Contract Tests..."
python -m pytest tests/ -v

# 2. Dataset Download & Verification
echo "[2/4] Verifying KaggleHub Dataset Access..."
python src/data_pipeline/kaggle_download.py --limit 5

# 3. Model Inference on Real Data
echo "[3/4] Running Model Inference on Real Images..."
python src/inference/test_real_data.py --limit 10

# 4. Full System Verification
echo "[4/4] Executing Complete Verification Scorecard..."
python scripts/verify_all.py

echo "=================================================================="
echo " All tests executed successfully!"
echo "=================================================================="
