# Full Project Test Suite Runner for Windows PowerShell
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " [SKYGUARD UAV: COMPLETE TEST SUITE RUNNER]" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Pytest Unit & Integration Tests
Write-Host "[1/4] Running Pytest Unit & Contract Tests..." -ForegroundColor Yellow
python -m pytest tests/ -v

# 2. Dataset Download & Verification
Write-Host "[2/4] Verifying KaggleHub Dataset Access..." -ForegroundColor Yellow
python src/data_pipeline/kaggle_download.py --limit 5

# 3. Model Inference on Real Data
Write-Host "[3/4] Running Model Inference on Real Images..." -ForegroundColor Yellow
python src/inference/test_real_data.py --limit 10

# 4. Full System Verification
Write-Host "[4/4] Executing Complete Verification Scorecard..." -ForegroundColor Yellow
python scripts/verify_all.py

Write-Host "==================================================================" -ForegroundColor Green
Write-Host " All tests executed successfully!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
