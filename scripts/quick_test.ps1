# Quick Test Script for Windows PowerShell
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " [SKYGUARD UAV: QUICK SMOKE TEST]" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Download/check 10 sample images via KaggleHub
Write-Host "[1/2] Fetching sample UAV dataset (10 images)..." -ForegroundColor Yellow
python src/data_pipeline/kaggle_download.py --limit 10

# 2. Run inference on those 10 images
Write-Host "[2/2] Running model inference on 10 sample images..." -ForegroundColor Yellow
python src/inference/test_real_data.py --limit 10

Write-Host "==================================================================" -ForegroundColor Green
Write-Host " Quick test completed successfully!" -ForegroundColor Green
Write-Host " Results and visualizations saved to: results/real_data_test/" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
