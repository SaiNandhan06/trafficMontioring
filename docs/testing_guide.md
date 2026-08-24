# 🧪 SkyGuard UAV Testing & Verification Guide

Comprehensive instructions for executing unit tests, contract validations, real dataset inference, hardware latency profiling, and system scorecards.

---

## 🚀 1. Quick Smoke Test (< 10 seconds)
Quickly test the pipeline with 10 images:

```powershell
# Windows PowerShell:
.\scripts\quick_test.ps1

# Or with Python:
python src/data_pipeline/kaggle_download.py --limit 10
python src/inference/test_real_data.py --limit 10
```

---

## 🎯 2. Complete Verification Suite
Run the automated verification scorecard across all 10 project components:

```powershell
python scripts/verify_all.py
```

### Expected Output:
```
===========================================================================
 [SKYGUARD UAV: COMPREHENSIVE SYSTEM VERIFICATION SUITE]
===========================================================================
  [PASS] Environment Variables               : Loaded .env
  [PASS] Core Dependencies                   : PyTorch | YOLOv8 | KaggleHub
  [PASS] Model File Available                : yolov8_uav_best.pt
  [PASS] Model Load & Architecture           : Loaded DetectionModel
  [PASS] Kaggle Dataset Download             : Dataset accessible
  [PASS] Single Image Inference              : Inference successful
  [PASS] Smart Contract Compilation          : Compiled 2 contracts
  [PASS] API Health Endpoint                 : HTTP 200 - HEALTHY
  [PASS] Security & HMAC Auth                : HMAC-SHA256 verified
  [PASS] End-to-End Pytest Suite             : 8/8 test suites passed
===========================================================================
 [VERIFICATION SUMMARY SCORECARD]
===========================================================================
Total Checks:   10
Passed:         10
Failed:         0
Blocked:        0
Overall Health: 100% HEALTHY
===========================================================================
```

---

## 📦 3. Master Pipeline (Download + Real Data Testing)
Downloads the latest VisDrone dataset from KaggleHub and runs inference on 50 images with side-by-side visualization exports:

```powershell
python scripts/run_full_pipeline.py --dataset visdrone --limit 50
```

- Results saved to: `results/real_data_test/visualizations/`
- Performance metrics JSON: `results/real_data_test/metrics_report.json`
- Pipeline status JSON: `results/pipeline_status.json`

---

## 🛠️ 4. Common Troubleshooting & Fixes

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'kagglehub'` | Package not installed | Run `pip install kagglehub` |
| `Kaggle authentication error` | Kaggle API credentials missing | Ensure dataset is public or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` |
| `OutOfMemoryError on CUDA` | Batch size too high | Reduce batch size: `python model/train.py --batch 8` |
| `Port 8501 already in use` | Previous Streamlit instance still running | Terminate old process or run on new port: `streamlit run dashboard/app.py --server.port 8502` |
