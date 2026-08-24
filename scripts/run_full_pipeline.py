"""
Master Pipeline Orchestrator for SkyGuard UAV.
Executes the unified end-to-end workflow:
1. Automated Dataset Download & Extraction via KaggleHub
2. Directory Schema Inspection & Cataloging
3. Real Dataset YOLOv8 Inference & Visual Analytics
4. Performance Latency Profiling (FPS, p95, p99)
5. Pipeline Status Checkpoint Generation (results/pipeline_status.json)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.data_pipeline.kaggle_download import download_dataset, inspect_dataset_directory
from src.inference.test_real_data import test_real_data

logger = setup_logger("master_pipeline")


def run_full_pipeline(
    dataset_handle: str = "kushagrapandya/visdrone-dataset",
    weights_path: Optional[str] = None,
    limit: int = 50,
    conf_threshold: float = 0.35
) -> Dict:
    """Executes the master dataset download and inference pipeline."""
    start_time = time.time()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    status_file = results_dir / "pipeline_status.json"

    print("\n" + "=" * 75)
    print(" [SKYGUARD UAV: MASTER DATASET DOWNLOAD & REAL DATA TESTING PIPELINE]")
    print("=" * 75)

    pipeline_status = {
        "pipeline_name": "skyguard_kaggle_real_data_pipeline",
        "started_at": start_time,
        "dataset_requested": dataset_handle,
        "stages": {}
    }

    # -------------------------------------------------------------
    # STAGE 1: KAGGLEHUB DATASET DOWNLOAD & INSPECTION
    # -------------------------------------------------------------
    logger.info("[STAGE 1/2] Initiating KaggleHub dataset download and inspection...")
    t0_dl = time.perf_counter()
    dataset_path = download_dataset(dataset_handle=dataset_handle, limit=limit)
    dl_duration = round(time.perf_counter() - t0_dl, 2)

    if dataset_path is None or not dataset_path.exists():
        logger.error("Dataset download failed or returned invalid path.")
        pipeline_status["stages"]["stage_1_download"] = {
            "status": "FAILED",
            "duration_seconds": dl_duration
        }
        pipeline_status["overall_status"] = "FAILED"
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(pipeline_status, f, indent=2)
        return pipeline_status

    pipeline_status["stages"]["stage_1_download"] = {
        "status": "SUCCESS",
        "dataset_path": str(dataset_path),
        "duration_seconds": dl_duration
    }

    # -------------------------------------------------------------
    # STAGE 2: REAL DATASET MODEL INFERENCE & VISUALIZATION
    # -------------------------------------------------------------
    logger.info("[STAGE 2/2] Running YOLOv8 inference and performance profiling...")
    t0_inf = time.perf_counter()
    test_metrics = test_real_data(
        dataset_dir=str(dataset_path),
        weights_path=weights_path,
        limit=limit,
        conf_threshold=conf_threshold,
        output_dir="results/real_data_test"
    )
    inf_duration = round(time.perf_counter() - t0_inf, 2)

    pipeline_status["stages"]["stage_2_inference_test"] = {
        "status": "SUCCESS",
        "images_evaluated": test_metrics.get("total_images_tested", 0),
        "total_detections": test_metrics.get("total_detections", 0),
        "throughput_fps": test_metrics.get("throughput_fps", 0.0),
        "mean_latency_ms": test_metrics.get("latency_metrics_ms", {}).get("mean", 0.0),
        "duration_seconds": inf_duration
    }

    total_duration = round(time.time() - start_time, 2)
    pipeline_status["overall_status"] = "COMPLETED"
    pipeline_status["completed_at"] = time.time()
    pipeline_status["total_pipeline_duration_seconds"] = total_duration

    # Write pipeline status checkpoint
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_status, f, indent=2)

    logger.info(f"Master pipeline completed in {total_duration}s. Status written to: {status_file}")

    print("\n" + "=" * 75)
    print(" [MASTER PIPELINE EXECUTION SUMMARY]")
    print("=" * 75)
    print(f"Overall Status:            SUCCESS (COMPLETED)")
    print(f"Total Execution Time:      {total_duration} seconds")
    print(f"Dataset Root:              {dataset_path}")
    print(f"Images Tested:             {test_metrics.get('total_images_tested', 0)}")
    print(f"Total Detections:          {test_metrics.get('total_detections', 0)}")
    print(f"Inference Speed:           {test_metrics.get('throughput_fps', 0.0)} FPS")
    print(f"Pipeline Status Checkpoint:'{status_file}'")
    print("=" * 75 + "\n")

    return pipeline_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master UAV Pipeline: KaggleHub Download + Real Inference Test")
    parser.add_argument("--dataset", type=str, default="kushagrapandya/visdrone-dataset",
                        help="Kaggle dataset handle or alias (visdrone, uavdt, ua_detrac)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to fine-tuned YOLOv8 weights (.pt)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of images to test")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold for predictions")
    args = parser.parse_args()

    run_full_pipeline(
        dataset_handle=args.dataset,
        weights_path=args.weights,
        limit=args.limit,
        conf_threshold=args.conf
    )
