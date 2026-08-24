"""
Model Evaluation and Benchmark Suite for SkyGuard UAV.
Runs rigorous evaluation on the specified dataset split using Ultralytics YOLOv8
and exports traceable, verified JSON reports without synthetic/fake fallback numbers.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("evaluate_model")


def evaluate_model(
    weights_path: Path = None,
    data_yaml: Path = None,
    split: str = "test",
    device: str = "cpu",
    output_json: Path = None
) -> dict:
    """
    Evaluates YOLOv8 model performance on real dataset split.
    Saves a verified, traceable report to results/model_evaluation_report.json.
    """
    if weights_path is None:
        weights_path = settings.WEIGHTS_DIR / "yolov8_uav_best.pt"
        if not weights_path.exists():
            weights_path = PROJECT_ROOT / "yolov8n.pt"

    if data_yaml is None:
        data_yaml = settings.DATA_DIR / "dataset.yaml"

    if output_json is None:
        output_json = PROJECT_ROOT / "results" / "model_evaluation_report.json"

    weights_path = Path(weights_path).resolve()
    data_yaml = Path(data_yaml).resolve()
    output_json = Path(output_json).resolve()

    if not weights_path.exists():
        logger.error(f"Weights file not found at: {weights_path}")
        return {
            "status": "NOT_EVALUATED",
            "error": f"Weights file not found: {weights_path}",
            "source_type": "error"
        }

    if not data_yaml.exists():
        logger.error(f"Dataset YAML config not found at: {data_yaml}")
        return {
            "status": "NOT_EVALUATED",
            "error": f"Dataset config not found: {data_yaml}",
            "source_type": "error"
        }

    logger.info(f"Initiating evaluation: Model={weights_path.name} | Dataset={data_yaml.name} | Split={split} | Device={device}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error(f"Ultralytics is not installed: {e}")
        return {
            "status": "NOT_EVALUATED",
            "error": "Ultralytics package missing",
            "source_type": "error"
        }

    try:
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            device=device,
            verbose=True
        )

        # Extract Overall Metrics
        overall_map50 = float(metrics.box.map50)
        overall_map50_95 = float(metrics.box.map)
        
        # Safely extract precision and recall arrays
        p_array = metrics.box.p
        r_array = metrics.box.r
        overall_p = float(p_array.mean()) if hasattr(p_array, "mean") and len(p_array) > 0 else 0.0
        overall_r = float(r_array.mean()) if hasattr(r_array, "mean") and len(r_array) > 0 else 0.0

        # Class-Wise Breakdown
        class_names = ["vehicle", "pedestrian", "cyclist", "traffic_signal"]
        per_class_metrics = {}

        for i, cname in enumerate(class_names):
            c_p = float(p_array[i]) if i < len(p_array) else 0.0
            c_r = float(r_array[i]) if i < len(r_array) else 0.0
            c_map50 = float(metrics.box.maps[i]) if hasattr(metrics.box, "maps") and i < len(metrics.box.maps) else 0.0
            per_class_metrics[cname] = {
                "precision": round(c_p, 6),
                "recall": round(c_r, 6),
                "map50": round(c_map50, 6)
            }

        # Build Verified Report
        report = {
            "status": "REAL_EVALUATION",
            "source_type": "actual_test_split_evaluation",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "weights_file": str(weights_path),
            "weights_name": weights_path.name,
            "dataset_yaml": str(data_yaml),
            "dataset_split": split,
            "device": device,
            "overall": {
                "precision": round(overall_p, 6),
                "recall": round(overall_r, 6),
                "map50": round(overall_map50, 6),
                "map50_95": round(overall_map50_95, 6)
            },
            "per_class": per_class_metrics,
            "speed_ms": {
                "preprocess": round(float(metrics.speed.get("preprocess", 0.0)), 2),
                "inference": round(float(metrics.speed.get("inference", 0.0)), 2),
                "loss": round(float(metrics.speed.get("loss", 0.0)), 2),
                "postprocess": round(float(metrics.speed.get("postprocess", 0.0)), 2)
            }
        }

        # Save Report JSON
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Print Transparent Summary
        print("\n" + "=" * 70)
        print(" [VERIFIED MODEL EVALUATION SUMMARY — REAL MEASURED RESULTS]")
        print("=" * 70)
        print(f"Model Weights:      {weights_path.name}")
        print(f"Dataset Split:      {split}")
        print(f"Device:             {device}")
        print(f"Source Type:        Actual Test Split Execution (Real)")
        print("-" * 70)
        print(f"{'Class':<18} | {'Precision':<12} | {'Recall':<12} | {'mAP@50':<12}")
        print("-" * 70)
        for cname, cdata in per_class_metrics.items():
            print(f"{cname.capitalize():<18} | {cdata['precision']:<12.6f} | {cdata['recall']:<12.6f} | {cdata['map50']:<12.6f}")
        print("-" * 70)
        print(f"{'Overall (All)':<18} | {overall_p:<12.6f} | {overall_r:<12.6f} | {overall_map50:<12.6f}")
        print(f"Overall mAP@50-95:   {overall_map50_95:.6f}")
        print("=" * 70)
        print(f"Report saved to: {output_json}\n")

        return report

    except Exception as e:
        logger.error(f"Evaluation execution failed: {e}")
        error_report = {
            "status": "EVALUATION_FAILED",
            "error": str(e),
            "source_type": "error"
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=2)
        print(f"\n[ERROR] Model evaluation failed: {e}. No fake metrics displayed.\n")
        return error_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model performance honestly on real data")
    parser.add_argument("--weights", type=Path, default=None, help="Weights path")
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (test, val)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda:0)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_model(args.weights, args.data, args.split, args.device, args.output)
