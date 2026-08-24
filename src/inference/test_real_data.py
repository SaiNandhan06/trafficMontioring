"""
Legacy real data testing compatibility layer.
Delegates to canonical module: src.inference.real_data_inference.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.real_data_inference import (
    run_real_data_inference,
    test_real_data,
    find_best_model_weights,
    find_test_images,
    draw_detections,
    create_side_by_side,
    CLASS_NAMES,
    CLASS_COLORS
)

__all__ = [
    "run_real_data_inference",
    "test_real_data",
    "find_best_model_weights",
    "find_test_images",
    "draw_detections",
    "create_side_by_side",
    "CLASS_NAMES",
    "CLASS_COLORS"
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test YOLOv8 on Real VisDrone/UAV Images (Compatibility CLI)")
    parser.add_argument("--dataset", type=str, default=None, help="Directory containing VisDrone/UAV images")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to test")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="results/real_data_test", help="Output directory")
    args = parser.parse_args()

    run_real_data_inference(
        dataset_dir=args.dataset,
        weights_path=args.weights,
        limit=args.limit,
        conf_threshold=args.conf,
        output_dir=args.output
    )
