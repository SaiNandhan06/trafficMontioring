"""
Master Pipeline Runner for SkyGuard UAV.
Entrypoint for automated KaggleHub dataset ingestion and real UAV inference profiling.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_full_pipeline import run_full_pipeline

# Canonical alias
run_pipeline = run_full_pipeline

__all__ = ["run_pipeline", "run_full_pipeline"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyGuard Master UAV Pipeline Runner")
    parser.add_argument("--dataset", type=str, default="kushagrapandya/visdrone-dataset",
                        help="Kaggle dataset handle or alias (visdrone, uavdt, ua_detrac)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to fine-tuned YOLOv8 weights (.pt)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of images to test")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold for predictions")
    args = parser.parse_args()

    run_pipeline(
        dataset_handle=args.dataset,
        weights_path=args.weights,
        limit=args.limit,
        conf_threshold=args.conf
    )
