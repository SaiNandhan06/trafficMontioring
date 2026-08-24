"""
YOLOv8 Model Training Pipeline for UAV Aerial Traffic Monitoring.
Trains YOLOv8 nano/small models on the unified VisDrone/UAV dataset with drone-optimized hyperparameters.
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("model_train")


def train_uav_yolo(
    data_yaml: Path = None,
    base_weights: str = "yolov8n.pt",
    model_size: str = "n",
    epochs: int = 10,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "cpu",
    run_name: str = "yolov8n_uav_traffic_phase4",
    workers: int = 2,
    seed: int = 42,
    patience: int = 15,
    use_wandb: bool = False
) -> Optional[Dict]:
    """Trains a YOLOv8 model with UAV-specific aerial hyperparameters."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run 'pip install ultralytics'.")
        return None

    if data_yaml is None:
        data_yaml = settings.DATA_DIR / "dataset.yaml"

    if not data_yaml.exists():
        logger.error(f"Dataset config not found at {data_yaml}. Run data preparation first.")
        return None

    # Preserve 1-epoch baseline weights as a reference benchmark if present
    current_best_weights = settings.WEIGHTS_DIR / "yolov8_uav_best.pt"
    baseline_backup = settings.WEIGHTS_DIR / "yolov8_uav_baseline_1epoch.pt"
    if current_best_weights.exists() and not baseline_backup.exists():
        settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_best_weights, baseline_backup)
        logger.info(f"Backed up 1-epoch baseline weights to: {baseline_backup}")

    logger.info(f"Initializing YOLO model with base weights: '{base_weights}'...")
    model = YOLO(base_weights)

    project_dir = settings.MODEL_DIR / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Aerial UAV drone-optimized training hyperparameters
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "workers": workers,
        "seed": seed,
        "patience": patience,
        # Aerial / Drone Augmentations
        "mosaic": 1.0,        # High mosaic for small objects in complex backgrounds
        "copy_paste": 0.3,     # Copy paste for dense vehicle clusters
        "flipud": 0.5,         # Drone views are top-down (vertically invariant)
        "fliplr": 0.5,         # Horizontal flip
        "degrees": 10.0,       # Slight rotation variation
        "scale": 0.5,          # Scale variation for altitude variance
        "cos_lr": True,        # Cosine learning rate schedule
        "save": True,
        "plots": True,
        "verbose": True
    }

    t0 = time.time()
    logger.info(f"Starting UAV YOLOv8 training: epochs={epochs}, batch={batch_size}, imgsz={img_size}, device={device}")
    results = model.train(**train_args)
    duration_s = round(time.time() - t0, 2)

    # Locate generated best weights
    best_pt = project_dir / run_name / "weights" / "best.pt"
    last_pt = project_dir / run_name / "weights" / "last.pt"

    if best_pt.exists():
        settings.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        target_pt = settings.WEIGHTS_DIR / "yolov8_uav_best.pt"
        shutil.copy2(best_pt, target_pt)
        logger.info(f"Updated active best model weights at: {target_pt}")

    # Extract metrics from training results
    training_summary = {
        "run_name": run_name,
        "base_weights": base_weights,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "image_size": img_size,
        "device": device,
        "training_duration_seconds": duration_s,
        "completed_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_checkpoint": str(best_pt) if best_pt.exists() else None,
        "last_checkpoint": str(last_pt) if last_pt.exists() else None,
        "results_csv": str(project_dir / run_name / "results.csv") if (project_dir / run_name / "results.csv").exists() else None
    }

    # Save summary report
    summary_file = PROJECT_ROOT / "results" / "training_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    logger.info(f"Training completed in {duration_s}s. Summary saved to: {summary_file}")
    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on UAV Traffic Dataset")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial pretrained weights")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l"], help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, 1)")
    parser.add_argument("--name", type=str, default="yolov8n_uav_traffic_phase4", help="Experiment run name")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    train_uav_yolo(
        base_weights=args.weights,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        run_name=args.name,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience
    )
