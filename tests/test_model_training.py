"""
Model Training & KaggleHub Registry Regression Tests.
Verifies dataset registry configuration, alias resolution,
model checkpoint integrity, and evaluation metrics provenance.
"""

import json
from pathlib import Path
import pytest

from src.data_pipeline.kaggle_download import DATASET_REGISTRY, DATASET_ALIASES
from config.settings import BASE_DIR, settings


def test_kaggle_dataset_registry_contains_all_three():
    """Verifies that the Kaggle dataset registry contains authoritative handles for VisDrone, UAVDT, and UA-DETRAC."""
    assert "visdrone" in DATASET_REGISTRY
    assert "uavdt" in DATASET_REGISTRY
    assert "ua_detrac" in DATASET_REGISTRY

    assert DATASET_REGISTRY["visdrone"]["handle"] == "kushagrapandya/visdrone-dataset"
    assert DATASET_REGISTRY["uavdt"]["handle"] == "shakaibkaggle/uavdt-dataset"
    assert DATASET_REGISTRY["ua_detrac"]["handle"] == "bratjay/ua-detrac-orig"


def test_dataset_aliases_resolve_correctly():
    """Verifies alias mappings resolve cleanly to canonical registry keys."""
    assert DATASET_ALIASES["visdrone"] == "visdrone"
    assert DATASET_ALIASES["kushagrapandya/visdrone-dataset"] == "visdrone"
    assert DATASET_ALIASES["uavdt"] == "uavdt"
    assert DATASET_ALIASES["shakaibkaggle/uavdt-dataset"] == "uavdt"
    assert DATASET_ALIASES["ua_detrac"] == "ua_detrac"
    assert DATASET_ALIASES["ua-detrac"] == "ua_detrac"
    assert DATASET_ALIASES["detrac"] == "ua_detrac"
    assert DATASET_ALIASES["bratjay/ua-detrac-orig"] == "ua_detrac"


def test_trained_model_checkpoint_exists_and_loads():
    """Verifies that the trained Phase 4 best model checkpoint exists and loads in Ultralytics YOLO."""
    weights_path = settings.WEIGHTS_DIR / "yolov8_uav_best.pt"
    assert weights_path.exists(), f"Weights file {weights_path} must exist after Phase 4 training"

    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    assert model is not None
    assert hasattr(model, "predict")


def test_evaluation_report_reflects_trained_model():
    """Verifies that the evaluation report reflects genuine trained model performance without fallbacks."""
    report_file = BASE_DIR / "results" / "model_evaluation_report.json"
    assert report_file.exists(), "Evaluation report must exist"

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "REAL_EVALUATION"
    assert data["source_type"] == "actual_test_split_evaluation"
    assert "overall" in data
    assert "map50" in data["overall"]

    # mAP@50 should reflect meaningful training (> 0.10)
    assert data["overall"]["map50"] > 0.10, "Trained model mAP@50 should be significantly greater than 1-epoch baseline"


def test_training_summary_recorded():
    """Verifies that the training summary JSON is generated with full provenance."""
    summary_file = BASE_DIR / "results" / "training_summary.json"
    assert summary_file.exists(), "Training summary JSON must exist"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "epochs_trained" in summary
    assert summary["epochs_trained"] >= 1
    assert "device" in summary
    assert "best_checkpoint" in summary
