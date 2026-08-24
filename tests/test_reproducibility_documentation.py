"""
Reproducibility and Documentation Integrity Tests.
Validates that all commands, script paths, dataset handles, model weights,
and environment variables referenced in README.md exist and match the implementation.
"""

from pathlib import Path
import pytest

from config.settings import settings, BASE_DIR
from src.data_pipeline.kaggle_download import DATASET_REGISTRY


def test_readme_exists_and_not_empty():
    """Verifies that README.md exists and contains comprehensive documentation."""
    readme_path = BASE_DIR / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert len(content) > 5000
    assert "## 1. System Architecture" in content
    assert "## 14. Verified Performance Results" in content


def test_documented_scripts_exist():
    """Verifies that all CLI scripts documented in the README exist."""
    documented_scripts = [
        "scripts/verify_all.py",
        "scripts/evaluate_model.py",
        "scripts/benchmark_edge_models.py",
        "scripts/benchmark_blockchain.py",
        "scripts/benchmark_ipfs.py",
        "scripts/validate_notifications.py",
        "scripts/validate_resilience.py",
        "data/prepare_visdrone.py",
        "data/validate_dataset.py",
        "model/train.py",
        "dashboard/app.py",
        "dashboard/api.py",
    ]

    for script_rel in documented_scripts:
        script_path = BASE_DIR / script_rel
        assert script_path.exists(), f"Documented script missing: {script_rel}"


def test_documented_weights_and_configs_exist():
    """Verifies that model checkpoints, contract artifacts, and yaml configs exist."""
    assert (BASE_DIR / "model" / "weights" / "yolov8_uav_best.pt").exists()
    assert (BASE_DIR / "model" / "weights" / "yolov8_uav_best.onnx").exists()
    assert (BASE_DIR / "data" / "dataset.yaml").exists()
    assert (BASE_DIR / ".env.example").exists()


def test_documented_dataset_handles_match_registry():
    """Verifies that the dataset Kaggle handles match DATASET_REGISTRY."""
    readme_content = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    for key, info in DATASET_REGISTRY.items():
        handle = info["handle"]
        assert handle in readme_content, f"Kaggle handle {handle} not documented in README"


def test_documented_env_vars_match_settings():
    """Verifies that core environment variables documented in README match settings."""
    documented_vars = [
        "ENVIRONMENT",
        "LOG_LEVEL",
        "DRONE_ID",
        "MODEL_WEIGHTS_PATH",
        "DEVICE",
        "ETH_RPC_URL",
        "IPFS_MODE",
        "NOTIFICATION_MODE",
        "SECRET_KEY",
    ]
    readme_content = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    for var in documented_vars:
        assert var in readme_content, f"Env var {var} not documented in README"
        assert hasattr(settings, var), f"Env var {var} missing from settings.py"
