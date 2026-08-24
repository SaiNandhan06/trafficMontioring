"""
Model Evaluation Truthfulness & Metric Integrity Regression Tests.
Guarantees that no fake or hardcoded benchmark metrics (0.912 mAP, 0.942 precision)
are returned or displayed as genuine model performance.
"""

import json
from pathlib import Path
import pytest
from scripts.evaluate_model import evaluate_model
from config.settings import BASE_DIR


def test_no_hardcoded_fake_metrics_in_evaluator():
    """Verifies evaluate_model source code does not contain hardcoded 0.912 or 0.942 fallback metrics."""
    eval_script = BASE_DIR / "scripts" / "evaluate_model.py"
    assert eval_script.exists(), "evaluate_model.py must exist"

    content = eval_script.read_text(encoding="utf-8")

    # The notorious fake values must not appear anywhere in evaluate_model.py
    assert "0.912" not in content, "Hardcoded fake mAP 0.912 must not exist in evaluate_model.py"
    assert "0.942" not in content, "Hardcoded fake precision 0.942 must not exist in evaluate_model.py"
    assert "0.956" not in content, "Hardcoded fake mAP 0.956 must not exist in evaluate_model.py"


def test_no_hardcoded_fake_metrics_in_dashboard():
    """Verifies dashboard/app.py does not contain hardcoded metric dataframes claiming 0.912 or 0.942."""
    app_file = BASE_DIR / "dashboard" / "app.py"
    assert app_file.exists(), "dashboard/app.py must exist"

    content = app_file.read_text(encoding="utf-8")

    # Ensure fake metric array is absent
    assert "0.912" not in content, "Hardcoded fake metric 0.912 must not exist in dashboard/app.py"
    assert "0.942" not in content, "Hardcoded fake metric 0.942 must not exist in dashboard/app.py"
    assert "0.956" not in content, "Hardcoded fake metric 0.956 must not exist in dashboard/app.py"


def test_evaluation_missing_weights_fails_cleanly():
    """Verifies that missing weights file produces error status instead of invented metrics."""
    non_existent_weights = BASE_DIR / "model" / "weights" / "non_existent_model_12345.pt"
    dummy_yaml = BASE_DIR / "data" / "dataset.yaml"
    temp_json = BASE_DIR / "results" / "temp_test_eval_fail.json"

    res = evaluate_model(
        weights_path=non_existent_weights,
        data_yaml=dummy_yaml,
        output_json=temp_json
    )

    assert res["status"] in ["NOT_EVALUATED", "EVALUATION_FAILED"]
    assert "error" in res
    assert "overall" not in res or res.get("overall") is None

    # Clean up temp file
    if temp_json.exists():
        temp_json.unlink()


def test_verified_evaluation_report_schema():
    """Verifies that results/model_evaluation_report.json contains complete provenance metadata."""
    report_file = BASE_DIR / "results" / "model_evaluation_report.json"
    assert report_file.exists(), "results/model_evaluation_report.json must exist"

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Provenance fields
    assert report["status"] == "REAL_EVALUATION"
    assert report["source_type"] == "actual_test_split_evaluation"
    assert "evaluation_timestamp" in report
    assert "weights_file" in report
    assert "dataset_yaml" in report
    assert report["dataset_split"] == "test"
    assert "device" in report

    # Metric structure
    overall = report["overall"]
    for k in ["precision", "recall", "map50", "map50_95"]:
        assert k in overall, f"Missing overall metric: {k}"
        assert isinstance(overall[k], (int, float)), f"Metric {k} must be numeric"

    # Per-class structure
    per_class = report["per_class"]
    for cname in ["vehicle", "pedestrian", "cyclist", "traffic_signal"]:
        assert cname in per_class, f"Missing class metric: {cname}"
        assert "precision" in per_class[cname]
        assert "recall" in per_class[cname]
        assert "map50" in per_class[cname]
