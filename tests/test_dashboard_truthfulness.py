"""
Streamlit Dashboard Truthfulness and Source Attribution Regression Tests.
Verifies that all metrics displayed across the dashboard are backed by verifiable
provenance reports and honest status tags (MEASURED, SIMULATED, MOCK, BLOCKED, N/A).
"""

import json
from pathlib import Path
import pytest

from dashboard.ui_components import (
    get_all_provenance_reports,
    get_source_badge_html,
    get_source_attribution_matrix
)
from config.settings import BASE_DIR


def test_ui_components_load_provenance_reports():
    """Verifies that all 6 provenance reports are loaded or cleanly handled."""
    reports = get_all_provenance_reports()
    assert "model_eval" in reports
    assert "training_summary" in reports
    assert "edge_benchmark" in reports
    assert "blockchain_validation" in reports
    assert "ipfs_validation" in reports
    assert "notification_validation" in reports


def test_ui_components_badge_generation():
    """Verifies that source badge HTML helper formats tags properly."""
    badge_measured = get_source_badge_html("MEASURED")
    assert "MEASURED" in badge_measured
    assert "#00dc82" in badge_measured

    badge_simulated = get_source_badge_html("SIMULATED")
    assert "SIMULATED" in badge_simulated
    assert "#facc15" in badge_simulated

    badge_mock = get_source_badge_html("MOCK")
    assert "MOCK" in badge_mock
    assert "#38bdf8" in badge_mock

    badge_blocked = get_source_badge_html("BLOCKED")
    assert "BLOCKED" in badge_blocked
    assert "#f87171" in badge_blocked


def test_source_attribution_matrix_structure():
    """Verifies that the attribution matrix has all mandatory provenance columns."""
    matrix = get_source_attribution_matrix()
    assert len(matrix) >= 8

    required_keys = {"Metric / Component", "Current Value", "Source", "Status", "Environment"}
    for item in matrix:
        assert required_keys.issubset(item.keys())
        assert item["Status"] in ["MEASURED", "SIMULATED", "MOCK", "ESTIMATED", "BLOCKED", "NOT MEASURED", "N/A"]


def test_source_attribution_matrix_honesty():
    """Verifies that hardware/backend constraints are honestly tagged."""
    matrix = get_source_attribution_matrix()
    matrix_dict = {item["Metric / Component"]: item for item in matrix}

    # TensorRT and Jetson must not be claimed as measured
    assert "TensorRT Acceleration" in matrix_dict
    assert matrix_dict["TensorRT Acceleration"]["Status"] in ["BLOCKED", "N/A"]

    assert "Jetson Nano Physical FPS" in matrix_dict
    assert matrix_dict["Jetson Nano Physical FPS"]["Status"] in ["NOT MEASURED", "N/A"]

    # Blockchain must be tagged SIMULATED
    assert "Blockchain Transaction Throughput" in matrix_dict
    assert matrix_dict["Blockchain Transaction Throughput"]["Status"] == "SIMULATED"

    # IPFS must be tagged MOCK
    assert "IPFS Evidence Pinning" in matrix_dict
    assert matrix_dict["IPFS Evidence Pinning"]["Status"] == "MOCK"


def test_dashboard_source_code_has_no_hardcoded_fake_metrics():
    """Verifies that dashboard/app.py does not contain hardcoded fake precision/mAP values."""
    app_file = BASE_DIR / "dashboard" / "app.py"
    assert app_file.exists()

    content = app_file.read_text(encoding="utf-8")
    assert "0.912" not in content, "Found stale hardcoded precision metric"
    assert "0.942" not in content, "Found stale hardcoded mAP metric"
    assert "30-45 FPS" not in content, "Found unverified Jetson claim"
