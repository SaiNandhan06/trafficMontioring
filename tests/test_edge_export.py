"""
Edge AI Model Export & ONNX Runtime Regression Tests.
Verifies ONNX model graph validity, ONNX Runtime execution provider compatibility,
hardware environment reporting, and device-aware fallback behavior.
"""

import json
from pathlib import Path
import pytest
import onnx
import onnxruntime as ort

from config.settings import BASE_DIR, settings
from model.export import export_edge_model
from edge.edge_pipeline import EdgeInferencePipeline


def test_onnx_model_file_exists_and_valid():
    """Verifies that the exported ONNX model exists and passes ONNX graph validation."""
    onnx_file = settings.WEIGHTS_DIR / "yolov8_uav_best.onnx"
    assert onnx_file.exists(), f"ONNX file must exist at {onnx_file}"

    model_proto = onnx.load(str(onnx_file))
    onnx.checker.check_model(model_proto)

    graph = model_proto.graph
    assert len(graph.input) == 1
    assert graph.input[0].name == "images"
    assert len(graph.output) == 1
    assert graph.output[0].name == "output0"


def test_onnx_runtime_session_execution():
    """Verifies that ONNX Runtime can load the model and execute with CPUExecutionProvider."""
    onnx_file = settings.WEIGHTS_DIR / "yolov8_uav_best.onnx"
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    assert input_meta.name == "images"
    assert output_meta.name == "output0"
    assert "CPUExecutionProvider" in session.get_providers()


def test_edge_benchmark_report_provenance():
    """Verifies that edge_benchmark.json explicitly distinguishes measured CPU/ONNX from unmeasured Jetson/TensorRT."""
    bench_file = BASE_DIR / "results" / "edge_benchmark.json"
    assert bench_file.exists(), "results/edge_benchmark.json must exist"

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "runtimes" in data
    runtimes = data["runtimes"]

    # PyTorch and ONNX Runtime must be measured on host CPU
    assert runtimes["pytorch"]["measured_or_estimated"] == "measured"
    assert runtimes["onnx_runtime"]["measured_or_estimated"] == "measured"
    assert runtimes["pytorch"]["fps"] > 0
    assert runtimes["onnx_runtime"]["fps"] > 0

    # TensorRT and Jetson Nano must be marked as not measured (no fake metrics)
    assert runtimes["tensorrt"]["measured_or_estimated"] == "not_measured"
    assert runtimes["jetson_nano"]["measured_or_estimated"] == "not_measured"


def test_edge_pipeline_loads_onnx_weights():
    """Verifies that EdgeInferencePipeline can initialize with ONNX weights cleanly."""
    onnx_file = settings.WEIGHTS_DIR / "yolov8_uav_best.onnx"
    pipeline = EdgeInferencePipeline(weights_path=str(onnx_file))
    assert pipeline.model is not None, "Pipeline must load ONNX model via Ultralytics"
