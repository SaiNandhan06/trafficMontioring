"""
YOLOv8 Edge Model Export Pipeline.
Exports trained PyTorch models to TensorRT (.engine) for NVIDIA Jetson Nano/Orin
and ONNX (.onnx) format for cross-platform edge acceleration.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("model_export")


def export_edge_model(
    weights_path: Path = None,
    export_format: str = "all",  # 'engine', 'onnx', or 'all'
    img_size: int = 640,
    half_precision: bool = True,
    device: str = "cpu"
):
    """Exports YOLOv8 weights to TensorRT Engine and ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run 'pip install ultralytics'.")
        return None

    if weights_path is None:
        weights_path = settings.WEIGHTS_DIR / "yolov8_uav_best.pt"
        if not weights_path.exists():
            weights_path = Path("yolov8n.pt")

    logger.info(f"Loading weights from {weights_path} for edge export...")
    model = YOLO(str(weights_path))

    exported_files = []

    # 1. Export to ONNX
    if export_format in ["onnx", "all"]:
        use_half = half_precision if device != "cpu" else False
        logger.info(f"Exporting to ONNX (imgsz={img_size}, half={use_half}, device={device})...")
        try:
            onnx_path = model.export(
                format="onnx",
                imgsz=img_size,
                half=use_half,
                dynamic=False,
                simplify=True,
                device=device
            )
            logger.info(f"ONNX export successful: {onnx_path}")
            exported_files.append(onnx_path)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    # 2. Export to TensorRT Engine (requires NVIDIA GPU & TensorRT)
    if export_format in ["engine", "all"]:
        logger.info(f"Exporting to TensorRT Engine (imgsz={img_size}, half={half_precision})...")
        try:
            engine_path = model.export(
                format="engine",
                imgsz=img_size,
                half=half_precision,
                device=device if device != "cpu" else "0",
                workspace=4  # 4GB workspace
            )
            logger.info(f"TensorRT export successful: {engine_path}")
            exported_files.append(engine_path)
        except Exception as e:
            logger.warning(f"TensorRT export note: {e} (Expected if CUDA/TensorRT environment is not present on host)")

    return exported_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model for Edge deployment")
    parser.add_argument("--weights", type=Path, default=None, help="Path to .pt weights")
    parser.add_argument("--format", type=str, default="all", choices=["onnx", "engine", "all"], help="Export format")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 half precision")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export")
    args = parser.parse_args()

    export_edge_model(
        weights_path=args.weights,
        export_format=args.format,
        img_size=args.img_size,
        half_precision=args.half,
        device=args.device
    )
