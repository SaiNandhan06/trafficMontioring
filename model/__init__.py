"""
SkyGuard UAV Model Package.
YOLOv8 aerial model training, ONNX export, and evaluation utilities.
"""

from model.export import export_edge_model
from model.train import train_uav_yolo

# Aliases
export_model = export_edge_model
train_model = train_uav_yolo

__all__ = [
    "export_edge_model",
    "export_model",
    "train_uav_yolo",
    "train_model",
]
