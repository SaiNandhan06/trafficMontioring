# Model Weights Directory

This directory holds the pre-trained and fine-tuned YOLOv8 weights for UAV traffic monitoring:

- `yolov8n.pt`: Nano model for ultra-low-power edge nodes (Jetson Nano / Raspberry Pi 4).
- `yolov8s.pt`: Small model with higher accuracy for higher-tier edge nodes (Jetson Orin / Xavier).
- `yolov8_uav_best.pt`: Best fine-tuned model weights on unified VisDrone/UAVDT/UA-DETRAC dataset.
- `yolov8_uav.onnx`: Exported ONNX fallback model.
- `yolov8_uav.engine`: Exported FP16 TensorRT engine for NVIDIA Jetson deployment.

Default YOLOv8 weights are automatically downloaded on first inference run by the Ultralytics framework.
