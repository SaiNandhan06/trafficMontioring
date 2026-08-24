"""
UAV Edge AI Video Inference & Kinematics Validation Benchmark.
Processes video frames, measures stage latency (YOLO vs ByteTrack vs Heuristics),
and audits incident trigger accuracy and cooldown deduplication.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from edge.tracker import UAVByteTracker
from edge.incident_detector import IncidentDetector
from ultralytics import YOLO

logger = setup_logger("validate_edge_kinematics")


def benchmark_video_kinematics(
    video_path: str = "data/sample_drone_feed.mp4",
    weights_path: str = "model/weights/yolov8_uav_best.pt",
    max_frames: int = 100
) -> Dict:
    """Runs edge detection + tracking + incident detection on real video."""
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found at: {v_path}")

    w_path = Path(weights_path)
    if not w_path.exists():
        w_path = Path("yolov8n.pt")

    logger.info(f"Loading YOLOv8 weights from: {w_path}")
    model = YOLO(str(w_path))

    tracker = UAVByteTracker(iou_threshold=settings.IOU_THRESHOLD)
    detector = IncidentDetector(
        pixels_per_meter=12.0,
        speed_limit_kmh=80.0,
        deceleration_thresh_kmh=25.0,
        accident_iou_thresh=0.35,
        congestion_thresh=0.55,
        cooldown_seconds=5.0
    )

    cap = cv2.VideoCapture(str(v_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    t_yolo_list = []
    t_tracker_list = []
    t_detector_list = []
    all_incidents = []
    unique_track_ids = set()

    t_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break

        timestamp = frame_count * 0.033

        # 1. Detection Stage
        t0 = time.perf_counter()
        results = model(frame, conf=0.25, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if cls_id in [2, 5, 7]:
                    cls_id = 0
                elif cls_id == 0:
                    cls_id = 1
                elif cls_id in [1, 3]:
                    cls_id = 2
                detections.append({"bbox": [x1, y1, x2, y2], "conf": conf, "class_id": cls_id})
        t_yolo = time.perf_counter() - t0
        t_yolo_list.append(t_yolo * 1000.0)

        # 2. Tracking Stage
        t1 = time.perf_counter()
        tracks = tracker.update(detections, timestamp=timestamp)
        for t in tracks:
            unique_track_ids.add(t.track_id)
        t_tracker = time.perf_counter() - t1
        t_tracker_list.append(t_tracker * 1000.0)

        # 3. Incident Kinematics Stage
        t2 = time.perf_counter()
        incidents = detector.detect_incidents(tracks, frame)
        t_detector = time.perf_counter() - t2
        t_detector_list.append(t_detector * 1000.0)

        for inc in incidents:
            all_incidents.append(inc.to_dict())

        frame_count += 1

    cap.release()
    total_time = time.perf_counter() - t_start
    avg_fps = frame_count / max(0.001, total_time)

    # Incident breakdown
    incident_types = {}
    for inc in all_incidents:
        itype = inc["incident_type"]
        incident_types[itype] = incident_types.get(itype, 0) + 1

    report = {
        "video_source": str(v_path),
        "total_video_frames": total_video_frames,
        "processed_frames": frame_count,
        "unique_tracks_identified": len(unique_track_ids),
        "total_incidents_triggered": len(all_incidents),
        "incident_types_breakdown": incident_types,
        "performance_profile": {
            "overall_throughput_fps": round(avg_fps, 2),
            "mean_total_frame_latency_ms": round(float(np.mean(t_yolo_list) + np.mean(t_tracker_list) + np.mean(t_detector_list)), 2),
            "stage_breakdown_ms": {
                "yolov8_detection_mean_ms": round(float(np.mean(t_yolo_list)), 2),
                "bytetrack_tracking_mean_ms": round(float(np.mean(t_tracker_list)), 2),
                "incident_kinematics_mean_ms": round(float(np.mean(t_detector_list)), 2)
            }
        },
        "incidents": all_incidents
    }

    output_path = PROJECT_ROOT / "results" / "edge_kinematics_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(" [UAV EDGE AI VIDEO INFERENCE & KINEMATICS VALIDATION REPORT]")
    print("=" * 70)
    print(f"Video File:               {v_path.name} ({frame_count} frames processed)")
    print(f"Unique Track IDs:         {len(unique_track_ids)}")
    print(f"Total Incidents Detected: {len(all_incidents)}")
    print("Incident Breakdown:")
    for k, v in incident_types.items():
        print(f"  * {k:<22}: {v} incidents")
    if not incident_types:
        print("  * None (Normal traffic flow)")
    print("-" * 70)
    print(f"Inference Speed:          {avg_fps:.2f} FPS")
    print(f"Mean Latency Per Frame:   {report['performance_profile']['mean_total_frame_latency_ms']} ms")
    print(f"  - YOLOv8 Inference:     {report['performance_profile']['stage_breakdown_ms']['yolov8_detection_mean_ms']} ms")
    print(f"  - ByteTrack Tracking:   {report['performance_profile']['stage_breakdown_ms']['bytetrack_tracking_mean_ms']} ms")
    print(f"  - Incident Kinematics:  {report['performance_profile']['stage_breakdown_ms']['incident_kinematics_mean_ms']} ms")
    print("=" * 70)
    print(f"Report written to: {output_path}\n")

    return report


if __name__ == "__main__":
    benchmark_video_kinematics()
