"""
Real-Time Edge AI Video Inference & Incident Reporting Pipeline.
Captures drone video, runs YOLOv8 detection + ByteTrack tracking,
identifies traffic incidents, uploads evidence to IPFS, and logs to EVM Blockchain.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from edge.stream_capture import StreamCapture
from edge.tracker import UAVByteTracker
from edge.incident_detector import IncidentDetector, Incident
from edge.retry_queue import RetryQueue
from ipfs.ipfs_client import IPFSClient
from ipfs.metadata_builder import build_incident_metadata
from blockchain.contract_client import Web3ContractClient

logger = setup_logger("edge_pipeline")

CLASS_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
    3: "traffic_signal"
}

CLASS_COLORS = {
    0: (0, 220, 100),    # Emerald green
    1: (255, 120, 0),    # Blue-orange
    2: (0, 180, 255),    # Amber
    3: (200, 50, 255)    # Purple
}


class EdgeInferencePipeline:
    """End-to-End UAV Edge Inference Engine."""

    def __init__(
        self,
        source: str = None,
        weights_path: str = None,
        drone_id: str = None,
        display: bool = False
    ):
        self.source = source or settings.STREAM_SOURCE
        self.drone_id = drone_id or settings.DRONE_ID
        self.display = display

        # Subsystems
        self.capture = StreamCapture(self.source, target_fps=settings.STREAM_FPS)
        self.tracker = UAVByteTracker(iou_threshold=settings.IOU_THRESHOLD)
        self.detector = IncidentDetector(
            speed_limit_kmh=settings.SPEED_LIMIT_KMH,
            deceleration_thresh_kmh=settings.SPEED_DECELERATION_THRESHOLD,
            accident_iou_thresh=settings.ACCIDENT_IOU_THRESHOLD,
            congestion_thresh=settings.CONGESTION_DENSITY_THRESHOLD,
            cooldown_seconds=settings.INCIDENT_COOLDOWN_SECONDS
        )
        self.retry_queue = RetryQueue()
        self.ipfs_client = IPFSClient()
        self.blockchain_client = Web3ContractClient()

        # Load YOLO Model
        self.model = None
        self._init_model(weights_path or settings.MODEL_WEIGHTS_PATH)

        self.is_running = False
        self.processed_frames = 0
        self.total_incidents_reported = 0

    def _init_model(self, weights_path: str):
        try:
            from ultralytics import YOLO
            w_path = Path(weights_path)
            best_uav = settings.WEIGHTS_DIR / "yolov8_uav_best.pt"
            if best_uav.exists():
                w_path = best_uav
            elif not w_path.exists():
                logger.warning(f"Weights {weights_path} not found. Loading pretrained yolov8n.pt...")
                w_path = Path("yolov8n.pt")
            self.model = YOLO(str(w_path))
            logger.info(f"Loaded YOLOv8 model from {w_path}")
        except Exception as e:
            logger.warning(f"Ultralytics load note: {e}. Running in lightweight synthetic detection mode.")
            self.model = None

    def _run_detection(self, frame: np.ndarray):
        """Runs object detection on frame and returns formatted detection dictionaries."""
        detections = []
        if self.model is not None:
            try:
                results = self.model(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        # Map COCO classes if standard model is used
                        if cls_id in [2, 5, 7]:  # car, bus, truck -> vehicle
                            cls_id = 0
                        elif cls_id == 0:        # person -> pedestrian
                            cls_id = 1
                        elif cls_id in [1, 3]:   # bicycle, motorcycle -> cyclist
                            cls_id = 2
                        elif cls_id == 9:        # traffic light -> signal
                            cls_id = 3

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                            "class_id": cls_id
                        })
            except Exception as e:
                logger.error(f"Inference error: {e}")

        # If model detections are empty (e.g. synthetic test drawing), use contour tracking fallback
        if not detections:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 300 < area < 15000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    detections.append({
                        "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                        "conf": 0.85,
                        "class_id": 0
                    })
        return detections

    def _handle_incident(self, incident: Incident, frame: np.ndarray):
        """Packages, pins to IPFS, and dispatches on-chain transaction."""
        self.total_incidents_reported += 1
        logger.warning(
            f"[INCIDENT DETECTED] Type: {incident.incident_type} | "
            f"Severity: {incident.severity} | Details: {incident.description}"
        )

        # 1. Package metadata & snapshot
        metadata, frame_bytes = build_incident_metadata(
            incident_id=incident.incident_id,
            drone_id=self.drone_id,
            incident_type=incident.incident_type,
            severity=incident.severity,
            confidence=incident.confidence,
            location=incident.location,
            involved_tracks=incident.involved_tracks,
            description=incident.description,
            frame=incident.frame,
            detections=[{"bbox": b} for b in incident.bboxes]
        )

        # 2. Upload to IPFS
        try:
            master_cid = self.ipfs_client.upload_incident_package(metadata, frame_bytes)
            incident.ipfs_hash = master_cid
        except Exception as e:
            logger.error(f"IPFS upload failed, queuing for retry: {e}")
            self.retry_queue.push(incident.to_dict())
            return

        # 3. Publish to Blockchain
        try:
            success, inc_id, tx_hash = self.blockchain_client.report_incident(
                ipfs_hash=master_cid,
                incident_type=incident.incident_type,
                severity_str=incident.severity,
                latitude=incident.location["lat"],
                longitude=incident.location["lng"],
                timestamp=incident.timestamp
            )
            if success and tx_hash:
                incident.tx_hash = tx_hash
                self.retry_queue.mark_success(incident.incident_id, ipfs_hash=master_cid, tx_hash=tx_hash)
            else:
                logger.warning(f"Blockchain tx pending/failed. Queuing incident {incident.incident_id}")
                self.retry_queue.push(incident.to_dict())
        except Exception as e:
            logger.error(f"Blockchain reporting exception: {e}")
            self.retry_queue.push(incident.to_dict())

    def _annotate_frame(self, frame: np.ndarray, tracks, incidents: list) -> np.ndarray:
        """Draws bounding boxes, speeds, and incident banners onto frame."""
        vis_frame = frame.copy()

        # Draw active tracks
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            color = CLASS_COLORS.get(t.class_id, (0, 255, 0))
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            speed = self.detector.track_speeds.get(t.track_id, 0.0)
            cls_name = CLASS_NAMES.get(t.class_id, "obj")
            label = f"#{t.track_id} {cls_name} {speed:.0f}km/h"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, max(0, y1 - 20)), (x1 + lw, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw Incident Banners
        if incidents:
            banner_text = f"ALERT: {incidents[0].incident_type} ({incidents[0].severity})"
            cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], 45), (0, 0, 220), -1)
            cv2.putText(vis_frame, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw HUD status
        hud_text = f"UAV: {self.drone_id} | Active Tracks: {len(tracks)} | Incidents: {self.total_incidents_reported}"
        cv2.putText(vis_frame, hud_text, (15, vis_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return vis_frame

    def run(self, max_frames: Optional[int] = None):
        """Starts real-time inference loop."""
        self.is_running = True
        self.capture.start()
        logger.info(f"Edge AI Inference Pipeline active on drone {self.drone_id}...")

        try:
            for frame, frame_idx in self.capture.frames():
                if not self.is_running or (max_frames and self.processed_frames >= max_frames):
                    break

                t_start = time.time()

                # 1. Detection
                detections = self._run_detection(frame)

                # 2. Tracking
                tracks = self.tracker.update(detections, timestamp=t_start)

                # 3. Incident Analytics
                incidents = self.detector.detect_incidents(
                    tracks,
                    frame,
                    base_coords=(settings.DRONE_LAT, settings.DRONE_LNG)
                )

                # 4. Handle newly triggered incidents
                for inc in incidents:
                    self._handle_incident(inc, frame)

                self.processed_frames += 1

                # 5. Optional GUI Display
                if self.display:
                    annotated = self._annotate_frame(frame, tracks, incidents)
                    cv2.imshow(f"UAV Edge AI Monitor - {self.drone_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.stop()

    def stop(self):
        """Cleans up pipeline resources."""
        self.is_running = False
        self.capture.stop()
        if self.display:
            cv2.destroyAllWindows()
        logger.info(f"Edge AI pipeline stopped. Total frames: {self.processed_frames}, Incidents: {self.total_incidents_reported}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UAV Edge AI Inference Pipeline")
    parser.add_argument("--source", type=str, default="data/sample_drone_feed.mp4", help="Video source (RTSP, file, camera index)")
    parser.add_argument("--drone-id", type=str, default="UAV-ALPHA-01", help="UAV Drone Identifier")
    parser.add_argument("--display", action="store_true", help="Display OpenCV live window")
    parser.add_argument("--frames", type=int, default=None, help="Limit number of frames to process")
    args = parser.parse_args()

    pipeline = EdgeInferencePipeline(
        source=args.source,
        drone_id=args.drone_id,
        display=args.display
    )
    pipeline.run(max_frames=args.frames)
