"""
Incident Detection Engine for UAV Aerial Traffic Monitoring.
Detects speeding, sudden deceleration, lane intrusion, collisions/accidents,
and roadway congestion with severity classification and metadata packaging.
"""

import time
import math
import uuid
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from edge.tracker import Track, calculate_iou

logger = setup_logger("incident_detector")


class Incident:
    """Represents a detected traffic incident."""

    def __init__(
        self,
        incident_type: str,
        severity: str,
        confidence: float,
        location: Dict[str, float],
        involved_tracks: List[int],
        description: str,
        frame: np.ndarray,
        bboxes: List[List[float]]
    ):
        self.incident_id = f"INC-{int(time.time())}-{str(uuid.uuid4())[:6]}"
        self.timestamp = time.time()
        self.incident_type = incident_type  # SPEEDING, COLLISION, LANE_VIOLATION, CONGESTION
        self.severity = severity            # LOW, MEDIUM, HIGH, CRITICAL
        self.confidence = confidence
        self.location = location            # {"lat": float, "lng": float}
        self.involved_tracks = involved_tracks
        self.description = description
        self.frame = frame.copy()
        self.bboxes = bboxes
        self.ipfs_hash: Optional[str] = None
        self.tx_hash: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "location": self.location,
            "involved_tracks": self.involved_tracks,
            "description": self.description,
            "ipfs_hash": self.ipfs_hash,
            "tx_hash": self.tx_hash,
            "bboxes": self.bboxes
        }


class IncidentDetector:
    """Analyzes vehicle tracks and frames to detect anomalies and dangerous incidents."""

    def __init__(
        self,
        pixels_per_meter: float = 12.0,
        speed_limit_kmh: float = 80.0,
        deceleration_thresh_kmh: float = 25.0,
        accident_iou_thresh: float = 0.35,
        congestion_thresh: float = 0.55,
        cooldown_seconds: float = 10.0
    ):
        self.ppm = pixels_per_meter
        self.speed_limit = speed_limit_kmh
        self.decel_thresh = deceleration_thresh_kmh
        self.accident_iou = accident_iou_thresh
        self.congestion_thresh = congestion_thresh
        self.cooldown = cooldown_seconds

        self.last_reported: Dict[str, float] = {}
        self.track_speeds: Dict[int, float] = {}

    def _is_on_cooldown(self, key: str) -> bool:
        now = time.time()
        if key in self.last_reported and (now - self.last_reported[key]) < self.cooldown:
            return True
        self.last_reported[key] = now
        return False

    def estimate_speed_kmh(self, track: Track, dt: float = 0.033) -> float:
        """Estimates vehicle ground speed in km/h from pixel displacement."""
        if len(track.history) < 3:
            return 0.0

        p1 = track.history[-3]
        p2 = track.history[-1]
        pixel_dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        time_elapsed = max(0.001, (p2[2] - p1[2]) if p2[2] > p1[2] else dt * 2)

        # pixels / (ppm * time) = meters / sec -> * 3.6 = km/h
        speed_mps = (pixel_dist / self.ppm) / time_elapsed
        speed_kmh = speed_mps * 3.6
        return speed_kmh

    def detect_incidents(
        self,
        tracks: List[Track],
        frame: np.ndarray,
        base_coords: Tuple[float, float] = (37.774929, -122.419418)
    ) -> List[Incident]:
        """Runs incident detection algorithms across all active tracks and frame context."""
        incidents: List[Incident] = []
        lat, lng = base_coords

        # 1. Speed and Deceleration Checks
        for track in tracks:
            # Only compute for vehicles / cyclists
            if track.class_id not in [0, 2]:
                continue

            if len(track.history) >= 3:
                current_speed = self.estimate_speed_kmh(track)
                prev_speed = self.track_speeds.get(track.track_id, current_speed)
                self.track_speeds[track.track_id] = current_speed
            else:
                current_speed = self.track_speeds.get(track.track_id, 0.0)
                prev_speed = current_speed
            track.velocities.append(current_speed)

            # Speeding Violation
            if current_speed > self.speed_limit:
                cd_key = f"speeding_{track.track_id}"
                if not self._is_on_cooldown(cd_key):
                    incidents.append(Incident(
                        incident_type="SPEEDING",
                        severity="MEDIUM" if current_speed < self.speed_limit + 25 else "HIGH",
                        confidence=min(0.99, track.confidence + 0.05),
                        location={"lat": lat, "lng": lng},
                        involved_tracks=[track.track_id],
                        description=f"Vehicle #{track.track_id} exceeded speed limit: {current_speed:.1f} km/h (Limit: {self.speed_limit} km/h)",
                        frame=frame,
                        bboxes=[track.bbox]
                    ))

            # Sudden Harsh Deceleration / Emergency Braking
            deceleration = prev_speed - current_speed
            if deceleration > self.decel_thresh:
                cd_key = f"decel_{track.track_id}"
                if not self._is_on_cooldown(cd_key):
                    incidents.append(Incident(
                        incident_type="SUDDEN_BRAKING",
                        severity="MEDIUM",
                        confidence=0.88,
                        location={"lat": lat, "lng": lng},
                        involved_tracks=[track.track_id],
                        description=f"Vehicle #{track.track_id} harsh braking: -{deceleration:.1f} km/h deceleration",
                        frame=frame,
                        bboxes=[track.bbox]
                    ))

        # 2. Vehicle Collision / Accident Detection (Pairwise IoU + Low Speed)
        n = len(tracks)
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tracks[i], tracks[j]
                if t1.class_id not in [0, 2] or t2.class_id not in [0, 2]:
                    continue

                iou = calculate_iou(t1.bbox, t2.bbox)
                if iou > self.accident_iou:
                    s1 = self.track_speeds.get(t1.track_id, 0.0)
                    s2 = self.track_speeds.get(t2.track_id, 0.0)

                    # Collision confirmed if bounding boxes overlap and vehicles are near stop
                    if s1 < 15.0 and s2 < 15.0:
                        cd_key = f"accident_{min(t1.track_id, t2.track_id)}_{max(t1.track_id, t2.track_id)}"
                        if not self._is_on_cooldown(cd_key):
                            incidents.append(Incident(
                                incident_type="COLLISION_ACCIDENT",
                                severity="CRITICAL",
                                confidence=0.94,
                                location={"lat": lat, "lng": lng},
                                involved_tracks=[t1.track_id, t2.track_id],
                                description=f"Collision detected between Track #{t1.track_id} and Track #{t2.track_id} (IoU: {iou:.2f})",
                                frame=frame,
                                bboxes=[t1.bbox, t2.bbox]
                            ))

        # 3. Roadway Traffic Congestion Density Scoring
        h, w = frame.shape[:2]
        total_frame_area = float(w * h)
        occupied_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in [t.bbox for t in tracks if t.class_id == 0])
        density_ratio = occupied_area / total_frame_area

        if density_ratio > self.congestion_thresh or (len(tracks) >= 8 and sum(self.track_speeds.values()) / max(1, len(tracks)) < 12.0):
            if not self._is_on_cooldown("congestion_global"):
                incidents.append(Incident(
                    incident_type="TRAFFIC_CONGESTION",
                    severity="HIGH" if density_ratio > 0.70 else "MEDIUM",
                    confidence=0.91,
                    location={"lat": lat, "lng": lng},
                    involved_tracks=[t.track_id for t in tracks[:6]],
                    description=f"Severe traffic congestion: {len(tracks)} active vehicles, area density {density_ratio*100:.1f}%",
                    frame=frame,
                    bboxes=[t.bbox for t in tracks]
                ))

        return incidents
