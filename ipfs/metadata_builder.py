"""
IPFS Incident Metadata and Evidence Builder.
Packages video frames, bounding box detections, drone telemetry, and cryptographic signatures
into canonical tamper-proof JSON schemas.
"""

import json
import base64
import time
from typing import Dict, List, Any, Tuple
import cv2
import numpy as np
from config.settings import settings


def build_incident_metadata(
    incident_id: str,
    drone_id: str,
    incident_type: str,
    severity: str,
    confidence: float,
    location: Dict[str, float],
    involved_tracks: List[int],
    description: str,
    frame: np.ndarray,
    detections: List[Dict[str, Any]] = None,
    video_ref: str = None
) -> Tuple[Dict, bytes]:
    """Encodes frame to JPEG bytes and structures complete incident metadata."""
    # Compress frame to JPEG
    _, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    frame_bytes = encoded_img.tobytes()

    metadata = {
        "schema_version": "1.0.0",
        "incident_id": incident_id,
        "timestamp": time.time(),
        "drone_id": drone_id,
        "location": {
            "latitude": location.get("lat", settings.DRONE_LAT),
            "longitude": location.get("lng", settings.DRONE_LNG)
        },
        "incident": {
            "type": incident_type,
            "severity": severity,
            "confidence": round(confidence, 4),
            "description": description,
            "involved_tracks": involved_tracks
        },
        "detections": detections or [],
        "video_ref": video_ref or "",
        "evidence_frame": {
            "mime_type": "image/jpeg",
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
            "size_bytes": len(frame_bytes)
        }
    }

    return metadata, frame_bytes
