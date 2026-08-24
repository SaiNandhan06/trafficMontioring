"""
Unit Tests for Edge Tracking and Incident Detection Algorithms.
"""

import numpy as np
import pytest
from edge.tracker import UAVByteTracker, Track, calculate_iou
from edge.incident_detector import IncidentDetector


def test_iou_calculation():
    box1 = [0, 0, 10, 10]
    box2 = [5, 0, 15, 10]
    iou = calculate_iou(box1, box2)
    # Intersection = 5x10 = 50. Union = 100 + 100 - 50 = 150 -> 50/150 = 0.3333
    assert 0.33 < iou < 0.34


def test_bytetrack_tracking():
    tracker = UAVByteTracker(iou_threshold=0.3)
    dets_frame1 = [
        {"bbox": [100, 100, 150, 200], "conf": 0.9, "class_id": 0},
        {"bbox": [300, 300, 350, 400], "conf": 0.85, "class_id": 0}
    ]
    tracks1 = tracker.update(dets_frame1, timestamp=0.0)
    assert len(tracks1) == 2
    id1 = tracks1[0].track_id

    # Small displacement in next frame
    dets_frame2 = [
        {"bbox": [105, 110, 155, 210], "conf": 0.92, "class_id": 0},
        {"bbox": [305, 310, 355, 410], "conf": 0.88, "class_id": 0}
    ]
    tracks2 = tracker.update(dets_frame2, timestamp=0.033)
    assert len(tracks2) == 2
    assert tracks2[0].track_id == id1  # ID persistence


def test_collision_incident_detection():
    detector = IncidentDetector(speed_limit_kmh=80.0, accident_iou_thresh=0.2)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    t1 = Track(track_id=1, bbox=[100, 100, 160, 200], class_id=0, confidence=0.9)
    t2 = Track(track_id=2, bbox=[110, 110, 170, 210], class_id=0, confidence=0.9)

    detector.track_speeds[1] = 5.0
    detector.track_speeds[2] = 4.0

    incidents = detector.detect_incidents([t1, t2], dummy_frame)
    collision_incidents = [i for i in incidents if i.incident_type == "COLLISION_ACCIDENT"]
    assert len(collision_incidents) >= 1
    assert collision_incidents[0].severity == "CRITICAL"
