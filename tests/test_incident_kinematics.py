"""
Incident Detection, Tracking & Kinematics Regression Test Suite.
Validates ByteTrack multi-object tracking persistence, bounded memory,
speed kinematics, braking heuristics, collision IoU filtering,
congestion density triggers, and cooldown deduplication.
"""

import json
import time
import pytest
import numpy as np

from edge.tracker import UAVByteTracker, Track, calculate_iou
from edge.incident_detector import IncidentDetector, Incident


@pytest.fixture
def dummy_frame():
    """Generates a dummy 1280x720 video frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_track_persistence_same_object():
    """Verifies that a continuous moving object maintains the same track ID across frames."""
    tracker = UAVByteTracker(max_lost_frames=5, iou_threshold=0.3)

    # Frame 1: Detection at [100, 100, 150, 150]
    dets_f1 = [{"bbox": [100.0, 100.0, 150.0, 150.0], "class_id": 0, "conf": 0.9}]
    tracks_f1 = tracker.update(dets_f1, timestamp=0.0)
    assert len(tracks_f1) == 1
    t_id = tracks_f1[0].track_id

    # Frame 2: Small shift [105, 102, 155, 152] (High IoU)
    dets_f2 = [{"bbox": [105.0, 102.0, 155.0, 152.0], "class_id": 0, "conf": 0.88}]
    tracks_f2 = tracker.update(dets_f2, timestamp=0.033)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t_id, "Track ID must persist across overlapping frames"

    # Frame 3: Further shift [110, 105, 160, 155]
    dets_f3 = [{"bbox": [110.0, 105.0, 160.0, 155.0], "class_id": 0, "conf": 0.91}]
    tracks_f3 = tracker.update(dets_f3, timestamp=0.066)
    assert len(tracks_f3) == 1
    assert tracks_f3[0].track_id == t_id


def test_track_bounded_history_and_memory():
    """Verifies that trajectory history length is strictly bounded to prevent memory leaks."""
    track = Track(track_id=1, bbox=[100, 100, 150, 150], class_id=0, confidence=0.9)
    assert track.history.maxlen == 30

    # Append 50 positions
    for i in range(50):
        track.update_center([100 + i, 100 + i, 150 + i, 150 + i], timestamp=i * 0.033)

    assert len(track.history) == 30, "Track history must be bounded at 30 items"


def test_track_pruning_after_lost_frames():
    """Verifies that disappeared tracks are pruned after max_lost_frames."""
    tracker = UAVByteTracker(max_lost_frames=3, iou_threshold=0.3)

    # Initial frame
    tracker.update([{"bbox": [100, 100, 150, 150], "class_id": 0, "conf": 0.9}])
    assert len(tracker.tracks) == 1

    # 3 empty frames (track lost)
    for _ in range(3):
        active = tracker.update([])
        assert len(active) == 0

    # 4th empty frame should prune track from memory
    tracker.update([])
    assert len(tracker.tracks) == 0, "Lost track must be pruned from tracker state"


def test_speed_estimation_formula():
    """Verifies speed estimation math: (pixels / ppm / dt) * 3.6 = km/h."""
    detector = IncidentDetector(pixels_per_meter=10.0)
    track = Track(track_id=1, bbox=[0, 0, 50, 50], class_id=0, confidence=0.9)

    # Simulate 3 observations over 0.1s total time (dt = 0.05s per step)
    # Movement: 100 pixels in 0.1 seconds -> 1000 px/s -> 100 m/s -> 360 km/h
    track.history.clear()
    track.history.append((0.0, 0.0, 1.0))
    track.history.append((50.0, 0.0, 1.05))
    track.history.append((100.0, 0.0, 1.10))

    speed_kmh = detector.estimate_speed_kmh(track)
    assert pytest.approx(speed_kmh, rel=1e-2) == 360.0


def test_speeding_threshold_boundary(dummy_frame):
    """Verifies speeding rule triggers strictly above configured speed limit."""
    detector = IncidentDetector(pixels_per_meter=10.0, speed_limit_kmh=80.0, cooldown_seconds=0.0)

    # Vehicle at 75 km/h (Under limit)
    # (75 / 3.6) * 10 * 0.1 = ~20.83 px displacement in 0.1s
    track_under = Track(track_id=1, bbox=[100, 100, 150, 150], class_id=0, confidence=0.9)
    track_under.history.clear()
    track_under.history.append((0.0, 0.0, 1.0))
    track_under.history.append((10.0, 0.0, 1.05))
    track_under.history.append((20.0, 0.0, 1.10))  # 20 px in 0.1s = 200 px/s = 20 m/s = 72 km/h

    incidents_under = detector.detect_incidents([track_under], dummy_frame)
    speeding_under = [i for i in incidents_under if i.incident_type == "SPEEDING"]
    assert len(speeding_under) == 0, "72 km/h must NOT trigger speeding under 80 km/h limit"

    # Vehicle at 90 km/h (Over limit)
    # 25 m/s * 10 ppm * 0.1s = 25 px displacement in 0.1s = 90 km/h
    track_over = Track(track_id=2, bbox=[200, 200, 250, 250], class_id=0, confidence=0.9)
    track_over.history.clear()
    track_over.history.append((0.0, 0.0, 1.0))
    track_over.history.append((12.5, 0.0, 1.05))
    track_over.history.append((25.0, 0.0, 1.10))

    incidents_over = detector.detect_incidents([track_over], dummy_frame)
    speeding_over = [i for i in incidents_over if i.incident_type == "SPEEDING"]
    assert len(speeding_over) == 1, "90 km/h MUST trigger speeding incident"
    assert speeding_over[0].severity == "MEDIUM"


def test_sudden_braking_heuristic(dummy_frame):
    """Verifies sudden harsh braking detection when speed drop exceeds threshold."""
    detector = IncidentDetector(pixels_per_meter=10.0, deceleration_thresh_kmh=25.0, cooldown_seconds=0.0)

    track = Track(track_id=1, bbox=[100, 100, 150, 150], class_id=0, confidence=0.9)
    # Set previous speed to 80 km/h
    detector.track_speeds[1] = 80.0

    # Current speed drops to 40 km/h (deceleration = 40.0 km/h > 25.0 km/h threshold)
    track.history.clear()
    track.history.append((0.0, 0.0, 1.0))
    track.history.append((5.55, 0.0, 1.05))
    track.history.append((11.11, 0.0, 1.10))  # ~40 km/h

    incidents = detector.detect_incidents([track], dummy_frame)
    braking_incidents = [i for i in incidents if i.incident_type == "SUDDEN_BRAKING"]
    assert len(braking_incidents) == 1, "40 km/h deceleration must trigger SUDDEN_BRAKING"


def test_collision_detection_true_positive(dummy_frame):
    """Verifies collision detection when 2 vehicle boxes overlap with low speed."""
    detector = IncidentDetector(accident_iou_thresh=0.35, cooldown_seconds=0.0)

    t1 = Track(track_id=1, bbox=[100.0, 100.0, 200.0, 200.0], class_id=0, confidence=0.95)
    t2 = Track(track_id=2, bbox=[120.0, 120.0, 220.0, 220.0], class_id=0, confidence=0.92)

    # Speeds are low (stopped / post-crash)
    detector.track_speeds[1] = 5.0
    detector.track_speeds[2] = 4.0

    incidents = detector.detect_incidents([t1, t2], dummy_frame)
    collisions = [i for i in incidents if i.incident_type == "COLLISION_ACCIDENT"]
    assert len(collisions) == 1, "Overlapping slow vehicles must trigger COLLISION_ACCIDENT"
    assert collisions[0].severity == "CRITICAL"
    assert set(collisions[0].involved_tracks) == {1, 2}


def test_collision_false_positive_rejection_high_speed(dummy_frame):
    """Verifies that high-speed vehicles in adjacent lanes do not trigger false collision alarms."""
    detector = IncidentDetector(accident_iou_thresh=0.35, cooldown_seconds=0.0)

    t1 = Track(track_id=1, bbox=[100.0, 100.0, 200.0, 200.0], class_id=0, confidence=0.95)
    t2 = Track(track_id=2, bbox=[120.0, 120.0, 220.0, 220.0], class_id=0, confidence=0.92)

    # Vehicles are moving fast (e.g. 60 km/h overtaking/lane change)
    detector.track_speeds[1] = 60.0
    detector.track_speeds[2] = 65.0

    incidents = detector.detect_incidents([t1, t2], dummy_frame)
    collisions = [i for i in incidents if i.incident_type == "COLLISION_ACCIDENT"]
    assert len(collisions) == 0, "High-speed moving vehicles must NOT trigger collision accident"


def test_traffic_congestion_detection(dummy_frame):
    """Verifies congestion trigger on high vehicle density or slow crawl."""
    detector = IncidentDetector(congestion_thresh=0.55, cooldown_seconds=0.0)

    # Empty frame
    assert len(detector.detect_incidents([], dummy_frame)) == 0

    # 10 slow vehicles (average speed = 5.0 km/h)
    tracks = []
    for i in range(10):
        t = Track(track_id=i + 1, bbox=[i * 50, 100, i * 50 + 40, 180], class_id=0, confidence=0.9)
        detector.track_speeds[i + 1] = 5.0
        tracks.append(t)

    incidents = detector.detect_incidents(tracks, dummy_frame)
    congestion = [i for i in incidents if i.incident_type == "TRAFFIC_CONGESTION"]
    assert len(congestion) == 1, "Dense slow crawl must trigger TRAFFIC_CONGESTION"


def test_incident_deduplication_cooldown(dummy_frame):
    """Verifies that repeated consecutive frames do not spam duplicate incidents during cooldown."""
    detector = IncidentDetector(speed_limit_kmh=80.0, cooldown_seconds=5.0)

    track = Track(track_id=1, bbox=[100, 100, 150, 150], class_id=0, confidence=0.9)
    track.history.clear()
    track.history.append((0.0, 0.0, 1.0))
    track.history.append((25.0, 0.0, 1.05))
    track.history.append((50.0, 0.0, 1.10))  # 180 km/h

    # 1st detection -> triggers
    inc_1 = detector.detect_incidents([track], dummy_frame)
    assert len(inc_1) == 1

    # 2nd detection immediately afterward -> suppressed by cooldown
    inc_2 = detector.detect_incidents([track], dummy_frame)
    assert len(inc_2) == 0, "Duplicate incident within cooldown window must be suppressed"


def test_incident_metadata_serialization(dummy_frame):
    """Verifies that Incident.to_dict() produces valid JSON-compliant serializable metadata."""
    incident = Incident(
        incident_type="COLLISION_ACCIDENT",
        severity="CRITICAL",
        confidence=0.95,
        location={"lat": 37.7749, "lng": -122.4194},
        involved_tracks=[1, 2],
        description="Collision between #1 and #2",
        frame=dummy_frame,
        bboxes=[[100, 100, 200, 200], [120, 120, 220, 220]]
    )

    data = incident.to_dict()
    assert data["incident_type"] == "COLLISION_ACCIDENT"
    assert data["severity"] == "CRITICAL"
    assert isinstance(data["involved_tracks"], list)

    # Verify JSON serialization
    serialized = json.dumps(data)
    loaded = json.loads(serialized)
    assert loaded["incident_id"] == data["incident_id"]
