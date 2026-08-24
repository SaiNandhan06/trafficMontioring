"""
Multi-Object Tracking (MOT) Module using ByteTrack principles.
Maintains persistent track IDs, trajectory histories, and kinematic states
across consecutive video frames.
"""

from collections import deque
import numpy as np
from typing import Dict, List, Tuple, Optional


class Track:
    """Represents a single tracked entity across video frames."""

    def __init__(self, track_id: int, bbox: List[float], class_id: int, confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.history = deque(maxlen=30)  # Store last 30 positions [(x_center, y_center, timestamp)]
        self.velocities = deque(maxlen=10)  # Pixels/sec or km/h
        self.age = 1
        self.time_since_update = 0
        self.update_center(bbox)

    def update_center(self, bbox: List[float], timestamp: float = 0.0):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.history.append((xc, yc, timestamp))
        self.age += 1
        self.time_since_update = 0

    def compute_displacement(self) -> Tuple[float, float]:
        """Returns (dx, dy) over the recent history."""
        if len(self.history) < 2:
            return 0.0, 0.0
        x_first, y_first, _ = self.history[0]
        x_last, y_last, _ = self.history[-1]
        return x_last - x_first, y_last - y_first


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class UAVByteTracker:
    """ByteTrack-style IoU and spatial tracker for aerial UAV video streams."""

    def __init__(self, max_lost_frames: int = 15, iou_threshold: float = 0.3):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def update(
        self,
        detections: List[Dict],  # List of {'bbox': [x1, y1, x2, y2], 'class_id': int, 'conf': float}
        timestamp: float = 0.0
    ) -> List[Track]:
        """Associates detections with existing tracks using IoU matching."""
        # Increment time since update for existing tracks
        for track in self.tracks.values():
            track.time_since_update += 1

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        matched_pairs = []

        if len(self.tracks) > 0 and len(detections) > 0:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = calculate_iou(self.tracks[tid].bbox, det["bbox"])

            # Greedy matching
            for _ in range(min(len(track_ids), len(detections))):
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]

                if max_iou < self.iou_threshold:
                    break

                t_idx, d_idx = max_idx
                tid = track_ids[t_idx]

                if tid in unmatched_tracks and d_idx in unmatched_dets:
                    matched_pairs.append((tid, d_idx))
                    unmatched_tracks.remove(tid)
                    unmatched_dets.remove(d_idx)
                    iou_matrix[t_idx, :] = -1.0
                    iou_matrix[:, d_idx] = -1.0

        # Update matched tracks
        for tid, d_idx in matched_pairs:
            det = detections[d_idx]
            self.tracks[tid].update_center(det["bbox"], timestamp)
            self.tracks[tid].confidence = det["conf"]

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = Track(
                track_id=self.next_track_id,
                bbox=det["bbox"],
                class_id=det["class_id"],
                confidence=det["conf"]
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # Prune dead tracks
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.time_since_update > self.max_lost_frames
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return currently active tracks
        return [track for track in self.tracks.values() if track.time_since_update == 0]
