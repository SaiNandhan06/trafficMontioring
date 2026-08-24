"""
SkyGuard UAV Edge Intelligence Layer Package.
Video ingestion, ByteTrack tracking, kinematic incident detection, and SQLite retry queue.
"""

from edge.incident_detector import IncidentDetector, Incident
from edge.tracker import UAVByteTracker, Track, calculate_iou
from edge.retry_queue import RetryQueue

__all__ = [
    "IncidentDetector",
    "Incident",
    "UAVByteTracker",
    "Track",
    "calculate_iou",
    "RetryQueue",
]
