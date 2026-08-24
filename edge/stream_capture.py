"""
Resilient UAV Video Stream Capture Module.
Supports RTSP, HLS, USB Webcams, and local video files with automatic reconnection,
frame rate throttling, and thread-safe buffer management.
"""

import time
import threading
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np
from config.logging_config import setup_logger

logger = setup_logger("stream_capture")


class StreamCapture:
    """Manages resilient video stream capture from RTSP, HLS, webcam, or video files."""

    def __init__(self, source: str | int = 0, target_fps: int = 30, reconnect_delay: float = 2.0):
        # Convert integer strings (e.g. "0") to int for webcam indices
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source

        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.033
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def _open_stream(self) -> bool:
        """Attempts to open the video capture stream."""
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Opening video stream source: {self.source}...")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            return False

        # Set buffer size to minimize latency for RTSP / Live
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"Successfully opened video stream: {self.source}")
        return True

    def _capture_loop(self):
        """Background thread continuously reading latest frames."""
        last_frame_time = 0.0

        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open_stream():
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # If reading from a file, loop back to the beginning for continuous simulation
                if isinstance(self.source, (str, Path)) and Path(str(self.source)).exists():
                    logger.debug("Looping video file source...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                else:
                    logger.warning("Stream disconnected or frame empty. Reconnecting...")
                    time.sleep(self.reconnect_delay)
                    self._open_stream()
                    continue

            current_time = time.time()
            if (current_time - last_frame_time) >= self.frame_interval:
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.frame_count += 1
                last_frame_time = current_time

            time.sleep(0.005)

    def start(self):
        """Starts asynchronous background frame capture."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Stream capture thread started.")

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Returns the latest captured frame and frame index."""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy(), self.frame_count
            return False, None, self.frame_count

    def frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Generator yielding latest frames continuously."""
        self.start()
        last_served = -1
        while self.is_running:
            success, frame, count = self.read()
            if success and count != last_served:
                last_served = count
                yield frame, count
            time.sleep(0.01)

    def stop(self):
        """Stops capture and releases resources."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("Stream capture stopped.")
