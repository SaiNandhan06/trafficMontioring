"""
Synthetic UAV Drone Traffic Footage & Annotation Generator.
Generates realistic aerial multi-lane traffic frames, ground truth YOLO annotations,
and an MP4 video feed for testing edge inference without requiring external dataset downloads.
"""

import sys
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("synthetic_generator")


class SyntheticVehicle:
    def __init__(self, track_id: int, lane: int, y: float, speed: float, vehicle_type: int = 0):
        self.track_id = track_id
        self.lane = lane
        self.y = y
        self.speed = speed
        self.vehicle_type = vehicle_type  # 0: vehicle, 2: cyclist
        self.width = 38 if vehicle_type == 0 else 18
        self.height = 65 if vehicle_type == 0 else 32
        self.color = (
            random.randint(50, 240),
            random.randint(50, 240),
            random.randint(50, 240)
        )
        self.is_stopped = False
        self.deceleration_rate = 0.0

    def update(self, dt: float, road_height: int):
        if self.is_stopped:
            self.speed = max(0.0, self.speed - self.deceleration_rate * dt)
        self.y += self.speed * dt
        if self.y > road_height + 100:
            self.y = -100


def generate_synthetic_uav_dataset(
    output_dir: Path,
    num_frames: int = 150,
    generate_video: bool = True,
    video_path: Path = None,
    fps: int = 25
):
    """Generates synthetic aerial drone traffic frames and YOLO annotations."""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    video_writer = None
    if generate_video:
        if video_path is None:
            video_path = output_dir / "sample_drone_feed.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Define Road Geometry
    road_left = 340
    road_right = 940
    road_width = road_right - road_left
    lane_width = road_width // 4
    lane_centers = [road_left + lane_width // 2 + i * lane_width for i in range(4)]

    # Spawn Vehicles
    vehicles: List[SyntheticVehicle] = []
    track_counter = 1
    for lane_idx in range(4):
        for slot in range(3):
            v_type = 2 if random.random() < 0.15 else 0
            v = SyntheticVehicle(
                track_id=track_counter,
                lane=lane_idx,
                y=slot * 250 + random.randint(0, 50),
                speed=random.uniform(120, 220),
                vehicle_type=v_type
            )
            vehicles.append(v)
            track_counter += 1

    # Simulate an accident at frame 60
    accident_vehicle = vehicles[1]

    dt = 1.0 / fps
    logger.info(f"Generating {num_frames} synthetic UAV frames...")

    for frame_idx in range(num_frames):
        # Background: Green foliage / city terrain
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 52, 40)  # Dark green asphalt edge

        # Road Surface
        cv2.rectangle(frame, (road_left, 0), (road_right, height), (60, 60, 60), -1)

        # Road Borders (solid white lines)
        cv2.line(frame, (road_left, 0), (road_left, height), (255, 255, 255), 4)
        cv2.line(frame, (road_right, 0), (road_right, height), (255, 255, 255), 4)

        # Lane dividers (dashed yellow/white)
        for i in range(1, 4):
            x_div = road_left + i * lane_width
            for y_dash in range((frame_idx * 5) % 40, height, 40):
                cv2.line(frame, (x_div, y_dash), (x_div, min(height, y_dash + 20)), (200, 200, 200), 2)

        # Pedestrian Crosswalk at top
        for x_cross in range(road_left + 10, road_right - 10, 30):
            cv2.rectangle(frame, (x_cross, 80), (x_cross + 18, 120), (240, 240, 240), -1)

        # Trigger accident/sudden braking at frame 50
        if frame_idx == 50:
            accident_vehicle.is_stopped = True
            accident_vehicle.deceleration_rate = 300.0

        yolo_annotations = []

        # Update & draw vehicles
        for v in vehicles:
            v.update(dt, height)
            x_center = lane_centers[v.lane]
            y_center = v.y

            if -50 <= y_center <= height + 50:
                top_left = (int(x_center - v.width // 2), int(y_center - v.height // 2))
                bot_right = (int(x_center + v.width // 2), int(y_center + v.height // 2))

                # Car body & windshield
                cv2.rectangle(frame, top_left, bot_right, v.color, -1)
                cv2.rectangle(frame, top_left, bot_right, (20, 20, 20), 2)

                # Windshield (darker top)
                ws_top = int(top_left[1] + 10)
                ws_bot = int(top_left[1] + 25)
                cv2.rectangle(frame, (top_left[0] + 4, ws_top), (bot_right[0] - 4, ws_bot), (40, 40, 40), -1)

                # YOLO normalized coords
                xc_norm = x_center / width
                yc_norm = y_center / height
                w_norm = v.width / width
                h_norm = v.height / height

                if 0 < xc_norm < 1 and 0 < yc_norm < 1:
                    yolo_annotations.append(
                        f"{v.vehicle_type} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    )

        # Pedestrian on sidewalk / crosswalk
        ped_x = road_left - 30
        ped_y = 100 + (frame_idx * 1.2) % 500
        cv2.circle(frame, (int(ped_x), int(ped_y)), 7, (220, 180, 140), -1)
        yolo_annotations.append(
            f"1 {ped_x / width:.6f} {ped_y / height:.6f} {14 / width:.6f} {14 / height:.6f}\n"
        )

        # Traffic Signal at intersection corner
        sig_x, sig_y = road_right + 40, 100
        cv2.rectangle(frame, (sig_x - 12, sig_y - 25), (sig_x + 12, sig_y + 25), (20, 20, 20), -1)
        cv2.circle(frame, (sig_x, sig_y - 12), 6, (0, 0, 255), -1)  # Red light
        cv2.circle(frame, (sig_x, sig_y + 12), 6, (0, 255, 0), -1)  # Green light
        yolo_annotations.append(
            f"3 {sig_x / width:.6f} {sig_y / height:.6f} {24 / width:.6f} {50 / height:.6f}\n"
        )

        # Save frame and YOLO label
        frame_name = f"synthetic_frame_{frame_idx:04d}"
        img_file = images_dir / f"{frame_name}.jpg"
        label_file = labels_dir / f"{frame_name}.txt"

        cv2.imwrite(str(img_file), frame)
        with open(label_file, "w", encoding="utf-8") as f:
            f.writelines(yolo_annotations)

        if video_writer:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
        logger.info(f"Synthetic video written to {video_path}")

    logger.info(f"Synthetic dataset created with {num_frames} frames in {output_dir}")


if __name__ == "__main__":
    generate_synthetic_uav_dataset(
        output_dir=settings.DATA_DIR / "synthetic",
        num_frames=100,
        generate_video=True,
        video_path=settings.DATA_DIR / "sample_drone_feed.mp4"
    )
