"""
UAV-Specific Data Augmentation Pipeline using Albumentations.
Implements transforms tailored for aerial / drone perspectives:
- Vertical and Horizontal flips (drones view from above)
- Perspective and rotation distortions
- Small object scaling and random cropping
- Weather/lighting simulation (day/night, fog, shadows, brightness)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
from config.logging_config import setup_logger

logger = setup_logger("augmentation")

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    HAS_ALBUMENTATIONS = False


def get_drone_augmentation_pipeline(img_size: int = 640):
    """Returns an Albumentations composition tailored for UAV imagery with YOLO bounding boxes."""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose(
        [
            # Drone nadir view is invariant to vertical & horizontal orientation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Subtle perspective shift simulating camera gimbal tilt
            A.Perspective(scale=(0.02, 0.08), p=0.4),

            # Scale and small object shift
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),

            # Weather and lighting variation (day, night, glare, fog)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                    A.RandomGamma(gamma_limit=(70, 130), p=0.8),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                ],
                p=0.6
            ),

            # Blur and atmospheric degradation
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(blur_limit=5, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                ],
                p=0.3
            ),

            # Final resize to model input dimensions
            A.LongestMaxSize(max_size=img_size, p=1.0),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=1.0
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
            min_area=16.0
        )
    )


def augment_image_and_labels(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_labels: List[int],
    pipeline=None
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Applies augmentation pipeline to an image and its bounding boxes."""
    if not HAS_ALBUMENTATIONS:
        # Graceful no-op return if albumentations is not installed
        return image, bboxes, class_labels

    if pipeline is None:
        pipeline = get_drone_augmentation_pipeline()

    # Filter out bboxes with non-positive dimensions
    valid_bboxes = []
    valid_labels = []
    for bbox, label in zip(bboxes, class_labels):
        xc, yc, w, h = bbox
        if w > 0.001 and h > 0.001 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0:
            valid_bboxes.append([xc, yc, w, h])
            valid_labels.append(label)

    if not valid_bboxes:
        return image, [], []

    try:
        transformed = pipeline(
            image=image,
            bboxes=valid_bboxes,
            class_labels=valid_labels
        )
        return transformed["image"], transformed["bboxes"], transformed["class_labels"]
    except Exception as e:
        logger.warning(f"Augmentation failed for frame: {e}. Returning original.")
        return image, valid_bboxes, valid_labels
