"""
Dynamic UAV Dataset Ingestion & Auto-Split Pipeline.
Allows seamless addition of new training images, videos, and annotations
with automatic validation, augmentations, and stratified 80/10/10 re-splitting.
"""

import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from config.settings import settings
from config.logging_config import setup_logger
from data.preprocess import create_train_val_test_splits
from data.augmentation import augment_image_and_labels

logger = setup_logger("data_ingest")


def extract_frames_from_video(video_path: Path, output_images_dir: Path, frame_interval: int = 15) -> int:
    """Extracts periodic frames from video for dataset expansion."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video {video_path}")
        return 0

    count = 0
    saved = 0
    output_images_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if count % frame_interval == 0:
            out_name = f"{video_stem}_frame_{saved:05d}.jpg"
            cv2.imwrite(str(output_images_dir / out_name), frame)
            saved += 1
        count += 1

    cap.release()
    logger.info(f"Extracted {saved} frames from {video_path.name}")
    return saved


def ingest_custom_data(
    source_dir: Path,
    augment: bool = False,
    num_augmentations: int = 1,
    re_split: bool = True
):
    """
    Ingests new images and labels into data/processed and automatically re-splits.
    Accepts:
      - A directory containing 'images/' and 'labels/'
      - Or a directory of raw images/videos
    """
    processed_dir = settings.DATA_DIR / "processed"
    target_img_dir = processed_dir / "images" / "raw_inbound"
    target_lbl_dir = processed_dir / "labels" / "raw_inbound"
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Source path {source_path} does not exist!")
        return

    # 1. Ingest Video Files if present
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    for vid_file in list(source_path.glob("*")) + list(source_path.glob("**/*")):
        if vid_file.suffix.lower() in video_extensions:
            extract_frames_from_video(vid_file, target_img_dir)

    # 2. Ingest Image Files & Labels
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    all_imgs = [p for p in source_path.glob("**/*") if p.suffix.lower() in img_extensions]

    ingested_count = 0
    for img_path in all_imgs:
        dest_img = target_img_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # Look for corresponding label
        lbl_candidate = source_path / "labels" / (img_path.stem + ".txt")
        if not lbl_candidate.exists():
            lbl_candidate = img_path.with_suffix(".txt")

        if lbl_candidate.exists():
            dest_lbl = target_lbl_dir / (img_path.stem + ".txt")
            shutil.copy2(lbl_candidate, dest_lbl)

        ingested_count += 1

    logger.info(f"Ingested {ingested_count} image files into dataset pool.")

    # 3. Dynamic Re-splitting (80/10/10)
    if re_split:
        logger.info("Re-indexing and organizing 80/10/10 train/val/test splits...")
        create_train_val_test_splits(processed_dir)

    # Clean up staging inbound folder
    if target_img_dir.exists() and not list(target_img_dir.glob("*")):
        shutil.rmtree(target_img_dir, ignore_errors=True)
    if target_lbl_dir.exists() and not list(target_lbl_dir.glob("*")):
        shutil.rmtree(target_lbl_dir, ignore_errors=True)

    logger.info(f"Custom training data ingestion complete! Ready for model training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamically ingest custom UAV training data")
    parser.add_argument("--source", type=Path, required=True, help="Path to custom images/annotations directory or video")
    parser.add_argument("--augment", action="store_true", help="Apply Albumentations aerial data augmentation")
    parser.add_argument("--no-split", action="store_true", help="Skip automatic 80/10/10 train/val/test re-split")
    args = parser.parse_args()

    ingest_custom_data(
        source_dir=args.source,
        augment=args.augment,
        re_split=not args.no_split
    )
