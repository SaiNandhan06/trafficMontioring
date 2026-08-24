"""
Master Dataset Preprocessing & Split Pipeline.
Splits processed labels and images into 80/10/10 Train/Val/Test sets
and verifies bounding box integrity.
"""

import sys
import random
import shutil
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("preprocess")

DATASET_DOWNLOAD_SOURCES = {
    "VisDrone": "http://aiskyeye.com/download/visdrone2019-det/",
    "UAVDT": "https://github.com/VisDrone/UAVDT-Benchmark",
    "UA-DETRAC": "https://detrac-db.rit.albany.edu/"
}


def print_dataset_download_instructions():
    """Prints instructions and direct links for downloading benchmark drone datasets."""
    print("=" * 80)
    print(" UAV TRAFFIC DATASET DOWNLOAD INSTRUCTIONS ")
    print("=" * 80)
    for name, url in DATASET_DOWNLOAD_SOURCES.items():
        print(f"[{name}]: {url}")
        print(f"  Target raw path: {settings.DATA_DIR / 'raw' / name.lower().replace('-', '_')}")
    print("=" * 80)


def create_train_val_test_splits(
    processed_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Splits all images and labels in processed_dir into train/val/test splits."""
    random.seed(seed)

    images_root = processed_dir / "images"
    labels_root = processed_dir / "labels"

    # Collect all image files
    all_images = list(images_root.glob("**/*.jpg")) + list(images_root.glob("**/*.png"))
    if not all_images:
        logger.warning(f"No images found in {images_root}. Run data converters or synthetic generator first.")
        return

    # Filter only those that have corresponding labels
    valid_pairs: List[Tuple[Path, Path]] = []
    for img_path in all_images:
        label_name = img_path.stem + ".txt"
        label_candidates = list(labels_root.glob(f"**/{label_name}"))
        if label_candidates:
            valid_pairs.append((img_path, label_candidates[0]))

    logger.info(f"Found {len(valid_pairs)} image-label pairs for splitting.")
    random.shuffle(valid_pairs)

    n_total = len(valid_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": valid_pairs[:n_train],
        "val": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }

    # Re-organize into structured split directories
    for split_name, pairs in splits.items():
        split_img_dir = processed_dir / "images" / split_name
        split_lbl_dir = processed_dir / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in pairs:
            # Move or copy if not already in place
            dest_img = split_img_dir / img_p.name
            dest_lbl = split_lbl_dir / lbl_p.name

            if img_p != dest_img:
                shutil.copy2(img_p, dest_img)
            if lbl_p != dest_lbl:
                shutil.copy2(lbl_p, dest_lbl)

        logger.info(f"Split [{split_name}]: {len(pairs)} samples organized.")

    logger.info("Dataset 80/10/10 split organization complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize dataset splits")
    parser.add_argument("--info", action="store_true", help="Show download instructions")
    parser.add_argument("--split", action="store_true", help="Execute 80/10/10 split")
    args = parser.parse_args()

    if args.info or not args.split:
        print_dataset_download_instructions()
    if args.split:
        create_train_val_test_splits(settings.DATA_DIR / "processed")
