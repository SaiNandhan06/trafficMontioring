"""
UAVDT (UAV Detection and Tracking) Dataset Annotation Converter to YOLOv8 Format.
Maps UAVDT vehicle categories to the unified taxonomy:
1: car -> 0 (vehicle), 2: truck -> 0 (vehicle), 3: bus -> 0 (vehicle).
"""

import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm
from config.logging_config import setup_logger

logger = setup_logger("uavdt_converter")

UAVDT_TO_UNIFIED = {
    1: 0,  # car -> vehicle
    2: 0,  # truck -> vehicle
    3: 0,  # bus -> vehicle
}


def convert_uavdt_sequence(
    gt_file: Path,
    sequence_images_dir: Path,
    output_labels_dir: Path,
    sequence_name: str
):
    """Processes a UAVDT sequence ground truth file and outputs frame-level YOLO labels."""
    if not gt_file.exists() or not sequence_images_dir.exists():
        return

    # Group annotations by frame index
    frames_annotations = defaultdict(list)
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue

            frame_idx = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            out_of_view = int(parts[6])
            category = int(parts[8])

            if out_of_view == 1 or category not in UAVDT_TO_UNIFIED:
                continue

            unified_cls = UAVDT_TO_UNIFIED[category]
            frames_annotations[frame_idx].append((unified_cls, x, y, w, h))

    for frame_idx, annotations in frames_annotations.items():
        img_name = f"img{frame_idx:06d}.jpg"
        img_path = sequence_images_dir / img_name
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []
        for cls_id, x, y, w, h in annotations:
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            if norm_w > 0 and norm_h > 0:
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        out_label_name = f"{sequence_name}_img{frame_idx:06d}.txt"
        out_label_path = output_labels_dir / out_label_name
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)


def process_uavdt_dataset(raw_dir: Path, output_dir: Path):
    """Iterates through all UAVDT sequence folders and converts annotations."""
    gt_dir = raw_dir / "UAV-benchmark-M"
    seq_dir = raw_dir / "UAV-benchmark-M"
    labels_out = output_dir / "labels"

    if not gt_dir.exists():
        logger.warning(f"UAVDT directory not found: {gt_dir}")
        return

    gt_files = list(gt_dir.glob("*_gt_whole.txt"))
    logger.info(f"Processing {len(gt_files)} UAVDT sequences...")

    for gt_file in tqdm(gt_files, desc="UAVDT Sequences"):
        seq_name = gt_file.stem.replace("_gt_whole", "")
        img_folder = seq_dir / seq_name
        convert_uavdt_sequence(gt_file, img_folder, labels_out, seq_name)

    logger.info("UAVDT conversion complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert UAVDT dataset to YOLO format")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/uavdt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    process_uavdt_dataset(args.raw_dir, args.output_dir)
