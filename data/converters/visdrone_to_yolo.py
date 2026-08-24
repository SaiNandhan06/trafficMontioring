"""
VisDrone Dataset Annotation Converter to YOLOv8 Format.
Converts VisDrone bounding box annotations (x, y, w, h) to normalized YOLO format
and maps categories to the unified 4-class taxonomy:
0: vehicle, 1: pedestrian, 2: cyclist, 3: traffic_signal.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm
from config.logging_config import setup_logger

logger = setup_logger("visdrone_converter")

# VisDrone category ID to Unified class mapping
VISDRONE_TO_UNIFIED = {
    1: 1,  # pedestrian -> pedestrian
    2: 1,  # people -> pedestrian
    3: 2,  # bicycle -> cyclist
    4: 0,  # car -> vehicle
    5: 0,  # van -> vehicle
    6: 0,  # truck -> vehicle
    7: 2,  # tricycle -> cyclist
    8: 2,  # awning-tricycle -> cyclist
    9: 0,  # bus -> vehicle
    10: 2, # motor -> cyclist
}


def convert_visdrone_annotation(
    anno_file: Path,
    image_file: Path,
    output_label_file: Path
) -> int:
    """Converts a single VisDrone annotation text file to YOLO format."""
    if not image_file.exists():
        return 0

    with Image.open(image_file) as img:
        img_w, img_h = img.size

    yolo_lines = []
    with open(anno_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            x, y, w, h = map(float, parts[:4])
            score = int(parts[4])
            category = int(parts[5])

            # Filter out ignored regions (score == 0 or category not in mapping)
            if score == 0 or category not in VISDRONE_TO_UNIFIED:
                continue

            unified_cls = VISDRONE_TO_UNIFIED[category]

            # Convert (x, y, w, h) top-left to normalized (x_center, y_center, w, h)
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Clip values within [0.0, 1.0]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            if norm_w > 0 and norm_h > 0:
                yolo_lines.append(f"{unified_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    output_label_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_file, "w", encoding="utf-8") as f:
        f.writelines(yolo_lines)

    return len(yolo_lines)


def process_visdrone_dataset(raw_dir: Path, output_dir: Path):
    """Processes entire VisDrone directory structure (images and annotations)."""
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    labels_out_dir = output_dir / "labels"

    if not annotations_dir.exists():
        logger.warning(f"Annotations directory not found: {annotations_dir}")
        return

    anno_files = list(annotations_dir.glob("*.txt"))
    logger.info(f"Converting {len(anno_files)} VisDrone annotation files...")

    total_objects = 0
    for anno_file in tqdm(anno_files, desc="VisDrone Converter"):
        img_name = anno_file.stem + ".jpg"
        img_path = images_dir / img_name
        label_out_path = labels_out_dir / (anno_file.stem + ".txt")

        count = convert_visdrone_annotation(anno_file, img_path, label_out_path)
        total_objects += count

    logger.info(f"VisDrone conversion complete: {len(anno_files)} files, {total_objects} objects processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert VisDrone dataset to YOLO format")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/visdrone"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    process_visdrone_dataset(args.raw_dir, args.output_dir)
