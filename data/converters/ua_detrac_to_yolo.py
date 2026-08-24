"""
UA-DETRAC Dataset XML Annotation Converter to YOLOv8 Format.
Parses XML sequence annotations and converts vehicle bounding boxes to normalized YOLO format.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm
from config.logging_config import setup_logger

logger = setup_logger("ua_detrac_converter")


def convert_ua_detrac_xml(
    xml_path: Path,
    sequence_images_dir: Path,
    output_labels_dir: Path,
    sequence_name: str
):
    """Parses a single UA-DETRAC XML annotation file and outputs frame-level YOLO labels."""
    if not xml_path.exists() or not sequence_images_dir.exists():
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num", 0))
        img_name = f"img{frame_num:05d}.jpg"
        img_path = sequence_images_dir / img_name

        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []
        target_list = frame.find("target_list")
        if target_list is not None:
            for target in target_list.findall("target"):
                box = target.find("box")
                if box is not None:
                    left = float(box.get("left", 0))
                    top = float(box.get("top", 0))
                    width = float(box.get("width", 0))
                    height = float(box.get("height", 0))

                    # UA-DETRAC targets are all vehicles (class 0)
                    cls_id = 0

                    x_center = (left + width / 2.0) / img_w
                    y_center = (top + height / 2.0) / img_h
                    norm_w = width / img_w
                    norm_h = height / img_h

                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    if norm_w > 0 and norm_h > 0:
                        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        out_label_name = f"{sequence_name}_img{frame_num:05d}.txt"
        out_label_path = output_labels_dir / out_label_name
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)


def process_ua_detrac_dataset(raw_dir: Path, output_dir: Path):
    """Processes all XML annotations and image sequences from UA-DETRAC."""
    xml_dir = raw_dir / "DETRAC-Train-Annotations-XML"
    seq_dir = raw_dir / "Insight-MVT_Annotation_Train"
    labels_out = output_dir / "labels"

    if not xml_dir.exists():
        logger.warning(f"UA-DETRAC XML directory not found: {xml_dir}")
        return

    xml_files = list(xml_dir.glob("*.xml"))
    logger.info(f"Processing {len(xml_files)} UA-DETRAC XML files...")

    for xml_file in tqdm(xml_files, desc="UA-DETRAC Converter"):
        seq_name = xml_file.stem
        img_folder = seq_dir / seq_name
        convert_ua_detrac_xml(xml_file, img_folder, labels_out, seq_name)

    logger.info("UA-DETRAC conversion complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC dataset to YOLO format")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ua_detrac"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    process_ua_detrac_dataset(args.raw_dir, args.output_dir)
