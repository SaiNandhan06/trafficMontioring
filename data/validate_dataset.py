"""
Automated UAV Traffic Dataset Validation & Integrity Checker.
Audits image-label pairings, YOLO coordinate normalization bounds,
class distribution across splits, and video sequence data leakage.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("validate_dataset")

UNIFIED_CLASS_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
    3: "traffic_signal"
}


def validate_yolo_label_file(label_path: Path) -> Tuple[bool, List[str], Dict[int, int]]:
    """
    Validates a single YOLO label file.
    Returns (is_valid, list_of_errors, class_counts).
    """
    errors = []
    class_counts = defaultdict(int)

    if not label_path.exists():
        return False, ["Label file does not exist"], class_counts

    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, [f"Unreadable file: {e}"], class_counts

    if not content:
        # Empty annotations are valid for background images in YOLO
        return True, [], class_counts

    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: Expected 5 fields, found {len(parts)} ('{line}')")
            continue

        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as e:
            errors.append(f"Line {idx}: Non-numeric bounding box value: {e}")
            continue

        # Class ID validation
        if cls_id not in UNIFIED_CLASS_NAMES:
            errors.append(f"Line {idx}: Invalid class_id {cls_id}. Allowed: {list(UNIFIED_CLASS_NAMES.keys())}")
            continue

        # Coordinates normalization validation [0.0, 1.0]
        if not (0.0 <= xc <= 1.0):
            errors.append(f"Line {idx}: x_center ({xc}) out of [0.0, 1.0] range")
        if not (0.0 <= yc <= 1.0):
            errors.append(f"Line {idx}: y_center ({yc}) out of [0.0, 1.0] range")
        if not (0.0 < w <= 1.0):
            errors.append(f"Line {idx}: width ({w}) must be in (0.0, 1.0]")
        if not (0.0 < h <= 1.0):
            errors.append(f"Line {idx}: height ({h}) must be in (0.0, 1.0]")

        class_counts[cls_id] += 1

    return (len(errors) == 0), errors, class_counts


def audit_dataset_split(
    processed_dir: Path,
    split: str = "train"
) -> Dict:
    """Audits image-label pairings, integrity, and class distribution for a single split."""
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split

    report = {
        "split": split,
        "image_dir_exists": img_dir.exists(),
        "label_dir_exists": lbl_dir.exists(),
        "total_images": 0,
        "total_labels": 0,
        "valid_labels": 0,
        "empty_labels": 0,
        "invalid_labels": 0,
        "missing_labels": 0,
        "orphaned_labels": 0,
        "unreadable_images": 0,
        "class_distribution": {name: 0 for name in UNIFIED_CLASS_NAMES.values()},
        "errors": []
    }

    if not img_dir.exists() or not lbl_dir.exists():
        return report

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in img_extensions]
    labels = [p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]

    report["total_images"] = len(images)
    report["total_labels"] = len(labels)

    img_stems = {img.stem for img in images}
    lbl_stems = {lbl.stem for lbl in labels}

    # Pairing checks
    missing_for_images = img_stems - lbl_stems
    orphaned_for_labels = lbl_stems - img_stems

    report["missing_labels"] = len(missing_for_images)
    report["orphaned_labels"] = len(orphaned_for_labels)

    # Validate image files
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            report["unreadable_images"] += 1
            report["errors"].append(f"Unreadable image {img_path.name}: {e}")

    # Validate label files
    for lbl_path in labels:
        is_valid, errs, counts = validate_yolo_label_file(lbl_path)
        if is_valid:
            report["valid_labels"] += 1
        else:
            report["invalid_labels"] += 1
            for err in errs[:3]:  # sample errors
                report["errors"].append(f"{lbl_path.name}: {err}")

        if lbl_path.stat().st_size == 0:
            report["empty_labels"] += 1

        for cid, count in counts.items():
            cname = UNIFIED_CLASS_NAMES.get(cid, f"class_{cid}")
            report["class_distribution"][cname] += count

    return report


def check_data_leakage(processed_dir: Path) -> Dict:
    """
    Checks for potential video sequence leakage across train, val, and test splits.
    Extracts sequence prefixes from image filenames (e.g. '0000006_...').
    """
    splits = ["train", "val", "test"]
    split_sequences = {}
    split_images = {}

    for s in splits:
        img_dir = processed_dir / "images" / s
        if not img_dir.exists():
            split_sequences[s] = set()
            split_images[s] = set()
            continue

        images = [p.name for p in img_dir.glob("*.jpg")]
        split_images[s] = set(images)

        # Extract sequence stems (first token before underscore)
        seq_prefixes = set()
        for name in images:
            if "_" in name:
                prefix = name.split("_")[0]
                seq_prefixes.add(prefix)
            else:
                seq_prefixes.add(name)
        split_sequences[s] = seq_prefixes

    # Check overlaps
    overlap_train_val = split_sequences.get("train", set()) & split_sequences.get("val", set())
    overlap_train_test = split_sequences.get("train", set()) & split_sequences.get("test", set())
    overlap_val_test = split_sequences.get("val", set()) & split_sequences.get("test", set())

    # Exact filename overlaps
    exact_train_val = split_images.get("train", set()) & split_images.get("val", set())
    exact_train_test = split_images.get("train", set()) & split_images.get("test", set())

    return {
        "sequence_overlap_train_val": list(overlap_train_val),
        "sequence_overlap_train_test": list(overlap_train_test),
        "sequence_overlap_val_test": list(overlap_val_test),
        "exact_image_overlap_train_val": list(exact_train_val),
        "exact_image_overlap_train_test": list(exact_train_test),
        "leakage_risk_detected": bool(exact_train_val or exact_train_test or overlap_train_val or overlap_train_test)
    }


def validate_entire_dataset(
    processed_dir: Path = None,
    output_manifest: Path = None
) -> Dict:
    """Master validation runner for the processed dataset."""
    if processed_dir is None:
        processed_dir = settings.DATA_DIR / "processed"

    if output_manifest is None:
        output_manifest = PROJECT_ROOT / "results" / "dataset_manifest.json"

    processed_dir = Path(processed_dir).resolve()
    output_manifest = Path(output_manifest).resolve()

    logger.info(f"Auditing dataset integrity at: {processed_dir}")

    split_reports = {}
    total_imgs = 0
    total_lbls = 0
    combined_class_counts = defaultdict(int)

    for split in ["train", "val", "test"]:
        sr = audit_dataset_split(processed_dir, split)
        split_reports[split] = sr
        total_imgs += sr["total_images"]
        total_lbls += sr["total_labels"]
        for cname, count in sr["class_distribution"].items():
            combined_class_counts[cname] += count

    leakage_report = check_data_leakage(processed_dir)

    manifest = {
        "dataset_name": "VisDrone2019-DET (Unified UAV Traffic Taxonomy)",
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processed_dir": str(processed_dir),
        "total_images": total_imgs,
        "total_labels": total_lbls,
        "split_counts": {
            "train": split_reports["train"]["total_images"],
            "val": split_reports["val"]["total_images"],
            "test": split_reports["test"]["total_images"]
        },
        "split_percentages": {
            "train": round((split_reports["train"]["total_images"] / max(1, total_imgs)) * 100, 1),
            "val": round((split_reports["val"]["total_images"] / max(1, total_imgs)) * 100, 1),
            "test": round((split_reports["test"]["total_images"] / max(1, total_imgs)) * 100, 1)
        },
        "class_distribution_total": dict(combined_class_counts),
        "split_reports": split_reports,
        "leakage_analysis": leakage_report,
        "validation_passed": (
            split_reports["train"]["invalid_labels"] == 0 and
            split_reports["val"]["invalid_labels"] == 0 and
            split_reports["test"]["invalid_labels"] == 0 and
            not leakage_report["exact_image_overlap_train_test"]
        )
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print Terminal Report
    print("\n" + "=" * 70)
    print(" [UAV TRAFFIC DATASET INTEGRITY & SPLIT AUDIT REPORT]")
    print("=" * 70)
    print(f"Total Processed Images:   {total_imgs}")
    print(f"Total Label Files:        {total_lbls}")
    print(f"Split Breakdown:          Train: {manifest['split_counts']['train']} ({manifest['split_percentages']['train']}%) | "
          f"Val: {manifest['split_counts']['val']} ({manifest['split_percentages']['val']}%) | "
          f"Test: {manifest['split_counts']['test']} ({manifest['split_percentages']['test']}%)")
    print("-" * 70)
    print("Class Distribution Across Dataset:")
    for cname, cnt in combined_class_counts.items():
        print(f"  * {cname:<16}: {cnt:>6} instances")
    print("-" * 70)
    print(f"Sequence Leakage Check:   {'[WARNING] Sequence Overlaps Detected' if leakage_report['leakage_risk_detected'] else '[PASS] Clean (No Overlaps)'}")
    print(f"Dataset Validation Status:{'[PASS] VALID' if manifest['validation_passed'] else '[WARNING] ISSUES DETECTED'}")
    print("=" * 70)
    print(f"Manifest written to: {output_manifest}\n")

    return manifest


if __name__ == "__main__":
    validate_entire_dataset()
