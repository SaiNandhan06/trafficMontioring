"""
VisDrone Dataset Automated Ingestion & YOLO Formatter.
Converts downloaded VisDrone2019-DET train/val/test splits to YOLOv8 format
with 4-class unified taxonomy mapping (vehicle, pedestrian, cyclist, traffic_signal).
"""

import sys
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("prepare_visdrone")

VISDRONE_TO_UNIFIED = {
    1: 1,   # pedestrian -> pedestrian
    2: 1,   # people -> pedestrian
    3: 2,   # bicycle -> cyclist
    4: 0,   # car -> vehicle
    5: 0,   # van -> vehicle
    6: 0,   # truck -> vehicle
    7: 2,   # tricycle -> cyclist
    8: 2,   # awning-tricycle -> cyclist
    9: 0,   # bus -> vehicle
    10: 2   # motor -> cyclist
}


def find_visdrone_root() -> Path:
    """Finds downloaded VisDrone directory in KaggleHub cache."""
    candidates = [
        Path.home() / ".cache" / "kagglehub" / "datasets" / "kushagrapandya" / "visdrone-dataset" / "versions" / "1",
        PROJECT_ROOT / "data" / "VisDrone",
        PROJECT_ROOT / "data" / "raw_kaggle"
    ]
    for c in candidates:
        if c.exists() and any(c.glob("VisDrone*")):
            return c
    raise FileNotFoundError("VisDrone dataset not found. Run 'python src/data_pipeline/kaggle_download.py' first.")


def process_split(split_name: str, src_dir: Path, target_img_dir: Path, target_lbl_dir: Path, max_samples: int = None):
    """Processes a single split (train/val/test) from VisDrone to YOLO."""
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_folder = None
    ann_folder = None

    for p in src_dir.rglob("images"):
        if p.is_dir():
            img_folder = p
            break
    for p in src_dir.rglob("annotations"):
        if p.is_dir():
            ann_folder = p
            break

    if not img_folder:
        logger.warning(f"Could not find images folder in {src_dir}")
        return 0

    all_images = list(img_folder.glob("*.jpg"))
    if max_samples:
        all_images = all_images[:max_samples]

    logger.info(f"Processing [{split_name}]: {len(all_images)} images from {img_folder}...")

    converted = 0
    for img_file in tqdm(all_images, desc=f"Converting {split_name}"):
        dest_img = target_img_dir / img_file.name
        if not dest_img.exists():
            shutil.copy2(img_file, dest_img)

        # Convert annotation if present
        if ann_folder:
            ann_file = ann_folder / (img_file.stem + ".txt")
            if ann_file.exists():
                try:
                    with Image.open(img_file) as im:
                        img_w, img_h = im.size
                except Exception:
                    continue

                yolo_lines = []
                with open(ann_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 8:
                            continue
                        x, y, w, h = map(float, parts[:4])
                        score = int(parts[4])
                        cat = int(parts[5])

                        if score == 0 or cat not in VISDRONE_TO_UNIFIED:
                            continue

                        cls_id = VISDRONE_TO_UNIFIED[cat]
                        xc = (x + w / 2.0) / img_w
                        yc = (y + h / 2.0) / img_h
                        nw = w / img_w
                        nh = h / img_h

                        xc = max(0.001, min(0.999, xc))
                        yc = max(0.001, min(0.999, yc))
                        nw = max(0.001, min(0.999, nw))
                        nh = max(0.001, min(0.999, nh))

                        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                dest_lbl = target_lbl_dir / (img_file.stem + ".txt")
                with open(dest_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines) + "\n")

        converted += 1

    return converted


def prepare_visdrone_dataset(max_train: int = 500, max_val: int = 100, max_test: int = 100):
    """Prepares stratified VisDrone train/val/test splits in data/processed."""
    root = find_visdrone_root()
    logger.info(f"Found VisDrone source directory at: {root}")

    out_base = PROJECT_ROOT / "data" / "processed"

    # Purge legacy synthetic artifacts from processed splits to prevent leakage
    for p in out_base.glob("**/*synthetic*"):
        if p.is_file():
            p.unlink()

    # Train Split
    train_src = root / "VisDrone2019-DET-train"
    if not train_src.exists():
        train_src = root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train"
    process_split("train", train_src, out_base / "images" / "train", out_base / "labels" / "train", max_samples=max_train)

    # Val Split
    val_src = root / "VisDrone2019-DET-val"
    if not val_src.exists():
        val_src = root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val"
    process_split("val", val_src, out_base / "images" / "val", out_base / "labels" / "val", max_samples=max_val)

    # Test Split
    test_src = root / "VisDrone2019-DET-test-dev"
    if not test_src.exists():
        test_src = root / "VisDrone2019-DET-test-dev" / "VisDrone2019-DET-test-dev"
    process_split("test", test_src, out_base / "images" / "test", out_base / "labels" / "test", max_samples=max_test)

    # Write master dataset.yaml (portable project-relative path)
    yaml_content = """# YOLOv8 Unified UAV Traffic Dataset Configuration
path: data/processed
train: images/train
val: images/val
test: images/test

names:
  0: vehicle
  1: pedestrian
  2: cyclist
  3: traffic_signal
"""
    yaml_path = PROJECT_ROOT / "data" / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info(f"Updated {yaml_path} with VisDrone paths. Ready for fine-tuning!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real VisDrone dataset for YOLOv8 UAV training")
    parser.add_argument("--max-train", type=int, default=500, help="Number of training images to format")
    parser.add_argument("--max-val", type=int, default=100, help="Number of validation images to format")
    parser.add_argument("--max-test", type=int, default=100, help="Number of test images to format")
    args = parser.parse_args()

    prepare_visdrone_dataset(
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test
    )
