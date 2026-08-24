"""
KaggleHub Automated UAV Dataset Downloader & Inspector.
Downloads aerial traffic benchmarks (VisDrone, UAVDT, UA-DETRAC) using kagglehub,
inspects directory schemas, and prepares staging image and annotation files.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("kaggle_download")

# Authoritative supported aerial datasets on Kaggle
DATASET_REGISTRY = {
    "visdrone": {
        "handle": "kushagrapandya/visdrone-dataset",
        "display_name": "VisDrone2019-DET Benchmark",
        "expected_format": "VisDrone text annotations (x,y,w,h,score,category,...)",
        "converter_module": "data.prepare_visdrone",
        "target_dir": "data/processed",
        "unified_classes": ["vehicle", "pedestrian", "cyclist", "traffic_signal"]
    },
    "uavdt": {
        "handle": "shakaibkaggle/uavdt-dataset",
        "display_name": "UAVDT Benchmark (UAV Detection & Tracking)",
        "expected_format": "UAVDT sequence text annotations (_gt_whole.txt)",
        "converter_module": "data.converters.uavdt_to_yolo",
        "target_dir": "data/processed/uavdt",
        "unified_classes": ["vehicle"]
    },
    "ua_detrac": {
        "handle": "bratjay/ua-detrac-orig",
        "display_name": "UA-DETRAC Benchmark (XML Sequences)",
        "expected_format": "UA-DETRAC XML annotation trees",
        "converter_module": "data.converters.ua_detrac_to_yolo",
        "target_dir": "data/processed/ua_detrac",
        "unified_classes": ["vehicle"]
    }
}

# Alias resolution mapping
DATASET_ALIASES = {
    "visdrone": "visdrone",
    "kushagrapandya/visdrone-dataset": "visdrone",
    "uavdt": "uavdt",
    "shakaibkaggle/uavdt-dataset": "uavdt",
    "ua_detrac": "ua_detrac",
    "ua-detrac": "ua_detrac",
    "detrac": "ua_detrac",
    "bratjay/ua-detrac-orig": "ua_detrac"
}

SUPPORTED_DATASETS = {k: v["handle"] for k, v in DATASET_REGISTRY.items()}
SUPPORTED_DATASETS["ua-detrac"] = DATASET_REGISTRY["ua_detrac"]["handle"]
SUPPORTED_DATASETS["detrac"] = DATASET_REGISTRY["ua_detrac"]["handle"]


def download_dataset(
    dataset_handle: str = "kushagrapandya/visdrone-dataset",
    custom_cache_dir: Optional[str] = None,
    limit: Optional[int] = None
) -> Optional[Path]:
    """
    Downloads a dataset using kagglehub with error handling, directory inspection,
    and optional staging file limit.
    """
    # 1. Configure custom cache directory if provided
    if custom_cache_dir:
        os.environ["KAGGLEHUB_CACHE"] = str(custom_cache_dir)
        Path(custom_cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Set KAGGLEHUB_CACHE to: {custom_cache_dir}")

    # Handle short names (e.g. 'visdrone', 'uavdt')
    dataset_key = dataset_handle.lower().strip()
    if dataset_key in SUPPORTED_DATASETS:
        dataset_handle = SUPPORTED_DATASETS[dataset_key]

    logger.info(f"Initiating KaggleHub download for dataset: '{dataset_handle}'...")

    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_handle)
        dataset_path = Path(path)
        logger.info(f"Dataset successfully downloaded/located at: {dataset_path}")
    except Exception as e:
        logger.warning(f"KaggleHub download encountered an issue: {e}")
        logger.info("Checking for local fallback datasets in 'data/processed' or 'data/'...")
        fallback_path = PROJECT_ROOT / "data" / "processed"
        if fallback_path.exists() and any((fallback_path / "images").glob("**/*.jpg")):
            logger.info(f"Using local cached dataset at: {fallback_path}")
            dataset_path = fallback_path
        else:
            logger.error(f"Failed to obtain dataset '{dataset_handle}'. Please check Kaggle credentials or network.")
            return None

    # 2. Inspect directory structure and catalog files
    stats = inspect_dataset_directory(dataset_path, limit=limit)
    print_dataset_summary(dataset_handle, dataset_path, stats)

    return dataset_path


def inspect_dataset_directory(dataset_path: Path, limit: Optional[int] = None) -> Dict:
    """Scans and categorizes images and annotation files in the dataset."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
    ann_exts = {".txt", ".json", ".csv", ".xml"}

    image_files: List[Path] = []
    annotation_files: List[Path] = []
    other_files: List[Path] = []
    subdirectories: List[Path] = []

    for item in dataset_path.rglob("*"):
        if item.is_dir():
            subdirectories.append(item)
        elif item.is_file():
            suffix = item.suffix.lower()
            if suffix in img_exts:
                image_files.append(item)
            elif suffix in ann_exts:
                annotation_files.append(item)
            else:
                other_files.append(item)

    if limit is not None and limit > 0:
        image_files = image_files[:limit]
        annotation_files = annotation_files[:limit]

    return {
        "total_dirs": len(subdirectories),
        "total_images": len(image_files),
        "total_annotations": len(annotation_files),
        "total_other": len(other_files),
        "sample_images": [str(p.relative_to(dataset_path)) for p in image_files[:5]],
        "sample_annotations": [str(p.relative_to(dataset_path)) for p in annotation_files[:5]],
        "subdirectories": [str(p.relative_to(dataset_path)) for p in subdirectories[:8]],
        "image_paths": image_files,
        "annotation_paths": annotation_files
    }


def print_dataset_summary(dataset_handle: str, dataset_path: Path, stats: Dict):
    """Prints a clear, structured summary of the downloaded dataset."""
    print("\n" + "=" * 70)
    print(f" [KAGGLE DATASET INSPECTION REPORT]: {dataset_handle}")
    print("=" * 70)
    print(f"Root Location:         {dataset_path}")
    print(f"Subdirectories:        {stats['total_dirs']}")
    print(f"Total Image Files:     {stats['total_images']} ({', '.join(set(Path(p).suffix for p in stats['sample_images'])) if stats['sample_images'] else 'None'})")
    print(f"Total Annotations:     {stats['total_annotations']} ({', '.join(set(Path(p).suffix for p in stats['sample_annotations'])) if stats['sample_annotations'] else 'None'})")
    print("-" * 70)
    print("Directory Structure (Sample):")
    for d in stats["subdirectories"]:
        print(f"  [DIR] /{d}")
    print("-" * 70)
    print("Sample Images:")
    for img in stats["sample_images"]:
        print(f"  [IMG] {img}")
    print("Sample Annotations:")
    for ann in stats["sample_annotations"]:
        print(f"  [ANN] {ann}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & inspect UAV datasets via KaggleHub")
    parser.add_argument("--dataset", type=str, default="kushagrapandya/visdrone-dataset",
                        help="Kaggle dataset handle or alias (visdrone, uavdt, ua_detrac)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom cache directory for KAGGLEHUB_CACHE")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images/annotations for quick testing")
    args = parser.parse_args()

    download_dataset(
        dataset_handle=args.dataset,
        custom_cache_dir=args.cache_dir,
        limit=args.limit
    )
