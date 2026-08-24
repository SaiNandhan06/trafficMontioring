"""
Real UAV Dataset Inference & Visualization Engine.
Runs fine-tuned or base YOLOv8 model on real/downloaded aerial images,
generates side-by-side visual comparisons, measures hardware latency (mean, p95, p99),
and exports comprehensive detection statistics to JSON.
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("real_data_inference")

# Unified taxonomy labels and color palette (BGR)
CLASS_NAMES = {0: "vehicle", 1: "pedestrian", 2: "cyclist", 3: "traffic_signal"}
CLASS_COLORS = {
    0: (0, 200, 255),   # Yellow/Orange for vehicles
    1: (255, 100, 50),  # Blue for pedestrians
    2: (50, 220, 50),   # Green for cyclists
    3: (0, 0, 255)      # Red for signals
}


def find_best_model_weights(explicit_path: Optional[str] = None) -> Path:
    """Finds best available trained model weights or falls back to nano baseline."""
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)

    search_candidates = [
        PROJECT_ROOT / "yolov8n.pt",
        PROJECT_ROOT / "model" / "weights" / "yolov8_uav_best.pt",
        PROJECT_ROOT / "model" / "runs" / "yolov8n_uav_traffic" / "weights" / "best.pt",
        PROJECT_ROOT / "model" / "runs" / "yolov8s_uav_traffic" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    ]

    for cand in search_candidates:
        if cand.exists():
            return cand

    return Path("yolov8n.pt")


def find_test_images(dataset_dir: Optional[Path] = None, limit: int = 50) -> List[Path]:
    """Finds available real or processed drone test images."""
    search_paths = []
    if dataset_dir and Path(dataset_dir).exists():
        search_paths.append(Path(dataset_dir))

    # Check KaggleHub cache directory for real VisDrone images
    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets"
    if kaggle_cache.exists():
        search_paths.append(kaggle_cache)

    search_paths.extend([
        PROJECT_ROOT / "data" / "processed" / "images" / "test",
        PROJECT_ROOT / "data" / "processed" / "images" / "val",
        PROJECT_ROOT / "data" / "processed" / "images" / "train",
        PROJECT_ROOT / "data" / "processed" / "images",
        PROJECT_ROOT / "data"
    ])

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    discovered = []

    for sp in search_paths:
        if sp.exists():
            for f in sp.rglob("*"):
                if f.is_file() and f.suffix.lower() in img_extensions:
                    discovered.append(f)
            if discovered:
                break

    if not discovered:
        logger.warning("No existing images found on disk. Generating a synthetic test frame for validation...")
        temp_dir = PROJECT_ROOT / "results" / "temp_test_feed"
        temp_dir.mkdir(parents=True, exist_ok=True)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Real Drone Feed Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(dummy_img, (200, 250), (320, 350), (0, 200, 255), -1)
        p = temp_dir / "synthetic_drone_sample.jpg"
        cv2.imwrite(str(p), dummy_img)
        discovered.append(p)

    random.seed(42)
    random.shuffle(discovered)
    return discovered[:limit]


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draws styled bounding boxes, class labels, and confidence tags on image."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det["class_id"]
        conf = det["conf"]
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label badge
        label = f"{cls_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(annotated, label, (x1 + 3, max(th + 2, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def create_side_by_side(orig: np.ndarray, pred: np.ndarray, title: str = "Real UAV Testing") -> np.ndarray:
    """Creates a side-by-side comparison canvas of original vs predicted detections."""
    h1, w1 = orig.shape[:2]
    h2, w2 = pred.shape[:2]
    h = max(h1, h2)
    
    # Resize to match height if different
    if h1 != h:
        orig = cv2.resize(orig, (int(w1 * h / h1), h))
    if h2 != h:
        pred = cv2.resize(pred, (int(w2 * h / h2), h))

    divider = np.full((h, 4, 3), (180, 180, 180), dtype=np.uint8)
    combined = np.hstack([orig, divider, pred])

    # Header banner
    header = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "ORIGINAL INPUT", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(header, f"YOLOv8 UAV PREDICTIONS: {title}", (orig.shape[1] + 20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

    return np.vstack([header, combined])


def run_real_data_inference(
    dataset_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
    limit: int = 50,
    conf_threshold: float = 0.35,
    output_dir: str = "results/real_data_test"
) -> Dict:
    """Main testing pipeline across real VisDrone/UAV images."""
    out_path = PROJECT_ROOT / output_dir
    vis_path = out_path / "visualizations"
    vis_path.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Model
    model_file = find_best_model_weights(weights_path)
    logger.info(f"Loading YOLOv8 weights from: {model_file}")

    from ultralytics import YOLO
    model = YOLO(str(model_file))

    # 2. Resolve Test Images
    image_paths = find_test_images(Path(dataset_dir) if dataset_dir else None, limit=limit)
    logger.info(f"Discovered {len(image_paths)} images for real-data inference test.")

    latencies: List[float] = []
    class_counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}
    class_confidences: Dict[str, List[float]] = {name: [] for name in CLASS_NAMES.values()}
    total_detections = 0
    saved_samples: List[str] = []

    print("\n" + "=" * 70)
    print(f" [RUNNING REAL UAV DATASET INFERENCE TESTING] ({len(image_paths)} images)")
    print("=" * 70)

    for idx, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        results = model(frame, conf=conf_threshold, verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                # Remap COCO classes if using standard weights
                if cls_id in [2, 3, 5, 6, 7, 8]:  # car, motorcycle, bus, train, truck, boat -> vehicle
                    cls_id = 0
                elif cls_id == 0:                # person -> pedestrian
                    cls_id = 1
                elif cls_id == 1:                # bicycle -> cyclist
                    cls_id = 2
                elif cls_id == 9:                # traffic light -> signal
                    cls_id = 3
                else:
                    cls_id = 0                   # default to vehicle for other mobile objects

                cls_name = CLASS_NAMES.get(cls_id, "vehicle")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                class_confidences.setdefault(cls_name, []).append(conf)
                total_detections += 1

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": cls_id
                })

        # Save visualization (for up to 20 images)
        if idx <= 20:
            pred_annotated = draw_detections(frame, detections)
            side_by_side = create_side_by_side(frame, pred_annotated, title=img_path.name)
            out_file = vis_path / f"test_result_{idx:03d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_file), side_by_side)
            saved_samples.append(str(out_file.relative_to(PROJECT_ROOT)))

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"  Processed [{idx}/{len(image_paths)}] images | Current Mean Latency: {np.mean(latencies):.2f} ms")

    # 3. Compute Aggregated Metrics
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99_lat = float(np.percentile(latencies, 99)) if latencies else 0.0
    fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

    avg_conf_per_class = {
        cls: float(np.mean(confs)) if confs else 0.0
        for cls, confs in class_confidences.items()
    }

    report = {
        "timestamp": time.time(),
        "model_weights": str(model_file),
        "total_images_tested": len(image_paths),
        "total_detections": total_detections,
        "class_distribution": class_counts,
        "average_confidence_per_class": avg_conf_per_class,
        "latency_metrics_ms": {
            "mean": round(mean_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2)
        },
        "throughput_fps": round(fps, 2),
        "saved_visualizations": saved_samples
    }

    # 4. Save JSON Report
    report_file = out_path / "metrics_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved real-data test report to: {report_file}")

    # 5. Print Formatted Console Report
    print("\n" + "=" * 70)
    print(" [REAL UAV DATASET TEST & INFERENCE PERFORMANCE REPORT]")
    print("=" * 70)
    print(f"Tested Images:          {len(image_paths)}")
    print(f"Total Detections:       {total_detections}")
    print(f"Inference FPS:          {fps:.2f} frames/sec")
    print(f"Mean Latency:           {mean_lat:.2f} ms (p95: {p95_lat:.2f} ms | p99: {p99_lat:.2f} ms)")
    print("-" * 70)
    print("Class-Wise Detection Summary:")
    for cls, count in class_counts.items():
        avg_c = avg_conf_per_class.get(cls, 0.0)
        print(f"  • {cls:<16}: {count:>4} detections (Avg Conf: {avg_c:.2f})")
    print("-" * 70)
    print(f"Visualizations Saved:   {len(saved_samples)} images in '{output_dir}/visualizations/'")
    print(f"Metrics Report:         '{report_file}'")
    print("=" * 70 + "\n")

    return report


# Canonical alias
test_real_data = run_real_data_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on Real VisDrone/UAV Images")
    parser.add_argument("--dataset", type=str, default=None, help="Directory containing VisDrone/UAV images")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to test")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="results/real_data_test", help="Output directory")
    args = parser.parse_args()

    run_real_data_inference(
        dataset_dir=args.dataset,
        weights_path=args.weights,
        limit=args.limit,
        conf_threshold=args.conf,
        output_dir=args.output
    )
