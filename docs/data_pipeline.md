# 📥 SkyGuard UAV Data Pipeline & KaggleHub Integration Guide

This guide covers automated dataset downloading, dataset formats, directory structures, and customization options for aerial traffic monitoring models.

---

## 🚁 Supported Public Datasets

SkyGuard UAV provides native support for popular aerial and drone datasets via `kagglehub`:

| Dataset Name | Kaggle Handle | Description |
| :--- | :--- | :--- |
| **VisDrone** | `kushagrapandya/visdrone-dataset` | High-density urban UAV traffic with vehicles, pedestrians, cyclists |
| **UAVDT** | `sshikamaru/uavdt-dataset` | UAV benchmark for vehicle detection and multi-object tracking |
| **UA-DETRAC** | `rohitrango/ua-detrac-dataset` | Complex real-world road and traffic surveillance sequences |

---

## ⚡ Automated Download with `kagglehub`

### Single-Command Download:
```powershell
python src/data_pipeline/kaggle_download.py --dataset kushagrapandya/visdrone-dataset
```

### Options & Arguments:
- `--dataset`: Kaggle dataset handle or alias (`visdrone`, `uavdt`, `ua_detrac`).
- `--limit`: Limit number of files to inspect for fast smoke testing (e.g. `--limit 10`).
- `--cache-dir`: Custom path for `KAGGLEHUB_CACHE` to avoid downloading to the default user cache.

### Custom Cache Directory:
```powershell
python src/data_pipeline/kaggle_download.py --dataset visdrone --cache-dir "data/raw_kaggle"
```

---

## 🗂️ Data Directory Layout

Once downloaded or processed, files are organized in standard YOLO format:

```
data/
├── dataset.yaml                 # Master YOLOv8 dataset configuration
├── processed/
│   ├── images/
│   │   ├── train/               # 80% Training images
│   │   ├── val/                 # 10% Validation images
│   │   └── test/                # 10% Testing images
│   └── labels/
│       ├── train/               # 80% Training YOLO .txt annotations
│       ├── val/                 # 10% Validation YOLO .txt annotations
│       └── test/                # 10% Testing YOLO .txt annotations
└── sample_drone_feed.mp4        # Sample video stream for edge testing
```

---

## 🏷️ Annotation Format
Each label file is a standard normalized YOLO format `.txt` file:
```
<class_id> <x_center> <y_center> <width> <height>
```
*Coordinates are normalized between 0.0 and 1.0 relative to image width and height.*

- `0`: `vehicle`
- `1`: `pedestrian`
- `2`: `cyclist`
- `3`: `traffic_signal`
