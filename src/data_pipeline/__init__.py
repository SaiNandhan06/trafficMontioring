"""
KaggleHub Automated UAV Dataset Downloader & Pipeline.
"""

from src.data_pipeline.kaggle_download import (
    download_dataset,
    inspect_dataset_directory,
    DATASET_REGISTRY,
    DATASET_ALIASES,
    SUPPORTED_DATASETS,
)

__all__ = [
    "download_dataset",
    "inspect_dataset_directory",
    "DATASET_REGISTRY",
    "DATASET_ALIASES",
    "SUPPORTED_DATASETS",
]
