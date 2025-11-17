import os
import zipfile
from pathlib import Path
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi

from . import config


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def download_stroke_dataset(raw_dir: Optional[str] = None) -> str:
    """
    Download the Kaggle Stroke Prediction Dataset programmatically.
    Requires that Kaggle API is configured (kaggle.json in %USERPROFILE%/.kaggle or ~/.kaggle).

    Returns path to the extracted CSV file.
    """
    raw_dir = raw_dir or config.RAW_DIR
    ensure_dir(raw_dir)

    dataset_ref = "fedesoriano/stroke-prediction-dataset"
    zip_path = os.path.join(raw_dir, "stroke-prediction-dataset.zip")

    api = KaggleApi()
    api.authenticate()

    # Download only if not already present
    if not os.path.exists(zip_path):
        api.dataset_download_files(dataset_ref, path=raw_dir, quiet=False, force=True)

    # Kaggle downloads as .zip. Extract if needed.
    # If CLI created .zip, it may be named dataset.csv.zip; handle any .zip in raw_dir
    extracted_csv = None
    for fname in os.listdir(raw_dir):
        if fname.endswith(".zip"):
            zf = zipfile.ZipFile(os.path.join(raw_dir, fname))
            zf.extractall(raw_dir)
            zf.close()

    # Find the CSV
    for fname in os.listdir(raw_dir):
        if fname.lower().endswith(".csv") and "stroke" in fname.lower():
            extracted_csv = os.path.join(raw_dir, fname)
            break

    if not extracted_csv:
        # Fallback: Known CSV name in this dataset
        candidate = os.path.join(raw_dir, "healthcare-dataset-stroke-data.csv")
        if os.path.exists(candidate):
            extracted_csv = candidate

    if not extracted_csv:
        raise FileNotFoundError("Could not locate extracted stroke dataset CSV in raw directory.")

    return extracted_csv
