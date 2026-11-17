"""Download and validate raw dataset."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import pandas as pd

UCI_RED_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/wine-quality/winequality-red.csv"
)
UCI_WHITE_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/wine-quality/winequality-white.csv"
)

EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


def download_file(url: str, dest_path: Path, token: str | None = None) -> None:
    """Download a file from url with optional authorization token."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    print(f"Downloading from {url} to {dest_path}...")
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        chunk_size = 64 * 1024
        while chunk := response.read(chunk_size):
            out_file.write(chunk)
    print(f"Saved {dest_path} ({dest_path.stat().st_size} bytes)")


def validate_dataset(file_path: Path) -> pd.DataFrame:
    """Validate that the dataset exists, is non-empty, and has expected schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Expected file {file_path} does not exist!")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Downloaded file {file_path} is empty!")

    df = pd.read_csv(file_path, sep=";")
    print(f"Validation: Loaded dataframe with shape {df.shape}")

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if len(df) < 100:
        raise ValueError(f"Dataset row count is suspiciously low: {len(df)} rows")

    print(
        f"Validation PASSED: {len(df)} rows, {len(df.columns)} columns. "
        f"Quality range: [{df['quality'].min()}, {df['quality'].max()}]"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw dataset for ML project.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw data",
    )
    parser.add_argument(
        "--include-white",
        action="store_true",
        help="Also download white wine dataset",
    )
    args = parser.parse_args()

    token = os.environ.get("DATASET_DOWNLOAD_TOKEN")

    red_dest = args.output_dir / "winequality-red.csv"
    download_file(UCI_RED_WINE_URL, red_dest, token=token)
    validate_dataset(red_dest)

    if args.include_white:
        white_dest = args.output_dir / "winequality-white.csv"
        download_file(UCI_WHITE_WINE_URL, white_dest, token=token)
        validate_dataset(white_dest)

    print("\nData download and validation finished successfully!")


if __name__ == "__main__":
    main()
