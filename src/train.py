"""Train ML model on Wine Quality dataset and save artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw dataset and split into features and target."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please run 'uv run python -m src.get_data' first."
        )

    df = pd.read_csv(data_path, sep=";")
    target_col = "quality"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def train_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    """Train Random Forest regressor and calculate regression metrics."""
    print(
        f"Training RandomForestRegressor(n_estimators={n_estimators}, "
        f"random_state={random_state})..."
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
    }

    return model, metrics


def save_artifacts(
    model: RandomForestRegressor,
    metrics: dict[str, float],
    artifacts_dir: Path,
) -> None:
    """Save trained model and metrics to the specified artifacts directory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.pkl"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path} ({model_path.stat().st_size} bytes)")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Metrics saved to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on Wine Quality dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/winequality-red.csv"),
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to save model and metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=== Step 1: Loading dataset ===")
    x, y = load_data(args.data_path)
    print(f"Loaded {len(x)} samples with {x.shape[1]} features.")

    print("\n=== Step 2: Splitting dataset ===")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

    print("\n=== Step 3: Model Training and Evaluation ===")
    model, metrics = train_and_evaluate(
        x_train,
        x_test,
        y_train,
        y_test,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("\nEvaluation Results on Test Set:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value}")

    print("\n=== Step 4: Saving Artifacts ===")
    save_artifacts(model, metrics, args.artifacts_dir)

    print("\nTraining workflow completed successfully!")


if __name__ == "__main__":
    main()
