"""
Train a student exam-score predictor and save it.

What this file does:
1. Loads the sample dataset from data/students.csv
2. Splits the data into a training set (80%) and a test set (20%)
3. Trains a Linear Regression model (easy to understand for beginners)
4. Measures how well it does on data it has never seen
5. Saves the model and the metrics so the Streamlit app can use them

Run it from the project folder:
    python train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "student_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

NUMERIC_FEATURES = [
    "hours_studied",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "assignments_completed",
]
CATEGORICAL_FEATURES = [
    "extra_curricular",
    "parental_support",
]
TARGET = "exam_score"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {path}. "
            "Make sure data/students.csv is in the project folder."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    if df.empty:
        raise ValueError("The dataset is empty. Add student rows to data/students.csv.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        bad = df[REQUIRED_COLUMNS].isna().sum()
        bad = bad[bad > 0]
        raise ValueError(
            "The dataset has missing values in: "
            + ", ".join(f"{name} ({count})" for name, count in bad.items())
        )

    return df


def build_pipeline() -> Pipeline:
    # Scale numbers so large columns (like attendance) do not dominate small ones.
    # Turn text columns into 0/1 flags the model can use.
    preprocess = ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )


def evaluate(y_true, y_pred) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> int:
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Rows in dataset: {len(df)}")
    print(f"Training rows:   {len(X_train)}")
    print(f"Test rows:       {len(X_test)}")
    print("Training Linear Regression...")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    report = {
        "model_type": "Linear Regression",
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": 0.20,
        "random_state": 42,
        "train": train_metrics,
        "test": test_metrics,
        "note": (
            "Test metrics are the honest score. They use students the model "
            "did not train on. Train metrics are usually a bit better."
        ),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=== Honest evaluation (test set: unseen students) ===")
    print(f"MAE  (mean absolute error): {test_metrics['mae']:.2f} score points")
    print(f"RMSE (root mean squared error): {test_metrics['rmse']:.2f}")
    print(f"R-squared: {test_metrics['r2']:.3f}")
    print()
    print("=== Training set (for comparison, not the final score) ===")
    print(f"MAE: {train_metrics['mae']:.2f}")
    print(f"RMSE: {train_metrics['rmse']:.2f}")
    print(f"R-squared: {train_metrics['r2']:.3f}")
    print()
    print(f"Saved model to:   {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print("Done. Next step:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Unexpected error while training: {exc}", file=sys.stderr)
        raise SystemExit(1)
