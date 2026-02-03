"""Predict movie genres using a trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict movie genres.")
    parser.add_argument(
        "plot",
        type=str,
        help="Movie plot synopsis to classify.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/genre_classifier.joblib"),
        help="Path to the trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found at {args.model}. Train the model first."
        )
    model = joblib.load(args.model)
    data = pd.DataFrame({"plot": [args.plot]})
    prediction = model.predict(data)[0]
    print(f"Predicted genre: {prediction}")


if __name__ == "__main__":
    main()
