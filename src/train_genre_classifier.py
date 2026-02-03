"""Train a movie genre classification model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a movie genre classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/movies.csv"),
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/genre_classifier.joblib"),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data to hold out for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")
    data = pd.read_csv(path)
    required_columns = {"plot", "genre"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return data


def build_pipeline() -> Pipeline:
    text_features = ColumnTransformer(
        transformers=[
            (
                "plot",
                TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
                "plot",
            )
        ]
    )
    model = LinearSVC()
    return Pipeline(
        steps=[
            ("features", text_features),
            ("classifier", model),
        ]
    )


def train_model(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[Pipeline, pd.Series, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        data[["plot"]],
        data["genre"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["genre"],
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    return pipeline, x_test["plot"], y_test, y_train


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> None:
    predictions = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions))


def main() -> None:
    args = parse_args()
    data = load_data(args.data)
    model, x_test, y_test, _ = train_model(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    evaluate_model(model, x_test, y_test)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()
