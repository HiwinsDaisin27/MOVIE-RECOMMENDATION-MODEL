"""Streamlit UI for the movie genre classifier."""
from __future__ import annotations

import subprocess
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/genre_classifier.joblib")
DATA_PATH = Path("data/movies.csv")


@st.cache_resource
def load_model() -> object:
    return joblib.load(MODEL_PATH)


def train_model() -> None:
    subprocess.run(
        [
            "python",
            "src/train_genre_classifier.py",
            "--data",
            str(DATA_PATH),
            "--model-out",
            str(MODEL_PATH),
        ],
        check=True,
    )


def main() -> None:
    st.title("Movie Genre Classification")
    st.write(
        "Enter a movie plot synopsis to predict its genre using a trained ML model."
    )

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Train one to enable predictions.")
        if st.button("Train model"):
            with st.spinner("Training model..."):
                train_model()
            st.success("Model trained and saved.")

    plot_text = st.text_area(
        "Movie plot",
        height=200,
        placeholder="A detective returns to her hometown to solve a string of ritualistic murders...",
    )

    if st.button("Predict genre"):
        if not MODEL_PATH.exists():
            st.error("Train the model first, then try again.")
            return
        if not plot_text.strip():
            st.error("Please enter a plot synopsis.")
            return
        model = load_model()
        data = pd.DataFrame({"plot": [plot_text.strip()]})
        prediction = model.predict(data)[0]
        st.success(f"Predicted genre: {prediction}")


if __name__ == "__main__":
    main()
