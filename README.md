# Movie Genre Classification Model

This project trains a machine-learning model to classify movie genres from plot summaries.
It uses a TF-IDF text representation with a Linear Support Vector Machine classifier.

## Dataset

The starter dataset lives at `data/movies.csv` with the following columns:

- `title`: Movie title.
- `plot`: Short synopsis used for training.
- `genre`: Target genre label.

You can replace the file with a larger dataset as long as the required columns remain.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_genre_classifier.py --data data/movies.csv --model-out models/genre_classifier.joblib
```

The script prints evaluation metrics and saves the trained model.

## Predict a genre

```bash
python src/predict_genre.py "A detective follows clues across a futuristic city to stop a rogue AI."
```

## UI (Streamlit)

```bash
streamlit run src/app.py
```

The UI lets you train the model (if it does not exist) and predict genres from plot text.

## Deployment (Docker)

Build and run the container:

```bash
docker build -t movie-genre-classifier .
docker run -p 8501:8501 movie-genre-classifier
```

Then open `http://localhost:8501` in your browser.

## Notes

- Expand `data/movies.csv` with real-world data for better accuracy.
- Consider experimenting with other models such as Logistic Regression or Random Forests.
