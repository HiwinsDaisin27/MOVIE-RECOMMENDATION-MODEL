FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src

RUN python src/train_genre_classifier.py --data data/movies.csv --model-out models/genre_classifier.joblib

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
