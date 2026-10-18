FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/features.py src/package_model.py src/app.py ./src/
COPY data/processed/heart_clean.csv ./data/processed/heart_clean.csv

RUN python src/package_model.py \
    && useradd --create-home appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "--app-dir", "src", "app:app", "--host", "0.0.0.0", "--port", "8000"]
