"""FastAPI service that serves the heart disease model."""
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from features import FEATURES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("heart-api")

MODEL_PATH = Path("models/heart_model.joblib")
model = joblib.load(MODEL_PATH)

REQUESTS = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
LATENCY = Histogram(
    "api_request_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTIONS = Counter(
    "predictions_total", "Predictions by class", ["prediction"]
)

app = FastAPI(title="Heart Disease Prediction API")


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        REQUESTS.labels(path, str(response.status_code)).inc()
        LATENCY.labels(path).observe(time.time() - start)
    return response


class Patient(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(patient: Patient):
    data = patient.model_dump()
    row = pd.DataFrame([data])[FEATURES]
    risk = float(model.predict_proba(row)[0, 1])
    prediction = int(risk > 0.5)
    confidence = risk if prediction == 1 else 1 - risk
    PREDICTIONS.labels(str(prediction)).inc()
    logger.info(
        "prediction=%s confidence=%.3f input=%s", prediction, confidence, data
    )
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "risk_probability": round(risk, 4),
    }
