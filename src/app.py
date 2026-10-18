"""FastAPI service that serves the heart disease model."""
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from features import FEATURES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("heart-api")

MODEL_PATH = Path("models/heart_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Heart Disease Prediction API")


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


@app.post("/predict")
def predict(patient: Patient):
    data = patient.model_dump()
    row = pd.DataFrame([data])[FEATURES]
    risk = float(model.predict_proba(row)[0, 1])
    prediction = int(risk > 0.5)
    confidence = risk if prediction == 1 else 1 - risk
    logger.info(
        "prediction=%s confidence=%.3f input=%s", prediction, confidence, data
    )
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "risk_probability": round(risk, 4),
    }
