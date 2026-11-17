import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_extractor import extract_features, FEATURE_NAMES

# Modelo de produccion: HistGradientBoosting (dataset v2) con su umbral de
# decision. Ambos se pueden cambiar por variable de entorno.
MODEL_PATH = os.getenv("MODEL_PATH", "modelos_comparacion/hist_gradient_boosting.pkl")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.31"))
# Desde esta probabilidad el riesgo es ALTO. Por debajo del umbral de decision
# es BAJO (legitima); entre ambos, MEDIO. Asi una URL clasificada como
# phishing nunca sale con riesgo BAJO.
HIGH_RISK_THRESHOLD = 0.70
# URL fija para la prediccion de calentamiento al arrancar.
WARMUP_URL = "http://secure-login.example.com/account/verify?id=123"

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        # La primera prediccion paga la inicializacion interna del modelo
        # (~2 s medidos con HistGradientBoosting). Se hace aqui, al arrancar,
        # para que no la pague la primera solicitud real.
        t0 = time.perf_counter()
        model.predict_proba(np.array(extract_features(WARMUP_URL)).reshape(1, -1))
        print(f"Modelo cargado: {MODEL_PATH} (umbral de decision: {DECISION_THRESHOLD}; "
              f"calentamiento: {(time.perf_counter() - t0) * 1000:.0f} ms)")
    else:
        print(f"ADVERTENCIA: modelo no encontrado en '{MODEL_PATH}'. "
              "Ejecuta evaluation/comparar_modelos.py primero.")
    yield


app = FastAPI(title="Phishing AI Service", version="1.1.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    is_phishing: bool
    probability: float
    risk_level: str
    features: dict


def risk_level(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    if prob >= DECISION_THRESHOLD:
        return "MEDIUM"
    return "LOW"


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta evaluation/comparar_modelos.py primero."
        )

    feats = extract_features(req.url)
    arr = np.array(feats).reshape(1, -1)

    prob = float(model.predict_proba(arr)[0][1])
    pred = int(prob >= DECISION_THRESHOLD)

    return AnalyzeResponse(
        is_phishing=bool(pred),
        probability=round(prob, 4),
        risk_level=risk_level(prob),
        features=dict(zip(FEATURE_NAMES, feats)),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "decision_threshold": DECISION_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
    }
