"""
ML Deployment Project — FastAPI Backend
The backend is fully automatic:
  - It knows exactly which features the model was trained on (from metadata)
  - When a CSV arrives, it auto-detects and drops the target column
  - It auto-drops any column not in the training features
  - It predicts and returns results — no client configuration needed
"""

import os, json, io, time
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── MODEL STATE ───────────────────────────────────────────────────────────────
model    = None
metadata = {}

@app.on_event("startup")
def load_model():
    global model, metadata
    if not os.path.exists(MODEL_PATH):
        logger.warning("No model found — run train.py first.")
        return
    try:
        model = joblib.load(MODEL_PATH)
        logger.success(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}"); return

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        logger.info(
            f"📋 {metadata.get('best_model')} | "
            f"features={metadata.get('feature_names',[])} | "
            f"target='{metadata.get('target_col','')}'"
        )

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    data: List[dict] = Field(..., example=[{"feature1": 1.0, "feature2": 2.0}])

class PredictionResult(BaseModel):
    row:                 int
    prediction:          str
    confidence:          Optional[float]
    class_probabilities: Optional[dict]

class PredictResponse(BaseModel):
    model_name:    str
    n_samples:     int
    predictions:   List[PredictionResult]
    elapsed_ms:    float
    columns_used:  List[str]
    columns_dropped: List[str]

# ── CORE LOGIC ────────────────────────────────────────────────────────────────

def _check_model():
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")


def _smart_prepare(df: pd.DataFrame):
    """
    Automatically prepare ANY dataframe for prediction:

    1. Identify the target column (stored in metadata from training).
       Drop it if present — the client always uploads their full dataset.
    2. Identify feature columns (stored in metadata).
       Drop every column that is NOT a training feature.
    3. Encode any remaining object/categorical columns automatically.
    4. Fill any missing feature columns with the column median (better than 0).
    5. Reorder columns to exactly match training order.

    Returns: (X as numpy array, list of used columns, list of dropped columns)
    """
    feature_names: list = metadata.get("feature_names", [])
    target_col:    str  = metadata.get("target_col", "")

    df = df.copy()
    dropped = []

    # ── Step 1: drop target column if it exists ──────────────────────────────
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
        dropped.append(target_col)
        logger.info(f"Auto-dropped target column: '{target_col}'")

    # ── Step 2: drop columns not in training features ─────────────────────────
    if feature_names:
        unknown = [c for c in df.columns if c not in feature_names]
        if unknown:
            df = df.drop(columns=unknown)
            dropped.extend(unknown)
            logger.info(f"Auto-dropped unknown columns: {unknown}")

    # ── Step 3: encode object/category columns ────────────────────────────────
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = pd.factorize(df[col])[0].astype(float)

    # ── Step 4: fill missing feature columns with 0.0 ────────────────────────
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                logger.warning(f"Feature '{col}' missing from upload — filling with 0.0")
                df[col] = 0.0

        # ── Step 5: enforce exact column order ───────────────────────────────
        df = df[feature_names]

    if df.empty:
        raise ValueError("No rows remain after preprocessing.")

    used = list(df.columns)
    return df.values.astype(float), used, dropped


def _predict(X: np.ndarray, used_cols: list, dropped_cols: list, t0: float) -> PredictResponse:
    """Run model.predict() and package the full response."""
    classes: list = metadata.get("classes", [])
    preds_idx     = model.predict(X)

    probas = None
    try:
        probas = model.predict_proba(X)
    except AttributeError:
        pass

    results = []
    for i, idx in enumerate(preds_idx):
        label    = classes[int(idx)] if classes else str(idx)
        conf     = None
        cls_map  = None
        if probas is not None:
            conf    = round(float(probas[i].max()), 4)
            cls_map = {
                (classes[j] if classes else str(j)): round(float(probas[i][j]), 4)
                for j in range(len(probas[i]))
            }
        results.append(PredictionResult(
            row=i + 1,
            prediction=str(label),
            confidence=conf,
            class_probabilities=cls_map,
        ))

    return PredictResponse(
        model_name=metadata.get("best_model", "unknown"),
        n_samples=len(results),
        predictions=results,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        columns_used=used_cols,
        columns_dropped=dropped_cols,
    )


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "ML Prediction API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status":        "ready" if model else "model_not_loaded",
        "model_name":    metadata.get("best_model"),
        "test_accuracy": metadata.get("test_accuracy"),
        "test_f1_macro": metadata.get("test_f1_macro"),
        "features":      metadata.get("feature_names", []),
        "classes":       metadata.get("classes", []),
        "target_col":    metadata.get("target_col", ""),
        "n_features":    len(metadata.get("feature_names", [])),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_json(body: PredictRequest):
    """Predict from a JSON array of row-dicts."""
    _check_model()
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame(body.data)
        X, used, dropped = _smart_prepare(df)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing error: {e}")
    return _predict(X, used, dropped, t0)


@app.post("/predict/csv", response_model=PredictResponse)
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload ANY CSV — even if it contains the target/label column.
    The system will automatically detect and remove it, then predict.
    """
    _check_model()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")

    t0 = time.perf_counter()

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(422, "Uploaded file is empty.")
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not read CSV: {e}")

    if df.empty:
        raise HTTPException(422, "CSV has no data rows.")

    logger.info(f"📂 Received: '{file.filename}' | shape={df.shape} | cols={list(df.columns)}")

    try:
        X, used, dropped = _smart_prepare(df)
    except Exception as e:
        raise HTTPException(422,
            f"Could not prepare features: {e}. "
            f"Model expects: {metadata.get('feature_names', [])}"
        )

    return _predict(X, used, dropped, t0)


# ── GLOBAL ERROR HANDLER ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(500, {"detail": "Internal server error", "error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
