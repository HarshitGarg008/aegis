from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json, joblib, os, numpy as np
import xgboost as xgb

router = APIRouter()

# Paths (relative to project root)
MODEL_PATH = "data/processed/fraud/xgb_model.bst"
SCALER_PATH = "data/processed/fraud/scaler.pkl"
FEATURES_PATH = "data/processed/fraud/features.json"

class Transaction(BaseModel):
    # Accept arbitrary JSON fields (we'll pick the numeric features needed)
    payload: dict

# Lazy-loaded artifacts
_model = None
_scaler = None
_features = None

def load_artifacts():
    global _model, _scaler, _features
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        m = xgboost = xgb.Booster()
        m.load_model(MODEL_PATH)
        _model = m
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
    if _features is None:
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"Features not found at {FEATURES_PATH}")
        with open(FEATURES_PATH) as f:
            _features = json.load(f)
    return _model, _scaler, _features

def prepare_row(payload, features):
    row = [float(payload.get(f, 0.0)) for f in features]
    return np.array(row).reshape(1, -1)

@router.post("/score")
def score_txn(body: Transaction):
    try:
        model, scaler, features = load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    X = prepare_row(body.payload, features)
    Xs = scaler.transform(X)
    dmat = xgb.DMatrix(Xs)
    prob = model.predict(dmat)[0]
    return {"fraud_probability": float(prob)}
