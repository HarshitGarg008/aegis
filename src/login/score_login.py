#!/usr/bin/env python3
"""
Score a single login event using the IsolationForest artifacts saved as login_isolation_*.pkl
"""
import os, json, joblib, numpy as np
FEATURES = "data/processed/login/login_isolation_features.json"
SCALER = "data/processed/login/login_isolation_scaler.pkl"
MODEL = "data/processed/login/login_isolation_model.pkl"

def load_artifacts():
    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    with open(FEATURES) as f:
        features = json.load(f)
    return model, scaler, features

def prepare_row(payload, features):
    row = [float(payload.get(f, 0.0)) for f in features]
    import numpy as np
    return np.array(row).reshape(1, -1)

def score(payload):
    model, scaler, features = load_artifacts()
    X = prepare_row(payload, features)
    Xs = scaler.transform(X)
    # IsolationForest.decision_function: higher means less anomalous
    score = model.decision_function(Xs)[0]
    # convert to anomaly probability
    # invert and scale to 0..1
    prob = 1 - ((score - (-1)) / (2))  # approximate mapping
    return float(max(0.0,min(1.0,prob)))

if __name__ == "__main__":
    sample = {"time_since_last_login":10, "failed_10min":5, "device_changed":1, "impossible_travel":1, "hour":3}
    print("Anomaly probability:", score(sample))
