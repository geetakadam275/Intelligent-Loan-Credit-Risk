"""Load the trained model and turn a default probability into a risk category."""
from pathlib import Path

import joblib
import pandas as pd

from src.features import CATEGORICAL, NUMERIC, add_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def categorize(prob: float):
    if prob < LOW_THRESHOLD:
        return "Low", "Approve"
    if prob < HIGH_THRESHOLD:
        return "Medium", "Manual Review"
    return "High", "Reject"


def predict_one(applicant: dict, bundle=None) -> dict:
    bundle = bundle or load_model()
    df = add_features(pd.DataFrame([applicant]))
    prob = float(bundle["pipeline"].predict_proba(df[NUMERIC + CATEGORICAL])[0, 1])
    risk, recommendation = categorize(prob)
    return {
        "model_name": bundle["name"],
        "default_prob": round(prob, 4),
        "risk_category": risk,
        "recommendation": recommendation,
        "emi_to_income": round(float(df["emi_to_income"].iloc[0]), 3),
        "loan_to_income": round(float(df["loan_to_income"].iloc[0]), 3),
    }
