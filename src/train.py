"""Train and compare Logistic Regression, Decision Tree and Random Forest.

Usage:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features import CATEGORICAL, NUMERIC, TARGET, add_features, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "loans.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10,
                                                min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=42),
    }


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df = add_features(pd.read_csv(DATA))
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results, fitted = {}, {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
        }
        fitted[name] = pipe
        print(f"{name:20s}", {k: round(v, 3) for k, v in results[name].items()})

        cm = confusion_matrix(y_te, pred)
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        plt.savefig(REPORTS / f"cm_{name.replace(' ', '_').lower()}.png", dpi=120)
        plt.close()

    best = max(results, key=lambda k: results[k]["f1"])
    joblib.dump({"name": best, "pipeline": fitted[best]}, MODELS / "best_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps({"best": best, "results": results}, indent=2))
    print(f"\nBest model by F1: {best}  -> saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
