"""Exploratory data analysis. Usage: python notebooks/eda.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "reports"
out.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "loans.csv")
print(df.describe(include="all").T)
print("\nMissing values:\n", df.isna().sum())
print("\nDefault rate:", round(df["default"].mean(), 3))

sns.countplot(x="default", data=df)
plt.title("Class balance"); plt.savefig(out / "eda_class_balance.png", dpi=120); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix"); plt.tight_layout()
plt.savefig(out / "eda_correlation.png", dpi=120); plt.close()

sns.boxplot(x="default", y="credit_score", data=df)
plt.title("Credit score vs default"); plt.savefig(out / "eda_credit_score.png", dpi=120); plt.close()

df.groupby("employment_status")["default"].mean().plot(kind="bar")
plt.title("Default rate by employment"); plt.ylabel("rate"); plt.tight_layout()
plt.savefig(out / "eda_employment.png", dpi=120); plt.close()
print("\nPlots saved to reports/")
