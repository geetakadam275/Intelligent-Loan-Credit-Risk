"""Feature engineering + preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default"
NUMERIC = [
    "age", "annual_income", "loan_amount", "loan_term_months", "credit_score",
    "existing_liabilities", "late_payments",
    "loan_to_income", "liability_to_income", "monthly_emi_est", "emi_to_income",
]
CATEGORICAL = ["employment_status"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered ratios commonly used in credit-risk analysis."""
    df = df.copy()
    income = df["annual_income"].clip(lower=1)
    df["loan_to_income"] = df["loan_amount"] / income
    df["liability_to_income"] = df["existing_liabilities"] / income
    df["monthly_emi_est"] = df["loan_amount"] / df["loan_term_months"].clip(lower=1)
    df["emi_to_income"] = df["monthly_emi_est"] / (income / 12)
    return df


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline(
        [("imp", SimpleImputer(strategy="most_frequent")),
         ("oh", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
