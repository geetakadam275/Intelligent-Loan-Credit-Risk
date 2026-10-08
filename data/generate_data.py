"""Generate a synthetic loan dataset (no real customer data) for training and demos."""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 6000


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    employment = rng.choice(
        ["Salaried", "Self-Employed", "Business", "Unemployed"], n, p=[0.55, 0.2, 0.18, 0.07]
    )
    income = np.clip(rng.lognormal(13.1, 0.55, n), 120_000, 5_000_000).round(-2)
    income = np.where(employment == "Unemployed", income * 0.3, income).round(-2)
    credit_score = np.clip(rng.normal(690, 80, n), 300, 900).astype(int)
    loan_amount = np.clip(income * rng.uniform(0.2, 3.0, n), 50_000, 8_000_000).round(-3)
    loan_term = rng.choice([12, 24, 36, 48, 60, 84, 120], n)
    liabilities = (income * rng.uniform(0, 0.8, n)).round(-2)
    late_payments = rng.poisson(1.0, n)

    # Latent risk -> default probability (so models have real signal to learn)
    z = (
        -1.0
        + 2.2 * (loan_amount / income) / 3
        + 1.8 * (liabilities / income)
        - 0.012 * (credit_score - 650)
        + 0.35 * late_payments
        + 1.2 * (employment == "Unemployed")
        + 0.3 * (employment == "Self-Employed")
        - 0.01 * (age - 40)
    )
    prob = 1 / (1 + np.exp(-(z - 3.0)))
    default = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "employment_status": employment,
            "annual_income": income,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term,
            "credit_score": credit_score,
            "existing_liabilities": liabilities,
            "late_payments": late_payments,
            "default": default,
        }
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "loans.csv"
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}  (default rate: {df['default'].mean():.1%})")
