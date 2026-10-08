"""PostgreSQL access layer (SQLAlchemy)."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL not set. Copy .env.example to .env and edit it.")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def init_db():
    """Create tables/view from sql/schema.sql."""
    sql = (ROOT / "sql" / "schema.sql").read_text()
    with get_engine().begin() as conn:
        conn.exec_driver_sql(sql)


def save_application(applicant: dict, result: dict) -> int:
    """Insert customer, application and prediction in one transaction."""
    with get_engine().begin() as conn:
        cid = conn.execute(
            text("""INSERT INTO customers (full_name, age, employment_status, annual_income)
                    VALUES (:name, :age, :emp, :inc) RETURNING customer_id"""),
            {"name": applicant["full_name"], "age": applicant["age"],
             "emp": applicant["employment_status"], "inc": applicant["annual_income"]},
        ).scalar()
        aid = conn.execute(
            text("""INSERT INTO loan_applications
                    (customer_id, loan_amount, loan_term_months, credit_score,
                     existing_liabilities, late_payments)
                    VALUES (:cid, :amt, :term, :cs, :liab, :late) RETURNING application_id"""),
            {"cid": cid, "amt": applicant["loan_amount"], "term": applicant["loan_term_months"],
             "cs": applicant["credit_score"], "liab": applicant["existing_liabilities"],
             "late": applicant["late_payments"]},
        ).scalar()
        conn.execute(
            text("""INSERT INTO predictions
                    (application_id, model_name, default_prob, risk_category, recommendation)
                    VALUES (:aid, :m, :p, :r, :rec)"""),
            {"aid": aid, "m": result["model_name"], "p": result["default_prob"],
             "r": result["risk_category"], "rec": result["recommendation"]},
        )
    return aid


def load_history(limit: int = 500) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text("SELECT * FROM v_prediction_history LIMIT :l"), conn,
                           params={"l": limit})
