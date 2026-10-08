"""Streamlit dashboard.  Run:  streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_one

st.set_page_config(page_title="Loan Risk System", page_icon="🏦", layout="wide")
st.title("🏦 Intelligent Loan Approval & Credit Risk Prediction")

page = st.sidebar.radio("Navigate", ["Predict", "History", "Model Performance"])


def db_available() -> bool:
    try:
        from src import db
        db.get_engine().connect().close()
        return True
    except Exception:
        return False


if page == "Predict":
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Full name", "Test Applicant")
    age = c1.number_input("Age", 18, 100, 35)
    emp = c1.selectbox("Employment status", ["Salaried", "Self-Employed", "Business", "Unemployed"])
    income = c2.number_input("Annual income (₹)", 0, 50_000_000, 900_000, step=10_000)
    liab = c2.number_input("Existing liabilities (₹)", 0, 50_000_000, 100_000, step=10_000)
    late = c2.number_input("Late payments (past)", 0, 50, 0)
    amount = c3.number_input("Loan amount (₹)", 10_000, 100_000_000, 1_000_000, step=10_000)
    term = c3.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 84, 120], index=2)
    score = c3.slider("Credit score", 300, 900, 700)

    if st.button("Assess risk", type="primary"):
        applicant = {
            "full_name": name, "age": age, "employment_status": emp,
            "annual_income": income, "loan_amount": amount, "loan_term_months": term,
            "credit_score": score, "existing_liabilities": liab, "late_payments": late,
        }
        try:
            result = predict_one(applicant, load_model())
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()

        colour = {"Low": "green", "Medium": "orange", "High": "red"}[result["risk_category"]]
        m1, m2, m3 = st.columns(3)
        m1.metric("Default probability", f"{result['default_prob']:.1%}")
        m2.markdown(f"**Risk category**\n\n### :{colour}[{result['risk_category']}]")
        m3.metric("Recommendation", result["recommendation"])
        st.caption(f"Model: {result['model_name']} · EMI/Income: {result['emi_to_income']:.1%} · "
                   f"Loan/Income: {result['loan_to_income']:.2f}x")

        if db_available():
            from src import db
            aid = db.save_application(applicant, result)
            st.success(f"Saved to PostgreSQL (application #{aid}).")
        else:
            st.info("PostgreSQL not configured - prediction not saved. See README.")

elif page == "History":
    st.subheader("Past predictions")
    if not db_available():
        st.warning("Database not reachable. Set DATABASE_URL in .env and run `python -m src.init_db`.")
    else:
        from src import db
        hist = db.load_history()
        if hist.empty:
            st.info("No predictions yet.")
        else:
            st.dataframe(hist, use_container_width=True)
            st.bar_chart(hist["risk_category"].value_counts())

else:
    st.subheader("Model comparison")
    mpath = ROOT / "models" / "metrics.json"
    if not mpath.exists():
        st.warning("Run `python -m src.train` first.")
    else:
        data = json.loads(mpath.read_text())
        st.success(f"Best model (by F1): **{data['best']}**")
        st.dataframe(pd.DataFrame(data["results"]).T.round(3), use_container_width=True)
        cols = st.columns(3)
        for col, name in zip(cols, data["results"]):
            img = ROOT / "reports" / f"cm_{name.replace(' ', '_').lower()}.png"
            if img.exists():
                col.image(str(img))
