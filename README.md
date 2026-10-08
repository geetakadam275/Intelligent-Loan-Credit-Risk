# 🏦 Intelligent Loan Approval & Credit Risk Prediction System

A machine-learning decision-support application that predicts the likelihood of loan default,
assigns a **Low / Medium / High** credit-risk category, and stores applicants and predictions
in **PostgreSQL**. A **Streamlit** dashboard lets users score applications and review history.

**Team:** Gita Parmeshwar Kadam (PRN 23022521242057) · Asmita Balaji Raut (PRN 23022521242059) · Keshav Gajendra Rajurkar (PRN 23022521242058)

## Features
- Data preprocessing (imputation, scaling, one-hot encoding) and feature engineering
  (loan-to-income, liability-to-income, estimated EMI, EMI-to-income)
- Models: Logistic Regression, Decision Tree, Random Forest, compared on
  accuracy, precision, recall, F1 and ROC-AUC; best model (by F1) is saved automatically
- PostgreSQL storage: `customers`, `loan_applications`, `predictions` + history view
- Streamlit dashboard: Predict, History, Model Performance pages

## Project structure
```
loan-risk-system/
├── app/dashboard.py        # Streamlit UI
├── data/generate_data.py   # synthetic dataset generator
├── notebooks/eda.py        # exploratory data analysis (saves plots to reports/)
├── sql/schema.sql          # PostgreSQL schema
├── src/
│   ├── features.py         # feature engineering + preprocessing
│   ├── train.py            # model training & evaluation
│   ├── predict.py          # inference + risk categories
│   ├── db.py               # database layer
│   └── init_db.py          # create tables
├── models/  reports/       # generated outputs
├── requirements.txt
└── .env.example
```

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Generate data and train
```bash
python data/generate_data.py      # creates data/loans.csv
python notebooks/eda.py           # optional: EDA plots in reports/
python -m src.train               # trains 3 models, saves the best
```
> The dataset is synthetic. To use a real dataset (e.g. Kaggle "Credit Risk" / "German Credit"),
> map its columns to those in `data/generate_data.py` and save it as `data/loans.csv`.

### 2. PostgreSQL (optional but recommended)
```bash
# in psql:  CREATE DATABASE loan_risk;
cp .env.example .env              # edit DATABASE_URL with your password
python -m src.init_db
```

### 3. Run the dashboard
```bash
streamlit run app/dashboard.py
```
Without a database, the Predict and Model Performance pages still work; History needs PostgreSQL.

## Risk categories
| Default probability | Category | Recommendation |
|---|---|---|
| < 30% | Low | Approve |
| 30% – 60% | Medium | Manual Review |
| ≥ 60% | High | Reject |

Thresholds are set in `src/predict.py`.

## Push to GitHub
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/loan-risk-system.git
git push -u origin main
```

## Disclaimer
Academic demonstration project. Predictions are decision support only and must not be used as the sole basis for real lending decisions.
