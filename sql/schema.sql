-- PostgreSQL schema for the Intelligent Loan Approval & Credit Risk Prediction System
-- Create the database first:  CREATE DATABASE loan_risk;

CREATE TABLE IF NOT EXISTS customers (
    customer_id        SERIAL PRIMARY KEY,
    full_name          VARCHAR(120) NOT NULL,
    age                INT          NOT NULL CHECK (age BETWEEN 18 AND 100),
    employment_status  VARCHAR(30)  NOT NULL,
    annual_income      NUMERIC(12,2) NOT NULL CHECK (annual_income >= 0),
    created_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS loan_applications (
    application_id        SERIAL PRIMARY KEY,
    customer_id           INT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    loan_amount           NUMERIC(12,2) NOT NULL CHECK (loan_amount > 0),
    loan_term_months      INT NOT NULL CHECK (loan_term_months > 0),
    credit_score          INT NOT NULL CHECK (credit_score BETWEEN 300 AND 900),
    existing_liabilities  NUMERIC(12,2) NOT NULL DEFAULT 0,
    late_payments         INT NOT NULL DEFAULT 0,
    applied_at            TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     SERIAL PRIMARY KEY,
    application_id    INT NOT NULL REFERENCES loan_applications(application_id) ON DELETE CASCADE,
    model_name        VARCHAR(60) NOT NULL,
    default_prob      NUMERIC(6,4) NOT NULL,
    risk_category     VARCHAR(10)  NOT NULL CHECK (risk_category IN ('Low','Medium','High')),
    recommendation    VARCHAR(20)  NOT NULL,
    predicted_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_customer ON loan_applications(customer_id);
CREATE INDEX IF NOT EXISTS idx_pred_app ON predictions(application_id);

-- Handy view for the dashboard
CREATE OR REPLACE VIEW v_prediction_history AS
SELECT p.prediction_id, c.full_name, c.age, c.employment_status, c.annual_income,
       a.loan_amount, a.loan_term_months, a.credit_score, a.existing_liabilities,
       a.late_payments, p.model_name, p.default_prob, p.risk_category,
       p.recommendation, p.predicted_at
FROM predictions p
JOIN loan_applications a ON a.application_id = p.application_id
JOIN customers c ON c.customer_id = a.customer_id
ORDER BY p.predicted_at DESC;
