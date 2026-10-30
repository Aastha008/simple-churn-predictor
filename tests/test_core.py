"""Tests for core pipeline, feature engineering, and model training."""

import pytest
import os
from src.models import CustomerRecord
from src.core import ChurnPipeline

@pytest.fixture
def sample_records():
    return [
        CustomerRecord(
            customer_id="C001",
            tenure_months=12,
            monthly_charges=50.0,
            total_charges=600.0,
            support_tickets=1,
            login_frequency_weekly=5.0,
            feature_usage_score=80.0,
            contract_type="one-year",
            payment_method="credit_card",
            churned=0
        ),
        CustomerRecord(
            customer_id="C002",
            tenure_months=2,
            monthly_charges=100.0,
            total_charges=200.0,
            support_tickets=5,
            login_frequency_weekly=0.5,
            feature_usage_score=10.0,
            contract_type="month-to-month",
            payment_method="electronic_check",
            churned=1
        ),
        CustomerRecord(
            customer_id="C003",
            tenure_months=24,
            monthly_charges=40.0,
            total_charges=960.0,
            support_tickets=0,
            login_frequency_weekly=7.0,
            feature_usage_score=95.0,
            contract_type="two-year",
            payment_method="bank_transfer",
            churned=0
        ),
        CustomerRecord(
            customer_id="C004",
            tenure_months=1,
            monthly_charges=120.0,
            total_charges=120.0,
            support_tickets=4,
            login_frequency_weekly=1.0,
            feature_usage_score=15.0,
            contract_type="month-to-month",
            payment_method="electronic_check",
            churned=1
        )
    ]


def test_feature_engineering(sample_records):
    pipeline = ChurnPipeline(model_path="models/test_model.joblib", metrics_path="models/test_metrics.json")
    df = pipeline.prepare_dataframe(sample_records)
    assert "avg_monthly_spend_ratio" in df.columns
    assert "tickets_per_month" in df.columns
    assert len(df) == 4
    # Cleanup
    if os.path.exists("models/test_model.joblib"):
        os.remove("models/test_model.joblib")
    if os.path.exists("models/test_metrics.json"):
        os.remove("models/test_metrics.json")


def test_model_training_and_prediction(sample_records):
    # Duplicate records to have enough samples for train_test_split with stratify
    training_data = sample_records * 3
    pipeline = ChurnPipeline(model_path="models/test_model.joblib", metrics_path="models/test_metrics.json")
    
    metrics = pipeline.train(training_data, test_size=0.25)
    assert metrics.roc_auc >= 0.0
    assert pipeline.model is not None

    predictions = pipeline.predict(sample_records[:2])
    assert len(predictions) == 2
    assert predictions[0].customer_id == "C001"
    assert 0.0 <= predictions[0].churn_probability <= 1.0
    assert predictions[0].risk_tier in ["Low", "Medium", "High", "Critical"]

    # Cleanup
    if os.path.exists("models/test_model.joblib"):
        os.remove("models/test_model.joblib")
    if os.path.exists("models/test_metrics.json"):
        os.remove("models/test_metrics.json")
