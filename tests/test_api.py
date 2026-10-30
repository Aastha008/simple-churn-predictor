"""Tests for FastAPI endpoints."""

import pytest
import os
from fastapi.testclient import TestClient
from src.main import app, pipeline
from src.models import CustomerRecord

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: train model before tests if not present
    sample = [
        CustomerRecord(customer_id="A1", tenure_months=10, monthly_charges=50.0, total_charges=500.0, support_tickets=0, login_frequency_weekly=4.0, feature_usage_score=70.0, contract_type="one-year", payment_method="credit_card", churned=0),
        CustomerRecord(customer_id="A2", tenure_months=2, monthly_charges=90.0, total_charges=180.0, support_tickets=3, login_frequency_weekly=1.0, feature_usage_score=20.0, contract_type="month-to-month", payment_method="electronic_check", churned=1),
        CustomerRecord(customer_id="A3", tenure_months=15, monthly_charges=60.0, total_charges=900.0, support_tickets=1, login_frequency_weekly=5.0, feature_usage_score=85.0, contract_type="two-year", payment_method="bank_transfer", churned=0),
        CustomerRecord(customer_id="A4", tenure_months=1, monthly_charges=110.0, total_charges=110.0, support_tickets=5, login_frequency_weekly=0.5, feature_usage_score=10.0, contract_type="month-to-month", payment_method="electronic_check", churned=1)
    ] * 3
    pipeline.train(sample)
    yield
    # Teardown artifacts
    if os.path.exists(pipeline.model_path):
        os.remove(pipeline.model_path)
    if os.path.exists(pipeline.metrics_path):
        os.remove(pipeline.metrics_path)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "records": [
            {
                "customer_id": "TEST-999",
                "tenure_months": 5,
                "monthly_charges": 75.0,
                "total_charges": 375.0,
                "support_tickets": 2,
                "login_frequency_weekly": 3.0,
                "feature_usage_score": 50.0,
                "contract_type": "month-to-month",
                "payment_method": "credit_card"
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["customer_id"] == "TEST-999"
    assert "churn_probability" in data["predictions"][0]
    assert "risk_tier" in data["predictions"][0]


def test_evaluate_endpoint():
    payload = {
        "records": [
            {
                "customer_id": "TEST-100",
                "tenure_months": 10,
                "monthly_charges": 50.0,
                "total_charges": 500.0,
                "support_tickets": 0,
                "login_frequency_weekly": 4.0,
                "feature_usage_score": 70.0,
                "contract_type": "one-year",
                "payment_method": "credit_card",
                "churned": 0
            }
        ]
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert data["total_samples"] == 1


def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert "latest_metrics" in data
