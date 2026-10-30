"""Pydantic schemas for telemetry input, account metadata, predictions, and training."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, conint, confloat


class CustomerRecord(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure_months: conint(ge=0) = Field(..., description="Months subscribed")
    monthly_charges: confloat(ge=0.0) = Field(..., description="Monthly billing amount")
    total_charges: confloat(ge=0.0) = Field(..., description="Total billing amount accumulated")
    support_tickets: conint(ge=0) = Field(..., description="Number of support tickets raised in last 90 days")
    login_frequency_weekly: confloat(ge=0.0) = Field(..., description="Average weekly logins")
    feature_usage_score: confloat(ge=0.0, le=100.0) = Field(..., description="Score representing product feature adoption (0-100)")
    contract_type: str = Field(..., description="Contract type: month-to-month, one-year, two-year")
    payment_method: str = Field(..., description="Payment method: credit_card, bank_transfer, electronic_check")
    churned: Optional[int] = Field(None, description="Ground truth churn label (1 for churned, 0 for retained)")


class PredictionRequest(BaseModel):
    records: List[CustomerRecord] = Field(..., description="List of customer records to evaluate")


class PredictionResult(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., description="Predicted probability of churning (0.0 to 1.0)")
    risk_tier: str = Field(..., description="Risk tier: Low, Medium, High, Critical")


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class BatchEvaluationRequest(BaseModel):
    records: List[CustomerRecord] = Field(..., description="Batch of customer records with ground truth churn labels for evaluation")


class EvaluationMetrics(BaseModel):
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: List[List[int]]


class BatchEvaluationResponse(BaseModel):
    metrics: EvaluationMetrics
    total_samples: int


class TrainingRequest(BaseModel):
    records: List[CustomerRecord] = Field(..., description="Historical records used for model training")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Proportion of dataset for validation")


class TrainingResponse(BaseModel):
    status: str
    message: str
    metrics: EvaluationMetrics
    model_path: str
