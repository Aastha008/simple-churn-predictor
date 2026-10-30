"""Feature engineering pipeline, Scikit-Learn training engine, persistence, and inference."""

import os
import json
import joblib
import pandas as pd
import numpy as np

from typing import List, Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from src.models import CustomerRecord, PredictionResult, EvaluationMetrics


class ChurnPipeline:
    def __init__(self, model_path: str = "models/churn_model.joblib", metrics_path: str = "models/metrics.json"):
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.model = None
        self._ensure_directories()
        if os.path.exists(self.model_path):
            self.load_model()

    def _ensure_directories(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def _build_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]))

        categorical_transformer = Pipeline(steps=([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]))

        preprocessor = ColumnTransformer(
            transformers=([
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        )
        return preprocessor

    def prepare_dataframe(self, records: List[CustomerRecord]) -> pd.DataFrame:
        data = [r.model_dump() for r in records]
        df = pd.DataFrame(data)
        
        # Feature Engineering: derived features
        if 'total_charges' in df.columns and 'tenure_months' in df.columns:
            # Avoid division by zero
            df['avg_monthly_spend_ratio'] = df['total_charges'] / (df['tenure_months'] * df['monthly_charges'] + 1e-5)
        else:
            df['avg_monthly_spend_ratio'] = 1.0

        if 'support_tickets' in df.columns and 'tenure_months' in df.columns:
            df['tickets_per_month'] = df['support_tickets'] / (df['tenure_months'] + 1.0)
        else:
            df['tickets_per_month'] = 0.0

        return df

    def train(self, records: List[CustomerRecord], test_size: float = 0.2) -> EvaluationMetrics:
        df = self.prepare_dataframe(records)
        
        if 'churned' not in df.columns or df['churned'].isnull().all():
            raise ValueError("Training records must include valid 'churned' labels.")

        # Drop identifier and target from features
        X = df.drop(columns=['customer_id', 'churned'])
        y = df['churned'].astype(int)

        numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = self._build_preprocessor(numeric_features, categorical_features)

        clf = RandomForestClassifier(random_state=42)

        pipeline = Pipeline(steps=([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ]))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = EvaluationMetrics(
            roc_auc=float(roc_auc_score(y_test, y_prob)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_test, y_pred, zero_division=0)),
            accuracy=float(accuracy_score(y_test, y_pred)),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist()
        )

        self.save_model()
        self.save_metrics(metrics)
        return metrics

    def predict(self, records: List[CustomerRecord]) -> List[PredictionResult]:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded. Please train or load a model first.")

        df = self.prepare_dataframe(records)
        X = df.drop(columns=['customer_id', 'churned'], errors='ignore')

        probabilities = self.model.predict_proba(X)[:, 1]

        results = []
        for customer_id, prob in zip(df['customer_id'], probabilities):
            risk_tier = self._assign_risk_tier(prob)
            results.append(PredictionResult(
                customer_id=customer_id,
                churn_probability=float(prob),
                risk_tier=risk_tier
            ))
        return results

    def evaluate_batch(self, records: List[CustomerRecord]) -> Tuple[EvaluationMetrics, int]:
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")

        df = self.prepare_dataframe(records)
        if 'churned' not in df.columns or df['churned'].isnull().any():
            raise ValueError("Batch evaluation records must include valid 'churned' labels for all items.")

        X = df.drop(columns=['customer_id', 'churned'], errors='ignore')
        y = df['churned'].astype(int)

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        metrics = EvaluationMetrics(
            roc_auc=float(roc_auc_score(y, y_prob)),
            precision=float(precision_score(y, y_pred, zero_division=0)),
            recall=float(recall_score(y, y_pred, zero_division=0)),
            f1_score=float(f1_score(y, y_pred, zero_division=0)),
            accuracy=float(accuracy_score(y, y_pred)),
            confusion_matrix=confusion_matrix(y, y_pred).tolist()
        )
        return metrics, len(records)

    def _assign_risk_tier(self, prob: float) -> str:
        if prob < 0.25:
            return "Low"
        elif prob < 0.50:
            return "Medium"
        elif prob < 0.75:
            return "High"
        else:
            return "Critical"

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def save_metrics(self, metrics: EvaluationMetrics):
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics.model_dump(), f, indent=2)

    def load_metrics(self) -> EvaluationMetrics:
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                data = json.load(f)
                return EvaluationMetrics(**data)
        raise FileNotFoundError("Metrics file not found. Train the model first.")
