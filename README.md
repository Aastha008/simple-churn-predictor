# Simple Churn Predictor

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready, lightweight machine learning service designed to analyze customer usage telemetry and account metadata for real-time churn risk prediction. Simple Churn Predictor unifies feature extraction, model training via Scikit-Learn, automated artifact serialization, and low-latency HTTP inference through FastAPI.

---

## 1. Problem Statement & Target Audience

### Problem Statement
Subscription and Software-as-a-Service (SaaS) businesses frequently discover a customer's intent to cancel too late. Traditional retention strategies rely heavily on reactive exit surveys and offboarding feedback. By the time a cancellation request is submitted, intervention efforts yield minimal success. 

Predicting customer attrition requires analyzing subtle, early warning signals embedded within multi-channel usage telemetry (e.g., declining login frequency, dropping feature usage) combined with account metadata (e.g., tenure, seat utilization, support ticket volume).

### Target Audience
* **SaaS Growth & Customer Success Teams:** Seeking automated, real-time risk scores to trigger targeted retention campaigns.
* **Data & Machine Learning Engineers:** Needing a modular, standardized blueprint for serving Scikit-Learn pipelines via lightweight REST endpoints.
* **Product Analysts:** Requiring structured batch evaluation reports (ROC-AUC, Precision, Recall) to audit churn propensity models across customer cohorts.

---

## 2. Key Features

* **Automated Data Cleaning & Feature Pipeline:** Standardizes incoming raw event logs and billing metadata, performing imputation, categorical encoding, scaling, and feature transformation.
* **Supervised Classification & Hyperparameter Tuning:** Utilizes Scikit-Learn estimators with randomized grid search optimization to output well-calibrated churn probabilities.
* **Low-Latency FastAPI Inference:** Offers single-record and bulk prediction HTTP endpoints returning churn probability scores, binary decisions, and categorized risk tiers (`LOW`, `MEDIUM`, `HIGH`).
* **Batch Evaluation & Artifact Management:** Serializes trained pipelines using Joblib with support for dynamic re-loading, versioning, and comprehensive metric evaluation (ROC-AUC, Precision, Recall, Confusion Matrix).

---

## 3. Architecture & System Design

The service follows a modular domain-driven layout separating data validation, business logic, model management, and HTTP transport layers.

```
                  +-------------------------------------------------------+
                  |                  Client Application                   |
                  +-------------------------------------------------------+
                                    |                   ^
                         HTTP POST  |                   | JSON Response
                       /predict     v                   |
+---------------------------------------------------------------------------------+
| FastAPI Service Layer (src/api.py & src/main.py)                                |
|                                                                                 |
|  +---------------------------+       +---------------------------------------+  |
|  | Request Pydantic Schemas  | ----> | Model Manager / In-Memory Artifacts   |  |
|  | (src/models.py)           |       | (Loaded via src/core.py)              |  |
|  +---------------------------+       +---------------------------------------+  |
|                                                          |                      |
|                                                          v                      |
|  +---------------------------------------------------------------------------+  |
|  | Feature Extraction & Scikit-Learn Pipeline Execution (src/core.py)        |  |
|  +---------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------+
                                                           |
                                                           v
                                         +-----------------------------------+
                                         | Model Artifact Storage (.joblib)  |
                                         +-----------------------------------+
```

### Module Responsibilities

* **`src/models.py`**: Defines strict Pydantic models for incoming event telemetry, customer account metadata, risk thresholds, inference results, evaluation metrics, and retraining payloads.
* **`src/core.py`**: Implements the feature engineering logic, Scikit-Learn training execution engine, hyperparameter tuning routines, model serialization helpers (Joblib), and risk-tier calculation routines.
* **`src/api.py`**: Contains the FastAPI route definitions handling prediction requests, health checks, model status metrics, batch performance evaluation, and trigger endpoints for model retraining.
* **`src/main.py`**: Acts as the application entrypoint. Configures server settings, loads initialized default model artifacts into state, and provides a Command Line Interface (CLI) for offline training execution.

---

## 4. Technology Stack

* **Language:** Python 3.11+
* **ML Framework:** Scikit-Learn 1.4+
* **Data Processing:** Pandas, NumPy
* **API Framework:** FastAPI, Uvicorn
* **Data Validation:** Pydantic v2
* **Serialization:** Joblib
* **Configuration:** Python-Dotenv
* **Testing & Quality:** pytest, HTTPX

---

## 5. Installation & Virtual Environment Setup

### Prerequisites
* Python 3.11 or higher
* `pip` package manager
* `git`

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dailyprojectfactory/simple-churn-predictor.git
   cd simple-churn-predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 6. Environment Variables Configuration

Copy the example environment configuration file to create your active `.env` file:

```bash
cp .env.example .env
```

### Supported Configuration Variables

| Variable | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `API_HOST` | String | `0.0.0.0` | IP interface binding for the server |
| `API_PORT` | Integer | `8000` | Port for the FastAPI HTTP server |
| `MODEL_PATH` | String | `models/churn_model.joblib` | Path to load/save model artifacts |
| `CHURN_THRESHOLD_HIGH` | Float | `0.70` | Probability score trigger for `HIGH` risk tier |
| `CHURN_THRESHOLD_MEDIUM`| Float | `0.35` | Probability score trigger for `MEDIUM` risk tier |
| `LOG_LEVEL` | String | `INFO` | Application logging verbosity (`DEBUG`, `INFO`, `WARN`, `ERROR`) |

---

## 7. Usage

### 1. Command Line Interface (CLI) Local Training

You can train a model locally using synthetic or custom datasets via `src/main.py`:

```bash
# Generate synthetic dataset and train model pipeline
python -m src.main train --output models/churn_model.joblib --samples 5000

# Evaluate an existing model artifact against test data
python -m src.main evaluate --model-path models/churn_model.joblib --data-path data/test_data.csv
```

### 2. Running the API Server

Start the production Uvicorn server via the main module:

```bash
python -m src.main serve
```

Or run Uvicorn directly:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

---

### API Endpoints & Request Examples

#### Health Check
```http
GET /health
```
*Response (`200 OK`):*
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

#### Single Prediction
```http
POST /predict
Content-Type: application/json
```

*Request Payload:*
```json
{
  "customer_id": "CUST-98421",
  "account_metadata": {
    "tenure_months": 14,
    "contract_type": "month-to-month",
    "monthly_charges": 89.50,
    "total_support_tickets": 6
  },
  "telemetry": {
    "days_since_last_login": 12,
    "weekly_active_hours": 1.5,
    "feature_usage_score": 0.22,
    "failed_payments_count": 1
  }
}
```

*Response (`200 OK`):*
```json
{
  "customer_id": "CUST-98421",
  "churn_probability": 0.784,
  "is_churn_predicted": true,
  "risk_tier": "HIGH",
  "evaluated_at": "2023-10-24T14:32:00Z"
}
```

---

#### Batch Evaluation
```http
POST /batch-evaluate
Content-Type: application/json
```

*Request Payload:*
```json
{
  "records": [
    {
      "customer_id": "CUST-1001",
      "account_metadata": { "tenure_months": 36, "contract_type": "annual", "monthly_charges": 45.0, "total_support_tickets": 0 },
      "telemetry": { "days_since_last_login": 1, "weekly_active_hours": 12.0, "feature_usage_score": 0.85, "failed_payments_count": 0 },
      "actual_churn": false
    },
    {
      "customer_id": "CUST-1002",
      "account_metadata": { "tenure_months": 2, "contract_type": "month-to-month", "monthly_charges": 115.0, "total_support_tickets": 8 },
      "telemetry": { "days_since_last_login": 20, "weekly_active_hours": 0.2, "feature_usage_score": 0.05, "failed_payments_count": 2 },
      "actual_churn": true
    }
  ]
}
```

*Response (`200 OK`):*
```json
{
  "total_evaluated": 2,
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "roc_auc": 1.0
  }
}
```

---

#### Model Retraining
```http
POST /train
Content-Type: application/json
```

*Request Payload:*
```json
{
  "hyperparameters": {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.05
  },
  "save_artifact": true
}
```

*Response (`200 OK`):*
```json
{
  "status": "success",
  "message": "Model retrained successfully",
  "metrics": {
    "accuracy": 0.912,
    "roc_auc": 0.945
  },
  "artifact_path": "models/churn_model.joblib"
}
```

---

## 8. Automated Testing Instructions

Tests are built using `pytest` and `httpx` for unit testing logic and API integration testing.

```bash
# Run all unit and API integration tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage output
pytest --cov=src --cov-report=term-missing
```

---

## 9. Project Structure Tree

```
simple-churn-predictor/
├── .env.example          # Template for environment configuration settings
├── .gitignore             # Git ignore patterns for Python, virtual environments, artifacts
├── README.md              # Complete project documentation
├── requirements.txt       # Project dependencies
├── src/                   # Source code directory
│   ├── __init__.py        # Package initialization
│   ├── api.py             # FastAPI routing and HTTP endpoint implementations
│   ├── core.py            # Feature engineering, ML training pipeline, & model execution
│   ├── main.py            # CLI commands and server entrypoint initialization
│   └── models.py          # Pydantic schemas for request/response payloads
└── tests/                 # Suite of automated test modules
    ├── __init__.py        # Test package marker
    ├── test_api.py        # API endpoint testing with TestClient
    └── test_core.py       # Unit tests for training, pipeline, and inference logic
```

---

## 10. Roadmap / Future Enhancements

- [ ] **Data Drift Monitoring:** Implement Evidently AI or Whylogs integration to detect feature drift over time.
- [ ] **Model Explainability:** Add SHAP (SHapley Additive exPlanations) values to prediction responses to highlight top factors driving churn risk per customer.
- [ ] **Database Persistence:** Integrate SQLAlchemy with PostgreSQL for logging historical predictions and actual churn outcomes.
- [ ] **Asynchronous Task Queue:** Add Celery/Redis background worker support for large batch dataset processing and long-running model retraining.
- [ ] **Containerization:** Complete Docker and Kubernetes deployment configurations with Prometheus metrics exposition.

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.