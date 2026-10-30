"""FastAPI router implementation defining inference, batch evaluation, model status, and retraining endpoints."""

from fastapi import APIRouter, HTTPException, status
from src.models import (
    PredictionRequest,
    PredictionResponse,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    TrainingRequest,
    TrainingResponse,
)
from src.core import ChurnPipeline


def create_router(pipeline: ChurnPipeline) -> APIRouter:
    router = APIRouter()

    @router.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        return {"status": "healthy", "model_loaded": pipeline.model is not None}

    @router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
    def predict_churn(request: PredictionRequest):
        try:
            predictions = pipeline.predict(request.records)
            return PredictionResponse(predictions=predictions)
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/evaluate", response_model=BatchEvaluationResponse, status_code=status.HTTP_200_OK)
    def evaluate_batch(request: BatchEvaluationRequest):
        try:
            metrics, total = pipeline.evaluate_batch(request.records)
            return BatchEvaluationResponse(metrics=metrics, total_samples=total)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/train", response_model=TrainingResponse, status_code=status.HTTP_200_OK)
    def train_model(request: TrainingRequest):
        try:
            metrics = pipeline.train(request.records, test_size=request.test_size)
            return TrainingResponse(
                status="success",
                message="Model successfully trained and serialized.",
                metrics=metrics,
                model_path=pipeline.model_path
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get("/status", status_code=status.HTTP_200_OK)
    def model_status():
        try:
            metrics = pipeline.load_metrics()
            return {
                "model_loaded": pipeline.model is not None,
                "model_path": pipeline.model_path,
                "latest_metrics": metrics.model_dump()
            }
        except FileNotFoundError:
            return {
                "model_loaded": pipeline.model is not None,
                "model_path": pipeline.model_path,
                "latest_metrics": None
            }

    return router
