from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Any, Dict
import numpy as np
import time
from ..core.model_manager import ModelManager
from ..core.exceptions import ModelNotFoundError
from .schemas import (
    ModelPerformanceMetrics,
    TrainingData,
    PredictionRequest,
    ModelResponse,
    PredictionResponse,
    AvailableModelsResponse,
)
from ..core import logger
import datetime

logger = logger.get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


@router.get("/models/available", response_model=AvailableModelsResponse)
async def get_available_models(
    manager: ModelManager = Depends(ModelManager),
) -> Dict[str, Any]:
    try:
        available_models = manager.get_available_models()
        active_models = {
            model_id: model.get_model_name()
            for model_id, model in manager.models.items()
        }

        return {"available_models": available_models, "active_models": active_models}
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/train", response_model=ModelResponse)
async def train_model(
    data: TrainingData, manager: ModelManager = Depends(ModelManager)
) -> ModelResponse:
    try:
        logger.info(f"Received training request for model type: {data.model_type}")

        # Convert features and labels to numpy arrays
        X = np.array(data.features, dtype=np.float32)
        y = np.array(data.labels, dtype=np.float32)

        logger.info(f"Data shapes - Features: {X.shape}, Labels: {y.shape}")

        result = manager.create_model(
            data.model_type, X, y, data.hyperparameters, data.model_id
        )

        # Create performance metrics
        performance_metrics = ModelPerformanceMetrics(
            best_score=float(result["results"]["best_score"]),
            best_params=result["results"]["best_params"],
        )

        # Create and validate response
        response = ModelResponse(
            model_id=result["model_id"],
            model_type=data.model_type,
            status="trained",
            performance_metrics=performance_metrics,
        )

        logger.info(f"Model training successful: {response.model_dump()}")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    data: PredictionRequest,
    manager: ModelManager = Depends(ModelManager),
) -> Dict[str, Any]:
    try:
        logger.info(f"Received prediction request for model: {data.model_id}")

        # Convert features to numpy array
        X = np.array(data.features, dtype=np.float32)

        logger.info(f"Features shape: {X.shape}")

        start_time = time.time()
        predictions = manager.predict(data.model_id, X)
        prediction_time = time.time() - start_time

        return {
            "model_id": data.model_id,
            "predictions": predictions.tolist(),
            "prediction_time": prediction_time,
        }
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str, manager: ModelManager = Depends(ModelManager)
) -> Dict[str, str]:
    try:
        logger.info(f"Received delete request for model: {model_id}")
        manager.delete_model(model_id)
        return {"status": f"Model {model_id} deleted successfully"}

    except ModelNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
