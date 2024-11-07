from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field, ConfigDict


class TrainingData(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    features: List[List[float]] = Field(
        ..., description="Training features as 2D array"
    )
    labels: List[float] = Field(..., description="Training labels")
    model_type: str = Field(..., description="Type of model to train")
    model_id: Optional[str] = Field(
        None, description="Optional custom model identifier"
    )
    hyperparameters: Optional[Dict[str, List[Union[int, float, str]]]] = Field(
        None, description="Optional hyperparameters"
    )


class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    features: List[List[float]] = Field(..., description="Features to predict")
    model_id: str = Field(..., description="ID of the model to use")


class ModelPerformanceMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    best_score: float = Field(..., description="Best cross-validation score")
    best_params: Dict[str, Union[int, float, str, None]] = Field(
        ..., description="Best hyperparameters"
    )


class ModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    model_type: str
    status: str
    performance_metrics: ModelPerformanceMetrics


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: List[float]
    prediction_time: float


class AvailableModelsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    available_models: List[str]
    active_models: Dict[str, str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
