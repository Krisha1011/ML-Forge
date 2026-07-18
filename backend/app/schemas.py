from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class UserSignup(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TrainRequest(BaseModel):
    dataset_id: str
    target: Optional[str] = ""
    features: List[str]
    problem_type: str = Field(..., description="Either 'classification', 'regression', or 'clustering'")
    scaling: str = Field("standard", description="scaling method: 'standard', 'minmax', 'robust', or 'none'")
    imputation: str = Field("median", description="imputation strategy: 'median', 'mean', or 'most_frequent'")
    categorical_imputation: str = Field("most_frequent", description="categorical imputation strategy: 'most_frequent' or 'constant'")
    categorical_encoding: str = Field("onehot", description="categorical encoding method: 'onehot' or 'ordinal'")
    selected_models: Optional[List[str]] = Field(default=None, description="List of model names to train")

class PreviewRequest(BaseModel):
    feature: str
    numeric_imputation: str = "median"
    scaling: str = "standard"
    categorical_imputation: str = "most_frequent"
    categorical_encoding: str = "onehot"

class RunResponse(BaseModel):
    id: str
    dataset_id: str
    target: str
    features: List[str]
    problem_type: str
    status: str
    best_model: Optional[str] = None
    error: Optional[str] = None
    leaderboard: List[Dict[str, Any]] = []
    selected_models: List[str] = []

class DatasetResponse(BaseModel):
    id: str
    name: str
    total_rows: int
    total_cols: int
    sample: Optional[List[Dict[str, Any]]] = None
    profile: Optional[Dict[str, Any]] = None

class SaveModelRequest(BaseModel):
    run_id: str
    model_name: str
    name: str

class ModelPredictionRequest(BaseModel):
    inputs: Dict[str, Any]

