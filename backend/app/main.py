import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pymongo.errors import PyMongoError
from typing import List

# Import modules
from .database import (
    save_dataset, get_dataset, create_run, get_run, create_user,
    get_user_by_username, get_user_datasets, get_user_runs,
    create_saved_model, get_saved_model, get_user_saved_models, delete_saved_model
)
from .profiler import profile_dataset
from .trainer import start_async_training, SAVED_MODELS_DIR
from .cleaner import preview_single_feature
from .schemas import (
    TrainRequest, PreviewRequest, RunResponse, UserSignup, UserLogin,
    UserResponse, TokenResponse, DatasetResponse, SaveModelRequest, ModelPredictionRequest
)
from .auth import hash_password, verify_password, create_access_token, get_current_user
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="MLForge - No-Code AutoML Platform", version="1.0.0")

@app.exception_handler(PyMongoError)
def pymongo_exception_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": "Database connection failed. Please ensure MongoDB service is running."}
    )

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory setup
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Authentication Endpoints
@app.post("/api/auth/signup", response_model=UserResponse)
def signup(request: UserSignup):
    """Registers a new user."""
    existing_user = get_user_by_username(request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username is already taken.")
    hashed_pwd = hash_password(request.password)
    user = create_user(request.username, hashed_pwd)
    return user

@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: UserLogin):
    """Authenticates user credentials and returns a JWT access token."""
    user = get_user_by_username(request.username)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = create_access_token(data={"sub": user["id"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns profile info for current authenticated user."""
    return current_user

@app.get("/api/datasets", response_model=List[DatasetResponse])
def get_my_datasets(current_user: dict = Depends(get_current_user)):
    """Returns all datasets uploaded by the current user."""
    return get_user_datasets(current_user["id"])

@app.get("/api/datasets/{dataset_id}")
def get_dataset_detail(dataset_id: str, current_user: dict = Depends(get_current_user)):
    """Returns a dataset with profile data and AutoML defaults for any uploaded CSV."""
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Dataset does not belong to you.")

    file_path = dataset.get("file_path")
    auto_target = None
    auto_problem_type = "classification"
    if file_path and os.path.exists(file_path):
        try:
            profile_res = profile_dataset(file_path)
            auto_target = profile_res["auto_target"]
            auto_problem_type = profile_res["auto_problem_type"]
            dataset["sample"] = profile_res["sample"]
            dataset["profile"] = profile_res["profile"]
            dataset["total_rows"] = profile_res["total_rows"]
            dataset["total_cols"] = profile_res["total_cols"]
        except Exception:
            pass

    return {
        **dataset,
        "auto_target": auto_target or (list(dataset.get("profile", {}).keys())[-1] if dataset.get("profile") else ""),
        "auto_problem_type": auto_problem_type,
    }

@app.get("/api/runs", response_model=List[RunResponse])
def get_my_runs(current_user: dict = Depends(get_current_user)):
    """Returns all training runs created by the current user."""
    return get_user_runs(current_user["id"])

# Dataset & AutoML Endpoints
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Handles CSV upload, profiles the data automatically, and saves to MongoDB."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")
        
    # Analyze and profile dataset
    try:
        profile_res = profile_dataset(file_path)
    except Exception as e:
        # Cleanup uploaded file if profiling fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Failed to profile dataset CSV: {str(e)}")
        
    # Save dataset profile metadata to MongoDB
    dataset = save_dataset(
        name=file.filename,
        sample_data=profile_res["sample"],
        total_rows=profile_res["total_rows"],
        total_cols=profile_res["total_cols"],
        column_profile=profile_res["profile"],
        file_path=file_path,
        user_id=current_user["id"]
    )
    
    return {
        "dataset_id": dataset["id"],
        "name": dataset["name"],
        "total_rows": dataset["total_rows"],
        "total_cols": dataset["total_cols"],
        "sample": dataset["sample"],
        "profile": dataset["profile"],
        "auto_target": profile_res["auto_target"],
        "auto_problem_type": profile_res["auto_problem_type"]
    }

@app.post("/api/train", response_model=RunResponse)
def train_model(request: TrainRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    """Initializes AutoML training run and runs training in the background."""
    dataset = get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    # Check ownership
    if dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Dataset does not belong to you.")
        
    file_path = dataset["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original CSV file not found on disk.")
        
    # Check that target is not in features
    if request.problem_type != "clustering" and request.target in request.features:
        raise HTTPException(status_code=400, detail="Target column cannot be a training feature.")
        
    # Create the run document
    run = create_run(
        dataset_id=request.dataset_id,
        target=request.target,
        features=request.features,
        problem_type=request.problem_type,
        user_id=current_user["id"],
        status="pending",
        selected_models=request.selected_models
    )
    
    # Delegate training to background task
    background_tasks.add_task(
        start_async_training,
        run_id=run["id"],
        file_path=file_path,
        target=request.target,
        features=request.features,
        problem_type=request.problem_type,
        scaling=request.scaling,
        imputation=request.imputation,
        categorical_imputation=request.categorical_imputation,
        categorical_encoding=request.categorical_encoding,
        selected_models=request.selected_models,
    )
    
    return run

@app.post("/api/datasets/{dataset_id}/preview_feature")
def preview_feature(
    dataset_id: str,
    request: PreviewRequest,
    current_user: dict = Depends(get_current_user),
):
    """Runs a single-column preprocessing pipeline and returns before/after preview rows."""
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if dataset.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Dataset does not belong to you.")

    file_path = dataset["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original CSV file not found on disk.")

    try:
        return preview_single_feature(
            file_path=file_path,
            feature=request.feature,
            numeric_imputation=request.numeric_imputation,
            scaling=request.scaling,
            categorical_imputation=request.categorical_imputation,
            categorical_encoding=request.categorical_encoding,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preview failed: {str(e)}")

@app.get("/api/runs/{run_id}", response_model=RunResponse)
def get_run_status(run_id: str, current_user: dict = Depends(get_current_user)):
    """Fetches status, metrics, leaderboard, and feature importances for a training run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="AutoML run not found.")
        
    # Check ownership
    if run.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Run does not belong to you.")
        
    return run

@app.get("/api/runs/{run_id}/models/{model_name}/download")
def download_model(run_id: str, model_name: str, current_user: dict = Depends(get_current_user)):
    """Downloads a specific model's serialized pickle pipeline file."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="AutoML run not found.")
        
    # Check ownership
    if run.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Run does not belong to you.")
        
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run is not completed.")
        
    model_slug = model_name.replace(" ", "_").lower()
    model_path = os.path.join(SAVED_MODELS_DIR, run_id, f"{model_slug}.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(SAVED_MODELS_DIR, run_id, "best_pipeline.pkl")
        
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found on server.")
        
    return FileResponse(
        path=model_path,
        filename=f"mlforge_{model_slug}_{run_id}.pkl",
        media_type="application/octet-stream"
    )

@app.get("/api/runs/{run_id}/models/{model_name}/export-code")
def export_model_code(run_id: str, model_name: str, current_user: dict = Depends(get_current_user)):
    """Generates deployable FastAPI service code for serving the selected model."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="AutoML run not found.")
        
    # Check ownership
    if run.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Run does not belong to you.")
        
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run is not completed.")
        
    target = run["target"]
    features = run["features"]
    problem_type = run["problem_type"]
    model_slug = model_name.replace(" ", "_").lower()
    response_type = "str" if problem_type == "classification" else ("int" if problem_type == "clustering" else "float")
    
    # Construct input fields for standard PredictionRequest class
    fields_code = ""
    for f in features:
        # Clean feature name for Pydantic field syntax
        f_clean = "".join([c if c.isalnum() or c == '_' else '_' for c in f])
        fields_code += f"    {f_clean}: float  # Feature column: {f}\n"
        
    code_template = f'''import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="MLForge AutoML Service",
    description="Exported model: {model_name} | Target: {target} | Type: {problem_type}",
    version="1.0.0"
)

# 1. Prediction request validation schema
class PredictionRequest(BaseModel):
{fields_code}

class PredictionResponse(BaseModel):
    prediction: {response_type}
    probabilities: Optional[Dict[str, float]] = None

# 2. Load the preprocessing and model pipeline
try:
    with open("{model_slug}.pkl", "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    pipeline = None
    print("CRITICAL: '{model_slug}.pkl' not found. Ensure it is in the active directory.")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded on server.")
    
    try:
        # Convert Pydantic request to dictionary matching actual feature names
        request_dict = request.model_dump()
        
        # Map cleaned Pydantic names back to exact training feature names
        # Features map:
        original_features = {features}
        input_dict = {{}}
        for clean_name, orig_name in zip(request_dict.keys(), original_features):
            input_dict[orig_name] = request_dict[clean_name]
            
        input_df = pd.DataFrame([input_dict])
        
        # Predict using pipeline (cleaning + scaling + classification/regression)
        pred = pipeline.predict(input_df)
        result = pred[0]
        
        # Handle label encoder mapping if classification model
        probabilities = None
        if "{problem_type}" == "classification":
            if hasattr(pipeline, "label_encoder"):
                result = pipeline.label_encoder.inverse_transform([result])[0]
                
            # Compute probabilities if the trained estimator supports it
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0]
                classes = pipeline.label_encoder.classes_ if hasattr(pipeline, "label_encoder") else range(len(proba))
                probabilities = {{str(c): float(p) for c, p in zip(classes, proba)}}
                
        # Cast to float or string or int for compatibility
        if "{problem_type}" == "clustering":
            result = int(result)
        elif isinstance(result, (int, float)):
            result = float(result)
        else:
            result = str(result)
            
        return PredictionResponse(
            prediction=result,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {{str(e)}}")

if __name__ == "__main__":
    import uvicorn
    print("Starting ML Service on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
    
    run_instructions = (
        "Deployment Guide:\n"
        "1. Create a fresh directory on your computer.\n"
        "2. Save this python script as `main.py`.\n"
        f"3. Download the pickled model pipeline from the dashboard and place it in the same directory as `{model_slug}.pkl`.\n"
        "4. Install dependencies: `pip install fastapi uvicorn pandas scikit-learn xgboost lightgbm`.\n"
        "5. Run: `python main.py`.\n"
        "6. Test the API at http://localhost:8080/docs."
    )
    
    return {
        "code": code_template,
        "instructions": run_instructions,
        "requirements": "fastapi\nuvicorn\npandas\nscikit-learn\nxgboost\nlightgbm\n"
    }

@app.post("/api/saved-models")
def save_model(request: SaveModelRequest, current_user: dict = Depends(get_current_user)):
    """Registers a specific model from a training run as a saved model."""
    run = get_run(request.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="AutoML run not found.")
        
    if run.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied: Run does not belong to you.")
        
    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="AutoML run has not completed.")
        
    # Find model in leaderboard to get its metrics
    model_metrics = None
    for m in run["leaderboard"]:
        if m["model_name"] == request.model_name:
            model_metrics = m["metrics"]
            break
            
    if model_metrics is None:
        raise HTTPException(status_code=400, detail="Model name not found in run leaderboard.")
        
    saved = create_saved_model(
        user_id=current_user["id"],
        run_id=request.run_id,
        model_name=request.model_name,
        name=request.name,
        metrics=model_metrics,
        problem_type=run["problem_type"],
        target=run["target"]
    )
    return saved

@app.get("/api/saved-models")
def list_saved_models(current_user: dict = Depends(get_current_user)):
    """Lists all registered models for the logged-in user."""
    return get_user_saved_models(current_user["id"])

@app.delete("/api/saved-models/{model_id}")
def delete_user_saved_model(model_id: str, current_user: dict = Depends(get_current_user)):
    """Deletes a saved model registration."""
    saved_model = get_saved_model(model_id)
    if not saved_model:
        raise HTTPException(status_code=404, detail="Saved model not found.")
        
    if saved_model.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied.")
        
    success = delete_saved_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete saved model.")
        
    return {"message": "Saved model deleted successfully."}

@app.get("/api/saved-models/{model_id}/details")
def get_saved_model_details(model_id: str, current_user: dict = Depends(get_current_user)):
    """Retrieves saved model metadata, run attributes, and column profile for prediction UI."""
    saved_model = get_saved_model(model_id)
    if not saved_model:
        raise HTTPException(status_code=404, detail="Saved model not found.")
        
    if saved_model.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied.")
        
    run = get_run(saved_model["run_id"])
    if not run:
        raise HTTPException(status_code=404, detail="Source AutoML run not found.")
        
    dataset = get_dataset(run["dataset_id"])
    if not dataset:
        raise HTTPException(status_code=404, detail="Source dataset not found.")
        
    # Filter dataset column profile to include only the active training features
    profile = dataset.get("profile", {})
    feature_profiles = {feat: profile[feat] for feat in run["features"] if feat in profile}
    
    return {
        "saved_model": saved_model,
        "run": run,
        "feature_profiles": feature_profiles
    }

@app.post("/api/saved-models/{model_id}/predict")
def predict_saved_model(model_id: str, request: ModelPredictionRequest, current_user: dict = Depends(get_current_user)):
    """Performs live prediction on model inputs using the registered model's pickle file."""
    saved_model = get_saved_model(model_id)
    if not saved_model:
        raise HTTPException(status_code=404, detail="Saved model not found.")
        
    if saved_model.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied.")
        
    run_id = saved_model["run_id"]
    model_name = saved_model["model_name"]
    problem_type = saved_model["problem_type"]
    
    model_slug = model_name.replace(" ", "_").lower()
    model_path = os.path.join(SAVED_MODELS_DIR, run_id, f"{model_slug}.pkl")
    
    # Fallback to best_pipeline.pkl
    if not os.path.exists(model_path):
        model_path = os.path.join(SAVED_MODELS_DIR, run_id, "best_pipeline.pkl")
        
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model pipeline file not found on server.")
        
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
            
        # Parse inputs
        input_data = request.inputs
        
        # Ensure values are typed correctly or handled
        cleaned_inputs = {}
        for key, val in input_data.items():
            if val == "" or val is None:
                cleaned_inputs[key] = None
            else:
                try:
                    cleaned_inputs[key] = float(val)
                except ValueError:
                    cleaned_inputs[key] = val
                    
        # Construct DataFrame
        input_df = pd.DataFrame([cleaned_inputs])
        
        # Run prediction
        pred = pipeline.predict(input_df)
        result = pred[0]
        
        probabilities = None
        if problem_type == "classification":
            # Map back using label encoder if available
            if hasattr(pipeline, "label_encoder"):
                result = pipeline.label_encoder.inverse_transform([result])[0]
                
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0]
                classes = pipeline.label_encoder.classes_ if hasattr(pipeline, "label_encoder") else range(len(proba))
                probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
                
        # Cast outputs
        if problem_type == "clustering":
            result = int(result)
        elif isinstance(result, (int, float, np.number)):
            result = float(result)
        else:
            result = str(result)
            
        return {
            "prediction": result,
            "probabilities": probabilities
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

