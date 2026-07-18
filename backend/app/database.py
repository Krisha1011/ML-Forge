import os
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "mlforge"

# Initialize MongoClient with a 2-second connection timeout
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
db = client[DB_NAME]

def get_db():
    return db

# User operations
def create_user(username: str, password_hash: str):
    """Creates a new user document in MongoDB."""
    user_doc = {
        "username": username,
        "password_hash": password_hash,
        "created_at": datetime.utcnow()
    }
    result = db.users.insert_one(user_doc)
    user_doc["id"] = str(result.inserted_id)
    del user_doc["_id"]
    return user_doc

def get_user_by_username(username: str):
    """Retrieves a user by username."""
    doc = db.users.find_one({"username": username})
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

def get_user_by_id(user_id: str):
    """Retrieves a user by ID."""
    if not ObjectId.is_valid(user_id):
        return None
    doc = db.users.find_one({"_id": ObjectId(user_id)})
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

# Dataset operations
def save_dataset(name: str, sample_data: list, total_rows: int, total_cols: int, column_profile: dict, file_path: str, user_id: str):
    """Saves dataset profile metadata into MongoDB, associated with a user."""
    dataset_doc = {
        "user_id": user_id,
        "name": name,
        "sample": sample_data,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "profile": column_profile,
        "file_path": file_path
    }
    result = db.datasets.insert_one(dataset_doc)
    dataset_doc["id"] = str(result.inserted_id)
    # Convert ObjectId to string for JSON serialization
    del dataset_doc["_id"]
    return dataset_doc

def get_dataset(dataset_id: str):
    """Retrieves dataset document by ID."""
    if not ObjectId.is_valid(dataset_id):
        return None
    doc = db.datasets.find_one({"_id": ObjectId(dataset_id)})
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

def get_user_datasets(user_id: str):
    """Retrieves all datasets for a specific user."""
    cursor = db.datasets.find({"user_id": user_id}).sort("_id", -1)
    datasets = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        datasets.append(doc)
    return datasets

# Training run operations
def create_run(dataset_id: str, target: str, features: list, problem_type: str, user_id: str, status: str = "pending", selected_models: list = None):
    """Creates a new AutoML training run document, associated with a user."""
    run_doc = {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "target": target,
        "features": features,
        "problem_type": problem_type,
        "status": status,
        "selected_models": selected_models or [],
        "leaderboard": [],
        "best_model": None,
        "error": None
    }
    result = db.runs.insert_one(run_doc)
    run_doc["id"] = str(result.inserted_id)
    del run_doc["_id"]
    return run_doc

def update_run_status(run_id: str, status: str, error: str = None):
    """Updates the status of a run (e.g., training, completed, failed)."""
    if not ObjectId.is_valid(run_id):
        print(f"Mock status change for {run_id}: {status}, error={error}")
        return
    update_data = {"status": status}
    if error:
        update_data["error"] = error
    db.runs.update_one({"_id": ObjectId(run_id)}, {"$set": update_data})

def update_run_leaderboard(run_id: str, leaderboard: list, best_model: str):
    """Saves the trained models leaderboard and updates run status to completed."""
    if not ObjectId.is_valid(run_id):
        print(f"Mock leaderboard update for {run_id}: {best_model}")
        return
    db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {
            "status": "completed",
            "leaderboard": leaderboard,
            "best_model": best_model
        }}
    )

def get_run(run_id: str):
    """Retrieves a run by ID."""
    if not ObjectId.is_valid(run_id):
        return None
    doc = db.runs.find_one({"_id": ObjectId(run_id)})
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

def get_user_runs(user_id: str):
    """Retrieves all training runs for a specific user."""
    cursor = db.runs.find({"user_id": user_id}).sort("_id", -1)
    runs = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        runs.append(doc)
    return runs

# Saved Model operations
def create_saved_model(user_id: str, run_id: str, model_name: str, name: str, metrics: dict, problem_type: str, target: str):
    """Registers a new saved model in the database."""
    saved_doc = {
        "user_id": user_id,
        "run_id": run_id,
        "model_name": model_name,
        "name": name,
        "metrics": metrics,
        "problem_type": problem_type,
        "target": target,
        "created_at": datetime.utcnow()
    }
    result = db.saved_models.insert_one(saved_doc)
    saved_doc["id"] = str(result.inserted_id)
    del saved_doc["_id"]
    return saved_doc

def get_saved_model(model_id: str):
    """Retrieves a saved model document by ID."""
    if not ObjectId.is_valid(model_id):
        return None
    doc = db.saved_models.find_one({"_id": ObjectId(model_id)})
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

def get_user_saved_models(user_id: str):
    """Retrieves all saved models for a user."""
    cursor = db.saved_models.find({"user_id": user_id}).sort("_id", -1)
    models = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        models.append(doc)
    return models

def delete_saved_model(model_id: str):
    """Deletes a saved model document from the database."""
    if not ObjectId.is_valid(model_id):
        return False
    result = db.saved_models.delete_one({"_id": ObjectId(model_id)})
    return result.deleted_count > 0

