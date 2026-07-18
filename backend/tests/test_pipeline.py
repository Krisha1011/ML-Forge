import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.profiler import profile_dataset
from backend.app.cleaner import build_preprocessor
from backend.app.trainer import run_automl_training

# Mock database calls to run independent pipeline unit tests
import backend.app.database as db
db.update_run_status = lambda run_id, status, error=None: print(f"Mock status change: {status}, error={error}")
db.update_run_leaderboard = lambda run_id, leaderboard, best_model: print(f"Mock leaderboard complete. Best model: {best_model}")

class TestAutoMLPipeline(unittest.TestCase):
    def setUp(self):
        self.csv_path = "test_titanic.csv"
        # Seed generator for test stability
        np.random.seed(42)
        rows = 100
        
        # Synthesize Titanic-like dataset
        self.df = pd.DataFrame({
            "PassengerId": range(1, rows + 1),               # ID Column
            "Age": np.random.normal(28, 12, rows),             # Numeric with missing
            "Fare": np.random.exponential(35.0, rows),         # Numeric continuous
            "Sex": np.random.choice(["male", "female"], rows), # Categorical
            "Embarked": np.random.choice(["C", "Q", "S"], rows),# Categorical with nulls
            "Survived": np.random.choice([0, 1], rows)         # Discrete target
        })
        
        # Inject some NaN values to test imputer robustness
        self.df.loc[10:20, "Age"] = np.nan
        self.df.loc[30:35, "Embarked"] = np.nan
        
        self.df.to_csv(self.csv_path, index=False)
        
    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
            
    def test_profiler_logic(self):
        """Verifies column category parsing, statistics aggregation, target detection, and task estimation."""
        profile = profile_dataset(self.csv_path)
        
        self.assertEqual(profile["total_rows"], 100)
        self.assertEqual(profile["total_cols"], 6)
        
        # Checks target Column Auto-Detection
        self.assertEqual(profile["auto_target"], "Survived")
        self.assertEqual(profile["auto_problem_type"], "classification")
        
        # Checks column profiles
        col_meta = profile["profile"]
        self.assertEqual(col_meta["PassengerId"]["type"], "id")
        self.assertEqual(col_meta["Age"]["type"], "numeric")
        self.assertEqual(col_meta["Sex"]["type"], "categorical")
        self.assertTrue(col_meta["Age"]["missing_count"] > 0)
        self.assertTrue(col_meta["Embarked"]["missing_count"] > 0)
        
    def test_training_pipeline(self):
        """Validates parallel model execution, metrics evaluation, SHAP mock creation, and serialization."""
        features = ["Age", "Fare", "Sex", "Embarked"]
        run_id = "test_run_pipeline"
        
        captured_leaderboard = []
        def mock_update(run_id_arg, leaderboard, best_model_arg):
            captured_leaderboard.append(leaderboard)
            
        import backend.app.trainer as trainer
        original_update = trainer.update_run_leaderboard
        trainer.update_run_leaderboard = mock_update
        
        try:
            # Executes automl
            run_automl_training(
                run_id=run_id,
                file_path=self.csv_path,
                target="Survived",
                features=features,
                problem_type="classification",
                scaling="standard",
                imputation="median"
            )
        finally:
            trainer.update_run_leaderboard = original_update
        
        # Confirms model pipeline file got serialized
        saved_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "saved_models", run_id, "best_pipeline.pkl")
        )
        self.assertTrue(os.path.exists(saved_file), "Trained model pipeline was not saved to disk.")
        
        # Verify the captured leaderboard is populated and has explainability optimizations
        self.assertEqual(len(captured_leaderboard), 1)
        leaderboard = captured_leaderboard[0]
        self.assertTrue(len(leaderboard) > 0)
        
        # Find best model
        best_model_entry = next((model for model in leaderboard if model["is_best"]), None)
        self.assertIsNotNone(best_model_entry, "No best model identified in leaderboard.")
        
        explainability = best_model_entry.get("explainability")
        self.assertIsNotNone(explainability, "Explainability data not found for best model.")
        
        # Check limited features
        self.assertTrue(len(explainability["global_importance"]) <= 10)
        self.assertTrue(len(explainability["feature_names"]) <= 10)
        
        # Verify structure of shap_detail
        shap_detail = explainability.get("shap_detail")
        self.assertIsNotNone(shap_detail)
        
        shap_values = shap_detail.get("shap_values")
        self.assertEqual(len(shap_values), 1, "shap_values must contain exactly 1 row (the precomputed average).")
        self.assertEqual(len(shap_values[0]), len(explainability["feature_names"]), "shap_values columns must match feature_names.")
        
        sample_values = shap_detail.get("sample_values")
        self.assertEqual(len(sample_values), 0, "sample_values must be empty to save database space.")
        
        # Cleanup
        import shutil
        test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", run_id))
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_formatted_numeric_cleaning(self):
        """Verifies currency, percentages, and commas are converted to numeric and can be trained."""
        dirty_csv = "test_dirty.csv"
        df = pd.DataFrame({
            "Price": ["₹69", "₹120", "₹85.50", "₹300", "₹1,500.00", "₹45", "₹99", "₹150", "₹200", "₹250"],
            "Discount": ["10%", "15%", "0%", "20%", "50%", "5%", "10%", "12.5%", "15%", "20%"],
            "Sales": ["1,000", "1,200", "800", "500", "100", "2,000", "1,500", "1,100", "950", "900"],
            "Target": ["₹10.50", "₹18.00", "₹0.00", "₹60.00", "₹75.00", "₹2.25", "₹9.90", "₹18.75", "₹30.00", "₹50.00"]
        })
        df.to_csv(dirty_csv, index=False)
        
        try:
            # 1. Verify profiler logic detects columns as numeric and problem_type as regression
            profile = profile_dataset(dirty_csv)
            self.assertEqual(profile["profile"]["Price"]["type"], "numeric")
            self.assertEqual(profile["profile"]["Discount"]["type"], "numeric")
            self.assertEqual(profile["profile"]["Sales"]["type"], "numeric")
            self.assertEqual(profile["profile"]["Target"]["type"], "numeric")
            self.assertEqual(profile["auto_problem_type"], "regression")
            
            # 2. Verify we can run training successfully on it
            features = ["Price", "Discount", "Sales"]
            run_id = "test_run_regression"
            
            run_automl_training(
                run_id=run_id,
                file_path=dirty_csv,
                target="Target",
                features=features,
                problem_type="regression",
                scaling="standard",
                imputation="median"
            )
            
            # Confirm serialization works
            saved_file = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "saved_models", run_id, "best_pipeline.pkl")
            )
            self.assertTrue(os.path.exists(saved_file), "Trained model pipeline was not saved for regression on cleaned columns.")
            
            # Cleanup saved model directory
            import shutil
            test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", run_id))
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)
                
        finally:
            if os.path.exists(dirty_csv):
                os.remove(dirty_csv)

if __name__ == "__main__":
    unittest.main()
