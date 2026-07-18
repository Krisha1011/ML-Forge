import pandas as pd
import numpy as np
import re
from .cleaner import load_and_clean_csv

def detect_column_type(series: pd.Series, name: str) -> str:
    """Classifies a column type based on its dtype, cardinality, and name patterns."""
    name_lower = name.lower()
    
    # Handle single-value constant columns
    if series.nunique(dropna=True) <= 1:
        return "constant"
    
    # Handle ID columns (e.g. index, id, uuid, key, code)
    if "id" in name_lower or "uuid" in name_lower or "key" in name_lower:
        if series.nunique() == len(series) or (pd.api.types.is_integer_dtype(series.dtype) and series.nunique() > len(series) * 0.9):
            return "id"
            
    # Check for Datetime format in string columns
    if pd.api.types.is_string_dtype(series.dtype):
        # Check a sample of non-null values
        sample = series.dropna().head(10)
        if len(sample) > 0:
            is_date = True
            for val in sample:
                if not isinstance(val, str):
                    is_date = False
                    break
                # Detect YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, and standard iso dates
                if not re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,4}', val) and not re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', val):
                    is_date = False
                    break
            if is_date:
                try:
                    pd.to_datetime(sample, errors='raise')
                    return "datetime"
                except Exception:
                    pass
                    
    # Boolean columns are treated as categorical for preprocessing
    if pd.api.types.is_bool_dtype(series.dtype):
        return "categorical"

    # Numeric types
    if pd.api.types.is_numeric_dtype(series.dtype):
        # Treat low-cardinality integer columns as categorical
        if pd.api.types.is_integer_dtype(series.dtype) and series.nunique() <= 10:
            return "categorical"
        return "numeric"
        
    # Text vs Categorical for string columns
    if pd.api.types.is_string_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
        sample = series.dropna().head(100)
        if len(sample) > 0:
            avg_len = sample.astype(str).str.len().mean()
            # If average length is high and cardinality is high, it is unstructured text
            if avg_len > 50 and series.nunique() > len(series) * 0.5:
                return "text"
        return "categorical"
        
    return "categorical"

def profile_dataset(file_path: str):
    """Generates column profiles, statistics, previews, and AutoML configurations."""
    df = load_and_clean_csv(file_path)
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Limit sample size to first 50 rows for frontend display and replace NaNs with empty string
    sample_df = df.head(50).replace({np.nan: ""})
    sample_data = sample_df.to_dict(orient="records")
    
    profile = {}
    for col in df.columns:
        series = df[col]
        col_type = detect_column_type(series, col)
        
        missing_count = int(series.isnull().sum())
        missing_pct = float((missing_count / total_rows) * 100)
        
        col_profile = {
            "type": col_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "unique_count": int(series.nunique()),
        }
        
        # Calculate stats for numeric fields
        if col_type == "numeric":
            col_profile["min"] = float(series.min()) if not pd.isna(series.min()) else None
            col_profile["max"] = float(series.max()) if not pd.isna(series.max()) else None
            col_profile["mean"] = float(series.mean()) if not pd.isna(series.mean()) else None
            col_profile["median"] = float(series.median()) if not pd.isna(series.median()) else None
            col_profile["std"] = float(series.std()) if not pd.isna(series.std()) else None
            
            # Simple histogram bins for numeric columns
            clean_series = series.dropna()
            if len(clean_series) > 0:
                counts, bins = np.histogram(clean_series, bins=10)
                col_profile["histogram"] = {
                    "counts": counts.tolist(),
                    "bins": [round(float(b), 4) for b in bins.tolist()]
                }
        elif col_type == "categorical":
            # Count distribution for top 10 categories
            vc = series.value_counts().head(10)
            col_profile["top_categories"] = [
                {"category": str(k), "count": int(v)} for k, v in vc.items()
            ]
            
        profile[col] = col_profile
        
    # Auto-detect target column by name patterns
    target_col = None
    target_keywords = [
        "target", "label", "class", "output", "price", "y", "prediction",
        "diagnose", "diagnosis", "survived", "churn", "heart_disease", "outcome"
    ]
    
    for kw in target_keywords:
        for col in df.columns:
            if col.lower() == kw or col.lower().endswith("_" + kw) or col.lower().startswith(kw + "_"):
                target_col = col
                break
        if target_col:
            break
            
    if not target_col:
        target_col = df.columns[-1]  # Default to last column
        
    # Auto-detect task type (classification or regression)
    problem_type = "classification"
    target_series = df[target_col]
    
    # If target is floating point or high-cardinality integer, treat as regression
    if pd.api.types.is_float_dtype(target_series.dtype) or (pd.api.types.is_integer_dtype(target_series.dtype) and target_series.nunique() > 15):
        problem_type = "regression"
        
    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "sample": sample_data,
        "profile": profile,
        "auto_target": target_col,
        "auto_problem_type": problem_type
    }
