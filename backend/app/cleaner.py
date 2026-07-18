import pandas as pd
import numpy as np
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder

def classify_preprocessing_type(series: pd.Series, col_name: str) -> str | None:
    """Maps profiler column types to numeric/categorical preprocessing, or None if dropped."""
    from .profiler import detect_column_type
    col_type = detect_column_type(series, col_name)
    if col_type == "numeric":
        return "numeric"
    if col_type in ("categorical", "datetime", "text"):
        return "categorical"
    return None

def split_features_by_type(df: pd.DataFrame, features: list) -> tuple[list, list]:
    """Splits feature columns into numeric and categorical lists for preprocessing."""
    numeric_features = []
    categorical_features = []
    for col in features:
        prep_type = classify_preprocessing_type(df[col], col)
        if prep_type == "numeric":
            numeric_features.append(col)
        elif prep_type == "categorical":
            categorical_features.append(col)
    return numeric_features, categorical_features

def prepare_feature_column(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Normalizes a single feature column so sklearn imputers/encoders accept any CSV dtype."""
    col = df[[feature]].copy()
    series = col[feature]
    if pd.api.types.is_bool_dtype(series.dtype):
        col[feature] = series.map({True: "True", False: "False"}).astype(object)
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        col[feature] = series.astype(str).replace("NaT", np.nan)
    elif pd.api.types.is_numeric_dtype(series.dtype):
        col[feature] = series
    else:
        col[feature] = series.astype(object).where(series.notna(), np.nan)
    return col

def prepare_features_dataframe(df: pd.DataFrame, features: list, categorical_features: list) -> pd.DataFrame:
    """Prepares all selected features, normalizing categorical columns for sklearn pipelines."""
    X = df[features].copy()
    for col in categorical_features:
        prepared = prepare_feature_column(X, col)
        X[col] = prepared[col]
    return X

def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
    numeric_imputation: str = "median",
    scaling: str = "standard",
    categorical_imputation: str = "most_frequent",
    categorical_encoding: str = "onehot"
):
    """
    Builds a scikit-learn ColumnTransformer that cleans and processes
    both numerical and categorical features.
    """
    # Define Numerical Pipeline
    num_steps = [("imputer", SimpleImputer(strategy=numeric_imputation))]
    
    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif scaling == "robust":
        num_steps.append(("scaler", RobustScaler()))
    # If "none", we just do imputation without scaling
    
    num_pipeline = Pipeline(num_steps)
    
    # Define Categorical Pipeline
    cat_imputer_strategy = categorical_imputation if categorical_imputation in ["most_frequent", "constant"] else "most_frequent"
    if cat_imputer_strategy == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        
    if categorical_encoding == "ordinal":
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
        
    cat_steps = [
        ("imputer", cat_imputer),
        ("encoder", cat_encoder)
    ]
    
    cat_pipeline = Pipeline(cat_steps)
    
    # Assemble preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"  # ID, constant, or unselected columns will be dropped
    )
    
    return preprocessor

def clean_numeric_strings(series: pd.Series) -> pd.Series:
    """Checks if an object or string column is formatted numeric data (currency, percentage, comma separators)
    and converts it to float if > 80% of non-null values can be parsed.
    """
    if not pd.api.types.is_string_dtype(series.dtype):
        return series
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return series
    
    null_placeholders = {'', 'n/a', 'na', 'null', 'none', '-', 'nan'}
    valid_series = clean_series[~clean_series.astype(str).str.strip().str.lower().isin(null_placeholders)]
    if len(valid_series) == 0:
        return series
        
    sample = valid_series.head(100)
    success_count = 0
    
    for val in sample:
        if not isinstance(val, str):
            continue
        # Strip currency symbols ($, ₹, €, £, ¥), percent signs, commas, and whitespace
        stripped = re.sub(r'[₹$€£¥%,\s]', '', val)
        try:
            float(stripped)
            success_count += 1
        except ValueError:
            pass
            
    # If most values can be parsed, convert the series
    if len(sample) > 0 and (success_count / len(sample)) > 0.8:
        def convert_val(x):
            if pd.isna(x):
                return x
            if not isinstance(x, str):
                try:
                    return float(x)
                except:
                    return np.nan
            val_str = x.strip().lower()
            if val_str in null_placeholders:
                return np.nan
            stripped = re.sub(r'[₹$€£¥%,\s]', '', x)
            try:
                return float(stripped)
            except ValueError:
                return np.nan
        return pd.to_numeric(series.apply(convert_val), errors='coerce')
        
    return series

def preview_single_feature(
    file_path: str,
    feature: str,
    numeric_imputation: str = "median",
    scaling: str = "standard",
    categorical_imputation: str = "most_frequent",
    categorical_encoding: str = "onehot",
    n_rows: int = 10,
):
    """Fits a single-column preprocessing pipeline and returns before/after preview rows."""
    df = load_and_clean_csv(file_path)
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataset.")

    series = df[feature]
    from .profiler import detect_column_type
    profiler_type = detect_column_type(series, feature)
    prep_type = classify_preprocessing_type(series, feature)

    if prep_type is None:
        return {
            "feature": feature,
            "column_type": profiler_type,
            "preview_available": False,
            "message": (
                f"'{feature}' is classified as {profiler_type} and is dropped during training. "
                "No preprocessing transformation applies."
            ),
            "before": [],
            "after_columns": [],
            "after": [],
        }

    is_numeric = prep_type == "numeric"
    full_col = prepare_feature_column(df, feature)

    if is_numeric:
        steps = [("imputer", SimpleImputer(strategy=numeric_imputation))]
        if scaling == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaling == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaling == "robust":
            steps.append(("scaler", RobustScaler()))
        pipeline = Pipeline(steps)
        column_type = "numeric"
    else:
        cat_imputer_strategy = categorical_imputation if categorical_imputation in ["most_frequent", "constant"] else "most_frequent"
        if cat_imputer_strategy == "constant":
            cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        else:
            cat_imputer = SimpleImputer(strategy="most_frequent")

        if categorical_encoding == "ordinal":
            cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)

        pipeline = Pipeline([
            ("imputer", cat_imputer),
            ("encoder", cat_encoder),
        ])
        column_type = "categorical"

    pipeline.fit(full_col)
    transformed = pipeline.transform(full_col.head(n_rows))

    if is_numeric:
        after_columns = [feature]
    else:
        encoder = pipeline.named_steps["encoder"]
        after_columns = list(encoder.get_feature_names_out([feature]))

    before = []
    for val in df[feature].head(n_rows).tolist():
        if pd.isna(val):
            before.append(None)
        elif isinstance(val, (np.integer, np.floating)):
            before.append(float(val) if isinstance(val, np.floating) else int(val))
        else:
            before.append(str(val))

    after = []
    for row in transformed:
        row_dict = {}
        for col_name, val in zip(after_columns, row):
            if isinstance(val, (np.integer, np.floating)):
                row_dict[col_name] = round(float(val), 4)
            else:
                row_dict[col_name] = val
        after.append(row_dict)

    return {
        "feature": feature,
        "column_type": column_type,
        "preview_available": True,
        "before": before,
        "after_columns": after_columns,
        "after": after,
    }

def load_and_clean_csv(file_path: str) -> pd.DataFrame:
    """Reads a CSV file and cleans any formatted numeric string columns to floats."""
    df = pd.read_csv(file_path)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = clean_numeric_strings(df[col])
    return df

