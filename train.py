"""
train.py — Model Training Module for Sales Prediction.

Trains three regression models, evaluates each, selects the best,
and persists the winner alongside its preprocessing artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import clean_dataframe, encode_and_scale, detect_target_column

# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────

MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  Core Training Pipeline
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """Compute R², MAE, RMSE for a fitted model."""
    preds = model.predict(X_test)
    return {
        "R² Score": round(r2_score(y_test, preds), 4),
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "predictions": preds,
    }


def train_all_models(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    End‑to‑end training pipeline.

    Returns
    -------
    results : dict with keys:
        'metrics'        – {model_name: {R², MAE, RMSE}}
        'best_name'      – name of the best model
        'best_model'     – fitted best model object
        'feature_names'  – list of feature column names
        'encoders'       – label encoders dict
        'scaler'         – fitted StandardScaler
        'X_test'         – test feature matrix
        'y_test'         – test target array
        'importances'    – feature importance array (or None)
        'target_col'     – name of the target column
    """
    # 1. Auto‑detect target
    if target_col is None:
        target_col = detect_target_column(df)

    # 2. Clean
    df = clean_dataframe(df)

    # 3. Encode & scale
    X, y, feature_names, encoders, scaler = encode_and_scale(df, target_col=target_col)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. Train & evaluate each model
    metrics = {}
    trained_models = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test)
        metrics[name] = {k: v for k, v in result.items() if k != "predictions"}
        metrics[name]["predictions"] = result["predictions"]
        trained_models[name] = model

    # 6. Select best model by R² (higher is better)
    best_name = max(metrics, key=lambda k: metrics[k]["R² Score"])
    best_model = trained_models[best_name]

    # 7. Feature importances
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_)

    # 8. Persist best model + artifacts
    save_model_artifacts(best_model, encoders, scaler, feature_names, target_col)

    return {
        "metrics": metrics,
        "best_name": best_name,
        "best_model": best_model,
        "feature_names": feature_names,
        "encoders": encoders,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "importances": importances,
        "target_col": target_col,
        "all_models": trained_models,
    }


# ──────────────────────────────────────────────────────────────
#  Persistence
# ──────────────────────────────────────────────────────────────

def save_model_artifacts(model, encoders, scaler, feature_names, target_col):
    """Save the best model and preprocessing objects to disk."""
    bundle = {
        "model": model,
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": feature_names,
        "target_col": target_col,
    }
    path = os.path.join(MODEL_DIR, "best_model_bundle.pkl")
    joblib.dump(bundle, path)
    return path


def load_model_artifacts() -> dict:
    """Load the saved model bundle from disk."""
    path = os.path.join(MODEL_DIR, "best_model_bundle.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("No trained model found. Please train a model first.")
    return joblib.load(path)
