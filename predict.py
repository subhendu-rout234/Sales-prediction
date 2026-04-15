"""
predict.py — Prediction Module for Sales Prediction.

Provides functions to:
    • Load the trained model bundle
    • Accept raw feature values and return a prediction
    • Compute a simple prediction interval (confidence range)
"""

import numpy as np
import pandas as pd

from train import load_model_artifacts
from utils import encode_and_scale


def predict_sales(input_dict: dict) -> dict:
    """
    Predict sales for a single observation.

    Parameters
    ----------
    input_dict : dict
        Feature name → value mapping. Must contain the same features
        the model was trained on.

    Returns
    -------
    result : dict with keys:
        'predicted_sales' – point estimate
        'lower_bound'     – lower 90 % interval estimate
        'upper_bound'     – upper 90 % interval estimate
        'model_type'      – string name of the model class
    """
    bundle = load_model_artifacts()
    model = bundle["model"]
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    # Build a single‑row DataFrame in the correct column order
    row = pd.DataFrame([input_dict], columns=feature_names)

    # Encode & scale using the saved artifacts (fit=False)
    X, _, _, _, _ = encode_and_scale(
        row,
        target_col=bundle["target_col"],
        fit=False,
        encoders=encoders,
        scaler=scaler,
    )

    prediction = model.predict(X)[0]

    # Simple error‑margin heuristic (±RMSE if tree model, else ±10 %)
    if hasattr(model, "estimators_"):
        # For ensemble models, use std of individual tree predictions
        if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "predict"):
            tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
            std = tree_preds.std()
        else:
            std = abs(prediction) * 0.10
    else:
        std = abs(prediction) * 0.10

    return {
        "predicted_sales": round(float(prediction), 2),
        "lower_bound": round(float(prediction - 1.645 * std), 2),
        "upper_bound": round(float(prediction + 1.645 * std), 2),
        "model_type": type(model).__name__,
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict sales for every row in a DataFrame.
    Returns the original DataFrame with a 'Predicted_Sales' column appended.
    """
    bundle = load_model_artifacts()
    model = bundle["model"]
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    pred_df = df.copy()

    # Ensure correct column order
    missing = set(feature_names) - set(pred_df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    X, _, _, _, _ = encode_and_scale(
        pred_df[feature_names],
        target_col=bundle["target_col"],
        fit=False,
        encoders=encoders,
        scaler=scaler,
    )

    pred_df["Predicted_Sales"] = np.round(model.predict(X), 2)
    return pred_df
