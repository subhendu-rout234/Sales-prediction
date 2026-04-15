"""
utils.py — Helper functions for the Sales Prediction project.

Contains utilities for:
    • Synthetic dataset generation
    • Data cleaning & preprocessing
    • Feature scaling & encoding
    • Visualization helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  Synthetic Dataset Generation
# ──────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic advertising → sales dataset.

    Columns produced:
        TV_Advertising        – spend on TV ads ($k)
        Radio_Advertising     – spend on radio ads ($k)
        Social_Media          – spend on social‑media ads ($k)
        Influencer_Marketing  – spend on influencer partnerships ($k)
        Platform              – categorical: 'Online', 'Retail', 'Hybrid'
        Audience_Segment      – categorical: 'Youth', 'Adults', 'Seniors'
        Sales                 – target (units sold, in $k)
    """
    rng = np.random.RandomState(seed)

    tv = rng.uniform(5, 300, n_samples)
    radio = rng.uniform(0, 60, n_samples)
    social = rng.uniform(0, 80, n_samples)
    influencer = rng.uniform(0, 50, n_samples)

    platforms = rng.choice(["Online", "Retail", "Hybrid"], n_samples, p=[0.45, 0.35, 0.20])
    segments = rng.choice(["Youth", "Adults", "Seniors"], n_samples, p=[0.30, 0.45, 0.25])

    # Platform multiplier
    platform_effect = np.where(platforms == "Online", 1.15,
                      np.where(platforms == "Hybrid", 1.05, 1.0))

    # Audience multiplier
    segment_effect = np.where(segments == "Youth", 1.10,
                    np.where(segments == "Adults", 1.0, 0.90))

    # Sales = weighted combination + noise
    sales = (
        0.045 * tv
        + 0.185 * radio
        + 0.120 * social
        + 0.095 * influencer
        + 3.5
    ) * platform_effect * segment_effect + rng.normal(0, 1.2, n_samples)

    sales = np.clip(sales, 1.0, None).round(2)

    df = pd.DataFrame({
        "TV_Advertising": np.round(tv, 1),
        "Radio_Advertising": np.round(radio, 1),
        "Social_Media": np.round(social, 1),
        "Influencer_Marketing": np.round(influencer, 1),
        "Platform": platforms,
        "Audience_Segment": segments,
        "Sales": sales,
    })

    return df


# ──────────────────────────────────────────────────────────────
#  Data Cleaning & Preprocessing
# ──────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
        1. Drop unnamed index columns.
        2. Fill numeric NaN with column median.
        3. Fill categorical NaN with mode.
        4. Drop duplicate rows.
    """
    # Drop common auto‑index columns
    drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ("float64", "int64"):
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def encode_and_scale(
    df: pd.DataFrame,
    target_col: str = "Sales",
    fit: bool = True,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
):
    """
    Encode categorical columns with LabelEncoder and scale numeric features.

    Parameters
    ----------
    df         : DataFrame (must already be cleaned).
    target_col : Name of the target column.
    fit        : If True, fit new encoders/scaler. Else, transform with existing.
    encoders   : dict of {col: LabelEncoder} for transform‑only mode.
    scaler     : fitted StandardScaler for transform‑only mode.

    Returns
    -------
    X : np.ndarray   – scaled feature matrix.
    y : np.ndarray   – target array (or None if target_col not in df).
    feature_names : list[str]
    encoders : dict
    scaler   : StandardScaler
    """
    df = df.copy()

    # Separate features and target
    y = df.pop(target_col).values if target_col in df.columns else None
    feature_names = list(df.columns)

    # Encode categoricals
    if encoders is None:
        encoders = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
            else:
                df[col] = 0  # fallback

    X = df.values.astype(float)

    # Scale
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is not None:
            X = scaler.transform(X)

    return X, y, feature_names, encoders, scaler


# ──────────────────────────────────────────────────────────────
#  Auto‑detect the target column
# ──────────────────────────────────────────────────────────────

def detect_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically determine which column is the target (Sales).
    Falls back to the last numeric column if nothing obvious is found.
    """
    for col in df.columns:
        if col.lower().strip() in ("sales", "revenue", "target", "y"):
            return col
    # Fall back to last numeric column
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric:
        return numeric[-1]
    raise ValueError("Cannot auto‑detect a numeric target column in the dataset.")


# ──────────────────────────────────────────────────────────────
#  Visualization Helpers
# ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Return a matplotlib Figure of the correlation heatmap."""
    numeric_df = df.select_dtypes(include=["number"])
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list[str]):
    """Return a horizontal bar chart of feature importances."""
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.45)))
    colors = sns.color_palette("viridis", len(feature_names))
    ax.barh(range(len(indices)), importances[indices], color=[colors[i] for i in indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


def plot_sales_distribution(df: pd.DataFrame, target_col: str = "Sales"):
    """Return a distribution plot for the target column."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram + KDE
    sns.histplot(df[target_col], kde=True, color="#6C5CE7", ax=axes[0], bins=30)
    axes[0].set_title("Sales Distribution", fontsize=13, weight="bold")
    axes[0].set_xlabel(target_col)

    # Box plot
    sns.boxplot(y=df[target_col], color="#00CEC9", ax=axes[1], width=0.35)
    axes[1].set_title("Sales Spread", fontsize=13, weight="bold")

    fig.tight_layout()
    return fig


def plot_pairwise(df: pd.DataFrame, target_col: str = "Sales"):
    """Return a pair‑plot figure (numeric columns only, max 5 cols)."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in numeric:
        # Keep target + top correlated features
        corr = df[numeric].corr()[target_col].abs().sort_values(ascending=False)
        keep = corr.head(5).index.tolist()
    else:
        keep = numeric[:5]

    g = sns.pairplot(df[keep], diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18})
    g.figure.suptitle("Pair‑wise Feature Relationships", y=1.02, fontsize=14, weight="bold")
    return g.figure


def plot_actual_vs_predicted(y_true, y_pred):
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=24, color="#6C5CE7", edgecolors="white", linewidth=0.4)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "--", color="#E17055", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Sales", fontsize=12)
    ax.set_ylabel("Predicted Sales", fontsize=12)
    ax.set_title("Actual vs Predicted", fontsize=14, weight="bold")
    ax.legend()
    fig.tight_layout()
    return fig
