"""
app.py — Streamlit Frontend for the Sales Prediction Web Application.

Run with:
    streamlit run app.py
"""

# ──────────────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────────────

import os
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import (
    generate_synthetic_dataset,
    clean_dataframe,
    detect_target_column,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_sales_distribution,
    plot_pairwise,
    plot_actual_vs_predicted,
)
from train import train_all_models, MODEL_DIR, load_model_artifacts
from predict import predict_sales


# ──────────────────────────────────────────────────────────────
#  Page Config & Global Styling
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sales Prediction AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject premium custom CSS ──
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0ff;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #e0e0ff !important;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea55;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(102,126,234,0.10);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        color: #a0a8d8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700;
        color: #667eea !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(102,126,234,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.45);
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(0,184,148,0.3);
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,184,148,0.45);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* ── Header banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 30px rgba(102,126,234,0.25);
    }
    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        color: white !important;
    }
    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        color: #e8e8ff !important;
    }

    /* ── Success / prediction box ── */
    .prediction-result {
        background: linear-gradient(135deg, #00b89422, #00cec922);
        border: 1px solid #00b89466;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,184,148,0.12);
    }
    .prediction-result h2 {
        color: #00b894 !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .prediction-result p {
        color: #636e72;
        margin: 0.3rem 0 0 0;
    }

    /* ── Info cards ── */
    .info-card {
        background: linear-gradient(135deg, #dfe6e922, #ffffff44);
        border: 1px solid #b2bec344;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* ── Problem statement callout ── */
    .problem-statement {
        background: linear-gradient(135deg, #fdcb6e22, #e1700522);
        border-left: 4px solid #fdcb6e;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
        font-size: 1.02rem;
        line-height: 1.6;
    }

    /* ── Summary card ── */
    .summary-card {
        background: linear-gradient(135deg, #a29bfe22, #6c5ce722);
        border: 1px solid #a29bfe44;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 0.8rem 0;
    }
    .summary-card h4 {
        margin: 0 0 0.5rem 0;
        color: #a29bfe !important;
    }

    /* ── Best model highlight ── */
    .best-model-banner {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,184,148,0.25);
    }

    /* ── Column description table ── */
    .col-desc-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0;
    }
    .col-desc-table th {
        background: #667eea33;
        color: #667eea;
        padding: 10px 14px;
        text-align: left;
        font-weight: 700;
        border-bottom: 2px solid #667eea55;
    }
    .col-desc-table td {
        padding: 8px 14px;
        border-bottom: 1px solid #dfe6e944;
    }
    .col-desc-table tr:hover td {
        background: #667eea0a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
#  Column Descriptions — used for dataset explanation section
# ──────────────────────────────────────────────────────────────

# Mapping of common column names to plain-English descriptions
COLUMN_DESCRIPTIONS = {
    # Real Advertising.csv columns
    "tv": "Budget spent on TV advertising (in $1,000s)",
    "radio": "Budget spent on Radio advertising (in $1,000s)",
    "newspaper": "Budget spent on Newspaper advertising (in $1,000s)",
    "sales": "Product sales revenue (target variable, in $1,000s)",
    # Synthetic dataset columns
    "tv_advertising": "Budget spent on TV advertising campaigns (in $1,000s)",
    "radio_advertising": "Budget spent on Radio advertising campaigns (in $1,000s)",
    "social_media": "Budget spent on Social Media advertising (in $1,000s)",
    "influencer_marketing": "Budget spent on Influencer Marketing partnerships (in $1,000s)",
    "platform": "Sales platform — Online, Retail, or Hybrid (categorical)",
    "audience_segment": "Target audience — Youth, Adults, or Seniors (categorical)",
}


def get_column_description(col_name: str) -> str:
    """Return a human-readable description for a column, or a sensible default."""
    desc = COLUMN_DESCRIPTIONS.get(col_name.lower().strip())
    if desc:
        return desc
    if col_name.lower().strip() in ("sales", "revenue", "target", "y"):
        return "Target variable to predict"
    return "Feature variable used for prediction"


# ──────────────────────────────────────────────────────────────
#  Session State Initialisation
# ──────────────────────────────────────────────────────────────

def init_state():
    """Initialise all session-state keys with safe defaults."""
    defaults = {
        "df": None,
        "df_source": None,           # 'uploaded' | 'synthetic'
        "target_col": None,
        "training_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────────────────────
#  Sidebar — Navigation & Dataset Controls
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Sales Predictor")
    st.markdown("---")

    # Navigation radio — now includes Summary page
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 EDA & Visualizations",
            "🤖 Train Model",
            "🎯 Predict Sales",
            "📝 Project Summary",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📂 Dataset")

    # ── CSV Upload with error handling ──
    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Validate: must have at least 2 columns and 10 rows
            if df.shape[1] < 2:
                st.error("❌ CSV must have at least 2 columns (features + target).")
            elif df.shape[0] < 10:
                st.error("❌ CSV must have at least 10 rows for meaningful analysis.")
            elif df.select_dtypes(include=["number"]).shape[1] < 1:
                st.error("❌ CSV must contain at least one numeric column.")
            else:
                st.session_state.df = df
                st.session_state.df_source = "uploaded"
                st.session_state.target_col = detect_target_column(df)
                st.success(f"✅ Loaded **{len(df)}** rows, **{df.shape[1]}** columns")
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty. Please upload a valid CSV.")
        except pd.errors.ParserError:
            st.error("❌ Could not parse the file. Ensure it is a valid CSV.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

    # ── Generate Synthetic Data ──
    if st.button("🔄 Generate Synthetic Data"):
        df = generate_synthetic_dataset()
        st.session_state.df = df
        st.session_state.df_source = "synthetic"
        st.session_state.target_col = "Sales"
        st.success(f"✅ Generated **{len(df)}** rows")

    # ── Dataset status indicator ──
    if st.session_state.df is not None:
        src = st.session_state.df_source
        st.info(
            f"Active: **{'Uploaded' if src == 'uploaded' else 'Synthetic'}** dataset  \n"
            f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns"
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;opacity:0.5;'>"
        "Built with Streamlit & Scikit-Learn</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════

if page == "🏠 Home":

    # ── Hero Banner ──
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📈 Sales Prediction AI</h1>
            <p>Upload your advertising dataset or generate synthetic data — train ML models,
            visualise insights, and predict future sales in seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 1️⃣  Problem Statement ──
    st.markdown("### 🎯 Problem Statement")
    st.markdown(
        """
        <div class="problem-statement">
            <strong>Objective:</strong> This project predicts product sales based on advertising
            budgets using machine learning models such as <b>Linear Regression</b>,
            <b>Random Forest</b>, and <b>Gradient Boosting</b>.<br><br>
            Businesses invest in multiple advertising channels (TV, Radio, Social Media, etc.).
            Understanding <em>which channel drives the most sales</em> enables smarter budget
            allocation and higher ROI. This application automates that analysis end-to-end.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Feature cards ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4>📊 Exploratory Analysis</h4>
                <p>Correlation heatmaps, scatter plots, distributions & pair plots.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4>🤖 Auto ML Training</h4>
                <p>Train 3 models, compare R²/MAE/RMSE, auto-select the best.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h4>🎯 Instant Prediction</h4>
                <p>Adjust sliders and get real-time sales prediction with confidence range.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 2️⃣  Dataset Explanation Section ──
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        target_col = st.session_state.target_col

        # --- Overview metrics ---
        st.markdown("### 📋 Dataset Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📐 Rows", df.shape[0])
        m2.metric("📏 Columns", df.shape[1])
        m3.metric("🎯 Target Column", target_col)
        m4.metric("❓ Missing Values", int(df.isnull().sum().sum()))

        st.markdown(f"**Column Names:** {', '.join(df.columns.tolist())}")

        st.markdown("")  # spacer

        # --- Column descriptions table ---
        st.markdown("### 📖 Column Descriptions")
        desc_rows = ""
        for col in df.columns:
            # Skip unnamed index columns
            if "unnamed" in col.lower():
                continue
            dtype_badge = "🔢 Numeric" if df[col].dtype in ("float64", "int64") else "🏷️ Categorical"
            role = "🎯 **Target**" if col == target_col else "📊 Feature"
            desc_rows += (
                f"<tr>"
                f"<td><b>{col}</b></td>"
                f"<td>{get_column_description(col)}</td>"
                f"<td>{dtype_badge}</td>"
                f"<td>{role}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <table class="col-desc-table">
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Description</th>
                        <th>Type</th>
                        <th>Role</th>
                    </tr>
                </thead>
                <tbody>{desc_rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # --- Data preview, statistics, info tabs ---
        st.markdown("### 🗂️ Dataset Preview")
        tab_preview, tab_stats, tab_info = st.tabs(
            ["📋 First 5 Rows", "📈 Descriptive Statistics", "ℹ️ Data Info"]
        )

        with tab_preview:
            st.dataframe(df.head(5), width="stretch")

        with tab_stats:
            st.dataframe(
                df.describe().T.style.format("{:.2f}"),
                width="stretch",
            )

        with tab_info:
            buf = io.StringIO()
            df.info(buf=buf)
            st.code(buf.getvalue(), language="text")

            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.warning("⚠️ Missing values detected:")
                st.dataframe(missing[missing > 0].rename("Missing Count"))
            else:
                st.success("✅ No missing values in the dataset!")
    else:
        st.info("👈 Upload a CSV or click **Generate Synthetic Data** in the sidebar to get started.")


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — EDA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

elif page == "📊 EDA & Visualizations":

    st.markdown(
        """
        <div class="hero-banner">
            <h1>📊 Exploratory Data Analysis</h1>
            <p>Visualise patterns, correlations, and distributions to understand what drives sales.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Guard: no data ──
    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload or generate data from the sidebar.")
        st.stop()

    df = clean_dataframe(st.session_state.df.copy())
    target_col = st.session_state.target_col

    # ── Tabs for different EDA sections ──
    tab_corr, tab_scatter, tab_dist, tab_pair = st.tabs(
        ["🔥 Correlation Heatmap", "📈 Scatter Plots (Ads vs Sales)", "📊 Distribution", "🔗 Pair Plot"]
    )

    # ── 3a. Correlation Heatmap ──
    with tab_corr:
        st.markdown("#### 🔥 Correlation Heatmap")
        st.caption(
            "Shows the linear relationship between every pair of numeric features. "
            "Values close to **+1** indicate strong positive correlation; close to **-1** indicate "
            "strong negative correlation."
        )
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)
        plt.close(fig)

        # Auto-generated insight
        numeric_df = df.select_dtypes(include=["number"])
        if target_col in numeric_df.columns:
            corr_with_target = numeric_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
            top_feat = corr_with_target.index[0]
            top_val = corr_with_target.iloc[0]
            st.info(
                f"💡 **Insight:** *{top_feat}* has the strongest correlation with *{target_col}* "
                f"(r = {top_val:.3f}). This feature is likely the most influential predictor."
            )

    # ── 3b. Scatter Plots — Ads vs Sales ──
    with tab_scatter:
        st.markdown("#### 📈 Advertising Spend vs Sales")
        st.caption("Each dot represents one observation. The dashed trend line shows the general direction.")

        numeric_cols = [c for c in df.select_dtypes("number").columns if c != target_col]

        if numeric_cols:
            # Show scatter for each numeric feature
            n_cols = min(len(numeric_cols), 3)
            for row_start in range(0, len(numeric_cols), n_cols):
                cols = st.columns(n_cols)
                for j, feat in enumerate(numeric_cols[row_start:row_start + n_cols]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.scatter(
                            df[feat], df[target_col],
                            alpha=0.45, s=20, color="#6C5CE7",
                            edgecolors="white", linewidth=0.3,
                        )
                        # Trend line
                        z = np.polyfit(df[feat], df[target_col], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
                        ax.plot(x_line, p(x_line), "--", color="#E17055", linewidth=2)
                        ax.set_xlabel(feat, fontsize=10)
                        ax.set_ylabel(target_col, fontsize=10)
                        ax.set_title(f"{feat} vs {target_col}", fontsize=11, weight="bold")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.info("No numeric feature columns available for scatter plots.")

    # ── 3c. Distribution ──
    with tab_dist:
        st.markdown(f"#### 📊 {target_col} Distribution")
        st.caption(
            "The histogram shows how sales values are spread. "
            "The box plot highlights the median, quartiles, and any outliers."
        )
        fig = plot_sales_distribution(df, target_col)
        st.pyplot(fig)
        plt.close(fig)

    # ── 3d. Pair Plot ──
    with tab_pair:
        st.markdown("#### 🔗 Pair-wise Feature Relationships")
        st.caption(
            "A matrix of scatter plots and density curves for the top correlated features. "
            "Useful for spotting multi-feature interactions."
        )
        with st.spinner("Generating pair plot …"):
            fig = plot_pairwise(df, target_col)
            st.pyplot(fig)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════

elif page == "🤖 Train Model":

    st.markdown(
        """
        <div class="hero-banner">
            <h1>🤖 Model Training</h1>
            <p>Train three regression models, compare performance metrics, and auto-select the best.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Guard: no data ──
    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Please upload or generate data from the sidebar.")
        st.stop()

    # ── 4. Training configuration ──
    st.markdown("### ⚙️ Training Configuration")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        test_size = st.slider(
            "Test split ratio (proportion of data held out for evaluation)",
            0.10, 0.40, 0.20, 0.05,
        )
    with col_r:
        st.markdown("#### 🎯 Target Column")
        st.code(st.session_state.target_col)

    # Models being trained
    st.markdown("#### 🧠 Models to Train")
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(
        '<div class="info-card"><h4>1. Linear Regression</h4>'
        "<p>Simple baseline model that fits a straight line.</p></div>",
        unsafe_allow_html=True,
    )
    mc2.markdown(
        '<div class="info-card"><h4>2. Random Forest</h4>'
        "<p>Ensemble of 200 decision trees for robust predictions.</p></div>",
        unsafe_allow_html=True,
    )
    mc3.markdown(
        '<div class="info-card"><h4>3. Gradient Boosting</h4>'
        "<p>Sequential boosted trees that learn from previous errors.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Train button with detailed spinner ──
    if st.button("🚀 Train All Models", use_container_width=True):
        with st.spinner("Training models, please wait..."):
            progress_bar = st.progress(0, text="Preparing data …")
            status_text = st.empty()

            try:
                # Step 1: Data preparation (show progress)
                status_text.info("🔄 **Step 1/4:** Cleaning and preprocessing data …")
                time.sleep(0.3)
                progress_bar.progress(15, text="Cleaning data …")

                # Step 2: Encoding & scaling
                status_text.info("🔄 **Step 2/4:** Encoding categorical variables & scaling features …")
                time.sleep(0.3)
                progress_bar.progress(30, text="Encoding & scaling …")

                # Step 3: Training
                status_text.info("🔄 **Step 3/4:** Training Linear Regression, Random Forest & Gradient Boosting …")
                progress_bar.progress(50, text="Training models …")

                results = train_all_models(
                    st.session_state.df.copy(),
                    target_col=st.session_state.target_col,
                    test_size=test_size,
                )
                st.session_state.training_results = results

                # Step 4: Evaluation
                status_text.info("🔄 **Step 4/4:** Evaluating models and selecting the best …")
                progress_bar.progress(90, text="Evaluating …")
                time.sleep(0.3)
                progress_bar.progress(100, text="Complete!")
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ Training complete! Best model selected: **{results['best_name']}**")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Training failed: {e}")
                st.stop()

    # ── 5️⃣  Model Performance Results ──
    results = st.session_state.training_results 

    if results is not None:
        st.markdown("---")

        # ── Best Model Banner ──
        best = results["best_name"]
        st.markdown(
            f'<div class="best-model-banner">'
            f'🏆 Best Model Selected: {best}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── 5a. Metric cards for best model ──
        st.markdown("### 📏 Best Model Performance Metrics")
        best_metrics = {k: v for k, v in results["metrics"][best].items() if k != "predictions"}

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", best_metrics["R² Score"])
        c2.metric("MAE", best_metrics["MAE"])
        c3.metric("RMSE", best_metrics["RMSE"])

        # Explain each metric
        with st.expander("📚 What do these metrics mean?"):
            st.markdown(
                """
                | Metric | Full Name | What It Tells You |
                |---|---|---|
                | **R² Score** | Coefficient of Determination | How much variance in sales the model explains. **1.0 = perfect**, 0.0 = no better than guessing the mean. |
                | **MAE** | Mean Absolute Error | Average absolute difference between predicted and actual sales. **Lower is better.** |
                | **RMSE** | Root Mean Squared Error | Similar to MAE but penalises large errors more. **Lower is better.** |
                """
            )

        st.markdown("---")

        # ── 5b. Full comparison table ──
        st.markdown("### 📋 All Models — Side-by-Side Comparison")
        comparison = pd.DataFrame(
            {
                name: {k: v for k, v in m.items() if k != "predictions"}
                for name, m in results["metrics"].items()
            }
        ).T
        comparison.index.name = "Model"

        def highlight_best(s):
            """Highlight the row of the best model."""
            if s.name == best:
                return ["background-color: #667eea22; font-weight: 700"] * len(s)
            return [""] * len(s)

        st.dataframe(
            comparison.style.apply(highlight_best, axis=1).format("{:.4f}"),
            width="stretch",
        )
        st.caption(f"_The highlighted row is the automatically selected best model: **{best}**_")

        st.markdown("---")

        # ── 5c. Actual vs Predicted chart ──
        st.markdown("### 🎯 Actual vs Predicted Sales")
        st.caption(
            "Each dot is a test-set observation. Points on the dashed orange line represent "
            "perfect predictions. Closer to the line = better model."
        )
        best_preds = results["metrics"][best]["predictions"]
        fig = plot_actual_vs_predicted(results["y_test"], best_preds)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # ── 6️⃣  Feature Importance ──
        if results["importances"] is not None:
            st.markdown("### 📊 Feature Importance")
            st.caption(
                "Shows which input features have the greatest influence on the model's predictions."
            )

            fig = plot_feature_importance(results["importances"], results["feature_names"])
            st.pyplot(fig)
            plt.close(fig)

            # ── Human-readable explanation ──
            st.markdown("#### 💡 What This Means")
            imp = results["importances"]
            names = results["feature_names"]
            order = np.argsort(imp)[::-1]

            top_name = names[order[0]]
            top_pct = imp[order[0]] / imp.sum() * 100

            st.success(f"{top_name} has the highest impact on sales")

            with st.expander("📝 Full Feature Importance Breakdown"):
                for rank, idx in enumerate(order, 1):
                    pct = imp[idx] / imp.sum() * 100
                    bar_filled = int(pct / 2)
                    bar = "█" * bar_filled + "░" * (50 - bar_filled)
                    st.markdown(f"**{rank}. {names[idx]}** — {pct:.1f}%  \n`{bar}`")

        st.markdown("---")

        # ── Download model ──
        st.markdown("### ⬇️ Download Trained Model")
        st.caption("Save the trained model bundle (.pkl) to use in production or share with your team.")
        model_path = os.path.join(MODEL_DIR, "best_model_bundle.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Trained Model (.pkl)",
                    data=f,
                    file_name="sales_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICT SALES
# ══════════════════════════════════════════════════════════════

elif page == "🎯 Predict Sales":

    st.markdown(
        """
        <div class="hero-banner">
            <h1>🎯 Predict Sales</h1>
            <p>Adjust input features and get an instant sales prediction with confidence range.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 9️⃣  Error handling: block if no model ──
    model_path = os.path.join(MODEL_DIR, "best_model_bundle.pkl")
    if not os.path.exists(model_path):
        st.warning(
            "⚠️ No trained model found. Please go to **🤖 Train Model** and train a model first."
        )
        st.stop()

    try:
        bundle = load_model_artifacts()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    feature_names = bundle["feature_names"]
    encoders = bundle["encoders"]

    # ── How it works explanation ──
    st.markdown(
        """
        <div class="problem-statement">
            <strong>How prediction works:</strong> The trained ML model has learned patterns
            from the dataset — relationships between advertising spend and sales. When you
            adjust the sliders below, the model uses those learned patterns to estimate the
            expected sales for the given input combination.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input widgets ──
    st.markdown("### 🔧 Input Features")
    st.caption("Adjust the values below to simulate different advertising scenarios.")

    input_values = {}
    cols = st.columns(min(len(feature_names), 3))

    for i, feat in enumerate(feature_names):
        col = cols[i % len(cols)]
        with col:
            if feat in encoders:
                # Categorical — dropdown
                classes = list(encoders[feat].classes_)
                input_values[feat] = st.selectbox(
                    f"📌 {feat}", classes, key=f"pred_{feat}"
                )
            else:
                # Numeric — slider with dataset-aware ranges
                if st.session_state.df is not None and feat in st.session_state.df.columns:
                    mn = float(st.session_state.df[feat].min())
                    mx = float(st.session_state.df[feat].max())
                    md = float(st.session_state.df[feat].median())
                else:
                    mn, mx, md = 0.0, 300.0, 50.0
                step = round((mx - mn) / 100, 2) if mx > mn else 0.1
                input_values[feat] = st.slider(
                    f"📌 {feat}", mn, mx, md, step=step, key=f"pred_{feat}",
                )

    st.markdown("---")

    # ── Predict button ──
    if st.button("⚡ Predict Sales", use_container_width=True):
        with st.spinner("Running prediction …"):
            try:
                result = predict_sales(input_values)

                # ── 7️⃣  Show predicted value clearly ──
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <p style="font-size:1.1rem;">Predicted Sales</p>
                        <h2>💰 {result['predicted_sales']:,.2f}</h2>
                        <p style="margin-top:0.8rem; font-size:0.95rem;">
                            📊 90% Confidence Range:
                            <b>{result['lower_bound']:,.2f}</b> — <b>{result['upper_bound']:,.2f}</b>
                        </p>
                        <p style="font-size:0.8rem; opacity:0.6; margin-top:0.5rem;">
                            Model: {result['model_type']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Explanation text ──
                st.markdown("")
                st.info("Prediction is based on trained machine learning model learning patterns between input features and sales.")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — PROJECT SUMMARY (For Viva)
# ══════════════════════════════════════════════════════════════

elif page == "📝 Project Summary":

    st.markdown(
        """
        <div class="hero-banner">
            <h1>📝 Project Summary</h1>
            <p>Complete overview of the Sales Prediction project — ideal for presentations and viva.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 🔟  Summary sections ──

    # Section 1: Problem Statement
    st.markdown(
        """
        <div class="summary-card">
            <h4>🎯 1. Problem Statement</h4>
            <p>
                Businesses invest significant budgets across multiple advertising channels such as
                TV, Radio, Social Media, and Influencer Marketing. However, it is often unclear which
                channel contributes the most to actual sales.<br><br>
                <b>Goal:</b> Build a machine learning system that predicts product sales based on
                advertising spend, enabling data-driven budget allocation for maximum ROI.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 2: Dataset Description
    st.markdown(
        """
        <div class="summary-card">
            <h4>📋 2. Dataset Description</h4>
            <p>
                The dataset contains advertising budgets across different channels and the
                corresponding sales figures. Key columns include:<br>
                • <b>Feature Variables:</b> TV, Radio, Newspaper / Social Media, Influencer Marketing<br>
                • <b>Target Variable:</b> Sales (the value we predict)<br><br>
                The dataset can be a real CSV upload (e.g., the classic Advertising dataset) or an
                auto-generated synthetic dataset with 500 samples and additional categorical features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 3: Models Used
    st.markdown(
        """
        <div class="summary-card">
            <h4>🤖 3. Models Used</h4>
            <p>
                Three regression models were trained and compared:<br><br>
                <b>1. Linear Regression</b> — A baseline model that assumes a linear relationship
                between features and the target. Simple and interpretable.<br><br>
                <b>2. Random Forest Regressor</b> — An ensemble of 200 decision trees that reduces
                overfitting through bagging. Captures non-linear patterns effectively.<br><br>
                <b>3. Gradient Boosting Regressor</b> — Builds trees sequentially, where each new
                tree corrects errors from the previous ones. Often achieves the highest accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 4: Key Findings — dynamic if training results exist
    results = st.session_state.training_results
    if results is not None:
        best = results["best_name"]
        best_m = {k: v for k, v in results["metrics"][best].items() if k != "predictions"}

        # Build comparison summary
        model_lines = ""
        for name, m in results["metrics"].items():
            badge = " ⭐ (Best)" if name == best else ""
            model_lines += (
                f"• <b>{name}{badge}</b> — "
                f"R² = {m['R² Score']}, MAE = {m['MAE']}, RMSE = {m['RMSE']}<br>"
            )

        # Feature importance insight
        imp_text = ""
        if results["importances"] is not None:
            imp = results["importances"]
            names = results["feature_names"]
            order = np.argsort(imp)[::-1]
            top_feat = names[order[0]]
            top_pct = imp[order[0]] / imp.sum() * 100
            imp_text = (
                f"<br><b>Most influential feature:</b> {top_feat} "
                f"({top_pct:.1f}% importance)"
            )

        st.markdown(
            f"""
            <div class="summary-card">
                <h4>🔍 4. Key Findings</h4>
                <p>
                    {model_lines}
                    <br>
                    <b>Best model:</b> {best} with R² = {best_m['R² Score']}
                    (explains {float(best_m['R² Score'])*100:.1f}% of sales variance).
                    {imp_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="summary-card">
                <h4>🔍 4. Key Findings</h4>
                <p><i>Train a model first to see the key findings here.</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 5: Methodology
    st.markdown(
        """
        <div class="summary-card">
            <h4>⚙️ 5. Methodology</h4>
            <p>
                <b>Step 1:</b> Data cleaning — handle missing values (median/mode imputation),
                drop duplicates, remove unnamed index columns.<br>
                <b>Step 2:</b> Feature engineering — encode categorical variables using Label Encoding,
                scale numeric features using Standard Scaler.<br>
                <b>Step 3:</b> Train-test split — hold out 20% of data for unbiased evaluation.<br>
                <b>Step 4:</b> Model training — fit three models on the training set.<br>
                <b>Step 5:</b> Evaluation — compare R², MAE, RMSE on the test set.<br>
                <b>Step 6:</b> Model selection — automatically select the model with the highest R² score.<br>
                <b>Step 7:</b> Prediction — use the best model for new sales predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 6: Conclusion
    st.markdown(
        """
        <div class="summary-card">
            <h4>✅ 6. Conclusion</h4>
            <p>
                This project demonstrates a complete end-to-end machine learning pipeline for
                sales prediction — from data ingestion and cleaning, through EDA and model training,
                to interactive prediction with confidence intervals.<br><br>
                <b>Key takeaways:</b><br>
                • Ensemble models (Random Forest, Gradient Boosting) consistently outperform
                simple Linear Regression on advertising datasets.<br>
                • Feature importance analysis reveals which advertising channels provide the
                highest return on investment.<br>
                • The Streamlit interface makes the entire workflow accessible to non-technical
                stakeholders, enabling data-driven business decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 7: Technologies Used
    st.markdown(
        """
        <div class="summary-card">
            <h4>🛠️ 7. Technologies Used</h4>
            <p>
                <b>Language:</b> Python 3.11<br>
                <b>ML Library:</b> Scikit-Learn (Linear Regression, Random Forest, Gradient Boosting)<br>
                <b>Data Processing:</b> Pandas, NumPy<br>
                <b>Visualization:</b> Matplotlib, Seaborn<br>
                <b>Web Framework:</b> Streamlit<br>
                <b>Model Persistence:</b> Joblib
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
