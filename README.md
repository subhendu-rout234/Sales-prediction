# 📈 Sales Prediction Web Application

A complete end-to-end **Machine Learning** web application that predicts product sales based on advertising spend and other features. Built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## ✨ Features

| Feature | Description |
|---|---|
| **CSV Upload** | Upload your own advertising dataset |
| **Synthetic Data** | Auto-generate a realistic dataset if none is available |
| **EDA Dashboard** | Correlation heatmaps, distributions, pair plots, trend lines |
| **Multi-Model Training** | Linear Regression, Random Forest, Gradient Boosting |
| **Auto Model Selection** | Automatically picks the best model by R² score |
| **Interactive Prediction** | Sliders & dropdowns for instant sales predictions |
| **Confidence Range** | 90% prediction interval for every estimate |
| **Model Download** | Download the trained model as a `.pkl` file |

---

## 📁 Project Structure

```
Sales prediction/
├── app.py                # Streamlit frontend
├── train.py              # Model training pipeline
├── predict.py            # Prediction logic
├── utils.py              # Helpers (data gen, cleaning, plots)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/                 # (optional) store datasets
└── models/               # Trained model artifacts
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open automatically at [http://localhost:8501](http://localhost:8501).

---

## 🖥️ App Pages

### 🏠 Home
- Dataset preview, statistics, and data-quality summary.

### 📊 EDA & Visualizations
- **Correlation Heatmap** — identify feature relationships.
- **Sales Distribution** — histogram + box plot.
- **Pair Plot** — multi-feature scatter matrix.
- **Trends** — scatter plot with trend line per feature.

### 🤖 Train Model
- Configure the train/test split ratio.
- Train three models in parallel.
- View the comparison table, actual-vs-predicted chart, and feature importance.
- Download the best model.

### 🎯 Predict Sales
- Adjust sliders (numeric features) and dropdowns (categorical features).
- Get an instant sales prediction with a 90% confidence range.

---

## 🤖 Models Trained

| Model | Type |
|---|---|
| Linear Regression | Baseline linear model |
| Random Forest Regressor | Ensemble of 200 decision trees |
| Gradient Boosting Regressor | Sequential boosted trees |

**Evaluation Metrics:** R² Score · MAE · RMSE

---

## 📊 Sample Dataset

The app ships with compatibility for the classic **Advertising.csv** dataset (TV, Radio, Newspaper → Sales). If no CSV is uploaded, the app can generate a **synthetic dataset** with additional features (Social Media, Influencer Marketing, Platform, Audience Segment).

---

## 📦 Dependencies

- Python 3.9+
- pandas · numpy · scikit-learn
- matplotlib · seaborn
- streamlit · joblib

---

## 📜 License

This project is open-source and available for educational purposes.
