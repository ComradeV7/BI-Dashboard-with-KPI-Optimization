# 📊 Predictive BI Dashboard for E-Commerce KPI Optimization

This project is a **full-stack application** featuring a **predictive Business Intelligence (BI) dashboard**.  
It leverages machine learning models to **forecast and analyze five critical Key Performance Indicators (KPIs)** for an e-commerce business, using the **Brazilian E-Commerce Public Dataset by Olist**.

The **backend** is built with Flask, serving trained models via a **REST API**, while the **frontend** (React/Vue) consumes these predictions to populate an interactive dashboard.  
The dashboard has also been prototyped in **Power BI** for additional BI visualization.

---

## 🚀 Features & Predictive Models
This dashboard is powered by five distinct, optimized machine learning models, each targeting a critical business KPI:

- **Customer Churn Prediction** – Classification (retention campaigns)  
- **Sales Volume Forecasting** – Time-series (90-day forecast)  
- **Customer Lifetime Value (CLV) Prediction** – Regression (customer segmentation)  
- **Order Review Score Prediction** – Classification (customer experience quality)  
- **Delivery Time Prediction** – Regression (accurate delivery expectations)  

---

## 🛠️ Technology Stack
- **Backend**: Python, Flask, Pandas, Scikit-learn, XGBoost, Pmdarima, Lifetimes  
- **Development**: VS Code, Jupyter Notebooks  
- **BI Prototyping**: Power BI  

---

## 📂 Project Structure
```bash
/
├── backend/
│   ├── models/           # Saved .pkl model files
│   ├── venv/             # Python virtual environment (ignored)
│   ├── app.py            # Main Flask application
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── node_modules/     # Frontend dependencies (ignored)
│   ├── public/
│   ├── src/              # React/Vue source code
│   └── package.json
├── data/
│   └── olist_prepared_dataset.csv # The primary dataset
├── notebooks/
│   ├── CLV.ipynb
│   ├── Customer-Churn.ipynb
│   └── ...               # Other analysis notebooks
├── .gitignore
├── package.json          # For running both servers concurrently
└── README.md
```

## 🏆 Final Model Selection

After extensive experimentation and hyperparameter tuning, the following final models were selected:

| KPI                       | Winning Model                  | Key Metric                    |
| ------------------------- | ------------------------------ | ----------------------------- |
| Customer Churn            | Tuned XGBoost Classifier       | F1-Score (Balanced)           |
| Sales Volume              | Tuned ARIMA Model              | MAE: 16,019                   |
| Customer Lifetime Value   | Tuned XGBoost Regressor        | MAE: 1.66                     |
| Order Review Score        | Tuned Random Forest Classifier | Macro Avg F1-Score (Balanced) |
| Delivery Time             | Tuned Random Forest Regressor  | MAE: 4.36 days                |

