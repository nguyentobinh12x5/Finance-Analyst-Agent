# VietStock AI-Quant Engine 🚀

> *"An end-to-end Machine Learning Quantitative Trading framework designed for the Vietnam Stock Market, featuring strict Walk-Forward validation and realistic bt engine backtesting."*

## 📖 Overview
This repository contains a full-stack algorithmic trading system for the Vietnam stock market. The dashboard uses the cached local dataset `raw_fundamental_data.csv`, applies rigorous feature engineering, and uses a suite of Machine Learning algorithms (XGBoost, LightGBM, Random Forest, Gradient Boosting) to forecast future returns.

`src/data/data_fetcher.py` is kept only as an optional data-refresh utility. Normal training, comparison, MLflow tracking, and backtesting do not need to call the data API because the raw data has already been saved in `raw_fundamental_data.csv`.

Instead of traditional Train/Test splits, the system operates on a professional **Walk-Forward Cross-Validation** to strictly eliminate Look-Ahead Bias. It then uses the `bt` algorithmic trading library to simulate historical capital growth (Equity Curve) and report KPIs like CAGR, Max Drawdown, and Sharpe Ratio.

---

## 🏗 System Architecture

The project strictly abides by the "Separation of Concerns" principle, mathematically isolating the "Predicting Brain" from the "Trading Arena".

1. **`raw_fundamental_data.csv` (Cached Dataset)**: The local source of truth used by the dashboard. It already contains financial statements, aligned quarterly prices (`adj_close_q`), and the ML target (`y_return`).
2. **`src/data/data_fetcher.py` (Optional Data Refresh Utility)**: Connects to the VCI source via vnstock only when you intentionally want to rebuild the cached CSV.
3. **`src/strategies/ml_strategy.py` (The AI Brain)**: Responsible for reading historical quarters (e.g., 2021-2023) and projecting out-of-sample expected values (`y_return`) for the subsequent quarter. Outputs a **Weights Matrix** to allocate capital across the top predicted stocks.
4. **`src/backtest/backtest_engine.py` (The Arena)**: Accepts the Weights Matrix and uses local quarterly prices from `raw_fundamental_data.csv` for simulation. It does not call the VNStock API during dashboard runs.
5. **`src/dashboard/app.py` (Interactive Web Dashboard)**: A simple `streamlit` application that acts as a UI for configuring the hyperparameters, picking the AI model, tracking runs with MLflow, and visualizing the equity curve dynamics interactively in your browser.

---

## ⚡ Quick Start / Usage

### 1. Prerequisites
Ensure you have Python 3.12+ and have installed all requirements:
```bash
pip install -r requirements.txt
```

### 2. Running a Complete Pipeline in Jupyter Notebook (`FinRL.ipynb`)
```python
from src.strategies.ml_strategy import EnsembleMLStrategy
from src.backtest.backtest_engine import BacktestEngine
import pandas as pd

raw_dataset = pd.read_csv("raw_fundamental_data.csv")

# 1. Define financial indicators to use as Features
features = ['EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio', 
            'debt_ratio', 'pe', 'pb', 'roe', 'net_income_ratio']

# 2. Spin up the Machine Learning strategy (Train 12 quarters -> Predict 1 quarter)
ml_agent = EnsembleMLStrategy(df=clean_dataset, features=features, target='y_return', train_window_quarters=12)

# 3. Exectute Walk-Forward Competition across Models
leaderboard, models = ml_agent.walk_forward_competition()

# 4. Visualize the prediction vs truth line chart
ml_agent.plot_model_comparison(leaderboard)
ml_agent.analyze_ticker("FPT")

# 5. Extract the Weights Matrix from the Champion Model
weights_matrix = ml_agent.generate_weights_matrix(top_k=5, chosen_model='XGBoost')

# 6. Throw the AI's predictions into the Backtest Arena using local CSV prices
prices = BacktestEngine.build_prices_from_fundamental_data(
    raw_dataset,
    tickers=weights_matrix.columns
)
engine = BacktestEngine(
    weights_df=weights_matrix,
    initial_capital=10000,
    prices_df=prices
)
engine.run_simulation()
engine.report_kpis() # Outputs Equity Curve
```

### 3. Running the Interactive Streamlit Dashboard 🖥️

If you prefer a UI instead of a Jupyter Notebook, you can launch the AI Quant Trading Dashboard directly in your browser:

```bash
streamlit run src/dashboard/app.py
```

Optional MLflow UI:
```bash
mlflow ui --backend-store-uri mlruns
```

*The dashboard reads `raw_fundamental_data.csv` directly. It provides model comparisons, configurable hyperparameters (Train Quarters, Top K Stocks), MLflow run tracking, exact transaction schedules, and the `bt` backtesting Equity Curve plotted natively inside the UI.*

### 4. Result
![alt text](result.png)
---

## 📘 Documentation
If you are coming from a Software Engineering background without prior knowledge in Quantitative Finance, reading our deep-dive documentation is **highly recommended**. It clarifies the exact *Why* behind the intense architectural decisions (Log Returns, Data Leakage, Walk-Forward Validation).

- 🇻🇳 **[Vietnamese Technical Guide](doc_vietnamese.md)** (Dành cho người mới)
- 🇬🇧 **[English Technical Guide](doc_english.md)** 

## Inspried by AI4Finacne-Foundation
- https://github.com/AI4Finance-Foundation/FinRL-Trading

## Author
- Nguyen To Binh
