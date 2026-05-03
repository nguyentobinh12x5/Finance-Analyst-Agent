import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the root project directory is in sys.path so we can import src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_process import DataProcess
from src.strategies.ml_strategy import EnsembleMLStrategy
from src.backtest.backtest_engine import BacktestEngine

# Set Streamlit page config
st.set_page_config(page_title="AI Quant Trading Dashboard", layout="wide", page_icon="📈")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    df_raw = pd.read_csv(file_path)
    
    # Remove delisted ticker VPL due to insufficient quarterly financial data
    df_raw = df_raw[df_raw['ticker'] != 'VPL'].copy()
    
    dp = DataProcess(df_raw)
    df_clean = dp.extract_features()
    return df_raw, df_clean

st.title("📈 AI Quantitative Trading Dashboard")
st.markdown("Tune hyperparameters, evaluate Machine Learning models, and test trading performance via Backtests.")

# Locate the data file
data_file_path = os.path.join(project_root, "raw_fundamental_data.csv")

try:
    df_raw, df_clean = load_and_preprocess_data(data_file_path)
except Exception as e:
    st.error(f"Error loading data from {data_file_path}: {e}")
    st.stop()

# ----- SIDEBAR CONFIGURATION -----
st.sidebar.header("⚙️ Hyperparameters Configuration")

st.sidebar.subheader("1. ML Strategy Settings")
train_window_quarters = st.sidebar.slider(
    "Train Window Quarters", 
    min_value=4, max_value=24, value=12, step=1,
    help="Number of past quarters to train the model before predicting the next quarter."
)

st.sidebar.subheader("2. Portfolio Settings")
top_k = st.sidebar.slider(
    "Top K Stocks to Pick", 
    min_value=1, max_value=20, value=5, step=1,
    help="Number of stocks with highest predicted returns to hold in the portfolio."
)

model_options = EnsembleMLStrategy.get_supported_model_names()
default_model = "XGBoost" if "XGBoost" in model_options else model_options[0]
chosen_model = st.sidebar.selectbox(
    "Select Model for Weight Allocation", 
    options=model_options,
    index=model_options.index(default_model)
)
availability = EnsembleMLStrategy.get_model_availability()
unavailable_models = [name for name, info in availability.items() if not info['available']]
if unavailable_models:
    st.sidebar.caption("Unavailable models on this machine:")
    for model_name in unavailable_models:
        st.sidebar.caption(f"- {model_name}: {availability[model_name]['reason']}")

st.sidebar.subheader("3. Backtesting Engine")
initial_capital = st.sidebar.number_input(
    "Initial Capital (VND mil)", 
    min_value=1000, value=10000, step=1000
)

st.sidebar.subheader("4. Experiment Tracking")
enable_mlflow = st.sidebar.checkbox(
    "Track runs with MLflow",
    value=True,
    help="Save parameters, metrics, model artifacts, predictions, and backtest outputs to MLflow."
)
mlflow_experiment_name = st.sidebar.text_input(
    "MLflow Experiment Name",
    value="QuantVN WalkForward",
    help="Runs with the same experiment name can be compared in the MLflow UI."
)
enable_model_registry = st.sidebar.checkbox(
    "Register best model to MLflow Registry",
    value=False,
    help="Automatically register the best walk-forward model after each run."
)
model_registry_name = st.sidebar.text_input(
    "Registry Model Name",
    value="QuantVN-WalkForward-BestModel",
    help="Name used in MLflow Model Registry."
)

run_button = st.sidebar.button("🚀 Run Full Analysis Pipeline", type="primary")


# ----- MAIN LAYOUT TABS -----
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🧠 Strategy Evaluation", "💸 Backtest Results"])

with tab1:
    st.header("Raw & Processed Data")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Raw Fundamentals ({df_raw.shape[0]} rows)")
        st.dataframe(df_raw.head(100))
    with col2:
        st.subheader(f"Cleaned Features ({df_clean.shape[0]} rows)")
        st.dataframe(df_clean.head(100))


if run_button:
    # 1. Define features manually as done in data_process
    features = [
        'EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio',
        'acc_rec_turnover', 'debt_ratio', 'debt_to_equity',
        'pe', 'ps', 'pb', 'roe', 'net_income_ratio'
    ]

    with st.spinner("Initializing and Evaluating ML Models (Walk-Forward Validation)..."):
        try:
            strategy = EnsembleMLStrategy(
                df=df_clean, 
                features=features, 
                target='y_return', 
                train_window_quarters=train_window_quarters,
                enable_mlflow=enable_mlflow,
                mlflow_experiment_name=mlflow_experiment_name,
                dataset_source_path=data_file_path,
                enable_model_registry=enable_model_registry,
                model_registry_name=model_registry_name,
                extra_run_params={
                    "dashboard.top_k": top_k,
                    "dashboard.chosen_model": chosen_model,
                    "dashboard.initial_capital_vnd_mil": initial_capital,
                    "data.source_file": os.path.basename(data_file_path),
                }
            )
            leaderboard, models = strategy.walk_forward_competition()
        except Exception as e:
            st.error(f"Error during Walk-Forward Evaluation: {e}")
            st.stop()

    with tab2:
        st.header("Model Performance Leaderboard")
        st.markdown(f"**Walk-Forward Rolling Window**: Train on `{train_window_quarters} Quarters`, Test on next `1 Quarter`.")

        if enable_mlflow:
            if strategy.mlflow_run_id:
                st.success(f"MLflow run saved: `{strategy.mlflow_run_id}`")
                st.code("mlflow ui --backend-store-uri mlruns", language="bash")
                if strategy.registered_model_version:
                    st.success(f"Registered model: `{strategy.registered_model_version}`")
            else:
                st.warning("MLflow tracking is enabled, but no run was saved. Check that `mlflow` is installed.")
        
        # Display Leaderboard
        ld_df = pd.DataFrame(list(leaderboard.items()), columns=['Model', 'Average MSE SCORE'])
        st.table(ld_df)

        st.subheader("Model Comparison Plots")
        # Ensure we capture the plot instead of plt.show()
        fig_comparison = plt.figure()
        strategy.plot_model_comparison(leaderboard)
        st.pyplot(plt.gcf())
        
    with st.spinner("Generating Portfolio Weights and Running Vectorized Backtest..."):
        try:
            weights_df = strategy.generate_weights_matrix(top_k=top_k, chosen_model=chosen_model)
            if weights_df is None or weights_df.empty:
                st.error("Weights matrix generation failed or returned empty.")
                st.stop()
            
            local_prices_df = BacktestEngine.build_prices_from_fundamental_data(
                df_raw,
                tickers=weights_df.columns
            )
            backtester = BacktestEngine(
                weights_df,
                initial_capital=initial_capital,
                prices_df=local_prices_df
            )
            res = backtester.run_simulation()
        except Exception as e:
            st.error(f"Error during Backtesting: {e}")
            st.stop()

    with tab3:
        st.header("Backtesting Results")
        st.markdown(f"**Model selected for allocation:** `{chosen_model}` | **Top K stocks:** `{top_k}` | **Initial Capital:** `{initial_capital:,.0f} VND (mil)`")
        
        st.subheader(f"📈 Portfolio Value From Local CSV Prices (Initial Capital: {initial_capital:,.0f} VND (mil))")
        # Calculate actual portfolio monetary value 
        real_money_curve = (backtester.res.prices / 100) * initial_capital
        real_money_curve.columns = ["Portfolio Value (VND mil)"]
        st.line_chart(real_money_curve)
        
        with st.expander("View detailed local-price balance changes"):
            st.dataframe(real_money_curve)

        st.subheader("📅 Reallocation Schedule & Capital Weights")
        st.markdown("The table below details exactly **which dates the portfolio is balanced** and the exact percentage of capital allocated to each stock.")
        pretty_weights = weights_df.copy()
        pretty_weights.index = pretty_weights.index.strftime('%Y-%m-%d')
        st.dataframe(pretty_weights.style.format("{:.2%}"))

        st.subheader("🛒 Transaction History")
        transactions = backtester.res.get_transactions()
        if not transactions.empty:
            st.dataframe(transactions)
        else:
            st.info("No transactions were executed during this period.")

        st.subheader("📊 Normalized Growth Curve (Base 100)")
        fig, ax = plt.subplots(figsize=(10, 5))
        backtester.res.plot(ax=ax)
        st.pyplot(fig)

        # Extract KPIs efficiently without logging whole string block to stdout if possible
        # `bt` prints to stdout using `res.display()`, to capture it:
        import io
        from contextlib import redirect_stdout
        
        st.subheader("Key Performance Indicators (KPIs)")
        f = io.StringIO()
        with redirect_stdout(f):
            backtester.res.display()
        out = f.getvalue()
        st.code(out, language="text")

        if enable_mlflow and strategy.mlflow_run_id:
            strategy.log_portfolio_artifacts(weights_df=weights_df, kpi_text=out)

else:
    with tab2:
        st.info("👈 Please set the hyperparameters and click **Run Full Analysis Pipeline** in the sidebar to view Strategy Evaluation.")
    with tab3:
        st.info("👈 Please set the hyperparameters and click **Run Full Analysis Pipeline** in the sidebar to view Backtest Results.")
