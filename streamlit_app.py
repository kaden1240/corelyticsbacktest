import streamlit as st
import pandas as pd
from backtest import run_backtest

st.title("Next Day Edge Backtester")
st.write("Customize your trading criteria and run a backtest on S&P 500 stocks.")

# --- User Inputs ---
st.header("Trading Criteria")
use_rsi = st.checkbox("Use RSI in criteria", value=True)
use_stoch = st.checkbox("Use Stoch K in criteria", value=True)

rsi_min, rsi_max = None, None
stoch_k_min, stoch_k_max = None, None

if use_rsi:
    rsi_min = st.slider("RSI Minimum", 0, 100, 50)
    rsi_max = st.slider("RSI Maximum", 0, 100, 65)

if use_stoch:
    stoch_k_min = st.slider("Stoch K Minimum", 0, 100, 60)
    stoch_k_max = st.slider("Stoch K Maximum", 0, 100, 75)

st.header("Date Range")
start_date = st.date_input("Start Date", pd.to_datetime("2025-06-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-09-29"))

# --- Run Backtest ---
if st.button("Run Backtest"):
    st.info("Running backtest... this may take a few minutes depending on the number of tickers.")

    df = run_backtest(
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        stoch_k_min=stoch_k_min,
        stoch_k_max=stoch_k_max,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    if df.empty:
        st.warning("No trades met the criteria in the selected date range.")
    else:
        st.success(f"Backtest complete! Found {len(df)} trades.")

        # --- Show trades ---
        st.subheader("Trades")
        st.dataframe(df)

        # --- Cumulative return plot ---
        df = df.sort_values("Date")
        df['Cumulative'] = (1 + df['Percent_Return']/100).cumprod()
        st.subheader("Cumulative Return Over Time")
        st.line_chart(df.set_index("Date")["Cumulative"])

        # --- Histogram of returns ---
        st.subheader("Distribution of Trade Returns")
        st.bar_chart(df['Percent_Return'])