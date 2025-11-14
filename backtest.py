import subprocess
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Install missing packages dynamically
packages = ["yfinance", "pandas", "numpy", "matplotlib"]
for package in packages:
    try:
        __import__(package)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

RSI_PERIOD = 14
RESULTS_CSV = "NextDayEdge_Backtest.csv"

# --- Helper functions ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_sp500():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers

# --- Main backtest function ---
def run_backtest(
    rsi_min=50, rsi_max=65,
    stoch_k_min=60, stoch_k_max=75,
    start_date="2025-06-01", end_date="2025-09-29",
    use_rsi=True, use_stoch=True, use_macd=True
):
    # Initialize CSV
    pd.DataFrame(columns=["Date", "Ticker", "Entry", "Exit", "Percent_Return"]).to_csv(RESULTS_CSV, index=False)
    sp500_tickers = load_sp500()
    print(f"Found {len(sp500_tickers)} tickers.")

    for idx, ticker in enumerate(sp500_tickers, 1):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 35:
                continue

            df = df.sort_index()

            # Indicators
            df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
            low14 = df['Low'].rolling(14).min()
            high14 = df['High'].rolling(14).max()
            df['Stoch_K'] = ((df['Close'] - low14) / (high14 - low14)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Line'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
            df['Prev_Hist'] = df['MACD_Hist'].shift(1)

            trades = []
            for i in range(len(df)-1):
                row = df.iloc[i].copy()
                next_day = df.iloc[i+1].copy()
                if row.isnull().any():
                    continue

                rsi_ok = not use_rsi or (rsi_min < row['RSI'] < rsi_max)
                stoch_ok = not use_stoch or ((stoch_k_min < row['Stoch_K'] < stoch_k_max) and (row['Stoch_K'] > row['Stoch_D']))
                macd_ok = not use_macd or ((row['MACD_Line'] > row['MACD_Signal']) and (row['MACD_Hist'] > row['Prev_Hist']))

                if rsi_ok and stoch_ok and macd_ok:
                    entry_price = float(next_day['Open'])
                    holding_days = 5
                    exit_day = df.iloc[i + holding_days] if i + holding_days < len(df) else df.iloc[-1]
                    exit_price = float(exit_day['Close'])
                    percent_return = (exit_price - entry_price) / entry_price * 100

                    trades.append({
                        "Date": next_day.name.date(),
                        "Ticker": ticker,
                        "Entry": entry_price,
                        "Exit": exit_price,
                        "Percent_Return": percent_return
                    })

            if trades:
                pd.DataFrame(trades).to_csv(RESULTS_CSV, mode='a', header=False, index=False)

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(sp500_tickers)} tickers")

            time.sleep(0.1)

        except Exception as e:
            print(f"Error for {ticker}: {e}")

    trades_df = pd.read_csv(RESULTS_CSV)
    return trades_df