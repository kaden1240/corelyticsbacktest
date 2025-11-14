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

# Now import everything
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

def passes_momentum_criteria(data, rsi_min, rsi_max, stoch_k_min, stoch_k_max):
    try:
        return (float(data["Volume"]) > 0.9 * float(data["avg_volume"])
                and float(data["Close"]) >= 0.9 * float(data["High"])
                and rsi_min < float(data["RSI"]) < rsi_max
                and stoch_k_min < float(data["Stoch_K"]) < stoch_k_max
                and float(data["Stoch_K"]) > float(data["Stoch_D"])
                and float(data["MACD_Line"]) > float(data["MACD_Signal"])
                and float(data["MACD_Hist"]) > float(data["Prev_Hist"]))
    except:
        return False

def load_sp500():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers

# --- Main backtest function ---
def run_backtest(rsi_min=50, rsi_max=65, stoch_k_min=60, stoch_k_max=75,
                 start_date="2025-06-01", end_date="2025-09-29"):
    pd.DataFrame(columns=["Date", "Ticker", "Entry", "Exit", "Percent_Return"]).to_csv(RESULTS_CSV, index=False)
    sp500_tickers = load_sp500()
    print(f"Found {len(sp500_tickers)} tickers.")

    for idx, ticker in enumerate(sp500_tickers, 1):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 35:
                continue

            df = df.sort_index()
            df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD_Line'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
            df['Prev_Hist'] = df['MACD_Hist'].shift(1)
            low14 = df['Low'].rolling(14).min()
            high14 = df['High'].rolling(14).max()
            df['Stoch_K'] = ((df['Close'] - low14) / (high14 - low14)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
            df['avg_volume'] = df['Volume'].rolling(30).mean()

            trades = []
            for i in range(len(df)-1):
                row = df.iloc[i].copy()
                next_day = df.iloc[i+1].copy()

                if row.isnull().any():
                    continue

                signal_data = {
                    "Close": float(row['Close']),
                    "High": float(row['High']),
                    "Volume": float(row['Volume']),
                    "avg_volume": float(row['avg_volume']),
                    "RSI": float(row['RSI']),
                    "Stoch_K": float(row['Stoch_K']),
                    "Stoch_D": float(row['Stoch_D']),
                    "MACD_Line": float(row['MACD_Line']),
                    "MACD_Signal": float(row['MACD_Signal']),
                    "MACD_Hist": float(row['MACD_Hist']),
                    "Prev_Hist": float(row['Prev_Hist'])
                }

                if passes_momentum_criteria(signal_data, rsi_min, rsi_max, stoch_k_min, stoch_k_max):
                    entry_price = float(next_day['Open'])
                    holding_days = 5

                    if i + holding_days < len(df):
                        exit_day = df.iloc[i + holding_days]
                    else:
                        exit_day = df.iloc[-1]

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