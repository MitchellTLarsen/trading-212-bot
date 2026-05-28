import yfinance as yf
import pandas as pd


def fetch_data(ticker, period="5y", interval="1d"):
    """Fetch historical OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def fetch_multiple(tickers, period="5y", interval="1d"):
    """Fetch data for multiple tickers, returns dict of DataFrames."""
    result = {}
    for ticker in tickers:
        try:
            df = fetch_data(ticker, period=period, interval=interval)
            if len(df) > 100:
                result[ticker] = df
                print(f"  {ticker}: {len(df)} bars loaded")
            else:
                print(f"  {ticker}: skipped (only {len(df)} bars)")
        except Exception as e:
            print(f"  {ticker}: failed ({e})")
    return result
