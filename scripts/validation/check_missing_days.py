#!/usr/bin/env python3
"""
Script to analyze stock data for missing trading days and identify gaps.
"""
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Load stock data
csv_path = "data/raw/stocks/egx_daily_12y.csv"
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])

print("=" * 80)
print("STOCK DATA ANALYSIS - Missing Trading Days")
print("=" * 80)

# Basic stats
print(f"\nTotal rows: {len(df):,}")
print(f"Unique tickers: {df['Ticker'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Check for gaps per ticker (focus on recent data - last 30 days)
recent_start = df['Date'].max() - timedelta(days=30)
recent_df = df[df['Date'] >= recent_start]

print(f"\n📅 Recent Data Analysis (Last 30 days from {recent_start.date()}):")
print("-" * 60)

# Get expected trading days (weekdays, excluding known holidays)
# EGX is closed on Fridays and Saturdays
def get_expected_trading_days(start_date, end_date):
    """Get expected trading days (Sun-Thu for EGX)"""
    dates = pd.date_range(start=start_date, end=end_date)
    # EGX is open Sun-Thu (weekday 0=Mon, 4=Fri, 5=Sat, 6=Sun)
    # In pandas: 0=Monday, 6=Sunday
    # EGX closed on Friday (4) and Saturday (5)
    trading_days = dates[~dates.dayofweek.isin([4, 5])]  # Exclude Fri, Sat
    return trading_days

# Check for each ticker
missing_dates_by_ticker = {}
tickers = df['Ticker'].unique()

for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].sort_values('Date')
    ticker_dates = set(ticker_df['Date'].dt.date)
    
    # Get expected trading days
    expected = get_expected_trading_days(ticker_df['Date'].min(), df['Date'].max())
    expected_set = set(expected.date)
    
    # Find missing
    missing = expected_set - ticker_dates
    
    # Only care about recent missing (last 30 days)
    recent_missing = [d for d in missing if d >= recent_start.date()]
    
    if recent_missing:
        missing_dates_by_ticker[ticker] = sorted(recent_missing)

# Print results
if missing_dates_by_ticker:
    print("\n⚠️  TICKERS WITH MISSING RECENT DATES:")
    for ticker, dates in sorted(missing_dates_by_ticker.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"   {ticker}: {len(dates)} missing days")
        if len(dates) <= 5:
            print(f"      Dates: {[str(d) for d in dates]}")
else:
    print("   ✅ No missing recent trading days found!")

# Check overall date coverage
print("\n📊 Date Coverage by Ticker (Last 7 days):")
print("-" * 60)

last_7_start = df['Date'].max() - timedelta(days=7)
for ticker in sorted(tickers)[:10]:  # First 10 tickers
    ticker_df = df[(df['Ticker'] == ticker) & (df['Date'] >= last_7_start)]
    dates = sorted(ticker_df['Date'].dt.date.unique())
    print(f"   {ticker}: {len(dates)} days - {[str(d) for d in dates]}")

# Check if Dec 16-30 is covered (the gap period)
print("\n🔍 Checking Dec 16-20 2025 Coverage (Previously reported gap):")
print("-" * 60)
gap_start = datetime(2025, 12, 16)
gap_end = datetime(2025, 12, 20)

for ticker in ['COMI.CA', 'TMGH.CA', 'HRHO.CA']:  # Sample tickers
    ticker_df = df[(df['Ticker'] == ticker) & (df['Date'] >= gap_start) & (df['Date'] <= gap_end)]
    dates = sorted(ticker_df['Date'].dt.date.unique())
    if dates:
        print(f"   {ticker}: {len(dates)} days - {[str(d) for d in dates]}")
    else:
        print(f"   {ticker}: ❌ NO DATA for Dec 16-20!")
