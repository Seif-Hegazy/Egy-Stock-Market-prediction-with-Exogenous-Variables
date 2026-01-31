#!/usr/bin/env python3
"""
Script to fetch and fill missing trading days (Dec 16-24, 2025) for all EGX tickers.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time

# Configuration
CSV_PATH = Path("data/raw/stocks/egx_daily_12y.csv")

EGX_TICKERS = [
    'COMI.CA', 'CIEB.CA', 'HDBK.CA', 'FAIT.CA', 'ADIB.CA', 'SAUD.CA',
    'EGBE.CA', 'EXPA.CA', 'HRHO.CA', 'EFIH.CA', 'FWRY.CA', 'BTFH.CA',
    'CICH.CA', 'TMGH.CA', 'PHDC.CA', 'OCDI.CA', 'MASR.CA', 'HELI.CA',
    'ORAS.CA', 'EMFD.CA', 'SWDY.CA', 'ABUK.CA', 'MFPC.CA', 'SKPC.CA',
    'AMOC.CA', 'ETEL.CA', 'EAST.CA', 'JUFO.CA', 'EFID.CA', 'ISPH.CA',
    'CLHO.CA', 'GBCO.CA', 'EKHO.CA', 'CCAP.CA', 'ORHD.CA',
]

# Load existing data
print("Loading existing data...")
df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])
print(f"Existing rows: {len(df):,}")

# Identify the gap: Dec 16 - Dec 24, 2025
gap_start = datetime(2025, 12, 16)
gap_end = datetime(2025, 12, 24)

print(f"\n🔍 Fetching missing data from {gap_start.date()} to {gap_end.date()}...")
print("=" * 60)

# Fetch missing data for each ticker
all_new_data = []

for i, ticker in enumerate(EGX_TICKERS, 1):
    print(f"[{i}/{len(EGX_TICKERS)}] Fetching {ticker}...", end=" ")
    
    try:
        stock = yf.Ticker(ticker)
        # Fetch data for the gap period (add 1 day buffer on each side)
        hist = stock.history(start=gap_start - timedelta(days=1), end=gap_end + timedelta(days=1))
        
        if hist.empty:
            print("❌ No data")
            continue
        
        # Reset index
        hist = hist.reset_index()
        hist['Ticker'] = ticker
        
        # Get existing metadata from our data
        existing_meta = df[df['Ticker'] == ticker][['ISIN', 'Sector']].drop_duplicates()
        if not existing_meta.empty:
            hist['ISIN'] = existing_meta['ISIN'].values[0]
            hist['Sector'] = existing_meta['Sector'].values[0]
        else:
            hist['ISIN'] = 'N/A'
            hist['Sector'] = 'N/A'
        
        # Format date
        hist['Date'] = pd.to_datetime(hist['Date']).dt.date
        
        # Filter to only the gap period
        hist = hist[(hist['Date'] >= gap_start.date()) & (hist['Date'] <= gap_end.date())]
        
        if hist.empty:
            print("❌ No data in gap period")
            continue
        
        # Select columns
        cols = ['Date', 'Ticker', 'ISIN', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in cols if c in hist.columns]
        hist = hist[available_cols]
        
        all_new_data.append(hist)
        print(f"✅ {len(hist)} rows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Small delay to avoid rate limiting
    time.sleep(0.3)

# Combine all new data
if all_new_data:
    new_df = pd.concat(all_new_data, ignore_index=True)
    print(f"\n📊 Total new rows to add: {len(new_df)}")
    
    # Check what dates we got
    new_dates = sorted(new_df['Date'].unique())
    print(f"   Dates covered: {new_dates}")
    
    # Remove any duplicates that might already exist in main df
    df['Date'] = df['Date'].dt.date  # Convert to date for comparison
    
    # Create a key for duplicate checking
    df['key'] = df['Date'].astype(str) + '_' + df['Ticker']
    new_df['key'] = new_df['Date'].astype(str) + '_' + new_df['Ticker']
    
    # Filter out duplicates
    existing_keys = set(df['key'])
    new_df = new_df[~new_df['key'].isin(existing_keys)]
    
    print(f"   After removing duplicates: {len(new_df)} new rows")
    
    if len(new_df) > 0:
        # Drop the key column
        new_df = new_df.drop('key', axis=1)
        df = df.drop('key', axis=1)
        
        # Append to existing data
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Sort by Date and Ticker
        combined_df = combined_df.sort_values(['Date', 'Ticker'])
        
        # Remove any duplicates (just in case)
        combined_df = combined_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
        
        # Save
        combined_df.to_csv(CSV_PATH, index=False)
        
        print(f"\n✅ Data saved! New total rows: {len(combined_df):,}")
    else:
        print("\n⚠️  No new data to add (all already exists)")
else:
    print("\n❌ No new data fetched")

print("\n" + "=" * 60)
print("Done!")
