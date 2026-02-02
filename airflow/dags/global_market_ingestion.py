"""
Global Market Data Ingestion DAG

Fetches daily global market indicators:
- Crude Oil (CL=F)
- Gold Futures (GC=F)
- S&P 500 (^GSPC)
- VIX Volatility Index (^VIX)
- EUR/USD (EURUSD=X)
- MSCI Emerging Markets ETF (EEM)

Schedule: Daily at 06:00 UTC (after US markets close)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import yfinance as yf
import os

# =============================================================================
# Configuration
# =============================================================================

GLOBAL_TICKERS = {
    'oil': 'CL=F',
    'gold_intl': 'GC=F',
    'sp500': '^GSPC',
    'vix': '^VIX',
    'eur_usd': 'EURUSD=X',
    'msci_em': 'EEM',
}

DATA_PATH = '/opt/airflow/data/raw/global/global_market_data.csv'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


# =============================================================================
# Task Functions
# =============================================================================

def fetch_global_data(**context):
    """Fetch latest global market data."""
    from datetime import datetime, timedelta
    
    print("Fetching global market data...")
    
    # Fetch last 7 days to ensure we get the latest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    all_data = []
    
    for name, ticker in GLOBAL_TICKERS.items():
        print(f"  Fetching {name} ({ticker})...")
        try:
            data = yf.download(
                ticker, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if len(data) == 0:
                print(f"    No data for {ticker}")
                continue
            
            data = data.reset_index()
            data = data.rename(columns={
                'Date': 'date',
                'Close': f'{name}_close',
            })
            
            # Keep only date and close
            data_simple = data[['date', f'{name}_close']].copy()
            all_data.append(data_simple)
            
            print(f"    Got {len(data)} rows")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    if len(all_data) == 0:
        raise ValueError("No data fetched!")
    
    # Merge new data
    merged = all_data[0]
    for df in all_data[1:]:
        merged = merged.merge(df, on='date', how='outer')
    
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Push to XCom for next task
    context['ti'].xcom_push(key='new_data', value=merged.to_json())
    
    print(f"Fetched {len(merged)} new rows")
    return len(merged)


def update_historical_data(**context):
    """Update the historical data file with new data."""
    import json
    
    print("Updating historical data...")
    
    # Get new data from XCom
    new_data_json = context['ti'].xcom_pull(key='new_data', task_ids='fetch_global_data')
    new_data = pd.read_json(new_data_json)
    new_data['date'] = pd.to_datetime(new_data['date'])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Load existing data if it exists
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH)
        existing['date'] = pd.to_datetime(existing['date'])
        
        print(f"  Existing data: {len(existing)} rows")
        print(f"  Last date: {existing['date'].max()}")
        
        # Combine and deduplicate
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)
        
        # Forward fill any gaps
        combined = combined.ffill()
        
    else:
        combined = new_data
        print("  No existing data - creating new file")
    
    # Save
    combined.to_csv(DATA_PATH, index=False)
    
    print(f"  Saved {len(combined)} total rows")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    
    return len(combined)


def check_data_quality(**context):
    """Verify data quality."""
    print("Checking data quality...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Check for recent data
    latest_date = df['date'].max()
    days_old = (datetime.now() - latest_date).days
    
    print(f"  Latest date: {latest_date.date()}")
    print(f"  Days old: {days_old}")
    
    if days_old > 5:
        print(f"  WARNING: Data is {days_old} days old!")
    
    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  Nulls: {null_counts.to_dict()}")
    else:
        print("  No nulls ✓")
    
    # Check row count
    print(f"  Total rows: {len(df)}")
    
    return True


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    'global_market_data_ingestion',
    default_args=default_args,
    description='Fetch daily global market indicators (Oil, Gold, S&P500, VIX, EUR/USD, MSCI EM)',
    schedule_interval='0 6 * * *',  # Daily at 06:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data', 'global', 'markets'],
) as dag:
    
    fetch_task = PythonOperator(
        task_id='fetch_global_data',
        python_callable=fetch_global_data,
        provide_context=True,
    )
    
    update_task = PythonOperator(
        task_id='update_historical_data',
        python_callable=update_historical_data,
        provide_context=True,
    )
    
    quality_task = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True,
    )
    
    fetch_task >> update_task >> quality_task
