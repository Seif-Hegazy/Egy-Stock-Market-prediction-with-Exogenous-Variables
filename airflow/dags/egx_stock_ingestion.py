"""
EgySentiment EGX Stock Data Ingestion DAG
=========================================

Production-grade Apache Airflow DAG for ingesting EGX100 stock data.

Features:
- Smart backfill/incremental detection
- Memory-efficient ticker-by-ticker processing
- Metadata caching (ISIN, Sector)
- Cairo timezone scheduling
- Robust error handling

Author: EgySentiment Team
Date: 2025-12-12
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pytz
import pandas as pd
import yfinance as yf
import json
import os
import logging
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# EGX100 Tickers (35 active tickers from app.py STOCK_DATA)
EGX100_TICKERS = [
    'COMI.CA',  # Commercial International Bank
    'CIEB.CA',  # Crédit Agricole Egypt
    'HDBK.CA',  # Housing & Development Bank
    'FAIT.CA',  # Faisal Islamic Bank
    'ADIB.CA',  # Abu Dhabi Islamic Bank
    'SAUD.CA',  # Al Baraka Bank
    'EGBE.CA',  # Egyptian Gulf Bank
    'EXPA.CA',  # Export Development Bank
    'HRHO.CA',  # EFG Hermes
    'EFIH.CA',  # E-Finance
    'FWRY.CA',  # Fawry
    'BTFH.CA',  # Beltone Financial
    'CICH.CA',  # CI Capital
    'TMGH.CA',  # Talaat Moustafa Group
    'PHDC.CA',  # Palm Hills Developments
    'OCDI.CA',  # SODIC
    'MASR.CA',  # Madinet Masr
    'HELI.CA',  # Heliopolis Housing
    'ORAS.CA',  # Orascom Construction
    'EMFD.CA',  # Emaar Misr
    'SWDY.CA',  # Elsewedy Electric
    'ABUK.CA',  # Abu Qir Fertilizers
    'MFPC.CA',  # MOPCO
    'SKPC.CA',  # Sidi Kerir Petrochemicals
    'AMOC.CA',  # Alexandria Mineral Oils
    'ETEL.CA',  # Telecom Egypt
    'EAST.CA',  # Eastern Company
    'JUFO.CA',  # Juhayna Food Industries
    'EFID.CA',  # Edita Food Industries
    'ISPH.CA',  # Ibnsina Pharma
    'CLHO.CA',  # Cleopatra Hospitals
    'GBCO.CA',  # GB Corp
    'EKHO.CA',  # Egypt Kuwait Holding
    'CCAP.CA',  # Qalaa Holdings
    'ORHD.CA',  # Oriental Weavers
]

# File paths (Docker volume mount points)
DATA_DIR = Path('/opt/airflow/data/raw/stocks')
METADATA_DIR = Path('/opt/airflow/data/stocks/metadata')
CSV_FILE = DATA_DIR / 'egx_daily_12y.csv'
METADATA_FILE = METADATA_DIR / 'ticker_info.json'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Timezone
CAIRO_TZ = pytz.timezone('Africa/Cairo')

# Fallback Metadata (for when yfinance fails) - VERIFIED ISINs from EGX
FALLBACK_METADATA = {
    # Banking Sector - VERIFIED
    'COMI.CA': {'isin': 'EGS60121C018', 'sector': 'Banking'},  # Commercial International Bank
    'CIEB.CA': {'isin': 'EGS60131C017', 'sector': 'Banking'},  # Credit Agricole Egypt
    'HDBK.CA': {'isin': 'EGS60091C013', 'sector': 'Banking'},  # Housing and Development Bank
    'FAIT.CA': {'isin': 'EGS60021C016', 'sector': 'Banking'},  # Faisal Islamic Bank
    'ADIB.CA': {'isin': 'EGS60141C016', 'sector': 'Banking'},  # Abu Dhabi Islamic Bank Egypt
    'SAUD.CA': {'isin': 'EGS60041C014', 'sector': 'Banking'},  # Banque Saudi Fransi Egypt
    'EGBE.CA': {'isin': 'EGS60051C013', 'sector': 'Banking'},  # Egyptian Gulf Bank
    'EXPA.CA': {'isin': 'EGS60151C015', 'sector': 'Banking'},  # Export Development Bank
    # Financial Services - VERIFIED
    'HRHO.CA': {'isin': 'EGS74081C017', 'sector': 'Financial Services'},  # EFG Hermes
    'EFIH.CA': {'isin': 'EGS74071C018', 'sector': 'Financial Services'},  # EFI Holding
    'FWRY.CA': {'isin': 'EGS745L1C014', 'sector': 'Financial Services'},  # Fawry
    'BTFH.CA': {'isin': 'EGS74061C019', 'sector': 'Financial Services'},  # Beltone Financial
    'CICH.CA': {'isin': 'EGS74091C016', 'sector': 'Financial Services'},  # CI Capital
    # Real Estate - VERIFIED
    'TMGH.CA': {'isin': 'EGS691S1C011', 'sector': 'Real Estate'},  # Talaat Moustafa Group
    'PHDC.CA': {'isin': 'EGS69091C014', 'sector': 'Real Estate'},  # Palm Hills
    'OCDI.CA': {'isin': 'EGS69061C017', 'sector': 'Real Estate'},  # SODIC
    'MASR.CA': {'isin': 'EGS69041C019', 'sector': 'Real Estate'},  # Madinet Masr
    'HELI.CA': {'isin': 'EGS69021C011', 'sector': 'Real Estate'},  # Heliopolis Housing
    'EMFD.CA': {'isin': 'EGS69101C021', 'sector': 'Real Estate'},  # Emaar Misr
    # Construction & Industrial - VERIFIED
    'ORAS.CA': {'isin': 'EGS65021C018', 'sector': 'Construction'},  # Orascom Construction
    'SWDY.CA': {'isin': 'EGS3G0Z1C014', 'sector': 'Industrial'},  # Elsewedy Electric
    # Chemicals & Petrochemicals
    'ABUK.CA': {'isin': 'EGS38191C017', 'sector': 'Chemicals'},  # Abu Qir Fertilizers
    'MFPC.CA': {'isin': 'EGS38201C014', 'sector': 'Chemicals'},  # MOPCO
    'SKPC.CA': {'isin': 'EGS38211C013', 'sector': 'Chemicals'},  # Sidi Kerir
    'AMOC.CA': {'isin': 'EGS38181C018', 'sector': 'Oil & Gas'},  # Alexandria Mineral Oils
    # Telecom
    'ETEL.CA': {'isin': 'EGS70251C011', 'sector': 'Telecom'},  # Telecom Egypt
    # Consumer Goods - VERIFIED
    'EAST.CA': {'isin': 'EGS30371C017', 'sector': 'Consumer'},  # Eastern Company
    'JUFO.CA': {'isin': 'EGS31151C014', 'sector': 'Food & Beverage'},  # Juhayna
    'EFID.CA': {'isin': 'EGS31161C013', 'sector': 'Food & Beverage'},  # Edita
    # Healthcare
    'ISPH.CA': {'isin': 'EGS74101C013', 'sector': 'Healthcare'},  # Ibn Sina Pharma
    'CLHO.CA': {'isin': 'EGS74111C012', 'sector': 'Healthcare'},  # Cleopatra Hospital
    # Diversified / Holdings
    'GBCO.CA': {'isin': 'EGS33161C013', 'sector': 'Automotive'},  # GB Auto
    'EKHO.CA': {'isin': 'EGS69071C016', 'sector': 'Holdings'},  # Egyptian Kuwaiti Holding
    'CCAP.CA': {'isin': 'EGS74051C010', 'sector': 'Holdings'},  # Qalaa Holdings (Citadel)
    'ORHD.CA': {'isin': 'EGS31131C016', 'sector': 'Textiles'},  # Oriental Weavers
}

# =============================================================================
# Utility Functions
# =============================================================================

def load_metadata_cache():
    """Load metadata cache from JSON file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata_cache(cache):
    """Save metadata cache to JSON file."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def fetch_ticker_metadata(ticker):
    """
    Fetch ISIN and Sector for a ticker.
    Uses cache first, falls back to yfinance API.
    """
    cache = load_metadata_cache()
    
    if ticker in cache:
        logging.info(f"✓ {ticker}: Using cached metadata")
        return cache[ticker]
    
    logging.info(f"⚡ {ticker}: Fetching metadata from yfinance")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metadata = {
            'isin': info.get('isin', 'N/A'),
            'sector': info.get('sector', 'N/A'),
        }
        
        # Update cache
        cache[ticker] = metadata
        save_metadata_cache(cache)
        
        return metadata
    except Exception as e:
        logging.warning(f"⚠️ {ticker}: Failed to fetch metadata - {e}")
        
        # Try fallback
        if ticker in FALLBACK_METADATA:
            logging.info(f"✓ {ticker}: Using FALLBACK metadata")
            return FALLBACK_METADATA[ticker]
            
        return {'isin': 'N/A', 'sector': 'N/A'}

def detect_execution_mode():
    """
    Determine execution mode: BACKFILL or INCREMENTAL.
    
    Returns:
        str: 'BACKFILL' if CSV is missing/empty, 'INCREMENTAL' otherwise
    """
    if not CSV_FILE.exists():
        logging.info("🔄 Mode: BACKFILL (CSV does not exist)")
        return 'BACKFILL'
    
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            logging.info("🔄 Mode: BACKFILL (CSV is empty)")
            return 'BACKFILL'
        
        logging.info(f"📈 Mode: INCREMENTAL (CSV has {len(df)} rows)")
        return 'INCREMENTAL'
    except Exception as e:
        logging.warning(f"⚠️ CSV read error: {e}. Defaulting to BACKFILL.")
        return 'BACKFILL'

def fetch_ticker_data(ticker, period):
    """
    Fetch historical data for a single ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): yfinance period ('12y' or '1d')
    
    Returns:
        pd.DataFrame: Stock data with metadata columns added
    """
    logging.info(f"📊 Fetching {period} data for {ticker}...")
    
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            logging.warning(f"⚠️ {ticker}: No data returned for period={period}")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        hist = hist.reset_index()
        
        # Add ticker column
        hist['Ticker'] = ticker
        
        # Fetch and add metadata
        metadata = fetch_ticker_metadata(ticker)
        hist['ISIN'] = metadata['isin']
        hist['Sector'] = metadata['sector']
        
        # Standardize date format
        hist['Date'] = pd.to_datetime(hist['Date']).dt.date
        
        # Reorder columns
        cols = ['Date', 'Ticker', 'ISIN', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Only keep columns that exist
        available_cols = [col for col in cols if col in hist.columns]
        hist = hist[available_cols]
        
        logging.info(f"✓ {ticker}: Fetched {len(hist)} rows")
        return hist
        
    except Exception as e:
        logging.error(f"❌ {ticker}: Failed to fetch data - {e}")
        return pd.DataFrame()

def backfill_mode():
    """
    Backfill Mode: Fetch 12 years of data for all tickers.
    
    Strategy:
    - Process ONE ticker at a time
    - Write immediately to CSV (append mode after first ticker)
    - Clear memory before next ticker
    """
    logging.info("=" * 80)
    logging.info("🚀 Starting BACKFILL Mode (12-year history)")
    logging.info("=" * 80)
    
    first_ticker = True
    success_count = 0
    
    for idx, ticker in enumerate(EGX100_TICKERS, 1):
        logging.info(f"\n[{idx}/{len(EGX100_TICKERS)}] Processing {ticker}...")
        
        # Fetch 12 years of data (use 'max' which works better for EGX stocks)
        df = fetch_ticker_data(ticker, period='max')
        
        if df.empty:
            continue
        
        # Write to CSV
        if first_ticker:
            # First write: create file with header
            df.to_csv(CSV_FILE, index=False, mode='w')
            first_ticker = False
            logging.info(f"📝 Created CSV with {len(df)} rows from {ticker}")
        else:
            # Subsequent writes: append without header
            df.to_csv(CSV_FILE, index=False, mode='a', header=False)
            logging.info(f"📝 Appended {len(df)} rows from {ticker}")
        
        success_count += 1
        
        # Explicitly clear memory
        del df
    
    logging.info("\n" + "=" * 80)
    logging.info(f"✅ BACKFILL Complete: {success_count}/{len(EGX100_TICKERS)} tickers processed")
    logging.info("=" * 80)

def incremental_mode():
    """
    Incremental Mode: Fetch latest data with catch-up logic.
    
    Strategy:
    - Read existing CSV to find last date for each ticker
    - Fetch last 5 days (to catch weekends/missed runs)
    - Filter for ONLY new dates
    - Append to CSV
    """
    logging.info("=" * 80)
    logging.info("📈 Starting INCREMENTAL Mode (Smart Catch-up)")
    logging.info("=" * 80)
    
    # 1. Load existing data to find last dates
    if not CSV_FILE.exists():
        logging.warning("⚠️ CSV file missing during incremental mode. Switching to BACKFILL.")
        backfill_mode()
        return

    try:
        existing_df = pd.read_csv(CSV_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.date
        
        # Get max date per ticker
        last_dates = existing_df.groupby('Ticker')['Date'].max().to_dict()
        logging.info(f"🔍 Loaded last dates for {len(last_dates)} tickers")
        
    except Exception as e:
        logging.error(f"❌ Error reading CSV: {e}")
        return

    success_count = 0
    total_new_rows = 0
    
    for idx, ticker in enumerate(EGX100_TICKERS, 1):
        logging.info(f"\n[{idx}/{len(EGX100_TICKERS)}] Processing {ticker}...")
        
        # Fetch last 5 days to be safe
        df = fetch_ticker_data(ticker, period='5d')
        
        if df.empty:
            continue
            
        # Filter for new data
        last_date = last_dates.get(ticker)
        
        if last_date:
            # Keep only rows AFTER the last known date
            new_data = df[df['Date'] > last_date]
            
            if new_data.empty:
                logging.info(f"  ✓ Up to date (Last: {last_date})")
                continue
                
            logging.info(f"  Found {len(new_data)} NEW rows since {last_date}")
            df = new_data
        else:
            logging.info(f"  New ticker found! Adding {len(df)} rows")

        # Append to CSV
        df.to_csv(CSV_FILE, index=False, mode='a', header=False)
        logging.info(f"📝 Appended {len(df)} rows")
        
        success_count += 1
        total_new_rows += len(df)
        
        # Clear memory
        del df
    
    logging.info("\n" + "=" * 80)
    logging.info(f"✅ INCREMENTAL Complete: {success_count} tickers updated")
    logging.info(f"📊 Total new rows added: {total_new_rows}")
    logging.info("=" * 80)

def execute_ingestion(**context):
    """
    Main ingestion function.
    Detects mode and executes appropriate strategy.
    """
    mode = detect_execution_mode()
    
    if mode == 'BACKFILL':
        backfill_mode()
    else:
        incremental_mode()
    
    # Log final CSV stats
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        logging.info(f"\n📊 Final CSV Stats:")
        logging.info(f"   - Total rows: {len(df):,}")
        logging.info(f"   - Unique tickers: {df['Ticker'].nunique()}")
        logging.info(f"   - Date range: {df['Date'].min()} to {df['Date'].max()}")

def sort_stock_data(**context):
    """
    Sorts and deduplicates the stock data CSV.
    Ensures data is always chronologically ordered.
    """
    logging.info("=" * 80)
    logging.info("🧹 Starting SORT & DEDUPLICATE Task")
    logging.info("=" * 80)
    
    if not CSV_FILE.exists():
        logging.warning("⚠️ CSV file missing. Skipping sort.")
        return

    try:
        df = pd.read_csv(CSV_FILE)
        original_count = len(df)
        
        # Convert Date to datetime for proper sorting
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by Date (Ascending) and Ticker
        df = df.sort_values(by=['Date', 'Ticker'], ascending=[True, True])
        
        # Drop Duplicates
        df = df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
        
        # Format Date back to string
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        final_count = len(df)
        duplicates_removed = original_count - final_count
        
        # Save back to CSV
        df.to_csv(CSV_FILE, index=False)
        
        logging.info(f"✅ Sort Complete")
        logging.info(f"   Original rows: {original_count}")
        logging.info(f"   Final rows:    {final_count}")
        logging.info(f"   Duplicates:    {duplicates_removed}")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"❌ Error sorting CSV: {e}")
        raise e

# =============================================================================
# DAG Definition
# =============================================================================

default_args = {
    'owner': 'egysentiment',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),  # Timeout for long backfills
}

with DAG(
    dag_id='egx_stock_ingestion',
    default_args=default_args,
    description='Production-grade EGX stock data ingestion with smart backfill/incremental logic',
    schedule_interval='30 14 * * *',  # 14:30 UTC = 16:30 Cairo Time (UTC+2)
    start_date=datetime(2025, 12, 1, tzinfo=CAIRO_TZ),
    catchup=False,
    tags=['egysentiment', 'egx', 'stocks', 'production'],
) as dag:
    
    ingest_task = PythonOperator(
        task_id='ingest_egx_stocks',
        python_callable=execute_ingestion,
        provide_context=True,
    )

    sort_task = PythonOperator(
        task_id='sort_stock_data',
        python_callable=sort_stock_data,
        provide_context=True,
    )

    # Glue: Trigger Sentiment Analysis Pipeline after Stock Ingestion
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    
    sentiment_task = TriggerDagRunOperator(
        task_id='trigger_sentiment_pipeline',
        trigger_dag_id='egy_sentiment_daily_collection',
        wait_for_completion=False,  # Fire and forget to avoid blocking stock ingestion
    )

    ingest_task >> sort_task >> sentiment_task
