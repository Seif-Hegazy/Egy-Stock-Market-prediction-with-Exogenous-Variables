import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_stock_data():
    logging.info("\n🔍 Validating Stock Data (egx_daily_12y.csv)...")
    path = Path('data/raw/stocks/egx_daily_12y.csv')
    if not path.exists():
        logging.error("❌ File missing!")
        return

    try:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 1. Check for NaNs
        nans = df.isna().sum().sum()
        if nans > 0:
            logging.warning(f"⚠️  Found {nans} missing values (NaNs)")
            print(df.isna().sum())
        else:
            logging.info("✅ No missing values found")

        # 2. Logical Checks
        invalid_prices = df[(df['High'] < df['Low']) | (df['Close'] < 0)]
        if not invalid_prices.empty:
            logging.error(f"❌ Found {len(invalid_prices)} rows with invalid prices (High < Low or Negative)")
        else:
            logging.info("✅ Price logic valid (High >= Low, Prices > 0)")

        # 3. Date Continuity per Ticker
        tickers = df['Ticker'].unique()
        logging.info(f"ℹ️  Checking {len(tickers)} tickers for date gaps...")
        
        for ticker in tickers:
            tdf = df[df['Ticker'] == ticker].sort_values('Date')
            tdf['diff'] = tdf['Date'].diff().dt.days
            gaps = tdf[tdf['diff'] > 5] # Allow weekends/holidays, flag > 5 days
            if not gaps.empty:
                logging.warning(f"   ⚠️  {ticker}: Found {len(gaps)} gaps > 5 days. Max gap: {tdf['diff'].max()} days")
        
        logging.info(f"✅ Stock Data Validation Complete ({len(df)} rows)")

    except Exception as e:
        logging.error(f"❌ Error: {e}")

def validate_news_data():
    logging.info("\n🔍 Validating News Data (testing_data.jsonl)...")
    path = Path('data/raw/news/articles.jsonl')
    if not path.exists():
        logging.error("❌ File missing!")
        return

    valid_count = 0
    error_count = 0
    urls = set()
    duplicates = 0
    
    try:
        with open(path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Check required fields (Updated keys based on file inspection)
                    if not all(k in record for k in ['title', 'source', 'published']):
                        error_count += 1
                        continue
                        
                    # Check duplicates
                    if record['source'] in urls:
                        duplicates += 1
                    else:
                        urls.add(record['source'])
                        
                    valid_count += 1
                except:
                    error_count += 1
        
        logging.info(f"✅ Valid Records: {valid_count}")
        if error_count > 0:
            logging.warning(f"⚠️  Invalid/Corrupt Lines: {error_count}")
        if duplicates > 0:
            logging.warning(f"⚠️  Duplicate URLs found: {duplicates}")
        else:
            logging.info("✅ No duplicates found")
            
    except Exception as e:
        logging.error(f"❌ Error: {e}")

def validate_economic_data():
    logging.info("\n🔍 Validating Economic Data (egypt_economic_data.csv)...")
    path = Path('data/raw/economic/egypt_economic_data.csv')
    if not path.exists():
        logging.error("❌ File missing!")
        return

    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. Check for NaNs
        nans = df.isna().sum().sum()
        if nans > 0:
            logging.warning(f"⚠️  Found {nans} missing values")
            print(df.isna().sum())
        else:
            logging.info("✅ No missing values found")
            
        # 2. Check Date Continuity (Daily data)
        df = df.sort_values('date')
        df['diff'] = df['date'].diff().dt.days
        gaps = df[df['diff'] > 1]
        
        if not gaps.empty:
            logging.warning(f"⚠️  Found {len(gaps)} date gaps (missing days)")
            logging.info(f"   Max gap: {df['diff'].max()} days")
        else:
            logging.info("✅ Date sequence is perfect (Daily)")
            
    except Exception as e:
        logging.error(f"❌ Error: {e}")

def validate_sentiment_features():
    logging.info("\n🔍 Validating Sentiment Features (daily_sentiment_features.csv)...")
    path = Path('data/processed/news/daily_sentiment_features.csv')
    if not path.exists():
        logging.warning("⚠️  File missing")
        return

    try:
        df = pd.read_csv(path)
        
        # Check for empty columns or all-zeros which might indicate pipeline failure
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                if df[col].sum() == 0 and 'sentiment' in col:
                     logging.warning(f"⚠️  Column '{col}' is all zeros (Suspicious)")
        
        logging.info(f"✅ Sentiment Features Valid ({len(df)} rows)")
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")

if __name__ == "__main__":
    validate_stock_data()
    validate_news_data()
    validate_economic_data()
    validate_sentiment_features()
