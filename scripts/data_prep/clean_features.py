#!/usr/bin/env python3
"""
clean_features.py - NON-DESTRUCTIVE Feature Transformation Script
==================================================================

Role: Integration Specialist & Defensive Coder

This script transforms raw market data into model-ready features WITHOUT:
- Modifying any existing data ingestion scripts
- Overwriting raw CSV files
- Changing variable names used by Airflow DAGs

Usage:
    from clean_features import prepare_model_data
    model_df = prepare_model_data(raw_df)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def prepare_model_data(raw_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Transform raw merged dataframe into model-ready features.
    
    NON-DESTRUCTIVE: Works on a copy, preserves original data.
    
    Args:
        raw_df: Merged dataframe containing price data and optionally sentiment.
                Expected columns: Date, Ticker, Open, High, Low, Close, Volume
                Optional columns: direct_count, sector_count, direct_sentiment, sector_sentiment
        verbose: If True, print verification info
    
    Returns:
        model_ready_df: DataFrame with engineered features only.
                        Excludes raw Open, High, Low to prevent data leakage.
    
    Engineered Features:
        - Return: Daily return (Close / Close.shift(1) - 1)
        - Target_Buffered: 1 if Return > 0.5%, 0 if Return < -0.5%, NaN otherwise
        - Open_Close_Ratio: Open / Close (stationarity fix)
        - High_Low_Ratio: High / Low (stationarity fix)
        - Vol_MA_Ratio: Volume / Volume_MA_20 (stationarity fix)
        - has_news: 1 if direct_count > 0 else 0
    """
    
    # ========== DEFENSIVE COPY ==========
    df = raw_df.copy()
    
    if verbose:
        print("=" * 60)
        print("🔒 NON-DESTRUCTIVE Feature Transformation")
        print("=" * 60)
        print(f"\n📥 INPUT COLUMNS ({len(raw_df.columns)}):")
        print(f"   {list(raw_df.columns)}")
    
    # ========== VALIDATION ==========
    required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Ensure proper types
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # ========== FEATURE ENGINEERING ==========
    
    # --- 1. Daily Return ---
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()
    
    # --- 2. Buffered Target ---
    # 1 if Return > 0.5%, 0 if Return < -0.5%, NaN otherwise (neutral zone)
    BUFFER_THRESHOLD = 0.005  # 0.5%
    
    conditions = [
        df['Return'] > BUFFER_THRESHOLD,   # Strong Up
        df['Return'] < -BUFFER_THRESHOLD,  # Strong Down
    ]
    choices = [1, 0]
    df['Target_Buffered'] = np.select(conditions, choices, default=np.nan)
    
    # --- 3. Stationarity Fixes (Ratios instead of raw prices) ---
    # These ratios are stationary by construction
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    df['High_Low_Ratio'] = df['High'] / df['Low']
    
    # Volume MA Ratio (avoid division by zero)
    df['Volume_MA_20'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df['Vol_MA_Ratio'] = df['Volume'] / df['Volume_MA_20'].replace(0, np.nan)
    
    # --- 4. Sentiment Flag ---
    if 'direct_count' in df.columns:
        df['has_news'] = (df['direct_count'] > 0).astype(int)
    else:
        # Default to 0 if sentiment data not merged
        df['has_news'] = 0
        if verbose:
            print("\n⚠️  Warning: 'direct_count' not found. Setting has_news = 0")
    
    # ========== SAFE OUTPUT ==========
    # Define columns to KEEP in model-ready output
    # Keys (always needed)
    key_cols = ['Date', 'Ticker']
    
    # Engineered features (safe, derived)
    feature_cols = [
        'Return',
        'Target_Buffered',
        'Open_Close_Ratio',
        'High_Low_Ratio',
        'Vol_MA_Ratio',
        'has_news',
    ]
    
    # Include Close for reference (needed for some analyses)
    # But EXCLUDE: Open, High, Low (to prevent data leakage)
    reference_cols = ['Close', 'Volume']
    
    # Include sentiment features if present
    sentiment_cols = ['direct_sentiment', 'sector_sentiment', 'direct_count', 'sector_count']
    available_sentiment = [c for c in sentiment_cols if c in df.columns]
    
    # Include sector if present
    if 'Sector' in df.columns:
        key_cols.append('Sector')
    
    # Final column selection
    output_cols = key_cols + feature_cols + reference_cols + available_sentiment
    
    # Create model-ready dataframe
    model_ready_df = df[output_cols].copy()
    
    # ========== VERIFICATION ==========
    if verbose:
        print(f"\n📤 OUTPUT COLUMNS ({len(model_ready_df.columns)}):")
        print(f"   {list(model_ready_df.columns)}")
        
        # Prove exclusion
        excluded = ['Open', 'High', 'Low']
        excluded_present = [c for c in excluded if c in model_ready_df.columns]
        
        print(f"\n🛡️  SAFETY CHECK:")
        print(f"   ✓ Excluded from output: {excluded}")
        if excluded_present:
            print(f"   ❌ WARNING: These should NOT be here: {excluded_present}")
        else:
            print(f"   ✓ Confirmed: No raw price leakage")
        
        # Target distribution
        target_dist = model_ready_df['Target_Buffered'].value_counts(dropna=False)
        print(f"\n📊 Target_Buffered Distribution:")
        print(f"   Up   (1): {target_dist.get(1, 0):,}")
        print(f"   Down (0): {target_dist.get(0, 0):,}")
        print(f"   Neutral (NaN): {model_ready_df['Target_Buffered'].isna().sum():,}")
        
        print("\n" + "=" * 60)
        print("✅ Transformation complete. Original data UNCHANGED.")
        print("=" * 60)
    
    return model_ready_df


def load_and_prepare(
    stock_path: str = "data/raw/stocks/egx_daily_12y.csv",
    sentiment_path: str = "data/processed/news/daily_sentiment_features.csv",
    economic_path: str = "data/raw/economic/egypt_economic_data.csv",
) -> pd.DataFrame:
    """
    Convenience function to load raw data and prepare model-ready features.
    
    This function READS files but NEVER WRITES to the original paths.
    
    Returns:
        model_ready_df: Transformed dataframe ready for ML
    """
    base_path = Path(__file__).parent.parent
    
    # --- Load Stock Data ---
    stock_file = base_path / stock_path
    if not stock_file.exists():
        raise FileNotFoundError(f"Stock data not found: {stock_file}")
    
    print(f"📂 Loading stock data from: {stock_file}")
    stocks_df = pd.read_csv(stock_file)
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    
    # --- Load Sentiment Data (optional) ---
    sentiment_file = base_path / sentiment_path
    if sentiment_file.exists():
        print(f"📂 Loading sentiment data from: {sentiment_file}")
        sentiment_df = pd.read_csv(sentiment_file)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge sentiment with stock data
        # Align column names: sentiment uses 'date'/'ticker', stocks use 'Date'/'Ticker'
        merged_df = stocks_df.merge(
            sentiment_df,
            left_on=['Date', 'Ticker'],
            right_on=['date', 'ticker'],
            how='left'
        )
        # Drop duplicate date/ticker columns from sentiment
        merged_df = merged_df.drop(columns=['date', 'ticker'], errors='ignore')
    else:
        print(f"⚠️  Sentiment data not found: {sentiment_file}")
        merged_df = stocks_df
    
    # --- Load Economic Data (optional, for future use) ---
    economic_file = base_path / economic_path
    if economic_file.exists():
        print(f"📂 Loading economic data from: {economic_file}")
        econ_df = pd.read_csv(economic_file)
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        
        # Merge economic data
        merged_df = merged_df.merge(
            econ_df,
            left_on='Date',
            right_on='date',
            how='left'
        )
        merged_df = merged_df.drop(columns=['date'], errors='ignore')
    else:
        print(f"⚠️  Economic data not found: {economic_file}")
    
    # --- Prepare Features ---
    model_df = prepare_model_data(merged_df, verbose=True)
    
    return model_df


def save_model_data(model_df: pd.DataFrame, output_path: str = None) -> str:
    """
    Save model-ready data to a NEW file (never overwrites raw data).
    
    Args:
        model_df: Output from prepare_model_data()
        output_path: Optional custom path. Default: data/model_ready/features.csv
    
    Returns:
        Path where data was saved
    """
    if output_path is None:
        base_path = Path(__file__).parent.parent
        output_dir = base_path / "data" / "model_ready"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features.csv"
    
    output_path = Path(output_path)
    model_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved model-ready data to: {output_path}")
    print(f"   Rows: {len(model_df):,}")
    print(f"   Columns: {len(model_df.columns)}")
    
    return str(output_path)


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    """
    Standalone execution for testing and verification.
    """
    print("\n" + "=" * 60)
    print("🚀 clean_features.py - Standalone Execution")
    print("=" * 60 + "\n")
    
    try:
        # Load, merge, and transform
        model_df = load_and_prepare()
        
        # Save to model_ready directory
        save_model_data(model_df)
        
        # Show sample
        print("\n📋 Sample Output (first 5 rows):")
        print(model_df.head().to_string())
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nTo test with mock data, use:")
        print("  from clean_features import prepare_model_data")
        print("  model_df = prepare_model_data(your_dataframe)")
