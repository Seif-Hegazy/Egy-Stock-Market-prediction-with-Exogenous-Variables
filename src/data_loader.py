"""
EGX Macro Significance Study - Data Loader Module
Handles EGX/Macro data merge with temporal integrity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Lag schema for exogenous variables
LAG_SCHEMA = [1, 2, 3, 5]

# Macro columns to use
MACRO_COLS = [
    'usd_sell_rate', 
    'gold_24k', 
    'cbe_deposit_rate', 
    'cbe_lending_rate', 
    'headline_inflation'
]


def load_stock_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load EGX stock price data.
    
    Returns:
        DataFrame with columns: Date, Ticker, ISIN, Sector, Open, High, Low, Close, Volume
    """
    if path is None:
        path = DATA_DIR / 'raw' / 'stocks' / 'egx_daily_12y.csv'
    
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    return df


def load_macro_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Egypt macroeconomic data.
    
    Returns:
        DataFrame with date and macro indicators
    """
    if path is None:
        path = DATA_DIR / 'raw' / 'economic' / 'egypt_economic_data.csv'
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def merge_stock_macro(df_stock: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock and macro data with temporal integrity.
    
    CRITICAL: Forward-fill BEFORE applying lags to prevent leakage.
    
    Steps:
    1. Left-join stock dates onto macro (stock dates = trading days only)
    2. Forward-fill NaN macro values (weekends/holidays use last known)
    3. Compute holiday gap (days since last trading day)
    
    Args:
        df_stock: Stock price DataFrame
        df_macro: Macro indicators DataFrame
        
    Returns:
        Merged DataFrame with forward-filled macro values
    """
    # Rename macro date column for clarity
    df_macro = df_macro.rename(columns={'date': 'Date'})
    
    # Left join: keep only trading days (from stock data)
    df_merged = df_stock.merge(
        df_macro[['Date'] + MACRO_COLS],
        on='Date',
        how='left'
    )
    
    # Forward-fill macro columns PER TICKER
    # This ensures we use the last known value, not future values
    for col in MACRO_COLS:
        df_merged[col] = df_merged.groupby('Ticker')[col].ffill()
    
    # Compute holiday gap (days since last trading day for this ticker)
    df_merged = df_merged.sort_values(['Ticker', 'Date'])
    df_merged['holiday_gap'] = df_merged.groupby('Ticker')['Date'].diff().dt.days
    df_merged['is_post_holiday'] = (df_merged['holiday_gap'] > 3).astype(int)
    
    return df_merged


def apply_lags(df: pd.DataFrame, cols: List[str], lags: List[int] = LAG_SCHEMA) -> pd.DataFrame:
    """
    Apply lag transformation to specified columns.
    
    IMPORTANT: Lags are applied PER TICKER to prevent cross-ticker leakage.
    
    Args:
        df: DataFrame with Ticker column
        cols: Columns to lag
        lags: List of lag periods (in trading days)
        
    Returns:
        DataFrame with additional lagged columns
    """
    df = df.copy()
    
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('Ticker')[col].shift(lag)
    
    return df


def compute_macro_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage changes for macro variables.
    
    Args:
        df: DataFrame with macro columns
        
    Returns:
        DataFrame with additional change columns
    """
    df = df.copy()
    
    # USD/EGP changes
    df['usd_change_1d'] = df.groupby('Ticker')['usd_sell_rate'].pct_change(1)
    df['usd_change_5d'] = df.groupby('Ticker')['usd_sell_rate'].pct_change(5)
    
    # Gold changes
    df['gold_change_1d'] = df.groupby('Ticker')['gold_24k'].pct_change(1)
    df['gold_change_5d'] = df.groupby('Ticker')['gold_24k'].pct_change(5)
    
    # Rate spread
    df['rate_spread'] = df['cbe_lending_rate'] - df['cbe_deposit_rate']
    
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target.
    
    Target: y = 1 if Close_{t+1} > Close_t, else 0
    
    Args:
        df: DataFrame with Close column
        
    Returns:
        DataFrame with 'target' column
    """
    df = df.copy()
    
    # Shift Close by -1 to get next day's close
    df['next_close'] = df.groupby('Ticker')['Close'].shift(-1)
    
    # Binary target
    df['target'] = (df['next_close'] > df['Close']).astype(int)
    
    # Drop the helper column
    df = df.drop(columns=['next_close'])
    
    return df


def load_and_prepare_data() -> pd.DataFrame:
    """
    Main function to load and prepare all data for modeling.
    
    Returns:
        Fully prepared DataFrame with:
        - Stock OHLCV
        - Forward-filled macro indicators
        - Lagged macro features
        - Macro changes
        - Binary target
    """
    print("Loading stock data...")
    df_stock = load_stock_data()
    print(f"  Stock data: {len(df_stock):,} rows, {df_stock['Ticker'].nunique()} tickers")
    
    print("Loading macro data...")
    df_macro = load_macro_data()
    print(f"  Macro data: {len(df_macro):,} rows")
    
    print("Merging with forward-fill...")
    df = merge_stock_macro(df_stock, df_macro)
    
    print("Applying lag schema...")
    df = apply_lags(df, MACRO_COLS, LAG_SCHEMA)
    
    print("Computing macro changes...")
    df = compute_macro_changes(df)
    
    print("Creating target variable...")
    df = create_target(df)
    
    # Sort final output
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print(f"Final dataset: {len(df):,} rows")
    
    return df


if __name__ == '__main__':
    # Test the data loader
    df = load_and_prepare_data()
    print("\nSample columns:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df[['Date', 'Ticker', 'Close', 'usd_sell_rate', 'usd_sell_rate_lag1', 'target']].head(10))
