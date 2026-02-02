"""
EGX Prediction Model v2 - Data Loader
Loads and merges EGX stocks with global + Egypt macro data.
Target: Weekly direction prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_stock_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load EGX stock data."""
    if path is None:
        path = DATA_DIR / 'raw' / 'stocks' / 'egx_daily_12y.csv'
    
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    return df


def load_egypt_macro(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Egypt macro data."""
    if path is None:
        path = DATA_DIR / 'raw' / 'economic' / 'egypt_economic_data.csv'
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Rename columns for clarity
    df = df.rename(columns={
        'usd_sell_rate': 'usd_egp',
        'gold_24k': 'gold_local',
        'cbe_deposit_rate': 'cbe_rate',
        'headline_inflation': 'inflation'
    })
    
    return df


def load_global_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load global market data."""
    if path is None:
        path = DATA_DIR / 'raw' / 'global' / 'global_market_data.csv'
    
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Rename for cleaner column names
    df = df.rename(columns={
        'oil_close': 'oil',
        'gold_intl_close': 'gold_intl',
        'sp500_close': 'sp500',
        'vix_close': 'vix',
        'eur_usd_close': 'eur_usd',
        'msci_em_close': 'msci_em'
    })
    
    return df


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily stock data to weekly (Friday close).
    
    Following Huang et al. (2005) and Kara et al. (2011):
    Weekly prediction reduces noise and improves signal.
    """
    weekly_dfs = []
    
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df = ticker_df.set_index('Date')
        
        # Resample to weekly (Friday)
        weekly = ticker_df.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Ticker': 'first',
            'Sector': 'first',
            'ISIN': 'first'
        }).dropna(subset=['Close'])
        
        weekly = weekly.reset_index()
        weekly_dfs.append(weekly)
    
    return pd.concat(weekly_dfs, ignore_index=True)


def merge_all_data(df_stock: pd.DataFrame, 
                   df_egypt: pd.DataFrame, 
                   df_global: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock data with Egypt macro and global indicators.
    
    Uses forward-fill before merge to prevent look-ahead bias.
    """
    # Rename date columns
    df_egypt = df_egypt.rename(columns={'date': 'Date'})
    df_global = df_global.rename(columns={'date': 'Date'})
    
    # Merge Egypt macro
    egypt_cols = ['Date', 'usd_egp', 'gold_local', 'cbe_rate', 'inflation']
    egypt_cols = [c for c in egypt_cols if c in df_egypt.columns]
    
    df = df_stock.merge(df_egypt[egypt_cols], on='Date', how='left')
    
    # Merge global data
    global_cols = ['Date', 'oil', 'gold_intl', 'sp500', 'vix', 'eur_usd', 'msci_em']
    global_cols = [c for c in global_cols if c in df_global.columns]
    
    df = df.merge(df_global[global_cols], on='Date', how='left')
    
    # Forward fill macro columns per ticker
    macro_cols = ['usd_egp', 'gold_local', 'cbe_rate', 'inflation', 
                  'oil', 'gold_intl', 'sp500', 'vix', 'eur_usd', 'msci_em']
    
    for col in macro_cols:
        if col in df.columns:
            df[col] = df.groupby('Ticker')[col].ffill()
    
    return df


def create_weekly_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: Next week UP (1) or DOWN (0).
    """
    df = df.copy()
    df = df.sort_values(['Ticker', 'Date'])
    
    # Next week's close
    df['next_close'] = df.groupby('Ticker')['Close'].shift(-1)
    
    # Target: 1 if next week is up
    df['target'] = (df['next_close'] > df['Close']).astype(int)
    
    df = df.drop(columns=['next_close'])
    
    return df


def load_and_prepare_weekly_data() -> pd.DataFrame:
    """
    Main function: Load all data, resample to weekly, merge, create target.
    """
    print("Loading stock data...")
    df_stock = load_stock_data()
    print(f"  Daily: {len(df_stock):,} rows, {df_stock['Ticker'].nunique()} tickers")
    
    print("Resampling to weekly...")
    df_weekly = resample_to_weekly(df_stock)
    print(f"  Weekly: {len(df_weekly):,} rows")
    
    print("Loading Egypt macro...")
    df_egypt = load_egypt_macro()
    print(f"  Egypt macro: {len(df_egypt):,} rows")
    
    print("Loading global data...")
    df_global = load_global_data()
    print(f"  Global: {len(df_global):,} rows")
    
    print("Merging all data...")
    df = merge_all_data(df_weekly, df_egypt, df_global)
    
    print("Creating target...")
    df = create_weekly_target(df)
    
    # Sort and reset index
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print(f"Final: {len(df):,} weekly observations")
    
    return df


if __name__ == '__main__':
    df = load_and_prepare_weekly_data()
    print("\nSample columns:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df[['Date', 'Ticker', 'Close', 'oil', 'vix', 'sp500', 'target']].tail(10))
