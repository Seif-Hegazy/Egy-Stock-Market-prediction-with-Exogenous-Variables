"""
EGX Prediction Model v3 - Data Loader (Research Framework)
Implements:
1. Fixed two-week rolling window scheme (Week 0, Week 1 -> Week 2 target)
2. Neutral zone labeling (m=0.001)
3. Strict chronological splitting (72% Train, 8% Val, 20% Test)
4. Feature standardization using training stats only
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw stock and macro data."""
    # Load stocks
    df_stock = pd.read_csv(DATA_DIR / 'raw' / 'stocks' / 'egx_daily_12y.csv')
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Load Egypt macro
    df_egypt = pd.read_csv(DATA_DIR / 'raw' / 'economic' / 'egypt_economic_data.csv')
    df_egypt['date'] = pd.to_datetime(df_egypt['date'])
    df_egypt = df_egypt.rename(columns={
        'date': 'Date',
        'usd_sell_rate': 'usd_egp',
        'gold_24k': 'gold_local',
        'cbe_deposit_rate': 'cbe_rate',
        'headline_inflation': 'inflation'
    })
    
    # Load Global macro
    df_global = pd.read_csv(DATA_DIR / 'raw' / 'global' / 'global_market_data.csv')
    df_global['date'] = pd.to_datetime(df_global['date'])
    df_global = df_global.rename(columns={
        'date': 'Date',
        'oil_close': 'oil',
        'gold_intl_close': 'gold_intl',
        'sp500_close': 'sp500',
        'vix_close': 'vix',
        'eur_usd_close': 'eur_usd',
        'msci_em_close': 'msci_em'
    })
    
    # Merge macro data
    df_macro = df_egypt.merge(df_global, on='Date', how='outer')
    df_macro = df_macro.sort_values('Date').ffill()  # Forward fill macro
    
    return df_stock, df_macro


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on the full timeframe (vectorized).
    Since we slice strictly by index later, past values are valid.
    """
    df = df.copy()
    
    # 1. Returns
    df['ret_1w'] = df.groupby('Ticker')['Close'].pct_change(5)
    df['ret_1m'] = df.groupby('Ticker')['Close'].pct_change(20)
    
    # 2. Volatility (20-day rolling std)
    df['vol_1m'] = df.groupby('Ticker')['Close'].rolling(20).std().reset_index(0, drop=True)
    
    # 3. RSI (14-day)
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('Ticker')['Close'].apply(rsi).reset_index(0, drop=True)
    
    # 4. Momentum (Price / 20-day MA)
    df['ma_20'] = df.groupby('Ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
    df['mom_20'] = df['Close'] / df['ma_20']
    
    # Fill NaNs (early periods)
    df = df.fillna(0)
    
    return df


def construct_rolling_windows(df_ticker: pd.DataFrame, 
                              df_macro: pd.DataFrame,
                              w: int = 5,
                              margin: float = 0.001) -> pd.DataFrame:
    """
    Construct rolling windows samples according to paper.
    Incorporates both Price features AND Technical features.
    """
    # 1. Merge ticker with macro
    df = df_ticker.merge(df_macro, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Fill missing macro
    macro_cols = ['usd_egp', 'gold_local', 'cbe_rate', 'inflation', 
                  'oil', 'gold_intl', 'sp500', 'vix', 'eur_usd', 'msci_em']
    df[macro_cols] = df[macro_cols].ffill()
    
    # 3. Add Technical Features
    df = add_technical_features(df)
    tech_cols = ['ret_1w', 'ret_1m', 'vol_1m', 'rsi_14', 'mom_20']
    
    samples = []
    
    total_len = len(df)
    max_idx = total_len - 3 * w
    
    for i in range(max_idx + 1):
        # Indices and data slices
        idx_week0 = slice(i, i + w)
        idx_week1 = slice(i + w, i + 2 * w)
        idx_week2 = slice(i + 2 * w, i + 3 * w)
        
        chunk_week0 = df.iloc[idx_week0]
        chunk_week1 = df.iloc[idx_week1]
        chunk_week2 = df.iloc[idx_week2]
        
        # --- Label Construction ---
        p_past = pd.concat([chunk_week0['Close'], chunk_week1['Close']]).mean()
        p_future = chunk_week2['Close'].mean()
        
        if p_past == 0: continue
            
        ratio = p_future / p_past
        
        if (1.0 - margin) <= ratio <= (1.0 + margin):
            continue
            
        target = 1 if ratio > (1.0 + margin) else 0
        
        # --- Feature Construction ---
        features = {}
        
        # Helper to add standard window features
        def add_window_features(prefix, col_name):
            # Week 0
            for d in range(w):
                features[f'{prefix}_w0_d{d}'] = chunk_week0[col_name].iloc[d]
            # Week 1
            for d in range(w):
                features[f'{prefix}_w1_d{d}'] = chunk_week1[col_name].iloc[d]

        # 1. Price (Endogenous)
        add_window_features('price', 'Close')
        
        # 2. Technicals (Endogenous)
        for col in tech_cols:
            add_window_features('tech_' + col, col)
            
        # 3. Macro (Exogenous)
        for col in macro_cols:
            add_window_features('macro_' + col, col)
        
        # Metadata
        features['Date'] = df.iloc[i + 2 * w - 1]['Date']
        features['Ticker'] = df['Ticker'].iloc[0]
        features['Target'] = target
        
        samples.append(features)
        
    return pd.DataFrame(samples)


def prepare_datasets(df_samples: pd.DataFrame, 
                     test_ratio: float = 0.20,
                     val_ratio: float = 0.10) -> dict:
    """
    Split data chronologically and normalize using training stats.
    
    Args:
        df_samples: Samples for ONE ticker (must be sorted by Date)
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(df_samples)
    
    # Chronological Split
    test_size = int(n * test_ratio)
    train_val_size = n - test_size
    val_size = int(train_val_size * val_ratio)
    train_size = train_val_size - val_size
    
    train = df_samples.iloc[:train_size]
    val = df_samples.iloc[train_size:train_size+val_size]
    test = df_samples.iloc[train_size+val_size:]
    
    # Feature columns (exclude metadata)
    feature_cols = [c for c in df_samples.columns if c not in ['Date', 'Ticker', 'Target']]
    
    X_train = train[feature_cols].copy()
    y_train = train['Target']
    
    X_val = val[feature_cols].copy()
    y_val = val['Target']
    
    X_test = test[feature_cols].copy()
    y_test = test['Target']
    
    # --- Z-Score Normalization ---
    # Compute mu and sigma on Training Set ONLY
    mu = X_train.mean()
    sigma = X_train.std().replace(0, 1)  # Avoid division by zero
    
    # Apply to all sets
    X_train_norm = (X_train - mu) / sigma
    X_val_norm = (X_val - mu) / sigma
    X_test_norm = (X_test - mu) / sigma
    
    return {
        'X_train': X_train_norm, 'y_train': y_train,
        'X_val': X_val_norm, 'y_val': y_val,
        'X_test': X_test_norm, 'y_test': y_test,
        'metadata_test': test[['Date', 'Ticker']]
    }


if __name__ == '__main__':
    print("Testing v3 data loader...")
    stocks, macro = load_raw_data()
    print(f"Loaded {len(stocks)} stock rows, {len(macro)} macro rows")
    
    # Test on one ticker
    ticker = 'COMI.CA'
    print(f"\nProcessing {ticker}...")
    df_ticker = stocks[stocks['Ticker'] == ticker].copy()
    
    samples = construct_rolling_windows(df_ticker, macro)
    print(f"Generated {len(samples)} samples (after neutral zone filter)")
    print(f"Sample features: {len(samples.columns)}")
    
    if len(samples) > 0:
        data = prepare_datasets(samples)
        print("\nSplit sizes:")
        print(f"Train: {len(data['y_train'])}")
        print(f"Val:   {len(data['y_val'])}")
        print(f"Test:  {len(data['y_test'])}")
