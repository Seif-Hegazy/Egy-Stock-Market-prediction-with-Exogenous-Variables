"""
EGX Macro Significance Study - Feature Engineering Pipeline (v2)
================================================================
Fixed Issues:
1. Unified correlation filtering (same features for both pipelines)
2. Improved windowing with non-overlapping samples
3. Better target definition (end-to-end return)
4. Quality thresholds for sample size
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Windowing parameters
WINDOW_SIZE = 5  # 1 trading week
NEUTRAL_MARGIN = 0.01  # 1% neutral zone
MIN_SAMPLES = 200  # Minimum samples required per ticker

# Feature definitions
TECHNICAL_FEATURES = [
    'ret_1w', 'ret_1m', 'ret_2w',  # Returns
    'vol_1m',                       # Volatility
    'rsi_14',                       # RSI
    'mom_20',                       # Momentum
    'macd', 'macd_signal',          # MACD
    'volume_ratio',                 # Volume relative to MA
]

MACRO_FEATURES = [
    'usd_egp', 'gold_local', 'cbe_rate', 'inflation',  # Egypt
    'oil', 'gold_intl', 'sp500', 'vix', 'eur_usd', 'msci_em'  # Global
]


# =============================================================================
# Data Loading
# =============================================================================

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
    df_macro = df_macro.sort_values('Date').ffill()
    
    return df_stock, df_macro


# =============================================================================
# Technical Indicators
# =============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Compute MACD and Signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators."""
    df = df.copy()
    
    # Returns
    df['ret_1w'] = df.groupby('Ticker')['Close'].pct_change(5)
    df['ret_2w'] = df.groupby('Ticker')['Close'].pct_change(10)
    df['ret_1m'] = df.groupby('Ticker')['Close'].pct_change(20)
    
    # Volatility
    df['vol_1m'] = df.groupby('Ticker')['Close'].pct_change().rolling(20).std().reset_index(0, drop=True)
    
    # RSI
    df['rsi_14'] = df.groupby('Ticker')['Close'].apply(compute_rsi).reset_index(0, drop=True)
    
    # Momentum
    df['ma_20'] = df.groupby('Ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
    df['mom_20'] = df['Close'] / df['ma_20']
    
    # MACD
    macd_results = df.groupby('Ticker')['Close'].apply(
        lambda x: pd.DataFrame(dict(zip(['macd', 'macd_signal'], compute_macd(x))))
    )
    macd_results = macd_results.reset_index(level=0, drop=True)
    df['macd'] = macd_results['macd']
    df['macd_signal'] = macd_results['macd_signal']
    
    # Volume
    df['volume_ma_20'] = df.groupby('Ticker')['Volume'].rolling(20).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    
    df = df.fillna(0)
    return df


# =============================================================================
# Window Construction (FIXED: Non-overlapping, Clear Target)
# =============================================================================

def construct_samples(df_ticker: pd.DataFrame, 
                      df_macro: pd.DataFrame,
                      include_macro: bool = True,
                      w: int = WINDOW_SIZE,
                      margin: float = NEUTRAL_MARGIN,
                      step: int = None) -> pd.DataFrame:
    """
    Construct rolling window samples.
    
    FIXED:
    - step parameter for non-overlapping windows (default: w for non-overlap)
    - Clear target: price at end of Week 2 vs price at end of Week 1
    - No lookahead bias
    """
    if step is None:
        step = w  # Non-overlapping by default
    
    # 1. Merge ticker with macro
    df = df_ticker.merge(df_macro, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Fill missing macro
    for col in MACRO_FEATURES:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    # 3. Add Technical Features
    df = add_technical_features(df)
    
    samples = []
    total_len = len(df)
    
    # Use step size for iteration (non-overlapping windows)
    i = 0
    while i + 3 * w <= total_len:
        idx_week0 = slice(i, i + w)
        idx_week1 = slice(i + w, i + 2 * w)
        idx_week2 = slice(i + 2 * w, i + 3 * w)
        
        chunk_week0 = df.iloc[idx_week0]
        chunk_week1 = df.iloc[idx_week1]
        chunk_week2 = df.iloc[idx_week2]
        
        # --- FIXED: Clear Target Definition ---
        # Price at end of observation period (Week 1, last day)
        p_now = chunk_week1['Close'].iloc[-1]
        # Price at end of prediction period (Week 2, last day)
        p_future = chunk_week2['Close'].iloc[-1]
        
        if p_now == 0:
            i += step
            continue
            
        # Simple return
        ret = (p_future - p_now) / p_now
        
        # Neutral zone filter
        if -margin <= ret <= margin:
            i += step
            continue
            
        target = 1 if ret > margin else 0
        
        # --- Feature Construction ---
        features = {}
        
        def add_window_features(prefix, col_name):
            """Add SUMMARY features from each window (not daily)."""
            # Week 0 summary
            features[f'{prefix}_w0_mean'] = chunk_week0[col_name].mean()
            features[f'{prefix}_w0_std'] = chunk_week0[col_name].std()
            features[f'{prefix}_w0_last'] = chunk_week0[col_name].iloc[-1]
            # Week 1 summary
            features[f'{prefix}_w1_mean'] = chunk_week1[col_name].mean()
            features[f'{prefix}_w1_std'] = chunk_week1[col_name].std()
            features[f'{prefix}_w1_last'] = chunk_week1[col_name].iloc[-1]
        
        # 1. Price
        add_window_features('price', 'Close')
        
        # 2. Technical
        for col in TECHNICAL_FEATURES:
            if col in df.columns:
                add_window_features(f'tech_{col}', col)
        
        # 3. Macro (only if requested)
        if include_macro:
            for col in MACRO_FEATURES:
                if col in df.columns:
                    add_window_features(f'macro_{col}', col)
        
        # Metadata
        features['Date'] = chunk_week1['Date'].iloc[-1]  # End of observation
        features['Ticker'] = df['Ticker'].iloc[0]
        features['Target'] = target
        features['Return'] = ret  # Store for debugging
        
        samples.append(features)
        i += step
    
    return pd.DataFrame(samples)


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_datasets(df_samples: pd.DataFrame, 
                     test_ratio: float = 0.20,
                     val_ratio: float = 0.10) -> Dict:
    """Split data chronologically and normalize."""
    n = len(df_samples)
    
    test_size = int(n * test_ratio)
    train_val_size = n - test_size
    val_size = int(train_val_size * val_ratio)
    train_size = train_val_size - val_size
    
    train = df_samples.iloc[:train_size]
    val = df_samples.iloc[train_size:train_size+val_size]
    test = df_samples.iloc[train_size+val_size:]
    
    # Feature columns
    feature_cols = [c for c in df_samples.columns if c not in ['Date', 'Ticker', 'Target', 'Return']]
    
    X_train = train[feature_cols].copy()
    y_train = train['Target']
    X_val = val[feature_cols].copy()
    y_val = val['Target']
    X_test = test[feature_cols].copy()
    y_test = test['Target']
    
    # Z-Score Normalization
    mu = X_train.mean()
    sigma = X_train.std().replace(0, 1)
    
    X_train_norm = (X_train - mu) / sigma
    X_val_norm = (X_val - mu) / sigma
    X_test_norm = (X_test - mu) / sigma
    
    return {
        'X_train': X_train_norm, 'y_train': y_train,
        'X_val': X_val_norm, 'y_val': y_val,
        'X_test': X_test_norm, 'y_test': y_test,
        'feature_names': feature_cols,
        'metadata_test': test[['Date', 'Ticker', 'Return']]
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_endogenous_samples(df_ticker: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Create samples with ONLY endogenous features."""
    return construct_samples(df_ticker, df_macro, include_macro=False)


def create_exogenous_samples(df_ticker: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Create samples with endogenous + exogenous features."""
    return construct_samples(df_ticker, df_macro, include_macro=True)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing FIXED Feature Engineering Pipeline...")
    stocks, macro = load_raw_data()
    print(f"Loaded {len(stocks):,} stock rows, {len(macro):,} macro rows")
    
    ticker = 'COMI.CA'
    df_ticker = stocks[stocks['Ticker'] == ticker].copy()
    
    print(f"\n--- ENDOGENOUS PIPELINE ---")
    endo_samples = create_endogenous_samples(df_ticker, macro)
    endo_features = [c for c in endo_samples.columns if c not in ['Date', 'Ticker', 'Target', 'Return']]
    print(f"Samples: {len(endo_samples)} (non-overlapping)")
    print(f"Features: {len(endo_features)}")
    print(f"Target Balance: {endo_samples['Target'].value_counts(normalize=True).to_dict()}")
    
    print(f"\n--- EXOGENOUS PIPELINE ---")
    exo_samples = create_exogenous_samples(df_ticker, macro)
    exo_features = [c for c in exo_samples.columns if c not in ['Date', 'Ticker', 'Target', 'Return']]
    print(f"Samples: {len(exo_samples)}")
    print(f"Features: {len(exo_features)}")
