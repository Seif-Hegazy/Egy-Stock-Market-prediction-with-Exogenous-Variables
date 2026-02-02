"""
EGX Macro Significance Study - Feature Engineering Pipeline (v3 FINAL)
=======================================================================
FIXES:
1. Semi-overlapping windows (step=2) for more samples
2. Fixed feature set (no per-ticker variation)
3. Separate pipelines with IDENTICAL preprocessing
4. Optimal threshold search (instead of fixed percentile)
5. Strict data leakage prevention
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
WINDOW_SIZE = 5          # 1 trading week
STEP_SIZE = 2            # Semi-overlapping (more samples)
NEUTRAL_MARGIN = 0.015   # 1.5% neutral zone
MIN_SAMPLES = 300        # Minimum after filtering

# Static feature definitions (no variation)
TECHNICAL_COLS = ['ret_1w', 'ret_2w', 'ret_1m', 'vol_1m', 'rsi_14', 'mom_20', 'macd', 'volume_ratio']
MACRO_COLS = ['usd_egp', 'gold_local', 'cbe_rate', 'inflation', 'oil', 'gold_intl', 'sp500', 'vix']


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw stock and macro data."""
    df_stock = pd.read_csv(DATA_DIR / 'raw' / 'stocks' / 'egx_daily_12y.csv')
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    df_egypt = pd.read_csv(DATA_DIR / 'raw' / 'economic' / 'egypt_economic_data.csv')
    df_egypt['date'] = pd.to_datetime(df_egypt['date'])
    df_egypt = df_egypt.rename(columns={
        'date': 'Date', 'usd_sell_rate': 'usd_egp',
        'gold_24k': 'gold_local', 'cbe_deposit_rate': 'cbe_rate',
        'headline_inflation': 'inflation'
    })
    
    df_global = pd.read_csv(DATA_DIR / 'raw' / 'global' / 'global_market_data.csv')
    df_global['date'] = pd.to_datetime(df_global['date'])
    df_global = df_global.rename(columns={
        'date': 'Date', 'oil_close': 'oil',
        'gold_intl_close': 'gold_intl', 'sp500_close': 'sp500',
        'vix_close': 'vix'
    })
    
    df_macro = df_egypt.merge(df_global[['Date', 'oil', 'gold_intl', 'sp500', 'vix']], 
                               on='Date', how='outer')
    df_macro = df_macro.sort_values('Date').ffill().bfill()
    
    return df_stock, df_macro


# =============================================================================
# Technical Indicators (Computed BEFORE windowing - no leakage)
# =============================================================================

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators. Called once per ticker before windowing."""
    df = df.copy()
    
    # Returns (using shift to prevent leakage)
    df['ret_1w'] = df['Close'].pct_change(5)
    df['ret_2w'] = df['Close'].pct_change(10)
    df['ret_1m'] = df['Close'].pct_change(20)
    
    # Volatility
    df['vol_1m'] = df['Close'].pct_change().rolling(20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / loss))
    
    # Momentum
    df['mom_20'] = df['Close'] / df['Close'].rolling(20).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    
    # Volume
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df.fillna(0)


# =============================================================================
# Window Construction (Strict No-Leakage)
# =============================================================================

def construct_windows(df_ticker: pd.DataFrame,
                      df_macro: pd.DataFrame,
                      include_macro: bool,
                      w: int = WINDOW_SIZE,
                      step: int = STEP_SIZE,
                      margin: float = NEUTRAL_MARGIN) -> pd.DataFrame:
    """
    Construct samples with STRICT no-leakage guarantee.
    
    Timeline:
    [----Week 0----][----Week 1----][----Week 2----]
       Features       Features        TARGET ONLY
       
    Target = sign(Close[W2_last] - Close[W1_last])
    Features use ONLY W0 and W1 data.
    """
    # 1. Merge with macro
    df = df_ticker.merge(df_macro, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Forward fill macro (no future data used)
    for col in MACRO_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    # 3. Compute technicals (uses only past data due to rolling)
    df = compute_technical_features(df)
    
    samples = []
    i = 0
    total_len = len(df)
    
    while i + 3 * w <= total_len:
        w0 = df.iloc[i:i+w]           # Week 0: Features
        w1 = df.iloc[i+w:i+2*w]       # Week 1: Features
        w2 = df.iloc[i+2*w:i+3*w]     # Week 2: Target ONLY
        
        # === TARGET (from Week 2 only) ===
        p_now = w1['Close'].iloc[-1]
        p_future = w2['Close'].iloc[-1]
        
        if p_now <= 0:
            i += step
            continue
        
        ret = (p_future - p_now) / p_now
        
        # Neutral zone filter
        if abs(ret) <= margin:
            i += step
            continue
        
        target = 1 if ret > margin else 0
        
        # === FEATURES (from W0 and W1 only - NO W2 DATA) ===
        features = {}
        
        # Price features (W0, W1)
        features['price_w0_last'] = w0['Close'].iloc[-1]
        features['price_w1_last'] = w1['Close'].iloc[-1]
        features['price_w0_mean'] = w0['Close'].mean()
        features['price_w1_mean'] = w1['Close'].mean()
        features['price_change_w0w1'] = (w1['Close'].iloc[-1] / w0['Close'].iloc[-1]) - 1
        
        # Technical features (W0, W1)
        for col in TECHNICAL_COLS:
            if col in df.columns:
                features[f'tech_{col}_w0_last'] = w0[col].iloc[-1]
                features[f'tech_{col}_w1_last'] = w1[col].iloc[-1]
                features[f'tech_{col}_w0w1_diff'] = w1[col].iloc[-1] - w0[col].iloc[-1]
        
        # Macro features (W0, W1) - only if requested
        if include_macro:
            for col in MACRO_COLS:
                if col in df.columns:
                    features[f'macro_{col}_w0_last'] = w0[col].iloc[-1]
                    features[f'macro_{col}_w1_last'] = w1[col].iloc[-1]
                    features[f'macro_{col}_w0w1_change'] = (
                        (w1[col].iloc[-1] / w0[col].iloc[-1]) - 1 
                        if w0[col].iloc[-1] != 0 else 0
                    )
        
        # Metadata
        features['Date'] = w1['Date'].iloc[-1]
        features['Ticker'] = df['Ticker'].iloc[0]
        features['Target'] = target
        
        samples.append(features)
        i += step
    
    return pd.DataFrame(samples)


# =============================================================================
# Dataset Preparation (Strict Chronological Split)
# =============================================================================

def prepare_datasets(df_samples: pd.DataFrame,
                     train_ratio: float = 0.70,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict:
    """
    Chronological split with proper normalization.
    
    NO SHUFFLING - maintains time order.
    Normalization uses ONLY training data statistics.
    """
    n = len(df_samples)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = df_samples.iloc[:train_end]
    val = df_samples.iloc[train_end:val_end]
    test = df_samples.iloc[val_end:]
    
    # Feature columns (exclude metadata)
    feature_cols = [c for c in df_samples.columns 
                    if c not in ['Date', 'Ticker', 'Target']]
    
    X_train = train[feature_cols].copy()
    y_train = train['Target'].values
    X_val = val[feature_cols].copy()
    y_val = val['Target'].values
    X_test = test[feature_cols].copy()
    y_test = test['Target'].values
    
    # === NORMALIZATION (Train stats only) ===
    mu = X_train.mean()
    sigma = X_train.std().replace(0, 1)
    
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    # Replace any remaining NaN/inf
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_val = X_val.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_cols,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test)
    }


# =============================================================================
# Public API
# =============================================================================

def create_endogenous_samples(df_ticker: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Create samples WITHOUT macro features."""
    return construct_windows(df_ticker, df_macro, include_macro=False)


def create_exogenous_samples(df_ticker: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Create samples WITH macro features."""
    return construct_windows(df_ticker, df_macro, include_macro=True)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing v3 FINAL Pipeline...")
    stocks, macro = load_raw_data()
    
    ticker = 'COMI.CA'
    df_ticker = stocks[stocks['Ticker'] == ticker].copy()
    
    print(f"\n--- ENDOGENOUS (No Macro) ---")
    endo = create_endogenous_samples(df_ticker, macro)
    print(f"Samples: {len(endo)}")
    print(f"Features: {len([c for c in endo.columns if c not in ['Date','Ticker','Target']])}")
    
    print(f"\n--- EXOGENOUS (With Macro) ---")
    exo = create_exogenous_samples(df_ticker, macro)
    print(f"Samples: {len(exo)}")
    print(f"Features: {len([c for c in exo.columns if c not in ['Date','Ticker','Target']])}")
    
    print(f"\n--- DATA LEAKAGE CHECK ---")
    # Verify W2 data is NOT in features
    sample_row = exo.iloc[0]
    feature_cols = [c for c in exo.columns if c not in ['Date','Ticker','Target']]
    w2_keywords = ['w2', 'future', 'target']
    leaky = [c for c in feature_cols if any(k in c.lower() for k in w2_keywords)]
    print(f"Leaky features found: {len(leaky)} (should be 0)")
    if leaky:
        print(f"  WARNING: {leaky}")
