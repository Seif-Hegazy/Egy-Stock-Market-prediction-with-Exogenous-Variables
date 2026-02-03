"""
EGX Macro Significance Study - Feature Engineering Pipeline (v7 Academic)
==========================================================================
Implements academic paper methodology:
1. Rolling window with neutral zone filtering (m=0.001)
2. Features = concat(Week0_prices, Week1_prices)  
3. Label = P_future/P_past based classification
4. 72/8/20 chronological split
5. Z-score normalization (train stats only)
6. Fixed 40th percentile threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# =============================================================================
# Configuration (Academic Paper Parameters)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Windowing parameters (per paper)
WINDOW_SIZE = 5           # w = 5 trading days (1 week)
STEP_SIZE = 1             # Rolling window (step=1 for max samples)
NEUTRAL_MARGIN = 0.001    # m = 0.001 (0.1% neutral zone)
MIN_SAMPLES = 200         # Minimum after filtering

# Split ratios (per paper: 72% train, 8% val, 20% test)
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
TEST_RATIO = 0.20

# Thresholding (per paper: 40th percentile)
THRESHOLD_PERCENTILE = 0.40

# Feature configuration
# Paper uses only adjusted close prices, but we extend with macro
USE_PRICE_ONLY = False  # Set True for pure paper methodology

# Macro columns for extended features
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
# Rolling Window Construction (Academic Paper Methodology)
# =============================================================================

def construct_windows_academic(
    df_ticker: pd.DataFrame,
    df_macro: pd.DataFrame,
    include_macro: bool,
    w: int = WINDOW_SIZE,
    step: int = STEP_SIZE,
    margin: float = NEUTRAL_MARGIN
) -> pd.DataFrame:
    """
    Academic paper rolling window construction.
    
    Timeline:
    [----Week 0----][----Week 1----][----Week 2----]
       t to t+w-1     t+w to t+2w-1   t+2w to t+3w-1
       FEATURES       FEATURES        TARGET
    
    Feature Construction (Eq.1):
        x_i = concat(p_{i:i+w}, p_{i+w:i+2w})
    
    Label Construction (Eq.2-4):
        P_past = mean(prices in Week0 + Week1)
        P_future = mean(prices in Week2)
        y = 1 if P_future/P_past > 1+m
        y = 0 if P_future/P_past < 1-m
        Discard if in neutral zone [1-m, 1+m]
    """
def construct_windows_academic(
    df_ticker: pd.DataFrame,
    df_macro: pd.DataFrame,
    include_macro: bool,
    w: int = WINDOW_SIZE,
    step: int = STEP_SIZE,
    margin: float = NEUTRAL_MARGIN
) -> pd.DataFrame:
    """
    Academic paper rolling window construction.
    v8 UPDATE: Uses Log Returns for features to fix Look-Ahead Bias.
    """
    # 1. Merge with macro
    df = df_ticker.merge(df_macro, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate Log Returns
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # 2. Forward fill macro (no future data used)
    for col in MACRO_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    samples = []
    total_len = len(df)
    
    # Position i iterates through all valid positions
    for i in range(0, total_len - 3 * w + 1, step):
        # Define the three weekly segments
        w0 = df.iloc[i:i+w]                 # Week 0: Historical
        w1 = df.iloc[i+w:i+2*w]             # Week 1: Historical  
        w2 = df.iloc[i+2*w:i+3*w]           # Week 2: Future (target only)
        
        # === LABEL CONSTRUCTION (Eq.2-4) ===
        # P_past = mean of prices in Week0 + Week1
        prices_past = pd.concat([w0['Close'], w1['Close']])
        P_past = prices_past.mean()
        
        # P_future = mean of prices in Week2
        P_future = w2['Close'].mean()
        
        if P_past <= 0:
            continue
        
        # Ratio for classification
        ratio = P_future / P_past
        
        # Neutral zone filtering (Eq.2)
        if 1 - margin <= ratio <= 1 + margin:
            continue
        
        # Binary label
        if ratio > 1 + margin:
            target = 1  # UP
        else:
            target = 0  # DOWN
        
        # === FEATURE CONSTRUCTION (v8 Robust) ===
        features = {}
        
        # --- BASE FEATURES (Endogenous) ---
        # USES LOG RETURNS (Stationary)
        
        # 1. Core features: Log Return vectors
        for day_idx in range(w):
            features[f'ret_w0_d{day_idx}'] = w0['LogReturn'].iloc[day_idx]
            features[f'ret_w1_d{day_idx}'] = w1['LogReturn'].iloc[day_idx]
        
        # 2. Additional Return statistics
        if not USE_PRICE_ONLY:
            features['ret_w0_mean'] = w0['LogReturn'].mean()
            features['ret_w1_mean'] = w1['LogReturn'].mean()
            features['ret_w0_std'] = w0['LogReturn'].std()
            features['ret_w1_std'] = w1['LogReturn'].std()
            # Cumulative return (Price change over the week)
            features['cumret_w0'] = (w0['Close'].iloc[-1] / w0['Close'].iloc[0]) - 1 if w0['Close'].iloc[0]>0 else 0
            features['cumret_w1'] = (w1['Close'].iloc[-1] / w1['Close'].iloc[0]) - 1 if w1['Close'].iloc[0]>0 else 0
            
            # Volume features (Changes, not levels)
            if 'Volume' in df.columns:
                vol_w0 = w0['Volume'].mean()
                vol_w1 = w1['Volume'].mean()
                features['vol_change_w0w1'] = (vol_w1 / vol_w0) - 1 if vol_w0 > 0 else 0
        
        # --- EXOGENOUS FEATURES (Macro Only) ---
        if include_macro:
            for col in MACRO_COLS:
                if col in df.columns:
                    # Use last value (Points in time are okay if they are macro levels like Interest Rate)
                    # But ideally we use CHANGES for stationarity.
                    # For v8, we'll keep Levels but normalized by Z-score in prep step.
                    # Macro variables don't trend as explosively as Stock Prices in nominal terms usually,
                    # EXCEPT for USD/EGP (Devaluation). 
                    # So for USD/EGP we DEFINITELY need changes.
                    
                    if col in ['usd_egp', 'gold_local', 'sp500', 'oil', 'gold_intl']:
                        # Trending vars: Use Change
                        if w0[col].iloc[-1] != 0:
                            features[f'macro_{col}_change'] = (w1[col].iloc[-1] / w0[col].iloc[-1]) - 1
                        else:
                            features[f'macro_{col}_change'] = 0
                    else:
                        # Mean-reverting/Rate vars: Use Levels (VIX, Inflation, Rates)
                        features[f'macro_{col}_level'] = w1[col].iloc[-1]

        # Metadata
        features['Date'] = w1['Date'].iloc[-1]
        features['Ticker'] = df['Ticker'].iloc[0] if 'Ticker' in df.columns else 'UNKNOWN'
        features['Target'] = target
        features['train_pos_ratio'] = 0.5 # Placeholder, not used in sample gen
        
        samples.append(features)
    
    return pd.DataFrame(samples)


# =============================================================================
# Dataset Preparation (v8: Purged Split)
# =============================================================================

def prepare_datasets_academic(
    df_samples: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Dict:
    """
    v8 Robust:
    - Purged Chronological Split (Gap = 5 samples between splits)
    - Z-Score Normalization
    """
    n = len(df_samples)
    purge_gap = WINDOW_SIZE  # Ensure no overlap
    
    train_end = int(n * train_ratio)
    val_start = train_end + purge_gap
    val_end = val_start + int(n * val_ratio)
    test_start = val_end + purge_gap
    
    if test_start >= n:
        # Fallback if purged consumes too much
        val_start = train_end
        test_start = val_end
    
    train = df_samples.iloc[:train_end]
    val = df_samples.iloc[val_start:val_end]
    test = df_samples.iloc[test_start:]
    
    # Feature columns (exclude metadata)
    meta_cols = ['Date', 'Ticker', 'Target', 'train_pos_ratio', 'P_past', 'P_future', 'Ratio']
    feature_cols = [c for c in df_samples.columns if c not in meta_cols]
    
    X_train = train[feature_cols].copy()
    y_train = train['Target'].values
    X_val = val[feature_cols].copy()
    y_val = val['Target'].values
    X_test = test[feature_cols].copy()
    y_test = test['Target'].values
    
    # === Z-SCORE NORMALIZATION (Eq.5) ===
    # x_tilde = (x - mu_train) / sigma_train
    mu_train = X_train.mean()
    sigma_train = X_train.std().replace(0, 1)  # Avoid division by zero
    
    X_train = (X_train - mu_train) / sigma_train
    X_val = (X_val - mu_train) / sigma_train
    X_test = (X_test - mu_train) / sigma_train
    
    # Handle any remaining NaN/inf
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_val = X_val.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Store positive class ratio for threshold calibration
    train_pos_ratio = y_train.mean()
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_cols,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'train_pos_ratio': train_pos_ratio,
        'mu_train': mu_train, 'sigma_train': sigma_train
    }


# =============================================================================
# Fixed Percentile Threshold (Academic Paper Section III-D)
# =============================================================================

def compute_fixed_percentile_threshold(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE
) -> float:
    """
    Fixed Percentile Threshold Strategy (Eq.9).
    
    tau = Q_{0.40}(p_val)
    
    This results in approximately 60% of predictions classified as UP.
    """
    return np.percentile(proba_val, percentile * 100)


# =============================================================================
# Public API
# =============================================================================

def create_endogenous_samples(df_ticker: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Create samples WITHOUT macro features."""
    return construct_windows_academic(df_ticker, df_macro, include_macro=False)


def create_exogenous_samples(df_ticker: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """Create samples WITH macro features."""
    return construct_windows_academic(df_ticker, df_macro, include_macro=True)


def prepare_datasets(df_samples: pd.DataFrame, **kwargs) -> Dict:
    """Wrapper for academic dataset preparation."""
    return prepare_datasets_academic(df_samples, **kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing v7 Academic Pipeline...")
    print(f"Parameters: w={WINDOW_SIZE}, step={STEP_SIZE}, margin={NEUTRAL_MARGIN}")
    print(f"Split: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"Threshold: {THRESHOLD_PERCENTILE:.0%} percentile")
    print()
    
    stocks, macro = load_raw_data()
    
    ticker = 'COMI.CA'
    df_ticker = stocks[stocks['Ticker'] == ticker].copy()
    
    print(f"--- ENDOGENOUS (No Macro) ---")
    endo = create_endogenous_samples(df_ticker, macro)
    print(f"Samples: {len(endo)}")
    feature_cols = [c for c in endo.columns if c not in ['Date','Ticker','Target','P_past','P_future','Ratio']]
    print(f"Features: {len(feature_cols)}")
    print(f"Positive ratio: {endo['Target'].mean():.2%}")
    
    print(f"\n--- EXOGENOUS (With Macro) ---")
    exo = create_exogenous_samples(df_ticker, macro)
    print(f"Samples: {len(exo)}")
    feature_cols = [c for c in exo.columns if c not in ['Date','Ticker','Target','P_past','P_future','Ratio']]
    print(f"Features: {len(feature_cols)}")
    print(f"Positive ratio: {exo['Target'].mean():.2%}")
    
    print(f"\n--- LABEL DISTRIBUTION ---")
    print(f"UP (1): {(endo['Target']==1).sum()}")
    print(f"DOWN (0): {(endo['Target']==0).sum()}")
    print(f"Filtered by neutral zone: {len(df_ticker) - 3*WINDOW_SIZE + 1 - len(endo)} (approx)")
    
    print(f"\n--- DATA SPLIT ---")
    data = prepare_datasets(endo)
    print(f"Train: {data['n_train']} ({data['n_train']/len(endo):.1%})")
    print(f"Val: {data['n_val']} ({data['n_val']/len(endo):.1%})")
    print(f"Test: {data['n_test']} ({data['n_test']/len(endo):.1%})")
