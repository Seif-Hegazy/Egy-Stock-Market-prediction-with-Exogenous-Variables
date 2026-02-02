"""
EGX Macro Significance Study - Feature Engineering Module
Implements endogenous and exogenous feature sets.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# =============================================================================
# Feature Definitions
# =============================================================================

ENDOGENOUS_FEATURES = [
    # Returns
    'return_1d', 'return_5d', 'return_10d',
    
    # Volatility
    'volatility_5d', 'volatility_10d', 'volatility_20d',
    
    # Price ratios
    'high_low_range', 'close_open_ratio',
    
    # Volume
    'volume_ma_ratio_5', 'volume_ma_ratio_10',
    
    # Technical
    'rsi_14',
    
    # Regime
    'is_extreme_volatility',
]

# 10-day lookback features (flattened OHLCV)
for i in range(1, 11):
    ENDOGENOUS_FEATURES.extend([
        f'close_norm_lag{i}',
        f'volume_norm_lag{i}',
    ])

EXOGENOUS_FEATURES = [
    # USD/EGP
    'usd_sell_rate_lag1', 'usd_sell_rate_lag2', 'usd_sell_rate_lag3', 'usd_sell_rate_lag5',
    'usd_change_1d', 'usd_change_5d',
    
    # Gold
    'gold_24k_lag1', 'gold_24k_lag2', 'gold_24k_lag3', 'gold_24k_lag5',
    'gold_change_1d', 'gold_change_5d',
    
    # Interest rates
    'cbe_deposit_rate_lag1',
    'cbe_lending_rate_lag1',
    'rate_spread_lag1',
    
    # Inflation
    'headline_inflation_lag1',
    
    # Calendar
    'holiday_gap',
    'is_post_holiday',
]

ALL_FEATURES = ENDOGENOUS_FEATURES + EXOGENOUS_FEATURES


# =============================================================================
# Feature Engineering Functions
# =============================================================================

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute return features."""
    df = df.copy()
    
    for period in [1, 5, 10]:
        df[f'return_{period}d'] = df.groupby('Ticker')['Close'].pct_change(period)
    
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling volatility features."""
    df = df.copy()
    
    # First compute daily returns if not present
    if 'return_1d' not in df.columns:
        df['return_1d'] = df.groupby('Ticker')['Close'].pct_change(1)
    
    for window in [5, 10, 20]:
        df[f'volatility_{window}d'] = df.groupby('Ticker')['return_1d'].transform(
            lambda x: x.rolling(window=window, min_periods=max(3, window//2)).std()
        )
    
    return df


def compute_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-based ratio features."""
    df = df.copy()
    
    # High-Low range as % of close
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    
    # Close/Open ratio
    df['close_open_ratio'] = df['Close'] / df['Open']
    
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features."""
    df = df.copy()
    
    for window in [5, 10]:
        ma = df.groupby('Ticker')['Volume'].transform(
            lambda x: x.rolling(window=window, min_periods=2).mean()
        )
        df[f'volume_ma_ratio_{window}'] = df['Volume'] / ma
    
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute RSI indicator."""
    df = df.copy()
    
    def calc_rsi(prices):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_14'] = df.groupby('Ticker')['Close'].transform(calc_rsi)
    
    return df


def compute_adaptive_volatility_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute adaptive volatility flag using 3-sigma rule.
    
    is_extreme_volatility = 1 if |return| > 3 * rolling_std_20
    """
    df = df.copy()
    
    if 'return_1d' not in df.columns:
        df['return_1d'] = df.groupby('Ticker')['Close'].pct_change(1)
    
    # 20-day rolling standard deviation
    df['rolling_std_20'] = df.groupby('Ticker')['return_1d'].transform(
        lambda x: x.rolling(window=20, min_periods=10).std()
    )
    
    # Adaptive extreme flag
    df['is_extreme_volatility'] = (
        df['return_1d'].abs() > 3 * df['rolling_std_20']
    ).astype(int)
    
    return df


def compute_lookback_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Compute flattened lookback features (normalized).
    
    Creates close_norm_lag1, close_norm_lag2, ..., close_norm_lag{lookback}
    """
    df = df.copy()
    
    # Normalized close (relative to current close)
    for lag in range(1, lookback + 1):
        lagged = df.groupby('Ticker')['Close'].shift(lag)
        df[f'close_norm_lag{lag}'] = lagged / df['Close']
    
    # Normalized volume (relative to current volume)
    for lag in range(1, lookback + 1):
        lagged = df.groupby('Ticker')['Volume'].shift(lag)
        df[f'volume_norm_lag{lag}'] = lagged / (df['Volume'] + 1)  # +1 to avoid div by zero
    
    return df


def apply_rate_spread_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lag to rate spread."""
    df = df.copy()
    
    if 'rate_spread' in df.columns:
        df['rate_spread_lag1'] = df.groupby('Ticker')['rate_spread'].shift(1)
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to engineer all features.
    
    Args:
        df: DataFrame from data_loader with merged stock/macro data
        
    Returns:
        DataFrame with all engineered features
    """
    print("Engineering features...")
    
    print("  Computing returns...")
    df = compute_returns(df)
    
    print("  Computing volatility...")
    df = compute_volatility(df)
    
    print("  Computing price ratios...")
    df = compute_price_ratios(df)
    
    print("  Computing volume features...")
    df = compute_volume_features(df)
    
    print("  Computing RSI...")
    df = compute_rsi(df)
    
    print("  Computing adaptive volatility flag...")
    df = compute_adaptive_volatility_flag(df)
    
    print("  Computing lookback features...")
    df = compute_lookback_features(df)
    
    print("  Computing rate spread lag...")
    df = apply_rate_spread_lag(df)
    
    # Drop intermediate columns
    df = df.drop(columns=['rolling_std_20'], errors='ignore')
    
    print(f"  Total features available: {len([c for c in df.columns if c not in ['Date', 'Ticker', 'ISIN', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']])}")
    
    return df


def get_feature_matrix(df: pd.DataFrame, feature_set: str = 'all') -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract feature matrix for modeling.
    
    Args:
        df: DataFrame with all features
        feature_set: 'endogenous', 'exogenous', or 'all'
        
    Returns:
        (DataFrame with features, list of feature names used)
    """
    if feature_set == 'endogenous':
        features = [f for f in ENDOGENOUS_FEATURES if f in df.columns]
    elif feature_set == 'exogenous':
        features = [f for f in EXOGENOUS_FEATURES if f in df.columns]
    else:  # 'all'
        features = [f for f in ALL_FEATURES if f in df.columns]
    
    return df[features], features


if __name__ == '__main__':
    from data_loader import load_and_prepare_data
    
    df = load_and_prepare_data()
    df = engineer_all_features(df)
    
    print("\nEndogenous features available:")
    endo = [f for f in ENDOGENOUS_FEATURES if f in df.columns]
    print(f"  {len(endo)}/{len(ENDOGENOUS_FEATURES)}")
    
    print("\nExogenous features available:")
    exo = [f for f in EXOGENOUS_FEATURES if f in df.columns]
    print(f"  {len(exo)}/{len(EXOGENOUS_FEATURES)}")
