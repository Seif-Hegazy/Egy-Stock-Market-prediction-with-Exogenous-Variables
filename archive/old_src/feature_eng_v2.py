"""
EGX Prediction Model v2 - Feature Engineering
Research-backed features from:
- Moskowitz, Ooi, Pedersen (2012) - Time Series Momentum
- Asness, Moskowitz, Pedersen (2013) - Value and Momentum Everywhere
- Gu, Kelly, Xiu (2020) - Empirical Asset Pricing via ML
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# =============================================================================
# Feature Definitions
# =============================================================================

TECHNICAL_FEATURES = [
    # Returns (momentum) - Moskowitz et al.
    'ret_1w', 'ret_2w', 'ret_4w', 'ret_8w', 'ret_12w',
    
    # Volatility
    'vol_4w', 'vol_8w',
    
    # Range and volume
    'weekly_range', 'volume_ratio_4w',
    
    # Mean reversion
    'rsi_4w', 'price_vs_ma4', 'price_vs_ma8',
]

GLOBAL_FEATURES = [
    # S&P 500 momentum
    'sp500_ret_1w', 'sp500_ret_4w',
    
    # VIX (risk sentiment) - Bekaert & Harvey
    'vix_level', 'vix_change_1w', 'vix_ma_ratio',
    
    # Oil momentum
    'oil_ret_1w', 'oil_ret_4w',
    
    # Gold momentum
    'gold_intl_ret_1w', 'gold_intl_ret_4w',
    
    # EM momentum
    'msci_em_ret_1w', 'msci_em_ret_4w',
    
    # EUR/USD (dollar strength)
    'eur_usd_ret_1w', 'eur_usd_ret_4w',
    
    # Cross-asset signals
    'em_vs_sp500',  # EM relative performance
    'oil_gold_ratio',  # Commodity risk
]

EGYPT_FEATURES = [
    # USD/EGP
    'usd_egp_change_1w', 'usd_egp_change_4w',
    
    # Local gold
    'gold_local_change_1w', 'gold_local_change_4w',
    
    # Gold premium (local vs international)
    'gold_premium',
    
    # Real interest rate
    'real_rate',
]

ALL_FEATURES = TECHNICAL_FEATURES + GLOBAL_FEATURES + EGYPT_FEATURES


# =============================================================================
# Technical Features
# =============================================================================

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-horizon returns (momentum factors).
    Based on Moskowitz, Ooi, Pedersen (2012).
    """
    df = df.copy()
    
    for weeks in [1, 2, 4, 8, 12]:
        df[f'ret_{weeks}w'] = df.groupby('Ticker')['Close'].pct_change(weeks)
    
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute realized volatility (weekly)."""
    df = df.copy()
    
    # First compute weekly returns if not present
    if 'ret_1w' not in df.columns:
        df['ret_1w'] = df.groupby('Ticker')['Close'].pct_change(1)
    
    for weeks in [4, 8]:
        df[f'vol_{weeks}w'] = df.groupby('Ticker')['ret_1w'].transform(
            lambda x: x.rolling(window=weeks, min_periods=2).std()
        )
    
    return df


def compute_range_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly range and volume features."""
    df = df.copy()
    
    # Weekly range
    df['weekly_range'] = (df['High'] - df['Low']) / df['Close']
    
    # Volume ratio vs 4-week MA
    df['volume_ma_4w'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=4, min_periods=2).mean()
    )
    df['volume_ratio_4w'] = df['Volume'] / df['volume_ma_4w']
    df = df.drop(columns=['volume_ma_4w'])
    
    return df


def compute_rsi(df: pd.DataFrame, period: int = 4) -> pd.DataFrame:
    """Compute weekly RSI."""
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
    
    df[f'rsi_{period}w'] = df.groupby('Ticker')['Close'].transform(calc_rsi)
    
    return df


def compute_price_vs_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Price relative to moving averages (mean reversion signal)."""
    df = df.copy()
    
    for weeks in [4, 8]:
        ma = df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window=weeks, min_periods=2).mean()
        )
        df[f'price_vs_ma{weeks}'] = df['Close'] / ma - 1
    
    return df


# =============================================================================
# Global Features
# =============================================================================

def compute_global_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum for global indicators.
    Based on Asness, Moskowitz, Pedersen (2013).
    """
    df = df.copy()
    
    global_cols = ['sp500', 'oil', 'gold_intl', 'msci_em', 'eur_usd']
    
    for col in global_cols:
        if col not in df.columns:
            continue
        
        for weeks in [1, 4]:
            df[f'{col}_ret_{weeks}w'] = df.groupby('Ticker')[col].pct_change(weeks)
    
    return df


def compute_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIX-based features.
    VIX level and changes are strong EM predictors (Bekaert & Harvey 1997).
    """
    df = df.copy()
    
    if 'vix' not in df.columns:
        return df
    
    # VIX level (normalized)
    df['vix_level'] = df['vix']
    
    # VIX change
    df['vix_change_1w'] = df.groupby('Ticker')['vix'].pct_change(1)
    
    # VIX vs 4-week MA (fear spike indicator)
    df['vix_ma_4w'] = df.groupby('Ticker')['vix'].transform(
        lambda x: x.rolling(window=4, min_periods=2).mean()
    )
    df['vix_ma_ratio'] = df['vix'] / df['vix_ma_4w']
    df = df.drop(columns=['vix_ma_4w'])
    
    return df


def compute_cross_asset_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-asset momentum signals.
    """
    df = df.copy()
    
    # EM vs S&P500 relative performance
    if 'msci_em_ret_1w' in df.columns and 'sp500_ret_1w' in df.columns:
        df['em_vs_sp500'] = df['msci_em_ret_1w'] - df['sp500_ret_1w']
    
    # Oil/Gold ratio (commodity risk)
    if 'oil' in df.columns and 'gold_intl' in df.columns:
        df['oil_gold_ratio'] = df['oil'] / df['gold_intl']
    
    return df


# =============================================================================
# Egypt Features
# =============================================================================

def compute_egypt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Egypt-specific macro features."""
    df = df.copy()
    
    # USD/EGP changes
    if 'usd_egp' in df.columns:
        for weeks in [1, 4]:
            df[f'usd_egp_change_{weeks}w'] = df.groupby('Ticker')['usd_egp'].pct_change(weeks)
    
    # Local gold changes
    if 'gold_local' in df.columns:
        for weeks in [1, 4]:
            df[f'gold_local_change_{weeks}w'] = df.groupby('Ticker')['gold_local'].pct_change(weeks)
    
    # Gold premium (local vs international, accounts for FX)
    if 'gold_local' in df.columns and 'gold_intl' in df.columns and 'usd_egp' in df.columns:
        # Convert international gold to EGP
        gold_intl_egp = df['gold_intl'] * df['usd_egp']
        df['gold_premium'] = (df['gold_local'] / gold_intl_egp) - 1
    
    # Real interest rate (monetary stance)
    if 'cbe_rate' in df.columns and 'inflation' in df.columns:
        df['real_rate'] = df['cbe_rate'] - df['inflation']
    
    return df


# =============================================================================
# Main Engineering Function
# =============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering.
    """
    print("Engineering features (v2 - research-backed)...")
    
    print("  Computing returns...")
    df = compute_returns(df)
    
    print("  Computing volatility...")
    df = compute_volatility(df)
    
    print("  Computing range & volume...")
    df = compute_range_volume(df)
    
    print("  Computing RSI...")
    df = compute_rsi(df)
    
    print("  Computing price vs MA...")
    df = compute_price_vs_ma(df)
    
    print("  Computing global momentum...")
    df = compute_global_momentum(df)
    
    print("  Computing VIX features...")
    df = compute_vix_features(df)
    
    print("  Computing cross-asset signals...")
    df = compute_cross_asset_signals(df)
    
    print("  Computing Egypt features...")
    df = compute_egypt_features(df)
    
    # Count available features
    available = [f for f in ALL_FEATURES if f in df.columns]
    print(f"  Total features: {len(available)}/{len(ALL_FEATURES)}")
    
    return df


def get_feature_matrix(df: pd.DataFrame, 
                       feature_set: str = 'all') -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract feature matrix for modeling.
    
    Args:
        df: DataFrame with features
        feature_set: 'technical', 'global', 'egypt', 'all'
    """
    if feature_set == 'technical':
        features = TECHNICAL_FEATURES
    elif feature_set == 'global':
        features = GLOBAL_FEATURES
    elif feature_set == 'egypt':
        features = EGYPT_FEATURES
    else:
        features = ALL_FEATURES
    
    available = [f for f in features if f in df.columns]
    
    return df[available], available


if __name__ == '__main__':
    from data_loader_v2 import load_and_prepare_weekly_data
    
    df = load_and_prepare_weekly_data()
    df = engineer_all_features(df)
    
    print("\nFeatures available:")
    for cat, feats in [('Technical', TECHNICAL_FEATURES), 
                       ('Global', GLOBAL_FEATURES), 
                       ('Egypt', EGYPT_FEATURES)]:
        avail = [f for f in feats if f in df.columns]
        print(f"  {cat}: {len(avail)}/{len(feats)}")
