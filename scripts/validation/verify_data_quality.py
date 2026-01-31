#!/usr/bin/env python3
"""
verify_data_quality.py - Pre-Training Data Quality Audit
=========================================================

Role: Lead Data Quality Assurance Engineer

This script audits train_ready_data.csv for:
1. Data Leakage (Perfect Predictor Trap)
2. Class Imbalance (All Zeros Risk)
3. Per-Ticker Health (Single-Stock Model Viability)
4. Stationarity & Range Sanity

Output: Go / No-Go Report with Safe Tickers list
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "model_ready" / "train_ready_data.csv"
SAFE_TICKERS_FILE = PROJECT_ROOT / "data" / "model_ready" / "safe_tickers.txt"

# Thresholds
LEAKAGE_THRESHOLD = 0.95     # Correlation threshold for leakage
MIN_CLASS_RATIO = 0.10       # Minimum acceptable minority class ratio
MAX_CLASS_RATIO = 0.90       # Maximum acceptable majority class ratio
MIN_ROWS_PER_TICKER = 1000   # ~4 years of daily data
RATIO_MAX = 2.0              # Max acceptable ratio value
RATIO_MIN = 0.1              # Min acceptable ratio value

# Dangerous columns that should NOT be in features
FORBIDDEN_COLUMNS = ['Open', 'High', 'Low', 'Volume', 'usd_sell_rate', 'gold_24k', 'Next_Return']


def log(status: str, msg: str):
    """Formatted logging with status indicator"""
    icons = {
        'PASS': '✅',
        'WARN': '⚠️',
        'FAIL': '❌',
        'INFO': 'ℹ️',
        'CHECK': '🔍'
    }
    icon = icons.get(status, '•')
    print(f"[{status}] {icon} {msg}")


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# AUDIT FUNCTIONS
# ============================================================

def audit_leakage(df: pd.DataFrame) -> tuple:
    """
    Audit Check 1: The "Perfect Predictor" Trap
    
    Calculate correlation of every feature against Target_Buffered.
    Flag if any feature has correlation > 0.95 or < -0.95.
    """
    print_header("AUDIT 1: Data Leakage Check (Perfect Predictor Trap)")
    
    # Get numeric columns (excluding target and non-features)
    exclude = ['Date', 'Ticker', 'Sector', 'Target_Buffered']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
    
    correlations = {}
    for col in feature_cols:
        corr = df[col].corr(df['Target_Buffered'])
        correlations[col] = corr
    
    # Check for leakage
    max_corr = max(correlations.values(), key=abs)
    max_corr_feature = [k for k, v in correlations.items() if v == max_corr][0]
    
    leaked_features = [k for k, v in correlations.items() if abs(v) > LEAKAGE_THRESHOLD]
    
    passed = len(leaked_features) == 0
    
    if passed:
        log('PASS', f"No Leaks detected (Max Correlation: {max_corr:.4f} for '{max_corr_feature}')")
    else:
        log('FAIL', f"LEAKAGE DETECTED! Features with correlation > {LEAKAGE_THRESHOLD}:")
        for feat in leaked_features:
            log('FAIL', f"  - {feat}: {correlations[feat]:.4f}")
    
    # Show all correlations for transparency
    print("\n  Feature Correlations with Target:")
    for col, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {col:25s}: {corr:+.4f}")
    
    return passed, correlations


def audit_class_balance(df: pd.DataFrame) -> tuple:
    """
    Audit Check 2: Class Balance (The "All Zeros" Risk)
    
    Check if Target=1 is between 10% and 90%.
    """
    print_header("AUDIT 2: Class Balance Check")
    
    # Global balance
    target_counts = df['Target_Buffered'].value_counts()
    total = len(df)
    buy_ratio = target_counts.get(1, 0) / total
    hold_ratio = target_counts.get(0, 0) / total
    
    global_passed = MIN_CLASS_RATIO <= buy_ratio <= MAX_CLASS_RATIO
    
    log('INFO', f"Global Target Distribution:")
    log('INFO', f"  Buy (1):       {target_counts.get(1, 0):,} ({buy_ratio*100:.1f}%)")
    log('INFO', f"  Hold/Sell (0): {target_counts.get(0, 0):,} ({hold_ratio*100:.1f}%)")
    
    if global_passed:
        log('PASS', f"Global class balance is healthy ({buy_ratio*100:.1f}% Buy)")
    else:
        log('FAIL', f"Class imbalance detected! Buy ratio {buy_ratio*100:.1f}% outside [{MIN_CLASS_RATIO*100:.0f}%, {MAX_CLASS_RATIO*100:.0f}%]")
    
    # Sample 3 tickers
    sample_tickers = df['Ticker'].unique()[:3]
    print("\n  Sample Ticker Distribution:")
    
    ticker_issues = []
    for ticker in sample_tickers:
        ticker_df = df[df['Ticker'] == ticker]
        ticker_buy_ratio = ticker_df['Target_Buffered'].mean()
        status = "OK" if MIN_CLASS_RATIO <= ticker_buy_ratio <= MAX_CLASS_RATIO else "WARN"
        print(f"    {ticker:12s}: Buy = {ticker_buy_ratio*100:5.1f}% [{status}]")
        if status == "WARN":
            ticker_issues.append(ticker)
    
    if ticker_issues:
        log('WARN', f"Some tickers have extreme class ratios: {ticker_issues}")
    
    return global_passed, buy_ratio


def audit_ticker_health(df: pd.DataFrame) -> tuple:
    """
    Audit Check 3: Per-Ticker Health
    
    Identify tickers with < 1000 rows (insufficient for training).
    """
    print_header("AUDIT 3: Per-Ticker Health Check")
    
    ticker_counts = df.groupby('Ticker').size().sort_values(ascending=False)
    
    healthy_tickers = ticker_counts[ticker_counts >= MIN_ROWS_PER_TICKER].index.tolist()
    unhealthy_tickers = ticker_counts[ticker_counts < MIN_ROWS_PER_TICKER].index.tolist()
    
    log('INFO', f"Total Tickers: {len(ticker_counts)}")
    log('INFO', f"Healthy Tickers (>= {MIN_ROWS_PER_TICKER} rows): {len(healthy_tickers)}")
    
    if unhealthy_tickers:
        log('WARN', f"Tickers with insufficient data:")
        for ticker in unhealthy_tickers:
            log('WARN', f"  - {ticker}: {ticker_counts[ticker]} rows (need {MIN_ROWS_PER_TICKER})")
    else:
        log('PASS', f"All tickers have sufficient data (>= {MIN_ROWS_PER_TICKER} rows)")
    
    # Show top and bottom 5
    print("\n  Rows per Ticker (Top 5):")
    for ticker in ticker_counts.head(5).index:
        print(f"    {ticker:12s}: {ticker_counts[ticker]:,} rows")
    
    print("\n  Rows per Ticker (Bottom 5):")
    for ticker in ticker_counts.tail(5).index:
        status = "⚠️" if ticker_counts[ticker] < MIN_ROWS_PER_TICKER else "✓"
        print(f"    {ticker:12s}: {ticker_counts[ticker]:,} rows {status}")
    
    passed = len(unhealthy_tickers) == 0
    return passed, healthy_tickers, unhealthy_tickers


def audit_stationarity_sanity(df: pd.DataFrame) -> tuple:
    """
    Audit Check 4: Stationarity & Range Sanity
    
    Check ratio features are in reasonable ranges.
    Verify no forbidden columns exist.
    """
    print_header("AUDIT 4: Stationarity & Range Sanity Check")
    
    issues = []
    
    # Check for forbidden columns
    forbidden_found = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
    
    if forbidden_found:
        for col in forbidden_found:
            log('FAIL', f"Forbidden column '{col}' found in dataset! Remove immediately.")
            issues.append(f"Forbidden: {col}")
    else:
        log('PASS', "No forbidden columns (raw prices) detected.")
    
    # Check ratio features
    ratio_cols = ['Open_Close_Ratio', 'High_Low_Ratio', 'Volume_Relative_20']
    
    print("\n  Ratio Feature Ranges:")
    for col in ratio_cols:
        if col not in df.columns:
            log('WARN', f"Expected column '{col}' not found")
            continue
        
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        
        range_ok = RATIO_MIN <= col_min and col_max <= RATIO_MAX
        
        status = "✓" if range_ok else "⚠️"
        print(f"    {col:25s}: min={col_min:.4f}, max={col_max:.4f}, mean={col_mean:.4f} {status}")
        
        if col_max > RATIO_MAX:
            log('WARN', f"'{col}' has suspicious max value {col_max:.4f} (expected <= {RATIO_MAX})")
            issues.append(f"Range: {col} max too high")
        
        if col_min < RATIO_MIN:
            log('WARN', f"'{col}' has suspicious min value {col_min:.4f} (expected >= {RATIO_MIN})")
            issues.append(f"Range: {col} min too low")
    
    # Check 'Close' column (should be reference only, warn if present)
    if 'Close' in df.columns:
        log('WARN', "'Close' column present. Ensure it's NOT used as a feature during training!")
    
    passed = len(forbidden_found) == 0
    return passed, issues


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run full data quality audit."""
    print("\n" + "=" * 60)
    print("  🔍 PRE-TRAINING DATA QUALITY AUDIT")
    print("  📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Load data
    log('INFO', f"Loading data from: {DATA_FILE}")
    
    if not DATA_FILE.exists():
        log('FAIL', f"Data file not found! Run prepare_training_data.py first.")
        return
    
    df = pd.read_csv(DATA_FILE)
    log('INFO', f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Run audits
    results = {}
    
    # Audit 1: Leakage
    results['leakage'], _ = audit_leakage(df)
    
    # Audit 2: Class Balance
    results['balance'], _ = audit_class_balance(df)
    
    # Audit 3: Ticker Health
    results['ticker_health'], safe_tickers, unsafe_tickers = audit_ticker_health(df)
    
    # Audit 4: Stationarity
    results['stationarity'], _ = audit_stationarity_sanity(df)
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print_header("🎯 FINAL VERDICT")
    
    all_passed = all(results.values())
    critical_passed = results['leakage'] and results['stationarity']
    
    print("\n  Audit Results Summary:")
    print(f"    1. Leakage Check:     {'PASS ✅' if results['leakage'] else 'FAIL ❌'}")
    print(f"    2. Class Balance:     {'PASS ✅' if results['balance'] else 'WARN ⚠️'}")
    print(f"    3. Ticker Health:     {'PASS ✅' if results['ticker_health'] else 'WARN ⚠️'}")
    print(f"    4. Stationarity:      {'PASS ✅' if results['stationarity'] else 'FAIL ❌'}")
    
    print("\n" + "-" * 60)
    
    if critical_passed:
        log('PASS', "🟢 GO FOR TRAINING - Critical checks passed!")
        
        # Save safe tickers
        SAFE_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SAFE_TICKERS_FILE, 'w') as f:
            for ticker in safe_tickers:
                f.write(ticker + '\n')
        
        log('INFO', f"Safe tickers ({len(safe_tickers)}) saved to: {SAFE_TICKERS_FILE}")
        
        if unsafe_tickers:
            log('WARN', f"Exclude these tickers from training: {unsafe_tickers}")
        
        print("\n  📋 SAFE TICKERS FOR TRAINING:")
        for i, ticker in enumerate(safe_tickers, 1):
            print(f"    {i:2d}. {ticker}")
        
    else:
        log('FAIL', "🔴 NO-GO - Critical issues detected! Fix before training.")
        
        if not results['leakage']:
            log('FAIL', "  → Remove leaked features from dataset")
        if not results['stationarity']:
            log('FAIL', "  → Remove forbidden columns and fix ratio ranges")
    
    print("\n" + "=" * 60)
    print("  📊 Audit Complete")
    print("=" * 60 + "\n")
    
    return all_passed, safe_tickers


if __name__ == "__main__":
    main()
