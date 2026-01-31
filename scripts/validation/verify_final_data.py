#!/usr/bin/env python3
"""
verify_final_data.py - Final Data Quality Verification
=======================================================

Role: Senior Data Quality Engineer

This script audits train_ready_data.csv to confirm:
1. New granular economic features exist
2. Macro logic is mathematically sound
3. Stationarity & scale is correct
4. No forbidden columns (sentiment, raw prices)

Output: READY / REVIEW status
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

# Required columns (must exist)
REQUIRED_ECON_COLUMNS = [
    'USD_Return_Daily',
    'Gold_Return_Daily',
    'Gas_Return',
    'GDP_Growth',
    'cbe_lending_rate',
]

# Forbidden columns (must NOT exist)
FORBIDDEN_COLUMNS = [
    # Raw prices
    'Open', 'High', 'Low', 'Volume',
    'usd_sell_rate', 'gold_24k', 'gasoline_92', 'gdp_usd_billion',
    # Sentiment
    'direct_sentiment', 'sector_sentiment', 'direct_count', 'sector_count',
    'sentiment', 'score', 'sentiment_score',
]

# Thresholds
GAS_ZERO_THRESHOLD = 0.95  # Gas should be 0 for >95% of rows


def log(status: str, msg: str):
    """Formatted logging"""
    icons = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'INFO': 'ℹ️', 'CHECK': '🔍'}
    icon = icons.get(status, '•')
    print(f"[{status}] {icon} {msg}")


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# VERIFICATION FUNCTIONS
# ============================================================

def verify_column_existence(df: pd.DataFrame) -> tuple:
    """
    Verification 1: Check that required economic columns exist.
    """
    print_header("VERIFY 1: Required Columns Check")
    
    missing = [c for c in REQUIRED_ECON_COLUMNS if c not in df.columns]
    
    if missing:
        log('FAIL', f"CRITICAL FAIL - Missing required columns: {missing}")
        return False, missing
    else:
        log('PASS', f"All {len(REQUIRED_ECON_COLUMNS)} required economic columns present:")
        for col in REQUIRED_ECON_COLUMNS:
            print(f"      ✓ {col}")
        return True, []


def verify_gasoline_logic(df: pd.DataFrame) -> tuple:
    """
    Verification 2a: Gasoline sanity check.
    
    Expectation: Gas prices rarely change.
    Gas_Return should be 0.0 for >95% of rows.
    """
    print_header("VERIFY 2a: Gasoline Logic Check")
    
    if 'Gas_Return' not in df.columns:
        log('FAIL', "Gas_Return column missing!")
        return False, 0, 0
    
    gas_returns = df['Gas_Return']
    
    # Count zeros (or near-zero due to float precision)
    zero_count = (gas_returns.abs() < 0.0001).sum()
    total = len(gas_returns)
    zero_pct = zero_count / total
    
    # Count spikes (non-zero changes)
    spike_count = total - zero_count
    
    log('INFO', f"Gas_Return Statistics:")
    log('INFO', f"  - Zero values: {zero_count:,} ({zero_pct*100:.1f}%)")
    log('INFO', f"  - Spike events: {spike_count:,}")
    
    if zero_pct == 0:
        log('FAIL', "Gas_Return is NEVER zero! Logic is wrong (likely capturing noise).")
        passed = False
    elif zero_pct < GAS_ZERO_THRESHOLD:
        log('WARN', f"Gas_Return is zero only {zero_pct*100:.1f}% of the time (expected >{GAS_ZERO_THRESHOLD*100:.0f}%)")
        passed = False
    else:
        log('PASS', f"Gas_Return is sparse as expected ({zero_pct*100:.1f}% zeros)")
        passed = True
    
    # Show some actual spikes
    if spike_count > 0:
        spikes = gas_returns[gas_returns.abs() > 0.0001]
        print(f"\n  Sample Gas Price Changes (first 5):")
        for idx, val in spikes.head(5).items():
            print(f"    Row {idx}: {val:+.4f} ({val*100:+.2f}%)")
    
    return passed, spike_count, zero_pct


def verify_usd_gold_separation(df: pd.DataFrame) -> tuple:
    """
    Verification 2b: USD vs Gold correlation check.
    
    They should be correlated but different.
    If correlation == 1.0, the separation failed.
    """
    print_header("VERIFY 2b: USD/Gold Separation Check")
    
    if 'USD_Return_Daily' not in df.columns or 'Gold_Return_Daily' not in df.columns:
        log('FAIL', "Missing USD_Return_Daily or Gold_Return_Daily!")
        return False, None
    
    usd = df['USD_Return_Daily']
    gold = df['Gold_Return_Daily']
    
    # Calculate correlation
    correlation = usd.corr(gold)
    
    log('INFO', f"Correlation(USD_Return, Gold_Return) = {correlation:.4f}")
    
    if abs(correlation) > 0.999:
        log('FAIL', "Correlation is ~1.0! Features are duplicates, not separate.")
        passed = False
    elif abs(correlation) > 0.9:
        log('WARN', f"High correlation ({correlation:.4f}). Features may be too similar.")
        passed = True  # Warning but not fail
    else:
        log('PASS', f"Features are appropriately separated (corr = {correlation:.4f})")
        passed = True
    
    # Show some statistics
    print(f"\n  USD_Return_Daily:  mean={usd.mean():.6f}, std={usd.std():.6f}")
    print(f"  Gold_Return_Daily: mean={gold.mean():.6f}, std={gold.std():.6f}")
    
    return passed, correlation


def verify_scale_and_outliers(df: pd.DataFrame) -> tuple:
    """
    Verification 3: Check scale and outliers.
    """
    print_header("VERIFY 3: Scale & Outlier Check")
    
    issues = []
    
    # GDP Growth check
    if 'GDP_Growth' in df.columns:
        gdp = df['GDP_Growth']
        gdp_max = gdp.max()
        gdp_min = gdp.min()
        
        print(f"\n  GDP_Growth:")
        print(f"    min: {gdp_min:.6f}, max: {gdp_max:.6f}, mean: {gdp.mean():.6f}")
        
        if gdp_max > 1.0:  # >100% growth is suspicious
            log('WARN', f"GDP_Growth max ({gdp_max:.4f}) seems too high. Check source data.")
            issues.append("GDP_Growth scale")
        else:
            log('PASS', "GDP_Growth scale looks reasonable")
    
    # CBE Lending Rate check
    if 'cbe_lending_rate' in df.columns:
        rate = df['cbe_lending_rate']
        rate_min = rate.min()
        rate_max = rate.max()
        rate_mean = rate.mean()
        
        print(f"\n  cbe_lending_rate:")
        print(f"    min: {rate_min:.2f}, max: {rate_max:.2f}, mean: {rate_mean:.2f}")
        
        # Determine scale (decimal vs percentage)
        if rate_max <= 1.0:
            scale = "Decimal (0-1)"
        elif rate_max <= 100:
            scale = "Percentage (0-100)"
        else:
            scale = "UNKNOWN (possibly wrong units)"
            issues.append("cbe_lending_rate scale")
        
        log('INFO', f"cbe_lending_rate appears to be in: {scale}")
    
    # Headline inflation check
    if 'headline_inflation' in df.columns:
        infl = df['headline_inflation']
        print(f"\n  headline_inflation:")
        print(f"    min: {infl.min():.2f}, max: {infl.max():.2f}, mean: {infl.mean():.2f}")
        
        if infl.max() <= 1.0:
            log('INFO', "headline_inflation appears to be in: Decimal (0-1)")
        elif infl.max() <= 100:
            log('INFO', "headline_inflation appears to be in: Percentage (0-100)")
    
    # Return features check
    return_cols = ['USD_Return_Daily', 'Gold_Return_Daily', 'Gas_Return', 'Price_ROC_1']
    print(f"\n  Return Features (expected range: -0.5 to +0.5):")
    
    for col in return_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            suspicious = col_max > 1.0 or col_min < -1.0
            status = "⚠️" if suspicious else "✓"
            print(f"    {col:20s}: [{col_min:+.4f}, {col_max:+.4f}] {status}")
            if suspicious:
                issues.append(f"{col} range")
    
    passed = len(issues) == 0
    return passed, issues


def verify_no_forbidden_columns(df: pd.DataFrame) -> tuple:
    """
    Verification 4: Confirm NO forbidden columns exist.
    """
    print_header("VERIFY 4: Forbidden Columns Check")
    
    found_forbidden = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
    
    if found_forbidden:
        log('FAIL', f"FORBIDDEN COLUMNS DETECTED: {found_forbidden}")
        for col in found_forbidden:
            log('FAIL', f"  ❌ {col} - REMOVE IMMEDIATELY!")
        return False, found_forbidden
    else:
        log('PASS', "No forbidden columns (raw prices, sentiment) detected")
        return True, []


def generate_summary_report(results: dict) -> bool:
    """
    Generate final summary report.
    """
    print_header("📊 FINAL SUMMARY REPORT")
    
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Gasoline Spikes Detected:    {results['gas_spikes']:5d} times           │")
    print(f"  │  Gold/USD Correlation:        {results['usd_gold_corr']:+.4f}               │")
    print(f"  │  Data Scale:                  {results['rate_scale']:20s} │")
    print("  └─────────────────────────────────────────────────────┘")
    
    # Determine final status
    critical_passed = results['columns_ok'] and results['no_forbidden']
    logic_ok = results['gas_ok'] and results['usd_gold_ok']
    scale_ok = results['scale_ok']
    
    print("\n  Verification Summary:")
    print(f"    1. Required Columns:    {'PASS ✅' if results['columns_ok'] else 'FAIL ❌'}")
    print(f"    2a. Gasoline Logic:     {'PASS ✅' if results['gas_ok'] else 'WARN ⚠️'}")
    print(f"    2b. USD/Gold Separation:{'PASS ✅' if results['usd_gold_ok'] else 'WARN ⚠️'}")
    print(f"    3. Scale & Outliers:    {'PASS ✅' if results['scale_ok'] else 'WARN ⚠️'}")
    print(f"    4. No Forbidden Cols:   {'PASS ✅' if results['no_forbidden'] else 'FAIL ❌'}")
    
    print("\n" + "-" * 60)
    
    if critical_passed and logic_ok and scale_ok:
        log('PASS', "🟢 READY - Data is verified and ready for training!")
        status = "READY"
    elif critical_passed:
        log('WARN', "🟡 REVIEW - Data passes critical checks but has warnings.")
        status = "REVIEW"
    else:
        log('FAIL', "🔴 NOT READY - Critical issues must be fixed before training!")
        status = "NOT_READY"
    
    print(f"\n  Final Status: [{status}]")
    print("\n" + "=" * 60)
    
    return status == "READY"


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run full verification audit."""
    print("\n" + "=" * 60)
    print("  🔍 FINAL DATA QUALITY VERIFICATION")
    print("  📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Load data
    log('INFO', f"Loading data from: {DATA_FILE}")
    
    if not DATA_FILE.exists():
        log('FAIL', f"Data file not found! Run prepare_training_data.py first.")
        return False
    
    df = pd.read_csv(DATA_FILE)
    log('INFO', f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Store results
    results = {}
    
    # Verify 1: Column existence
    results['columns_ok'], missing = verify_column_existence(df)
    
    # Verify 2a: Gasoline logic
    results['gas_ok'], results['gas_spikes'], gas_zero_pct = verify_gasoline_logic(df)
    
    # Verify 2b: USD/Gold separation
    results['usd_gold_ok'], results['usd_gold_corr'] = verify_usd_gold_separation(df)
    if results['usd_gold_corr'] is None:
        results['usd_gold_corr'] = 0.0
    
    # Verify 3: Scale & outliers
    results['scale_ok'], scale_issues = verify_scale_and_outliers(df)
    
    # Determine rate scale for report
    if 'cbe_lending_rate' in df.columns:
        rate_max = df['cbe_lending_rate'].max()
        if rate_max <= 1.0:
            results['rate_scale'] = "Decimal (0-1)"
        elif rate_max <= 100:
            results['rate_scale'] = "Percentage (0-100)"
        else:
            results['rate_scale'] = "Unknown"
    else:
        results['rate_scale'] = "N/A"
    
    # Verify 4: No forbidden columns
    results['no_forbidden'], forbidden_found = verify_no_forbidden_columns(df)
    
    # Generate summary
    all_passed = generate_summary_report(results)
    
    return all_passed


if __name__ == "__main__":
    main()
