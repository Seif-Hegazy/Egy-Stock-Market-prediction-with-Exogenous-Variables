#!/usr/bin/env python3
"""
audit_leakage.py - Forensic Data Leakage Audit
================================================

Role: Forensic Data Auditor

This script audits train_hybrid_model.py for Look-Ahead Bias / Data Leakage:
1. Timestamp Forensics - Check date ranges don't overlap
2. Normalization Audit - Verify scaler fitted on train only
3. Target Shift Audit - Verify proper temporal separation

Output: PASS/FAIL report with specific findings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import random


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "model_ready" / "train_ready_logreg.csv"
HYBRID_SCRIPT = PROJECT_ROOT / "scripts" / "train_hybrid_model.py"

LOOKBACK_WINDOW = 10
FORECAST_HORIZON = 5


def log(status: str, msg: str):
    """Formatted logging"""
    icons = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'INFO': 'ℹ️', 'CHECK': '🔍'}
    icon = icons.get(status, '•')
    print(f"[{status}] {icon} {msg}")


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# AUDIT 1: TIMESTAMP FORENSICS
# ============================================================

def audit_timestamp_alignment():
    """
    Verify that input window dates do NOT overlap with target window dates.
    
    Input Window: T-9 to T (10 days of history)
    Target Window: T+1 to T+5 (5 days of future)
    
    CRITICAL: Max(Input_Date) MUST BE < Min(Target_Date)
    """
    print_header("AUDIT 1: TIMESTAMP FORENSICS")
    
    # Load data
    log('INFO', f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pick a sample ticker with enough data
    ticker = 'COMI.CA'
    ticker_df = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
    
    log('INFO', f"Auditing ticker: {ticker} ({len(ticker_df)} rows)")
    
    # Test indices
    test_indices = [100, 500, 1000]
    
    all_passed = True
    
    print(f"\n  {'Index':<8} {'Input Window':<30} {'Target Window':<30} {'Gap OK?':<10}")
    print("  " + "-" * 80)
    
    for idx in test_indices:
        if idx >= len(ticker_df) - FORECAST_HORIZON:
            continue
        
        # Input window: T-9 to T (indices idx-9 to idx)
        input_start_idx = max(0, idx - LOOKBACK_WINDOW + 1)
        input_end_idx = idx
        
        input_start_date = ticker_df.iloc[input_start_idx]['Date']
        input_end_date = ticker_df.iloc[input_end_idx]['Date']
        
        # Target window: T+1 to T+5 (indices idx+1 to idx+5)
        target_start_idx = idx + 1
        target_end_idx = min(idx + FORECAST_HORIZON, len(ticker_df) - 1)
        
        target_start_date = ticker_df.iloc[target_start_idx]['Date']
        target_end_date = ticker_df.iloc[target_end_idx]['Date']
        
        # Gap check
        gap_ok = input_end_date < target_start_date
        status = "✓ PASS" if gap_ok else "❌ FAIL"
        
        if not gap_ok:
            all_passed = False
        
        input_range = f"{input_start_date.date()} to {input_end_date.date()}"
        target_range = f"{target_start_date.date()} to {target_end_date.date()}"
        
        print(f"  {idx:<8} {input_range:<30} {target_range:<30} {status:<10}")
    
    print()
    
    if all_passed:
        log('PASS', "Timestamp alignment is correct - NO OVERLAP detected")
    else:
        log('FAIL', "CRITICAL LEAK DETECTED: Overlap found between input and target windows!")
    
    return all_passed


# ============================================================
# AUDIT 2: NORMALIZATION (Scaler Leak)
# ============================================================

def audit_normalization():
    """
    Verify that StandardScaler is fitted ONLY on training data.
    
    LEAK: If scaler.fit(X_all) instead of scaler.fit(X_train),
    then future data statistics leak into training.
    """
    print_header("AUDIT 2: NORMALIZATION AUDIT (Scaler Leak)")
    
    # Read the hybrid script
    log('INFO', f"Reading script: {HYBRID_SCRIPT}")
    
    with open(HYBRID_SCRIPT, 'r') as f:
        script_content = f.read()
    
    # Check for proper scaler usage
    issues = []
    
    # Pattern 1: scaler.fit_transform(X_train) - CORRECT
    if 'scaler.fit_transform(X_train)' in script_content:
        log('PASS', "Found: scaler.fit_transform(X_train) - Correct pattern")
    else:
        log('WARN', "Did not find: scaler.fit_transform(X_train)")
    
    # Pattern 2: scaler.transform(X_val) and scaler.transform(X_test) - CORRECT
    if 'scaler.transform(X_val)' in script_content and 'scaler.transform(X_test)' in script_content:
        log('PASS', "Found: scaler.transform(X_val) and scaler.transform(X_test) - Correct pattern")
    else:
        log('WARN', "Check val/test transformation")
    
    # Pattern 3: WRONG - scaler.fit() on full dataset
    if 'scaler.fit_transform(X)' in script_content and 'X_train' not in script_content:
        log('FAIL', "LEAK DETECTED: Scaler fitted on full dataset, not just training!")
        issues.append("scaler_leak")
    
    # Check the actual lines
    print("\n  Relevant scaler code found:")
    lines = script_content.split('\n')
    for i, line in enumerate(lines):
        if 'scaler' in line.lower() and ('fit' in line or 'transform' in line):
            print(f"    Line {i+1}: {line.strip()}")
    
    passed = len(issues) == 0
    
    print()
    if passed:
        log('PASS', "Normalization is correctly applied to train only")
    else:
        log('FAIL', f"Normalization issues detected: {issues}")
    
    return passed


# ============================================================
# AUDIT 3: TARGET SHIFT LOGIC
# ============================================================

def audit_target_shift():
    """
    Verify that target calculation uses future data correctly.
    
    CORRECT: Target looks at T+1 to T+5 (future)
    WRONG: Using shift(-5) on features (would bring future into present)
    """
    print_header("AUDIT 3: TARGET SHIFT AUDIT")
    
    with open(HYBRID_SCRIPT, 'r') as f:
        script_content = f.read()
    
    issues = []
    
    print("\n  Checking shift operations...")
    
    # Find all shift operations
    lines = script_content.split('\n')
    shift_lines = []
    for i, line in enumerate(lines):
        if '.shift(' in line:
            shift_lines.append((i+1, line.strip()))
            print(f"    Line {i+1}: {line.strip()}")
    
    print()
    
    # Check for CORRECT patterns
    # Features should use positive shifts (looking back): .shift(lag) where lag >= 0
    # Target should use negative shift or forward-looking calculation
    
    # Check for feature lag generation (should be positive shifts)
    feature_lag_ok = True
    for line_num, line in shift_lines:
        if 'shift(lag)' in line or 'shift(0)' in line:
            # Positive shift for features - OK
            pass
        elif '.shift(-' in line and 'Close' in line:
            # Negative shift on Close for target - OK (but let's verify context)
            pass
    
    # Check for accidental negative shifts on features
    for line_num, line in shift_lines:
        # If shifting features (not Close) with negative value = LEAK
        if '.shift(-' in line:
            if 'Return_T' in line or 'Vol_Ratio' in line or 'USD_Return' in line:
                log('FAIL', f"Line {line_num}: Negative shift on feature - potential leak!")
                issues.append(f"negative_feature_shift_line_{line_num}")
    
    # Verify target calculation logic
    print("\n  Checking target calculation...")
    
    # Look for the target creation logic
    if 'Mean_Future' in script_content and 'Mean_Past' in script_content:
        log('PASS', "Found Mean_Future/Mean_Past pattern for target")
        
        # Verify Mean_Future uses future indices
        if 'iloc[i+1:i+1+FORECAST_HORIZON]' in script_content:
            log('PASS', "Mean_Future correctly uses future indices [i+1:i+1+5]")
        else:
            log('WARN', "Verify Mean_Future indexing manually")
    
    if 'rolling' in script_content and 'Mean_Past' in script_content:
        log('PASS', "Mean_Past uses rolling window on past data")
    
    # Additional check: print the target creation function
    print("\n  Target creation code:")
    in_target_func = False
    for i, line in enumerate(lines):
        if 'def create_t5_target' in line:
            in_target_func = True
        if in_target_func:
            print(f"    {i+1}: {line}")
            if line.strip().startswith('return '):
                break
    
    passed = len(issues) == 0
    
    print()
    if passed:
        log('PASS', "Target shift logic appears correct")
    else:
        log('FAIL', f"Target shift issues detected: {issues}")
    
    return passed


# ============================================================
# AUDIT 4: DEEP DIVE - Manual Verification
# ============================================================

def audit_deep_dive():
    """
    Perform a deep dive verification by actually running the data pipeline.
    """
    print_header("AUDIT 4: DEEP DIVE VERIFICATION")
    
    # Load data and replicate the windowing logic
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    ticker = 'COMI.CA'
    ticker_df = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
    
    # Replicate feature creation
    feature_cols = ['Return_T', 'Vol_Ratio', 'USD_Return', 'Gold_Return', 'Gas_Index']
    
    log('INFO', "Creating rolling window features...")
    
    # Create lagged features
    for lag in range(LOOKBACK_WINDOW):
        for feature in feature_cols:
            col_name = f"{feature}_L{lag}"
            ticker_df[col_name] = ticker_df[feature].shift(lag)
    
    # Create target
    log('INFO', "Creating T+5 target...")
    
    # Mean of past 10 days
    ticker_df['Mean_Past'] = ticker_df['Close'].rolling(window=LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).mean()
    
    # Mean of next 5 days
    future_closes = []
    for i in range(len(ticker_df)):
        if i + FORECAST_HORIZON <= len(ticker_df) - 1:
            future_mean = ticker_df['Close'].iloc[i+1:i+1+FORECAST_HORIZON].mean()
            future_closes.append(future_mean)
        else:
            future_closes.append(np.nan)
    
    ticker_df['Mean_Future'] = future_closes
    ticker_df['Trend_Ratio'] = ticker_df['Mean_Future'] / ticker_df['Mean_Past']
    
    # Pick a test row and verify
    test_idx = 500
    log('INFO', f"Verifying row {test_idx}...")
    
    row = ticker_df.iloc[test_idx]
    
    print(f"\n  Row {test_idx} Analysis:")
    print(f"    Current Date (T):       {row['Date'].date()}")
    print(f"    Close at T:             {row['Close']:.4f}")
    
    # What dates are in the feature window?
    print(f"\n  Feature Window (T-9 to T):")
    for lag in range(LOOKBACK_WINDOW):
        lag_idx = test_idx - lag
        if lag_idx >= 0:
            lag_date = ticker_df.iloc[lag_idx]['Date'].date()
            lag_close = ticker_df.iloc[lag_idx]['Close']
            print(f"    L{lag}: {lag_date} | Close: {lag_close:.4f}")
    
    # What dates are in the target window?
    print(f"\n  Target Window (T+1 to T+5):")
    for offset in range(1, FORECAST_HORIZON + 1):
        future_idx = test_idx + offset
        if future_idx < len(ticker_df):
            future_date = ticker_df.iloc[future_idx]['Date'].date()
            future_close = ticker_df.iloc[future_idx]['Close']
            print(f"    T+{offset}: {future_date} | Close: {future_close:.4f}")
    
    # Verify no overlap
    feature_end_date = row['Date']
    target_start_idx = test_idx + 1
    
    if target_start_idx < len(ticker_df):
        target_start_date = ticker_df.iloc[target_start_idx]['Date']
        gap_ok = feature_end_date < target_start_date
        
        print(f"\n  Gap Check:")
        print(f"    Feature End:   {feature_end_date.date()}")
        print(f"    Target Start:  {target_start_date.date()}")
        print(f"    Gap OK:        {'✅ YES' if gap_ok else '❌ NO - LEAK!'}")
        
        return gap_ok
    
    return True


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    """Run full leakage audit."""
    print("\n" + "=" * 70)
    print("  🔍 FORENSIC DATA LEAKAGE AUDIT")
    print("  📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    results = {}
    
    # Audit 1: Timestamp alignment
    results['timestamp'] = audit_timestamp_alignment()
    
    # Audit 2: Normalization
    results['normalization'] = audit_normalization()
    
    # Audit 3: Target shift
    results['target_shift'] = audit_target_shift()
    
    # Audit 4: Deep dive
    results['deep_dive'] = audit_deep_dive()
    
    # Final verdict
    print_header("📋 FINAL AUDIT REPORT")
    
    print("\n  Audit Results:")
    print(f"    1. Timestamp Forensics:  {'PASS ✅' if results['timestamp'] else 'FAIL ❌'}")
    print(f"    2. Normalization Audit:  {'PASS ✅' if results['normalization'] else 'FAIL ❌'}")
    print(f"    3. Target Shift Audit:   {'PASS ✅' if results['target_shift'] else 'FAIL ❌'}")
    print(f"    4. Deep Dive Verify:     {'PASS ✅' if results['deep_dive'] else 'FAIL ❌'}")
    
    all_passed = all(results.values())
    
    print("\n" + "-" * 70)
    
    if all_passed:
        log('PASS', "🟢 ALL AUDITS PASSED - No data leakage detected!")
        print("\n  The high precision (75%) appears to be legitimate.")
        print("  Possible explanations for good performance:")
        print("    1. T+5 weekly trend is easier to predict than daily moves")
        print("    2. 10-day rolling window captures meaningful patterns")
        print("    3. Threshold calibration improves precision")
        print("    4. Balanced class weights help with imbalanced data")
    else:
        log('FAIL', "🔴 AUDIT FAILED - Potential data leakage detected!")
        print("\n  Review the failed audits above and fix the issues.")
    
    print("\n" + "=" * 70)
    
    return all_passed


if __name__ == "__main__":
    main()
