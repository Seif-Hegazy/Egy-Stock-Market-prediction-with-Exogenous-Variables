#!/usr/bin/env python3
"""
prepare_training_data.py - Master Fix Data Preparation Script
==============================================================

Role: Senior Quantitative Developer & Data Engineer

This script:
1. Ingests raw stock and economic data
2. Fixes all identified quality issues
3. Outputs a clean, model-ready dataset

Input:
    - data/stocks/raw/egx_daily_12y.csv
    - data/economic/egypt_economic_data.csv

Output:
    - data/model_ready/train_ready_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
STOCK_FILE = PROJECT_ROOT / "data" / "raw" / "stocks" / "egx_daily_12y.csv"
ECON_FILE = PROJECT_ROOT / "data" / "raw" / "economic" / "egypt_economic_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
OUTPUT_FILE = OUTPUT_DIR / "train_ready_data.csv"

# Feature Engineering Parameters
VOLUME_MA_WINDOW = 20
BUFFER_THRESHOLD = 0.005  # 0.5% noise filter


def log(msg: str):
    """Timestamped logging"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================
# 1. DATA CLEANING (The "Suspended Days" Fix)
# ============================================================

def load_and_clean_stock_data(filepath: Path) -> pd.DataFrame:
    """
    Load stock data and remove suspended/zero-volume days.
    """
    log(f"📂 Loading stock data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Count before
    total_rows = len(df)
    
    # === FIX: Drop zero volume (suspended days) ===
    df = df[df['Volume'] > 0].copy()
    
    dropped = total_rows - len(df)
    pct = (dropped / total_rows) * 100
    log(f"🗑️  Dropped {dropped:,} rows ({pct:.1f}%) due to zero volume/suspension.")
    
    # Sort by Ticker and Date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    log(f"✓ Stock data: {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    return df


def load_economic_data(filepath: Path) -> pd.DataFrame:
    """
    Load economic/macro data.
    """
    log(f"📂 Loading economic data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    
    log(f"✓ Economic data: {len(df):,} rows")
    return df


# ============================================================
# 2. FEATURE ENGINEERING (Stationarity & Ratios)
# ============================================================

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stationary price-based features (ratios, not raw values).
    """
    log("⚙️  Creating price-based features...")
    
    # --- Ratio Features (Stationary by construction) ---
    
    # Open/Close Ratio: Measures overnight gap vs. intraday movement
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    
    # High/Low Ratio: Intraday volatility
    df['High_Low_Ratio'] = df['High'] / df['Low']
    
    # Volume Relative to 20-day MA
    df['Volume_MA_20'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=VOLUME_MA_WINDOW, min_periods=1).mean()
    )
    df['Volume_Relative_20'] = df['Volume'] / df['Volume_MA_20']
    
    # Price Rate of Change (1-day return) - grouped by ticker!
    df['Price_ROC_1'] = df.groupby('Ticker')['Close'].pct_change()
    
    log("   ✓ Open_Close_Ratio, High_Low_Ratio, Volume_Relative_20, Price_ROC_1")
    return df


# ============================================================
# 3. MACRO-ECONOMIC LOGIC (Granular Economic Features)
# ============================================================

def create_economic_features(econ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create granular economic features (separate, not combined proxies).
    
    Features created:
    - USD_Return_Daily: Daily % change in official USD rate
    - Gold_Return_Daily: Daily % change in gold price
    - Gas_Return: % change in gasoline price (sparse, step-function)
    - Gas_Update_Flag: 1 if gas price changed today, else 0
    - GDP_Growth: % change in GDP (sparse, annual signal)
    """
    log("⚙️  Creating granular economic features...")
    
    df = econ_df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # --- 1. FX Features (Separate, not combined) ---
    df['USD_Return_Daily'] = df['usd_sell_rate'].pct_change()
    df['Gold_Return_Daily'] = df['gold_24k'].pct_change()
    
    log("   ✓ USD_Return_Daily, Gold_Return_Daily")
    
    # --- 2. Gasoline Features ---
    # Gas prices are step-functions: stable for months, then jump
    # Use gasoline_92 as benchmark
    if 'gasoline_92' in df.columns:
        df['Gas_Return'] = df['gasoline_92'].pct_change()
        # Flag: 1 if price changed today (significant event)
        df['Gas_Update_Flag'] = (df['Gas_Return'].abs() > 0.001).astype(int)
        log("   ✓ Gas_Return, Gas_Update_Flag (from gasoline_92)")
    else:
        df['Gas_Return'] = 0
        df['Gas_Update_Flag'] = 0
        log("   ⚠️  gasoline_92 not found, Gas features set to 0")
    
    # --- 3. GDP Feature ---
    if 'gdp_usd_billion' in df.columns:
        # GDP is annual, forward-filled daily
        # pct_change will be 0 most days, signals annual shift
        df['GDP_Growth'] = df['gdp_usd_billion'].pct_change()
        # Fill NaN/inf with 0
        df['GDP_Growth'] = df['GDP_Growth'].replace([np.inf, -np.inf], 0).fillna(0)
        log("   ✓ GDP_Growth (annual signal)")
    else:
        df['GDP_Growth'] = 0
        log("   ⚠️  gdp_usd_billion not found, GDP_Growth set to 0")
    
    # --- 4. Keep headline_inflation and cbe_lending_rate as-is ---
    # These are already in appropriate form (forward-filled from announcements)
    log("   ✓ headline_inflation, cbe_lending_rate (kept as-is)")
    
    return df


def merge_economic_data(stock_df: pd.DataFrame, econ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left join economic data onto stock data, with forward-fill for gaps.
    """
    log("🔗 Merging stock and economic data...")
    
    # Prepare economic data with all features
    econ_df = create_economic_features(econ_df)
    
    # Select columns to merge (granular features)
    econ_cols = [
        'date',
        # FX features
        'USD_Return_Daily',
        'Gold_Return_Daily',
        # Gasoline features
        'Gas_Return',
        'Gas_Update_Flag',
        # GDP
        'GDP_Growth',
        # Inflation & rates
        'headline_inflation',
        'cbe_lending_rate',
    ]
    
    # Only include columns that exist
    available_econ = [c for c in econ_cols if c in econ_df.columns]
    econ_subset = econ_df[available_econ].copy()
    econ_subset = econ_subset.rename(columns={'date': 'Date'})
    
    # Left join (keep all stock rows)
    merged = stock_df.merge(econ_subset, on='Date', how='left')
    
    # Forward-fill for level variables (inflation, rates)
    # These don't change daily, so forward-fill is correct
    level_vars = ['headline_inflation', 'cbe_lending_rate']
    for col in level_vars:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    
    # Return variables: Fill NaN with 0 (no change on that day)
    return_vars = ['USD_Return_Daily', 'Gold_Return_Daily', 'Gas_Return', 'GDP_Growth']
    for col in return_vars:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    
    # Flag variables: Fill with 0
    flag_vars = ['Gas_Update_Flag']
    for col in flag_vars:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    
    # Fill any remaining NaN in level vars with median
    for col in level_vars:
        if col in merged.columns:
            median_val = merged[col].median()
            missing = merged[col].isna().sum()
            if missing > 0:
                log(f"   ⚠️  {missing} rows missing {col}, filling with median ({median_val:.2f})")
                merged[col] = merged[col].fillna(median_val)
    
    log(f"   ✓ Merged: {len(merged):,} rows with {len(available_econ)-1} economic features")
    return merged


# ============================================================
# 4. TARGET GENERATION (The "Noise Buffer" Fix)
# ============================================================

def create_buffered_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Target_Buffered with noise filtering.
    
    Logic:
    - Compute forward 1-day return (what we're predicting)
    - Apply buffer: Only count as BUY if return > 0.5%
    - Drop last row of each ticker (no future data)
    """
    log("🎯 Creating Target_Buffered...")
    
    # Next-day return (forward-looking) - MUST be grouped by ticker!
    df['Next_Return'] = df.groupby('Ticker')['Close'].shift(-1) / df['Close'] - 1
    
    # Buffered target: 1 = Strong Buy, 0 = Hold/Sell
    df['Target_Buffered'] = np.where(
        df['Next_Return'] > BUFFER_THRESHOLD,
        1,  # Buy signal (strong upward movement)
        0   # Hold/Sell (neutral or down)
    )
    
    # Count distribution before dropping
    buy_count = (df['Target_Buffered'] == 1).sum()
    hold_count = (df['Target_Buffered'] == 0).sum()
    
    # Drop last row of each ticker (target is NaN/undefined)
    df = df.groupby('Ticker').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    
    # Recalculate after dropping
    final_buy = (df['Target_Buffered'] == 1).sum()
    final_hold = (df['Target_Buffered'] == 0).sum()
    
    log(f"   Buffer threshold: {BUFFER_THRESHOLD*100:.1f}%")
    log(f"   ✓ Target distribution:")
    log(f"      Buy (1):  {final_buy:,} ({final_buy/(final_buy+final_hold)*100:.1f}%)")
    log(f"      Hold (0): {final_hold:,} ({final_hold/(final_buy+final_hold)*100:.1f}%)")
    
    return df


# ============================================================
# 5. FINAL CLEANUP & OUTPUT
# ============================================================

def finalize_and_save(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Select final columns and save to CSV.
    
    CRITICAL: Exclude raw prices to prevent data leakage!
    """
    log("🧹 Final cleanup...")
    
    # === KEEP: These are safe features ===
    keep_columns = [
        # Keys
        'Date',
        'Ticker',
        'Sector',  # Useful for grouping
        
        # TARGET
        'Target_Buffered',
        
        # Price-based ratios (stationary)
        'Open_Close_Ratio',
        'High_Low_Ratio',
        'Volume_Relative_20',
        'Price_ROC_1',
        
        # Macro features (granular)
        'USD_Return_Daily',
        'Gold_Return_Daily',
        'Gas_Return',
        'Gas_Update_Flag',
        'GDP_Growth',
        'headline_inflation',
        'cbe_lending_rate',
        
        # Reference (for analysis, but exclude from model features)
        'Close',  # Keep for reference/debugging only
    ]
    
    # Filter to available columns
    available = [c for c in keep_columns if c in df.columns]
    missing = [c for c in keep_columns if c not in df.columns]
    
    if missing:
        log(f"   ⚠️  Missing columns (skipped): {missing}")
    
    final_df = df[available].copy()
    
    # === VERIFY: Raw prices are EXCLUDED ===
    dangerous_cols = ['Open', 'High', 'Low', 'Volume', 'usd_sell_rate', 'gold_24k', 'gasoline_92', 'gdp_usd_billion']
    leaked = [c for c in dangerous_cols if c in final_df.columns]
    
    if leaked:
        log(f"   ❌ ERROR: Data leakage detected! Columns still present: {leaked}")
        raise ValueError(f"Data leakage: {leaked}")
    else:
        log(f"   ✓ Safety check passed: Raw prices excluded")
    
    # Drop any remaining NaN rows
    before_drop = len(final_df)
    final_df = final_df.dropna()
    dropped = before_drop - len(final_df)
    if dropped > 0:
        log(f"   ✓ Dropped {dropped:,} rows with NaN values")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    final_df.to_csv(output_path, index=False)
    log(f"💾 Saved to: {output_path}")
    
    return final_df


def print_verification(df: pd.DataFrame):
    """Print verification info to prove data is clean."""
    print("\n" + "=" * 60)
    print("📋 VERIFICATION")
    print("=" * 60)
    
    print(f"\n📊 Final Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"📅 Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"📈 Tickers: {df['Ticker'].nunique()}")
    
    print("\n📝 Final Columns:")
    for i, col in enumerate(df.columns):
        print(f"   {i+1}. {col}")
    
    print("\n🔍 Sample Data (first 5 rows):")
    print(df.head().to_string())
    
    print("\n📊 Target Distribution:")
    target_dist = df['Target_Buffered'].value_counts()
    for val, count in target_dist.items():
        label = "Buy" if val == 1 else "Hold/Sell"
        print(f"   {label} ({val}): {count:,} ({count/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute the full data preparation pipeline."""
    print("\n" + "=" * 60)
    print("🚀 MASTER FIX DATA PREPARATION SCRIPT")
    print("=" * 60 + "\n")
    
    # 1. Load and clean stock data
    stock_df = load_and_clean_stock_data(STOCK_FILE)
    
    # 2. Load economic data
    econ_df = load_economic_data(ECON_FILE)
    
    # 3. Create price-based features
    stock_df = create_price_features(stock_df)
    
    # 4. Merge with economic data (including FX impulse)
    merged_df = merge_economic_data(stock_df, econ_df)
    
    # 5. Create buffered target
    merged_df = create_buffered_target(merged_df)
    
    # 6. Finalize and save
    final_df = finalize_and_save(merged_df, OUTPUT_FILE)
    
    # 7. Verification
    print_verification(final_df)
    
    return final_df


if __name__ == "__main__":
    main()
