#!/usr/bin/env python3
"""
prepare_logreg_data.py - Logistic Regression Dataset Preparation
=================================================================

Role: Lead Data Scientist & Forensic Data Auditor

This script creates a mathematically perfect dataset for Logistic Regression:
1. Source Data Forensics (Reality Check)
2. Feature Engineering (Lag "Memory" Layer)
3. Target & Sanitization (Outlier Removal)
4. Optimization (Collinearity & Scaling)
5. Final Output & Balance Check

Output: train_ready_logreg.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
STOCK_FILE = PROJECT_ROOT / "data" / "raw" / "stocks" / "egx_daily_12y.csv"
ECON_FILE = PROJECT_ROOT / "data" / "raw" / "economic" / "egypt_economic_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
OUTPUT_FILE = OUTPUT_DIR / "train_ready_logreg.csv"

# Parameters
BUFFER_THRESHOLD = 0.005      # 0.5% buffer for target
RETURN_CLIP_MIN = -0.50       # Maximum allowed daily drop
RETURN_CLIP_MAX = 0.50        # Maximum allowed daily gain
COLLINEARITY_THRESHOLD = 0.85 # Drop features with corr > this
LAG_PERIODS = [1, 2]          # Lag 1 and Lag 2


def log(msg: str, level: str = "INFO"):
    """Formatted logging"""
    icons = {'INFO': 'ℹ️', 'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'DATA': '📊'}
    icon = icons.get(level, '•')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")


def print_header(phase: int, title: str):
    """Print phase header"""
    print("\n" + "=" * 70)
    print(f"  PHASE {phase}: {title}")
    print("=" * 70)


# ============================================================
# PHASE 1: SOURCE DATA FORENSICS
# ============================================================

def phase1_data_forensics() -> tuple:
    """
    Load raw data and generate health report.
    """
    print_header(1, "SOURCE DATA FORENSICS (Reality Check)")
    
    # --- Load Stock Data ---
    log(f"Loading stock data: {STOCK_FILE}")
    stock_df = pd.read_csv(STOCK_FILE)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    log(f"Stock data: {len(stock_df):,} rows, {stock_df['Ticker'].nunique()} tickers")
    
    # --- Load Economic Data ---
    log(f"Loading economic data: {ECON_FILE}")
    econ_df = pd.read_csv(ECON_FILE)
    econ_df['date'] = pd.to_datetime(econ_df['date'])
    log(f"Economic data: {len(econ_df):,} rows")
    
    # === DATA HEALTH REPORT ===
    print("\n" + "-" * 50)
    print("  📋 DATA HEALTH REPORT")
    print("-" * 50)
    
    # Price Check
    close_lt_001 = (stock_df['Close'] < 0.01).sum()
    close_nan = stock_df['Close'].isna().sum()
    print(f"\n  Price Check:")
    print(f"    Close < 0.01:  {close_lt_001:,} rows")
    print(f"    Close is NaN:  {close_nan:,} rows")
    
    if close_lt_001 > 0 or close_nan > 0:
        log("Price anomalies detected!", "WARN")
    else:
        log("Price data looks clean", "PASS")
    
    # Event Scan - Top 3 Biggest Moves
    stock_df['Daily_Return'] = stock_df.groupby('Ticker')['Close'].pct_change()
    
    print(f"\n  🔻 Top 3 BIGGEST DAILY DROPS:")
    biggest_drops = stock_df.nsmallest(3, 'Daily_Return')[['Date', 'Ticker', 'Close', 'Daily_Return']]
    for _, row in biggest_drops.iterrows():
        print(f"    {row['Date'].date()} | {row['Ticker']:10s} | {row['Daily_Return']*100:+.1f}% | Close: {row['Close']:.4f}")
    
    print(f"\n  🔺 Top 3 BIGGEST DAILY GAINS:")
    biggest_gains = stock_df.nlargest(3, 'Daily_Return')[['Date', 'Ticker', 'Close', 'Daily_Return']]
    for _, row in biggest_gains.iterrows():
        print(f"    {row['Date'].date()} | {row['Ticker']:10s} | {row['Daily_Return']*100:+.1f}% | Close: {row['Close']:.4f}")
    
    # Macro Check
    print(f"\n  Macro Data Check:")
    usd_empty = econ_df['usd_sell_rate'].isna().all()
    gold_empty = econ_df['gold_24k'].isna().all()
    gas_exists = 'gasoline_92' in econ_df.columns
    
    print(f"    usd_sell_rate: {'❌ EMPTY' if usd_empty else '✓ OK'}")
    print(f"    gold_24k:      {'❌ EMPTY' if gold_empty else '✓ OK'}")
    print(f"    gasoline_92:   {'✓ OK' if gas_exists else '❌ MISSING'}")
    
    # Drop the temporary column
    stock_df = stock_df.drop(columns=['Daily_Return'])
    
    return stock_df, econ_df


# ============================================================
# PHASE 2: FEATURE ENGINEERING (The "Memory" Layer)
# ============================================================

def phase2_feature_engineering(stock_df: pd.DataFrame, econ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features with lag memory for Logistic Regression.
    """
    print_header(2, "FEATURE ENGINEERING (The 'Memory' Layer)")
    
    # --- Filter: Drop Volume == 0 ---
    before = len(stock_df)
    stock_df = stock_df[stock_df['Volume'] > 0].copy()
    log(f"Dropped {before - len(stock_df):,} suspended rows (Volume=0)")
    
    # Sort
    stock_df = stock_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # === MACRO FEATURES ===
    log("Creating Macro features...")
    econ_df = econ_df.sort_values('date').reset_index(drop=True)
    
    # USD Return
    econ_df['USD_Return'] = econ_df['usd_sell_rate'].pct_change()
    
    # Gold Return
    econ_df['Gold_Return'] = econ_df['gold_24k'].pct_change()
    
    # Gas Index (normalized to start at 1.0)
    if 'gasoline_92' in econ_df.columns:
        first_valid = econ_df['gasoline_92'].first_valid_index()
        if first_valid is not None:
            base_gas = econ_df.loc[first_valid, 'gasoline_92']
            econ_df['Gas_Index'] = econ_df['gasoline_92'] / base_gas
        else:
            econ_df['Gas_Index'] = 1.0
    else:
        econ_df['Gas_Index'] = 1.0
    
    log("   ✓ USD_Return, Gold_Return, Gas_Index", "PASS")
    
    # === CORE PRICE FEATURES ===
    log("Creating Core features...")
    
    # Return_T = Daily % change
    stock_df['Return_T'] = stock_df.groupby('Ticker')['Close'].pct_change()
    
    # Vol_Ratio = Volume / 20-day MA
    stock_df['Vol_MA_20'] = stock_df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    stock_df['Vol_Ratio'] = stock_df['Volume'] / stock_df['Vol_MA_20']
    
    # High_Low_Ratio
    stock_df['High_Low_Ratio'] = stock_df['High'] / stock_df['Low']
    
    log("   ✓ Return_T, Vol_Ratio, High_Low_Ratio", "PASS")
    
    # === MERGE MACRO DATA ===
    log("Merging Macro data...")
    econ_cols = ['date', 'USD_Return', 'Gold_Return', 'Gas_Index', 
                 'headline_inflation', 'cbe_lending_rate']
    econ_subset = econ_df[[c for c in econ_cols if c in econ_df.columns]].copy()
    econ_subset = econ_subset.rename(columns={'date': 'Date'})
    
    merged_df = stock_df.merge(econ_subset, on='Date', how='left')
    
    # Forward-fill macro data
    macro_cols = ['USD_Return', 'Gold_Return', 'Gas_Index', 'headline_inflation', 'cbe_lending_rate']
    for col in macro_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill().fillna(0)
    
    log(f"   ✓ Merged: {len(merged_df):,} rows", "PASS")
    
    # === LAG GENERATION (Critical for LogReg) ===
    log("Generating Lag features (memory layer)...")
    
    lag_features = ['Return_T', 'Vol_Ratio', 'Gold_Return']
    
    for feature in lag_features:
        if feature not in merged_df.columns:
            continue
        for lag in LAG_PERIODS:
            lag_col = f"{feature}_Lag{lag}"
            merged_df[lag_col] = merged_df.groupby('Ticker')[feature].shift(lag)
            log(f"   ✓ Created {lag_col}", "PASS")
    
    return merged_df


# ============================================================
# PHASE 3: TARGET & SANITIZATION
# ============================================================

def phase3_target_sanitization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target and sanitize outliers.
    """
    print_header(3, "TARGET & SANITIZATION")
    
    # === TARGET GENERATION ===
    log("Creating buffered Target...")
    
    # Next-day return (forward-looking)
    df['Next_Close'] = df.groupby('Ticker')['Close'].shift(-1)
    df['Next_Return'] = (df['Next_Close'] / df['Close']) - 1
    
    # Target = 1 if next day gain > 0.5%
    df['Target'] = (df['Next_Return'] > BUFFER_THRESHOLD).astype(int)
    
    log(f"   Target = 1 if Next_Return > {BUFFER_THRESHOLD*100:.1f}%", "PASS")
    
    # === GARBAGE FILTER (Outliers) ===
    log(f"Filtering extreme returns outside [{RETURN_CLIP_MIN}, {RETURN_CLIP_MAX}]...")
    
    before = len(df)
    df = df[(df['Return_T'] >= RETURN_CLIP_MIN) & (df['Return_T'] <= RETURN_CLIP_MAX)].copy()
    dropped = before - len(df)
    log(f"   Dropped {dropped:,} extreme outlier rows", "WARN" if dropped > 0 else "PASS")
    
    # === DROP LOW-SIGNAL FEATURES ===
    log("Dropping low-signal features...")
    drop_cols = ['GDP_Growth', 'Next_Close', 'Next_Return', 'Vol_MA_20']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            log(f"   Dropped: {col}", "INFO")
    
    # === IMPUTATION ===
    log("Imputing missing values...")
    
    # Drop last row per ticker (no target)
    before = len(df)
    df = df.groupby('Ticker', group_keys=False).apply(lambda x: x.iloc[:-1])
    log(f"   Dropped {before - len(df)} tail rows (no target)")
    
    # Drop remaining NaNs
    before = len(df)
    df = df.dropna()
    log(f"   Dropped {before - len(df)} NaN rows")
    
    return df


# ============================================================
# PHASE 4: OPTIMIZATION (Collinearity & Scaling)
# ============================================================

def phase4_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove multicollinearity and apply standardization.
    """
    print_header(4, "OPTIMIZATION (Collinearity & Scaling)")
    
    # === IDENTIFY FEATURE COLUMNS ===
    # Exclude keys and target
    exclude_cols = ['Date', 'Ticker', 'Sector', 'ISIN', 'Target', 'Close', 
                    'Open', 'High', 'Low', 'Volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
    
    log(f"Analyzing {len(feature_cols)} features for collinearity...")
    
    # === MULTICOLLINEARITY PURGE ===
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find pairs > threshold
    dropped_features = []
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            feat_i = feature_cols[i]
            feat_j = feature_cols[j]
            
            # Skip if already dropped
            if feat_i in dropped_features or feat_j in dropped_features:
                continue
            
            corr = corr_matrix.loc[feat_i, feat_j]
            
            if corr > COLLINEARITY_THRESHOLD:
                # Drop the one less correlated with Target
                if 'Target' in df.columns:
                    corr_i_target = abs(df[feat_i].corr(df['Target']))
                    corr_j_target = abs(df[feat_j].corr(df['Target']))
                    
                    to_drop = feat_i if corr_i_target < corr_j_target else feat_j
                else:
                    to_drop = feat_j  # Arbitrary
                
                dropped_features.append(to_drop)
                log(f"   Dropped '{to_drop}' (corr with '{feat_i if to_drop == feat_j else feat_j}' = {corr:.2f})", "WARN")
    
    # Remove dropped features
    for feat in dropped_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])
            feature_cols.remove(feat)
    
    if len(dropped_features) == 0:
        log("   No multicollinear features found", "PASS")
    else:
        log(f"   Removed {len(dropped_features)} collinear features", "WARN")
    
    # === STANDARDIZATION (Z-Score) ===
    log("Applying StandardScaler (Z-Score normalization)...")
    
    # Only scale numeric feature columns (NOT Target)
    remaining_features = [c for c in feature_cols if c in df.columns]
    
    scaler = StandardScaler()
    df[remaining_features] = scaler.fit_transform(df[remaining_features])
    
    # Verify
    sample_col = remaining_features[0] if remaining_features else None
    if sample_col:
        log(f"   Sample: {sample_col} -> mean={df[sample_col].mean():.4f}, std={df[sample_col].std():.4f}", "PASS")
    
    return df


# ============================================================
# PHASE 5: FINAL OUTPUT & BALANCE CHECK
# ============================================================

def phase5_final_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check class balance and save output.
    """
    print_header(5, "FINAL OUTPUT & BALANCE CHECK")
    
    # === SELECT FINAL COLUMNS ===
    # Keys + Target + Features
    exclude_raw = ['Open', 'High', 'Low', 'Volume', 'ISIN', 
                   'usd_sell_rate', 'gold_24k', 'gasoline_92', 'gdp_usd_billion']
    
    final_cols = [c for c in df.columns if c not in exclude_raw]
    final_df = df[final_cols].copy()
    
    log(f"Final dataset: {len(final_df):,} rows, {len(final_df.columns)} columns")
    
    # === BALANCE CHECK ===
    target_1 = (final_df['Target'] == 1).sum()
    target_0 = (final_df['Target'] == 0).sum()
    total = len(final_df)
    
    ratio_1 = target_1 / total
    ratio_0 = target_0 / total
    
    print("\n" + "-" * 50)
    print("  📊 CLASS BALANCE CHECK")
    print("-" * 50)
    print(f"\n  Target = 1 (Buy):       {target_1:,} ({ratio_1*100:.1f}%)")
    print(f"  Target = 0 (Hold/Sell): {target_0:,} ({ratio_0*100:.1f}%)")
    print(f"  Total:                  {total:,}")
    
    if ratio_1 < 0.20 or ratio_1 > 0.80:
        log("⚠️  WARNING: Severe Class Imbalance detected!", "WARN")
        log("   Consider: SMOTE, class weights, or threshold adjustment", "WARN")
    else:
        log("Class balance is acceptable", "PASS")
    
    # === SHOW FINAL FEATURES ===
    print("\n  📝 Final Feature Set:")
    feature_cols = [c for c in final_df.columns if c not in ['Date', 'Ticker', 'Sector', 'Target', 'Close']]
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {col}")
    
    # === SAVE ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    log(f"💾 Saved to: {OUTPUT_FILE}", "PASS")
    
    # === VERIFICATION SUMMARY ===
    print("\n" + "=" * 70)
    print("  ✅ LOGISTIC REGRESSION DATASET READY")
    print("=" * 70)
    print(f"\n  📊 Shape: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
    print(f"  📅 Date Range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"  📈 Tickers: {final_df['Ticker'].nunique()}")
    print(f"  🎯 Target Balance: {ratio_1*100:.1f}% positive")
    print(f"  📁 Output: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    
    return final_df


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute full LogReg data preparation pipeline."""
    print("\n" + "=" * 70)
    print("  🧮 LOGISTIC REGRESSION DATA PREPARATION")
    print("  🎯 Mathematically Perfect Dataset Generator")
    print("  📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Phase 1: Data Forensics
    stock_df, econ_df = phase1_data_forensics()
    
    # Phase 2: Feature Engineering
    df = phase2_feature_engineering(stock_df, econ_df)
    
    # Phase 3: Target & Sanitization
    df = phase3_target_sanitization(df)
    
    # Phase 4: Optimization
    df = phase4_optimization(df)
    
    # Phase 5: Final Output
    final_df = phase5_final_output(df)
    
    return final_df


if __name__ == "__main__":
    main()
