#!/usr/bin/env python3
"""
train_hybrid_model.py - Hybrid Forecasting Pipeline
====================================================

Role: Financial Machine Learning Engineer

This script implements the research paper methodology (Section IV):
1. Multi-Variate Rolling Window (10-day history flattened)
2. T+5 Weekly Trend Target (not daily noise)
3. Strict Time-Splitting with Embargo Gap
4. Threshold Calibration on Validation Set

Input: data/model_ready/train_ready_logreg.csv
Output:
    - data/model_ready/hybrid_results.csv
    - models/hybrid_models/{TICKER}_hybrid.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "model_ready" / "train_ready_logreg.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
MODELS_DIR = PROJECT_ROOT / "models" / "hybrid_models"

# Window & Target Parameters
LOOKBACK_WINDOW = 10         # 10-day history window
FORECAST_HORIZON = 5         # T+5 prediction (weekly trend)
EMBARGO_GAP = 5              # Gap between train/val and val/test

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Target Thresholds (Paper's Logic)
UP_THRESHOLD = 1.001         # Ratio > 1.001 = UP
DOWN_THRESHOLD = 0.999       # Ratio < 0.999 = DOWN

# Model Configuration
MODEL_CONFIG = {
    'penalty': 'l2',
    'C': 1.0,                  # Less regularization for hybrid
    'class_weight': 'balanced',
    'solver': 'liblinear',
    'random_state': 42,
    'max_iter': 1000,
}

# Threshold Calibration
THRESHOLD_PERCENTILE = 40    # Use 40th percentile of validation probabilities

# Minimum requirements
MIN_ROWS_PER_TICKER = 200    # Need enough for windows + splits


def log(msg: str, level: str = "INFO"):
    """Formatted logging"""
    icons = {'INFO': 'ℹ️', 'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'TRAIN': '🏋️', 'EVAL': '📊'}
    icon = icons.get(level, '•')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# PHASE 1: DATA CONSTRUCTION (Multi-Variate Rolling Window)
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """Identify base feature columns for windowing."""
    # Core features to include in window
    base_features = [
        'Return_T', 'Vol_Ratio', 'USD_Return', 'Gold_Return', 'Gas_Index'
    ]
    
    # Only use features that exist
    available = [f for f in base_features if f in df.columns]
    return available


def create_rolling_window_features(ticker_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Create flattened 10-day rolling window features.
    
    For 5 features × 10 days = 50-dimensional input vector.
    Window slides from (t-9) to (t).
    """
    df = ticker_df.sort_values('Date').reset_index(drop=True)
    
    # Create lagged features for each base feature
    window_features = []
    window_col_names = []
    
    for lag in range(LOOKBACK_WINDOW):
        for feature in feature_cols:
            col_name = f"{feature}_L{lag}"
            df[col_name] = df[feature].shift(lag)
            window_col_names.append(col_name)
    
    return df, window_col_names


def create_t5_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create T+5 Weekly Trend Target (Paper's Logic).
    
    Mean_future = Avg(Close[t+1] ... Close[t+5])
    Mean_past = Avg(Close[t-9] ... Close[t])
    Ratio = Mean_future / Mean_past
    
    Label:
        1 (UP) if Ratio > 1.001
        0 (DOWN) if Ratio < 0.999
        NaN (Neutral) otherwise - dropped
    """
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate rolling mean of past 10 days (t-9 to t)
    df['Mean_Past'] = df['Close'].rolling(window=LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).mean()
    
    # Calculate rolling mean of future 5 days (t+1 to t+5)
    # Shift by -5, then rolling mean of 5
    future_closes = []
    for i in range(len(df)):
        if i + FORECAST_HORIZON <= len(df) - 1:
            future_mean = df['Close'].iloc[i+1:i+1+FORECAST_HORIZON].mean()
            future_closes.append(future_mean)
        else:
            future_closes.append(np.nan)
    
    df['Mean_Future'] = future_closes
    
    # Calculate ratio
    df['Trend_Ratio'] = df['Mean_Future'] / df['Mean_Past']
    
    # Create target labels
    def label_trend(ratio):
        if pd.isna(ratio):
            return np.nan
        elif ratio > UP_THRESHOLD:
            return 1  # UP trend
        elif ratio < DOWN_THRESHOLD:
            return 0  # DOWN trend
        else:
            return np.nan  # Neutral - drop
    
    df['Target_T5'] = df['Trend_Ratio'].apply(label_trend)
    
    return df


# ============================================================
# PHASE 2: STRICT TIME-SPLITTING WITH EMBARGO
# ============================================================

def embargo_split(df: pd.DataFrame) -> tuple:
    """
    Split data with embargo gaps between sets.
    
    70% Train | GAP(5) | 10% Val | GAP(5) | 20% Test
    
    The gaps prevent look-ahead bias since target uses T+5.
    """
    n = len(df)
    
    train_end = int(n * TRAIN_RATIO)
    val_start = train_end + EMBARGO_GAP
    val_end = val_start + int(n * VAL_RATIO)
    test_start = val_end + EMBARGO_GAP
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()
    
    return train_df, val_df, test_df


# ============================================================
# PHASE 3: THRESHOLD CALIBRATION
# ============================================================

def calibrate_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Calibrate decision threshold using validation set.
    
    Returns the 40th percentile of predicted probabilities.
    This helps identify high-confidence predictions.
    """
    # Get probability of class 1 (UP)
    probs = model.predict_proba(X_val)[:, 1]
    
    # Calculate 40th percentile as threshold
    threshold = np.percentile(probs, THRESHOLD_PERCENTILE)
    
    return threshold, probs


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Apply calibrated threshold to generate predictions."""
    return (probs >= threshold).astype(int)


# ============================================================
# PHASE 4: TRAINING PIPELINE
# ============================================================

def train_hybrid_ticker(ticker: str, ticker_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train hybrid model for a single ticker.
    """
    # Step 1: Create rolling window features
    df, window_cols = create_rolling_window_features(ticker_df, feature_cols)
    
    # Step 2: Create T+5 target
    df = create_t5_target(df)
    
    # Step 3: Drop rows with NaN (from windowing, target, or neutral labels)
    required_cols = window_cols + ['Target_T5', 'Close', 'Date']
    df_clean = df.dropna(subset=[c for c in required_cols if c in df.columns])
    
    if len(df_clean) < 100:
        return None  # Insufficient data
    
    # Step 4: Embargo split
    train_df, val_df, test_df = embargo_split(df_clean)
    
    if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 20:
        return None  # Insufficient data in splits
    
    # Prepare feature matrices
    X_train = train_df[window_cols].values
    y_train = train_df['Target_T5'].values.astype(int)
    
    X_val = val_df[window_cols].values
    y_val = val_df['Target_T5'].values.astype(int)
    
    X_test = test_df[window_cols].values
    y_test = test_df['Target_T5'].values.astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Step 5: Train model
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    # Step 6: Calibrate threshold on validation set
    threshold, val_probs = calibrate_threshold(model, X_val, y_val)
    
    # Step 7: Apply to test set
    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = apply_threshold(test_probs, threshold)
    
    # Step 8: Calculate metrics
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Win rate = same as precision for class 1
    win_rate = precision
    
    result = {
        'ticker': ticker,
        'model': model,
        'scaler': scaler,
        'threshold': threshold,
        'window_cols': window_cols,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'win_rate': win_rate,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'feature_dim': len(window_cols),
        'test_buy_signals': (y_pred == 1).sum(),
        'actual_ups': (y_test == 1).sum(),
    }
    
    return result


def save_hybrid_model(result: dict, models_dir: Path):
    """Save trained hybrid model with metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    ticker = result['ticker']
    clean_ticker = ticker.replace('.', '_').replace(':', '_')
    
    # Save model bundle
    bundle = {
        'model': result['model'],
        'scaler': result['scaler'],
        'threshold': result['threshold'],
        'window_cols': result['window_cols'],
        'ticker': ticker,
    }
    
    model_path = models_dir / f"{clean_ticker}_hybrid.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    
    return model_path


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_all_hybrid_models():
    """Train hybrid models for all tickers."""
    print_header("🔬 HYBRID FORECASTING PIPELINE")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📊 Lookback Window: {LOOKBACK_WINDOW} days")
    print(f"  🎯 Forecast Horizon: T+{FORECAST_HORIZON}")
    print(f"  🛡️ Embargo Gap: {EMBARGO_GAP} rows")
    
    # Load data
    log(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get base feature columns
    feature_cols = get_feature_columns(df)
    log(f"Base features for windowing ({len(feature_cols)}): {feature_cols}")
    log(f"Flattened feature dimension: {len(feature_cols)} × {LOOKBACK_WINDOW} = {len(feature_cols) * LOOKBACK_WINDOW}")
    
    # Get unique tickers
    tickers = df['Ticker'].unique()
    log(f"Total tickers: {len(tickers)}")
    
    # Results storage
    results = []
    skipped_tickers = []
    
    print_header("TRAINING HYBRID MODELS")
    
    for i, ticker in enumerate(tickers, 1):
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # Skip if insufficient data
        if len(ticker_df) < MIN_ROWS_PER_TICKER:
            skipped_tickers.append((ticker, len(ticker_df)))
            continue
        
        # Train hybrid model
        result = train_hybrid_ticker(ticker, ticker_df, feature_cols)
        
        if result is None:
            skipped_tickers.append((ticker, len(ticker_df), "split_fail"))
            continue
        
        # Save model
        model_path = save_hybrid_model(result, MODELS_DIR)
        result['model_path'] = str(model_path)
        
        # Store result (without model objects)
        result_copy = {k: v for k, v in result.items() 
                      if k not in ['model', 'scaler', 'window_cols']}
        results.append(result_copy)
        
        # Progress log
        log(f"[{i:2d}/{len(tickers)}] {ticker:12s} | "
            f"Precision: {result['precision']:.3f} | "
            f"Recall: {result['recall']:.3f} | "
            f"τ: {result['threshold']:.3f} | "
            f"Train: {result['train_size']}", "TRAIN")
    
    # Report skipped
    if skipped_tickers:
        print_header("⚠️ SKIPPED TICKERS")
        for item in skipped_tickers:
            if len(item) == 3:
                log(f"{item[0]}: {item[1]} rows ({item[2]})", "WARN")
            else:
                log(f"{item[0]}: {item[1]} rows (insufficient)", "WARN")
    
    return results


# ============================================================
# RESULTS ANALYSIS
# ============================================================

def analyze_hybrid_results(results: list) -> pd.DataFrame:
    """Analyze and save hybrid results."""
    print_header("📊 HYBRID RESULTS ANALYSIS")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Select summary columns
    summary_cols = ['ticker', 'precision', 'recall', 'win_rate', 'accuracy',
                   'threshold', 'train_size', 'test_size']
    
    available_cols = [c for c in summary_cols if c in results_df.columns]
    summary_df = results_df[available_cols].copy()
    
    # Rename for output
    column_map = {
        'ticker': 'Ticker',
        'precision': 'Test_Precision',
        'recall': 'Test_Recall',
        'win_rate': 'Win_Rate',
        'accuracy': 'Test_Accuracy',
        'threshold': 'Calibrated_Threshold',
        'train_size': 'Train_Size',
        'test_size': 'Test_Size',
    }
    summary_df = summary_df.rename(columns=column_map)
    
    # Sort by precision
    summary_df = summary_df.sort_values('Test_Precision', ascending=False)
    
    # Save
    output_path = OUTPUT_DIR / "hybrid_results.csv"
    summary_df.to_csv(output_path, index=False)
    log(f"Results saved to: {output_path}", "PASS")
    
    return summary_df


def print_hybrid_report(summary_df: pd.DataFrame):
    """Print final hybrid performance report."""
    print_header("📈 HYBRID MODEL PERFORMANCE REPORT")
    
    # Overall statistics
    avg_precision = summary_df['Test_Precision'].mean()
    avg_recall = summary_df['Test_Recall'].mean()
    avg_win_rate = summary_df['Win_Rate'].mean()
    std_precision = summary_df['Test_Precision'].std()
    
    avg_threshold = summary_df['Calibrated_Threshold'].mean()
    
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Average Precision (Win Rate):  {avg_precision:.3f} ± {std_precision:.3f}      │")
    print(f"  │  Average Recall:                {avg_recall:.3f}               │")
    print(f"  │  Average Calibrated τ:          {avg_threshold:.3f}               │")
    print("  └─────────────────────────────────────────────────────┘")
    
    # Top performers
    print("\n  🏆 TOP 5 PERFORMERS (by Precision):")
    print("-" * 60)
    top5 = summary_df.head(5)
    for _, row in top5.iterrows():
        print(f"    {row['Ticker']:12s} | Precision: {row['Test_Precision']:.3f} | "
              f"Recall: {row['Test_Recall']:.3f} | τ: {row['Calibrated_Threshold']:.3f}")
    
    # Precision distribution
    print("\n  📊 PRECISION DISTRIBUTION:")
    print("-" * 60)
    
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
    for low, high in bins:
        count = ((summary_df['Test_Precision'] >= low) & 
                (summary_df['Test_Precision'] < high)).sum()
        bar = "█" * count
        print(f"    {low:.1f}-{high:.1f}: {count:2d} tickers {bar}")
    
    # Improvement check
    print("\n" + "-" * 60)
    if avg_precision >= 0.50:
        log("🎯 Excellent! Average precision ≥ 50%", "PASS")
    elif avg_precision >= 0.40:
        log("Good performance - Average precision ≥ 40%", "PASS")
    else:
        log("Consider adding more features or tuning hyperparameters", "WARN")
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"  ✅ HYBRID TRAINING COMPLETE")
    print(f"  📊 Models trained: {len(summary_df)}")
    print(f"  🎯 Average Win Rate: {avg_win_rate:.1%}")
    print(f"  📁 Models saved to: {MODELS_DIR}")
    print(f"  📋 Results saved to: {OUTPUT_DIR / 'hybrid_results.csv'}")
    print("=" * 70)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute full hybrid training pipeline."""
    print("\n" + "=" * 70)
    print("  🔬 HYBRID FORECASTING PIPELINE")
    print("  📚 Based on Research Paper Methodology (Section IV)")
    print("  🎯 T+5 Weekly Trend Prediction with Threshold Calibration")
    print("=" * 70)
    
    # Train all models
    results = train_all_hybrid_models()
    
    if not results:
        log("No models were trained!", "FAIL")
        return
    
    # Analyze and save
    summary_df = analyze_hybrid_results(results)
    
    # Print report
    print_hybrid_report(summary_df)
    
    return summary_df


if __name__ == "__main__":
    main()
