#!/usr/bin/env python3
"""
train_models.py - Per-Ticker Logistic Regression Training
==========================================================

Role: Senior Quant Researcher & ML Engineer

This script:
1. Trains individual LogReg models for each stock ticker
2. Uses strict time-based train/test split (no data leakage)
3. Evaluates with precision, recall, and directional accuracy
4. Saves models and results summary

Input: data/model_ready/train_ready_logreg.csv
Output:
    - data/model_ready/results_summary.csv
    - models/saved_models/{TICKER}_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "model_ready" / "train_ready_logreg.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

# Training Parameters
TRAIN_RATIO = 0.80           # 80% train, 20% test
MIN_ROWS_PER_TICKER = 500    # Skip tickers with insufficient data

# Model Configuration
MODEL_CONFIG = {
    'penalty': 'l2',           # Ridge regularization
    'C': 0.1,                  # Strong regularization (prevent overfitting)
    'class_weight': 'balanced', # Handle 35/65 class imbalance
    'solver': 'liblinear',     # Good for smaller datasets
    'random_state': 42,        # Reproducibility
    'max_iter': 1000,
}


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
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the prepared LogReg dataset."""
    log(f"Loading data from: {DATA_FILE}")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    log(f"Loaded {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Identify feature columns (exclude keys and target)."""
    exclude = ['Date', 'Ticker', 'Sector', 'Target', 'Close']
    features = [c for c in df.columns if c not in exclude]
    return features


# ============================================================
# TIME-BASED TRAIN/TEST SPLIT
# ============================================================

def time_split(ticker_df: pd.DataFrame, train_ratio: float = 0.80) -> tuple:
    """
    Strict time-based split for a single ticker.
    
    Args:
        ticker_df: DataFrame for one ticker, MUST be sorted by Date
        train_ratio: Fraction for training (rest is test)
    
    Returns:
        train_df, test_df
    """
    # Ensure sorted by date
    ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
    
    split_idx = int(len(ticker_df) * train_ratio)
    
    train_df = ticker_df.iloc[:split_idx].copy()
    test_df = ticker_df.iloc[split_idx:].copy()
    
    return train_df, test_df


# ============================================================
# MODEL TRAINING
# ============================================================

def train_ticker_model(ticker: str, ticker_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Train a Logistic Regression model for a single ticker.
    
    Returns:
        Dictionary with model, metrics, and metadata
    """
    # Time-based split
    train_df, test_df = time_split(ticker_df, TRAIN_RATIO)
    
    # Prepare features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Target'].values
    
    # Train model
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    # Precision for class 1 (Buy signals)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Recall for class 1
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    # Directional accuracy (overall)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Additional stats
    buy_signals = (y_pred == 1).sum()
    actual_buys = (y_test == 1).sum()
    
    result = {
        'ticker': ticker,
        'model': model,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'buy_signals_predicted': buy_signals,
        'actual_buys_in_test': actual_buys,
        'train_start': train_df['Date'].min(),
        'train_end': train_df['Date'].max(),
        'test_start': test_df['Date'].min(),
        'test_end': test_df['Date'].max(),
    }
    
    return result


def save_model(model, ticker: str, models_dir: Path):
    """Save trained model as pickle file."""
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean ticker name for filename
    clean_ticker = ticker.replace('.', '_').replace(':', '_')
    model_path = models_dir / f"{clean_ticker}_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_all_models():
    """Train models for all tickers."""
    print_header("🏋️ LOGISTIC REGRESSION TRAINING PIPELINE")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    feature_cols = get_feature_columns(df)
    
    log(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}... ")
    
    # Get unique tickers
    tickers = df['Ticker'].unique()
    log(f"Total tickers: {len(tickers)}")
    
    # Results storage
    results = []
    skipped_tickers = []
    
    print_header("TRAINING INDIVIDUAL MODELS")
    
    for i, ticker in enumerate(tickers, 1):
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # Skip if insufficient data
        if len(ticker_df) < MIN_ROWS_PER_TICKER:
            skipped_tickers.append((ticker, len(ticker_df)))
            continue
        
        # Train model
        result = train_ticker_model(ticker, ticker_df, feature_cols)
        
        # Save model
        model_path = save_model(result['model'], ticker, MODELS_DIR)
        result['model_path'] = str(model_path)
        
        # Store result (without model object)
        result_copy = {k: v for k, v in result.items() if k != 'model'}
        results.append(result_copy)
        
        # Progress log
        log(f"[{i:2d}/{len(tickers)}] {ticker:12s} | "
            f"Precision: {result['precision']:.3f} | "
            f"Recall: {result['recall']:.3f} | "
            f"Accuracy: {result['accuracy']:.3f} | "
            f"Train: {result['train_size']:,}", "TRAIN")
    
    # Report skipped tickers
    if skipped_tickers:
        print_header("⚠️ SKIPPED TICKERS (Insufficient Data)")
        for ticker, count in skipped_tickers:
            log(f"{ticker}: {count} rows (need {MIN_ROWS_PER_TICKER})", "WARN")
    
    return results, feature_cols


# ============================================================
# RESULTS ANALYSIS
# ============================================================

def analyze_results(results: list) -> pd.DataFrame:
    """Analyze and save results summary."""
    print_header("📊 RESULTS ANALYSIS")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Select columns for summary
    summary_cols = [
        'ticker', 'precision', 'recall', 'accuracy', 
        'train_size', 'test_size', 'buy_signals_predicted', 'actual_buys_in_test'
    ]
    
    summary_df = results_df[summary_cols].copy()
    summary_df.columns = [
        'Ticker', 'Test_Precision', 'Test_Recall', 'Test_Accuracy',
        'Training_Sample_Size', 'Test_Sample_Size', 'Buy_Signals_Predicted', 'Actual_Buys'
    ]
    
    # Sort by precision
    summary_df = summary_df.sort_values('Test_Precision', ascending=False)
    
    # Save to CSV
    summary_path = OUTPUT_DIR / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"Results saved to: {summary_path}", "PASS")
    
    return summary_df


def print_final_report(summary_df: pd.DataFrame):
    """Print final performance report."""
    print_header("📈 FINAL PERFORMANCE REPORT")
    
    # Overall statistics
    avg_precision = summary_df['Test_Precision'].mean()
    avg_recall = summary_df['Test_Recall'].mean()
    avg_accuracy = summary_df['Test_Accuracy'].mean()
    
    std_precision = summary_df['Test_Precision'].std()
    
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Average Precision (Win Rate):  {avg_precision:.3f} ± {std_precision:.3f}      │")
    print(f"  │  Average Recall:                {avg_recall:.3f}               │")
    print(f"  │  Average Directional Accuracy:  {avg_accuracy:.3f}               │")
    print("  └─────────────────────────────────────────────────────┘")
    
    # Top performers
    print("\n  🏆 TOP 5 PERFORMERS (by Precision):")
    print("-" * 60)
    top5 = summary_df.head(5)
    for _, row in top5.iterrows():
        print(f"    {row['Ticker']:12s} | Precision: {row['Test_Precision']:.3f} | "
              f"Recall: {row['Test_Recall']:.3f} | Acc: {row['Test_Accuracy']:.3f}")
    
    # Bottom performers
    print("\n  📉 BOTTOM 5 PERFORMERS (by Precision):")
    print("-" * 60)
    bottom5 = summary_df.tail(5)
    for _, row in bottom5.iterrows():
        print(f"    {row['Ticker']:12s} | Precision: {row['Test_Precision']:.3f} | "
              f"Recall: {row['Test_Recall']:.3f} | Acc: {row['Test_Accuracy']:.3f}")
    
    # Precision distribution
    print("\n  📊 PRECISION DISTRIBUTION:")
    print("-" * 60)
    
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
    for low, high in bins:
        count = ((summary_df['Test_Precision'] >= low) & (summary_df['Test_Precision'] < high)).sum()
        bar = "█" * count
        print(f"    {low:.1f}-{high:.1f}: {count:2d} tickers {bar}")
    
    # Warning check
    print("\n" + "-" * 60)
    if avg_precision < 0.40:
        log("⚠️ Average precision below 40% - consider hyperparameter tuning", "WARN")
    elif avg_precision < 0.50:
        log("Average precision is moderate (40-50%)", "INFO")
    else:
        log("🎯 Average precision above 50% - Good signal!", "PASS")
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  📊 Models trained: {len(summary_df)}")
    print(f"  📁 Models saved to: {MODELS_DIR}")
    print(f"  📋 Summary saved to: {OUTPUT_DIR / 'results_summary.csv'}")
    print("=" * 70)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Execute full training pipeline."""
    print("\n" + "=" * 70)
    print("  🧮 LOGISTIC REGRESSION TRAINING SYSTEM")
    print("  🎯 Per-Ticker Model Training with Time-Based Split")
    print("=" * 70)
    
    # Train all models
    results, feature_cols = train_all_models()
    
    if not results:
        log("No models were trained!", "FAIL")
        return
    
    # Analyze and save results
    summary_df = analyze_results(results)
    
    # Print final report
    print_final_report(summary_df)
    
    return summary_df


if __name__ == "__main__":
    main()
