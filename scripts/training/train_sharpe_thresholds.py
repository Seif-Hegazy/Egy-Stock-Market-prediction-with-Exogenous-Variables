#!/usr/bin/env python3
"""
train_sharpe_thresholds.py - Sharpe Ratio Threshold Optimization
=================================================================

Role: Quantitative Portfolio Manager

This script optimizes the trading threshold using Sharpe Ratio Maximization:
1. Multi-threshold search (0.40 to 0.90)
2. Annualized Sharpe Ratio calculation
3. Walk-Forward win rate constraint (>50%)
4. Visual threshold curve plots

Input: data/model_ready/train_ready_logreg.csv
Output:
    - data/model_ready/sharpe_results.csv
    - plots/threshold_curves/{TICKER}_sharpe_curve.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped.")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "model_ready" / "train_ready_logreg.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
MODELS_DIR = PROJECT_ROOT / "models" / "sharpe_models"
PLOTS_DIR = PROJECT_ROOT / "plots" / "threshold_curves"

# Window & Target Parameters
LOOKBACK_WINDOW = 10
FORECAST_HORIZON = 5
EMBARGO_GAP = 5

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Target Thresholds (Paper's Logic)
UP_THRESHOLD = 1.001
DOWN_THRESHOLD = 0.999

# Sharpe Optimization Parameters
THRESHOLD_MIN = 0.40
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01
MIN_TRADES = 10              # Minimum trades for valid Sharpe
MIN_WIN_RATE = 0.50          # Walk-forward constraint
TRADING_DAYS_PER_YEAR = 252

# Model Configuration
MODEL_CONFIG = {
    'penalty': 'l2',
    'C': 1.0,
    'class_weight': 'balanced',
    'solver': 'liblinear',
    'random_state': 42,
    'max_iter': 1000,
}

MIN_ROWS_PER_TICKER = 200


def log(msg: str, level: str = "INFO"):
    """Formatted logging"""
    icons = {'INFO': 'ℹ️', 'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'TRAIN': '🏋️', 'SHARPE': '📈'}
    icon = icons.get(level, '•')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# DATA PREPARATION (Same as hybrid model)
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    base_features = ['Return_T', 'Vol_Ratio', 'USD_Return', 'Gold_Return', 'Gas_Index']
    return [f for f in base_features if f in df.columns]


def create_rolling_window_features(ticker_df: pd.DataFrame, feature_cols: list) -> tuple:
    df = ticker_df.sort_values('Date').reset_index(drop=True)
    window_col_names = []
    
    for lag in range(LOOKBACK_WINDOW):
        for feature in feature_cols:
            col_name = f"{feature}_L{lag}"
            df[col_name] = df[feature].shift(lag)
            window_col_names.append(col_name)
    
    return df, window_col_names


def create_t5_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['Mean_Past'] = df['Close'].rolling(window=LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW).mean()
    
    future_closes = []
    next_returns = []
    
    for i in range(len(df)):
        if i + FORECAST_HORIZON <= len(df) - 1:
            future_mean = df['Close'].iloc[i+1:i+1+FORECAST_HORIZON].mean()
            future_closes.append(future_mean)
            # Actual return for Sharpe calculation
            actual_return = (df['Close'].iloc[i+FORECAST_HORIZON] / df['Close'].iloc[i]) - 1
            next_returns.append(actual_return)
        else:
            future_closes.append(np.nan)
            next_returns.append(np.nan)
    
    df['Mean_Future'] = future_closes
    df['Next_Return'] = next_returns
    df['Trend_Ratio'] = df['Mean_Future'] / df['Mean_Past']
    
    def label_trend(ratio):
        if pd.isna(ratio):
            return np.nan
        elif ratio > UP_THRESHOLD:
            return 1
        elif ratio < DOWN_THRESHOLD:
            return 0
        else:
            return np.nan
    
    df['Target_T5'] = df['Trend_Ratio'].apply(label_trend)
    
    return df


def embargo_split(df: pd.DataFrame) -> tuple:
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_start = train_end + EMBARGO_GAP
    val_end = val_start + int(n * VAL_RATIO)
    test_start = val_end + EMBARGO_GAP
    
    return df.iloc[:train_end].copy(), df.iloc[val_start:val_end].copy(), df.iloc[test_start:].copy()


# ============================================================
# SHARPE RATIO CALCULATION
# ============================================================

def calculate_annualized_sharpe(returns: np.ndarray) -> float:
    """
    Calculate Annualized Sharpe Ratio.
    
    Sharpe = (Mean(Returns) / Std(Returns)) * sqrt(252)
    
    This rewards high returns but heavily penalizes volatility.
    """
    if len(returns) < MIN_TRADES:
        return -999.0  # Invalid - too few trades
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0 or np.isnan(std_return):
        return -999.0  # Invalid - no volatility (suspicious)
    
    # Annualized Sharpe
    sharpe = (mean_return / std_return) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return sharpe


def calculate_cagr(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate Compound Annual Growth Rate from returns."""
    if len(returns) == 0:
        return 0.0
    
    # Cumulative return
    cum_return = np.prod(1 + returns) - 1
    
    # Number of years
    n_years = len(returns) / periods_per_year
    
    if n_years == 0:
        return 0.0
    
    # CAGR
    if cum_return <= -1:
        return -1.0  # Total loss
    
    cagr = (1 + cum_return) ** (1 / n_years) - 1
    
    return cagr


# ============================================================
# OPTIMAL THRESHOLD FINDING (Sharpe Maximization)
# ============================================================

def find_optimal_threshold_sharpe(
    model, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    val_returns: np.ndarray
) -> tuple:
    """
    Find optimal threshold by maximizing Sharpe Ratio on validation set.
    
    Walk-Forward Constraint: Selected threshold must have Win Rate > 50%
    
    Returns:
        optimal_threshold, sharpe_curve (for plotting)
    """
    # Get probabilities for class 1 (UP)
    probs = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    sharpe_curve = []
    win_rate_curve = []
    
    for t in thresholds:
        # Identify Buy signals
        buy_mask = probs >= t
        
        if buy_mask.sum() < MIN_TRADES:
            sharpe_curve.append(-999.0)
            win_rate_curve.append(0.0)
            continue
        
        # Get returns for Buy signals
        signal_returns = val_returns[buy_mask]
        
        # Calculate Sharpe
        sharpe = calculate_annualized_sharpe(signal_returns)
        sharpe_curve.append(sharpe)
        
        # Calculate Win Rate (for walk-forward constraint)
        wins = (signal_returns > 0).sum()
        win_rate = wins / len(signal_returns) if len(signal_returns) > 0 else 0
        win_rate_curve.append(win_rate)
    
    sharpe_curve = np.array(sharpe_curve)
    win_rate_curve = np.array(win_rate_curve)
    
    # Apply Walk-Forward Constraint
    valid_mask = (sharpe_curve > -999) & (win_rate_curve >= MIN_WIN_RATE)
    
    if not valid_mask.any():
        # No valid threshold found, use default
        log("No threshold meets win rate constraint, using default", "WARN")
        return 0.50, thresholds, sharpe_curve, win_rate_curve
    
    # Find maximum Sharpe among valid thresholds
    valid_sharpes = np.where(valid_mask, sharpe_curve, -999)
    optimal_idx = np.argmax(valid_sharpes)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, thresholds, sharpe_curve, win_rate_curve


# ============================================================
# PLOTTING
# ============================================================

def plot_sharpe_curve(ticker: str, thresholds: np.ndarray, sharpe_curve: np.ndarray, 
                      optimal_t: float, plots_dir: Path):
    """Plot Sharpe Ratio curve for threshold selection."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Filter valid Sharpe values
    valid_mask = sharpe_curve > -999
    
    plt.plot(thresholds[valid_mask], sharpe_curve[valid_mask], 
             'g-', linewidth=2, label='Sharpe Ratio')
    
    # Mark optimal threshold
    optimal_sharpe = sharpe_curve[np.argmin(np.abs(thresholds - optimal_t))]
    plt.axvline(x=optimal_t, color='r', linestyle='--', label=f'Optimal τ = {optimal_t:.2f}')
    plt.scatter([optimal_t], [optimal_sharpe], color='r', s=100, zorder=5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Annualized Sharpe Ratio', fontsize=12)
    plt.title(f'{ticker} - Sharpe Ratio vs Threshold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    clean_ticker = ticker.replace('.', '_').replace(':', '_')
    plt.savefig(plots_dir / f"{clean_ticker}_sharpe_curve.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_sharpe_ticker(ticker: str, ticker_df: pd.DataFrame, feature_cols: list) -> dict:
    """Train model with Sharpe-optimized threshold for a single ticker."""
    
    # Create features and target
    df, window_cols = create_rolling_window_features(ticker_df, feature_cols)
    df = create_t5_target(df)
    
    # Clean
    required_cols = window_cols + ['Target_T5', 'Next_Return', 'Close', 'Date']
    df_clean = df.dropna(subset=[c for c in required_cols if c in df.columns])
    
    if len(df_clean) < 100:
        return None
    
    # Split
    train_df, val_df, test_df = embargo_split(df_clean)
    
    if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 20:
        return None
    
    # Prepare matrices
    X_train = train_df[window_cols].values
    y_train = train_df['Target_T5'].values.astype(int)
    
    X_val = val_df[window_cols].values
    y_val = val_df['Target_T5'].values.astype(int)
    val_returns = val_df['Next_Return'].values
    
    X_test = test_df[window_cols].values
    y_test = test_df['Target_T5'].values.astype(int)
    test_returns = test_df['Next_Return'].values
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    # Find optimal threshold via Sharpe maximization
    optimal_t, thresholds, sharpe_curve, win_rate_curve = find_optimal_threshold_sharpe(
        model, X_val, y_val, val_returns
    )
    
    # Apply to test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_signals = test_probs >= optimal_t
    
    # Calculate test metrics
    if test_signals.sum() >= MIN_TRADES:
        signal_returns = test_returns[test_signals]
        test_sharpe = calculate_annualized_sharpe(signal_returns)
        test_cagr = calculate_cagr(signal_returns)
        test_win_rate = (signal_returns > 0).mean()
        num_trades = len(signal_returns)
    else:
        test_sharpe = 0.0
        test_cagr = 0.0
        test_win_rate = 0.0
        num_trades = 0
    
    result = {
        'ticker': ticker,
        'model': model,
        'scaler': scaler,
        'optimal_threshold': optimal_t,
        'test_sharpe': test_sharpe,
        'test_cagr': test_cagr,
        'test_win_rate': test_win_rate,
        'num_trades': num_trades,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'thresholds': thresholds,
        'sharpe_curve': sharpe_curve,
        'win_rate_curve': win_rate_curve,
    }
    
    return result


def train_all_sharpe_models():
    """Train all models with Sharpe optimization."""
    print_header("📈 SHARPE RATIO THRESHOLD OPTIMIZATION")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📊 Threshold Range: {THRESHOLD_MIN:.2f} to {THRESHOLD_MAX:.2f}")
    print(f"  🎯 Min Win Rate Constraint: {MIN_WIN_RATE*100:.0f}%")
    
    # Load data
    log(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    feature_cols = get_feature_columns(df)
    log(f"Features: {feature_cols}")
    
    tickers = df['Ticker'].unique()
    log(f"Total tickers: {len(tickers)}")
    
    results = []
    skipped = []
    top_tickers = []  # For plotting top 3
    
    print_header("TRAINING WITH SHARPE OPTIMIZATION")
    
    for i, ticker in enumerate(tickers, 1):
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        if len(ticker_df) < MIN_ROWS_PER_TICKER:
            skipped.append((ticker, len(ticker_df)))
            continue
        
        result = train_sharpe_ticker(ticker, ticker_df, feature_cols)
        
        if result is None:
            skipped.append((ticker, len(ticker_df), "split_fail"))
            continue
        
        results.append(result)
        top_tickers.append((result['test_sharpe'], result))
        
        log(f"[{i:2d}/{len(tickers)}] {ticker:12s} | "
            f"Sharpe: {result['test_sharpe']:+.2f} | "
            f"CAGR: {result['test_cagr']*100:+.1f}% | "
            f"τ: {result['optimal_threshold']:.2f} | "
            f"WR: {result['test_win_rate']*100:.1f}%", "SHARPE")
    
    # Plot top 3 by Sharpe
    if MATPLOTLIB_AVAILABLE:
        top_tickers.sort(key=lambda x: x[0], reverse=True)
        print_header("📊 GENERATING PLOTS FOR TOP 3 TICKERS")
        for sharpe, result in top_tickers[:3]:
            plot_sharpe_curve(
                result['ticker'],
                result['thresholds'],
                result['sharpe_curve'],
                result['optimal_threshold'],
                PLOTS_DIR
            )
            log(f"Saved plot for {result['ticker']}", "PASS")
    
    # Report skipped
    if skipped:
        print_header("⚠️ SKIPPED TICKERS")
        for item in skipped[:5]:
            log(f"{item[0]}: {item[1]} rows", "WARN")
    
    return results


# ============================================================
# RESULTS ANALYSIS
# ============================================================

def analyze_sharpe_results(results: list) -> pd.DataFrame:
    """Analyze and save Sharpe results."""
    print_header("📊 SHARPE RESULTS ANALYSIS")
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Ticker': r['ticker'],
            'Test_Sharpe': r['test_sharpe'],
            'Test_CAGR': r['test_cagr'],
            'Optimal_Threshold': r['optimal_threshold'],
            'Win_Rate': r['test_win_rate'],
            'Num_Trades': r['num_trades'],
            'Train_Size': r['train_size'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test_Sharpe', ascending=False)
    
    # Save
    output_path = OUTPUT_DIR / "sharpe_results.csv"
    summary_df.to_csv(output_path, index=False)
    log(f"Results saved to: {output_path}", "PASS")
    
    return summary_df


def print_sharpe_report(summary_df: pd.DataFrame):
    """Print final Sharpe performance report."""
    print_header("📈 SHARPE OPTIMIZATION REPORT")
    
    valid_df = summary_df[summary_df['Test_Sharpe'] > -999]
    
    avg_sharpe = valid_df['Test_Sharpe'].mean()
    avg_cagr = valid_df['Test_CAGR'].mean()
    avg_win_rate = valid_df['Win_Rate'].mean()
    avg_threshold = valid_df['Optimal_Threshold'].mean()
    
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Average Sharpe Ratio:       {avg_sharpe:+.2f}                   │")
    print(f"  │  Average CAGR:               {avg_cagr*100:+.1f}%                  │")
    print(f"  │  Average Win Rate:           {avg_win_rate*100:.1f}%                  │")
    print(f"  │  Average Optimal τ:          {avg_threshold:.2f}                   │")
    print("  └─────────────────────────────────────────────────────┘")
    
    # Top 5
    print("\n  🏆 TOP 5 BY SHARPE RATIO:")
    print("-" * 70)
    for _, row in valid_df.head(5).iterrows():
        print(f"    {row['Ticker']:12s} | Sharpe: {row['Test_Sharpe']:+.2f} | "
              f"CAGR: {row['Test_CAGR']*100:+.1f}% | τ: {row['Optimal_Threshold']:.2f} | "
              f"WR: {row['Win_Rate']*100:.1f}%")
    
    # Sharpe distribution
    print("\n  📊 SHARPE DISTRIBUTION:")
    print("-" * 70)
    bins = [(-999, 0), (0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 999)]
    labels = ['Negative', '0.0-0.5', '0.5-1.0', '1.0-2.0', '>2.0']
    
    for (low, high), label in zip(bins, labels):
        count = ((valid_df['Test_Sharpe'] >= low) & (valid_df['Test_Sharpe'] < high)).sum()
        bar = "█" * count
        print(f"    {label:12s}: {count:2d} tickers {bar}")
    
    # Summary
    positive_sharpe = (valid_df['Test_Sharpe'] > 0).sum()
    total = len(valid_df)
    
    print("\n" + "-" * 70)
    if avg_sharpe > 1.0:
        log(f"🎯 Excellent! Average Sharpe > 1.0", "PASS")
    elif avg_sharpe > 0.5:
        log(f"Good risk-adjusted returns (Sharpe > 0.5)", "PASS")
    elif avg_sharpe > 0:
        log(f"Positive Sharpe, but room for improvement", "WARN")
    else:
        log(f"Negative average Sharpe - review strategy", "FAIL")
    
    print(f"\n  {positive_sharpe}/{total} tickers have positive Sharpe Ratio")
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"  ✅ SHARPE OPTIMIZATION COMPLETE")
    print(f"  📊 Models trained: {len(summary_df)}")
    print(f"  📈 Positive Sharpe: {positive_sharpe}/{total}")
    print(f"  📁 Results: {OUTPUT_DIR / 'sharpe_results.csv'}")
    if MATPLOTLIB_AVAILABLE:
        print(f"  📊 Plots: {PLOTS_DIR}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  📈 SHARPE RATIO THRESHOLD OPTIMIZATION")
    print("  🎯 Risk-Adjusted Returns for Egyptian Market")
    print("=" * 70)
    
    results = train_all_sharpe_models()
    
    if not results:
        log("No models trained!", "FAIL")
        return
    
    summary_df = analyze_sharpe_results(results)
    print_sharpe_report(summary_df)
    
    return summary_df


if __name__ == "__main__":
    main()
