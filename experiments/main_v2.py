#!/usr/bin/env python3
"""
EGX Prediction Model v2 - Main Runner

Weekly direction prediction using XGBoost with global indicators.
Based on:
- Gu, Kelly, Xiu (2020) - ML for Asset Pricing
- Moskowitz et al. (2012) - Time Series Momentum
- Kara et al. (2011) - Weekly Stock Prediction

Usage:
    python main_v2.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, classification_report
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader_v2 import load_and_prepare_weekly_data
from src.feature_eng_v2 import (
    engineer_all_features, 
    TECHNICAL_FEATURES, GLOBAL_FEATURES, EGYPT_FEATURES, ALL_FEATURES
)
from src.models_v2 import train_model, predict_proba, predict, get_feature_importance
from src.validation import PurgedWalkForwardCV, diebold_mariano_test, compute_squared_loss, is_significant

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

# Window sizes in WEEKS (not days)
WINDOW_SIZES = {
    'Short': 26,    # 6 months
    'Medium': 52,   # 1 year
    'Long': 104,    # 2 years
}

# Minimum history in weeks
MIN_HISTORY = {
    26: 26 + 13 + 2,   # window + test + gap = 41 weeks
    52: 52 + 13 + 2,   # 67 weeks
    104: 104 + 13 + 2, # 119 weeks (~2.3 years)
}

RESULTS_DIR = Path(__file__).parent / 'results_v2'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Validation
# =============================================================================

class WeeklyPurgedCV:
    """Purged Walk-Forward CV for weekly data."""
    
    def __init__(self, window_size: int, test_size: int = 13, gap: int = 1):
        """
        Args:
            window_size: Training window in weeks
            test_size: Test size in weeks (13 = 1 quarter)
            gap: Embargo gap in weeks
        """
        self.window_size = window_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        
        min_start = self.window_size + self.gap
        current_test_start = min_start
        
        while current_test_start + self.test_size <= n:
            train_end = current_test_start - self.gap
            train_start = max(0, train_end - self.window_size)
            
            if train_end - train_start >= self.window_size * 0.8:
                train_idx = indices[train_start:train_end]
                test_idx = indices[current_test_start:current_test_start + self.test_size]
                yield train_idx, test_idx
            
            current_test_start += self.test_size
    
    def get_n_splits(self, X):
        return sum(1 for _ in self.split(X))


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc'] = np.nan
    return metrics


# =============================================================================
# Tournament Runner
# =============================================================================

def run_ticker_tournament(df_ticker: pd.DataFrame, ticker: str,
                          tech_features: list, all_features: list) -> pd.DataFrame:
    """Run tournament for a single ticker."""
    results = []
    
    for window_name, window_size in WINDOW_SIZES.items():
        n_weeks = len(df_ticker)
        min_required = MIN_HISTORY.get(window_size, window_size + 15)
        
        if n_weeks < min_required:
            print(f"    {window_name}: Skipped ({n_weeks} < {min_required} weeks)")
            continue
        
        cv = WeeklyPurgedCV(window_size=window_size, test_size=13, gap=1)
        n_splits = cv.get_n_splits(df_ticker)
        
        if n_splits == 0:
            print(f"    {window_name}: No valid folds")
            continue
        
        # Collect predictions
        all_y_true = []
        all_proba_a = []
        all_proba_b = []
        all_metrics_a = []
        all_metrics_b = []
        
        for train_idx, test_idx in cv.split(df_ticker):
            train = df_ticker.iloc[train_idx]
            test = df_ticker.iloc[test_idx]
            
            y_train = train['target']
            y_test = test['target']
            
            # Skip if only one class
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            
            # Model A: Technical only (baseline)
            X_train_a = train[tech_features].copy()
            X_test_a = test[tech_features].copy()
            
            model_a, _ = train_model(X_train_a, y_train)
            proba_a = predict_proba(model_a, X_test_a)
            pred_a = (proba_a > 0.5).astype(int)
            
            # Model B: All features (enhanced)
            X_train_b = train[all_features].copy()
            X_test_b = test[all_features].copy()
            
            model_b, _ = train_model(X_train_b, y_train)
            proba_b = predict_proba(model_b, X_test_b)
            pred_b = (proba_b > 0.5).astype(int)
            
            # Store
            all_y_true.extend(y_test.values)
            all_proba_a.extend(proba_a)
            all_proba_b.extend(proba_b)
            
            all_metrics_a.append(compute_metrics(y_test, pred_a, proba_a))
            all_metrics_b.append(compute_metrics(y_test, pred_b, proba_b))
        
        if len(all_y_true) < 20:
            print(f"    {window_name}: Insufficient test samples")
            continue
        
        y_true_arr = np.array(all_y_true)
        proba_a_arr = np.array(all_proba_a)
        proba_b_arr = np.array(all_proba_b)
        
        # DM test
        loss_a = compute_squared_loss(y_true_arr, proba_a_arr)
        loss_b = compute_squared_loss(y_true_arr, proba_b_arr)
        dm_stat, p_value = diebold_mariano_test(loss_a, loss_b)
        
        # Aggregate metrics
        avg_metrics_a = {k: np.mean([m[k] for m in all_metrics_a]) for k in all_metrics_a[0]}
        avg_metrics_b = {k: np.mean([m[k] for m in all_metrics_b]) for k in all_metrics_b[0]}
        
        # Lifts
        f1_lift = ((avg_metrics_b['f1'] - avg_metrics_a['f1']) / avg_metrics_a['f1'] * 100 
                   if avg_metrics_a['f1'] > 0 else np.nan)
        auc_lift = ((avg_metrics_b['auc'] - avg_metrics_a['auc']) / avg_metrics_a['auc'] * 100
                    if avg_metrics_a['auc'] > 0 else np.nan)
        
        result = {
            'Ticker': ticker,
            'Sector': df_ticker['Sector'].iloc[0],
            'Window': window_name,
            'Window_Weeks': window_size,
            'N_Folds': n_splits,
            'N_Test': len(y_true_arr),
            
            # Baseline (Technical only)
            'F1_Tech': avg_metrics_a['f1'],
            'AUC_Tech': avg_metrics_a['auc'],
            'Precision_Tech': avg_metrics_a['precision'],
            'Recall_Tech': avg_metrics_a['recall'],
            
            # Enhanced (All features)
            'F1_Enhanced': avg_metrics_b['f1'],
            'AUC_Enhanced': avg_metrics_b['auc'],
            'Precision_Enhanced': avg_metrics_b['precision'],
            'Recall_Enhanced': avg_metrics_b['recall'],
            
            # Comparison
            'F1_Lift_%': f1_lift,
            'AUC_Lift_%': auc_lift,
            'DM_Stat': dm_stat,
            'P_Value': p_value,
            'Significant': 'Yes' if is_significant(p_value) else 'No',
        }
        
        results.append(result)
        
        sig = "✓" if is_significant(p_value) else "✗"
        print(f"    {window_name}: F1={avg_metrics_b['f1']:.1%} (lift {f1_lift:+.1f}%), "
              f"AUC={avg_metrics_b['auc']:.3f} {sig}")
    
    return pd.DataFrame(results)


def run_full_tournament(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run tournament across all tickers."""
    
    # Get available features
    tech_features = [f for f in TECHNICAL_FEATURES if f in df.columns]
    global_features = [f for f in GLOBAL_FEATURES if f in df.columns]
    egypt_features = [f for f in EGYPT_FEATURES if f in df.columns]
    all_features = tech_features + global_features + egypt_features
    
    print(f"\nFeatures:")
    print(f"  Technical: {len(tech_features)}")
    print(f"  Global: {len(global_features)}")
    print(f"  Egypt: {len(egypt_features)}")
    print(f"  Total: {len(all_features)}")
    
    tickers = df['Ticker'].unique()
    print(f"\nRunning tournament for {len(tickers)} tickers...")
    
    all_results = []
    
    for ticker in tqdm(tickers, desc="Tickers"):
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        df_ticker = df_ticker.dropna(subset=['target'])
        
        if len(df_ticker) < MIN_HISTORY[26]:
            print(f"\n  {ticker}: Skipped ({len(df_ticker)} weeks)")
            continue
        
        print(f"\n  {ticker} ({len(df_ticker)} weeks):")
        
        ticker_results = run_ticker_tournament(df_ticker, ticker, tech_features, all_features)
        
        if len(ticker_results) > 0:
            all_results.append(ticker_results)
    
    if len(all_results) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    report = pd.concat(all_results, ignore_index=True)
    
    # Winners
    winners = report.loc[report.groupby('Ticker')['F1_Enhanced'].idxmax()]
    winners = winners[['Ticker', 'Sector', 'Window', 'F1_Enhanced', 'AUC_Enhanced', 'F1_Lift_%', 'Significant']]
    
    return report, winners


def generate_summary(report: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("TOURNAMENT SUMMARY (v2 - Weekly XGBoost)")
    print("=" * 70)
    
    total = len(report)
    significant = (report['Significant'] == 'Yes').sum()
    
    print(f"\nTotal combinations: {total}")
    print(f"Statistically significant: {significant} ({100*significant/total:.1f}%)")
    
    print(f"\n{'Metric':<15} {'Technical':>12} {'Enhanced':>12} {'Improvement':>12}")
    print("-" * 51)
    
    for metric in ['F1', 'AUC', 'Precision', 'Recall']:
        tech = report[f'{metric}_Tech'].mean()
        enh = report[f'{metric}_Enhanced'].mean()
        imp = (enh - tech) / tech * 100 if tech > 0 else 0
        print(f"{metric:<15} {tech:>11.1%} {enh:>11.1%} {imp:>+11.1f}%")
    
    print("\nBy Window:")
    for window in ['Short', 'Medium', 'Long']:
        subset = report[report['Window'] == window]
        if len(subset) > 0:
            f1 = subset['F1_Enhanced'].mean()
            auc = subset['AUC_Enhanced'].mean()
            sig_pct = (subset['Significant'] == 'Yes').mean() * 100
            print(f"  {window}: F1={f1:.1%}, AUC={auc:.3f}, Sig={sig_pct:.0f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("EGX PREDICTION MODEL v2")
    print("Weekly XGBoost with Global Indicators")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n[1] Loading weekly data...")
    df = load_and_prepare_weekly_data()
    
    # Engineer features
    print("\n[2] Engineering features...")
    df = engineer_all_features(df)
    
    # Drop NaN targets
    initial = len(df)
    df = df.dropna(subset=['target'])
    print(f"  Rows: {len(df):,} (dropped {initial - len(df):,})")
    
    # Run tournament
    print("\n[3] Running tournament...")
    report, winners = run_full_tournament(df)
    
    if len(report) == 0:
        print("No results!")
        return
    
    # Save
    print("\n[4] Saving results...")
    report.to_csv(RESULTS_DIR / 'significance_report_v2.csv', index=False)
    winners.to_csv(RESULTS_DIR / 'tournament_winners_v2.csv', index=False)
    print(f"  Saved to {RESULTS_DIR}")
    
    # Summary
    generate_summary(report)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
