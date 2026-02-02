#!/usr/bin/env python3
"""
EGX Macro Significance Study - Main Orchestration Script

Runs the tournament to prove macroeconomic variables add significant 
predictive value to EGX30 stock direction.

Usage:
    python main.py
"""

import os
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_and_prepare_data
from src.feature_eng import engineer_all_features, ENDOGENOUS_FEATURES, EXOGENOUS_FEATURES
from src.models import train_model, predict_proba, predict
from src.validation import (
    create_cv_for_window,
    check_sufficient_history,
    diebold_mariano_test,
    compute_squared_loss,
    compute_metrics,
    compute_lift,
    is_significant,
    MIN_HISTORY
)

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

WINDOW_SIZES = {
    'Reactionary': 126,   # ~6 months
    'Tactical': 378,      # ~18 months
    'Strategic': 756,     # ~3 years
}

# Top 10 liquid EGX tickers for faster PoC (set to None for all 35 tickers)
FOCUS_TICKERS = None  # Run all 35 tickers

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Tournament Runner
# =============================================================================

def run_ticker_tournament(df_ticker: pd.DataFrame, ticker: str, 
                          endo_features: list, all_features: list) -> pd.DataFrame:
    """
    Run tournament for a single ticker across all window sizes.
    
    Args:
        df_ticker: Data for single ticker (sorted by date)
        ticker: Ticker symbol
        endo_features: Endogenous feature names
        all_features: All feature names (endo + exo)
        
    Returns:
        DataFrame with results for each window
    """
    results = []
    
    for window_name, window_size in WINDOW_SIZES.items():
        # Check sufficient history
        if not check_sufficient_history(len(df_ticker), window_size):
            print(f"    {window_name} ({window_size}d): Skipped - insufficient history")
            continue
        
        # Create CV splitter
        cv = create_cv_for_window(window_size)
        n_splits = cv.get_n_splits(df_ticker)
        
        if n_splits == 0:
            print(f"    {window_name}: Skipped - no valid folds")
            continue
        
        # Collect predictions across folds
        all_y_true = []
        all_proba_a = []
        all_proba_b = []
        all_metrics_a = []
        all_metrics_b = []
        
        for train_idx, test_idx in cv.split(df_ticker):
            train_data = df_ticker.iloc[train_idx]
            test_data = df_ticker.iloc[test_idx]
            
            # Get features and target
            X_train_endo = train_data[endo_features].copy()
            X_train_all = train_data[all_features].copy()
            y_train = train_data['target'].copy()
            
            X_test_endo = test_data[endo_features].copy()
            X_test_all = test_data[all_features].copy()
            y_test = test_data['target'].copy()
            
            # Skip if target has only one class
            if y_train.nunique() < 2:
                continue
            
            # Model A: Baseline (endogenous only)
            model_a, _ = train_model(X_train_endo, y_train)
            proba_a = predict_proba(model_a, X_test_endo)
            pred_a = (proba_a > 0.5).astype(int)
            
            # Model B: Enhanced (all features)
            model_b, _ = train_model(X_train_all, y_train)
            proba_b = predict_proba(model_b, X_test_all)
            pred_b = (proba_b > 0.5).astype(int)
            
            # Store for DM test
            all_y_true.extend(y_test.values)
            all_proba_a.extend(proba_a)
            all_proba_b.extend(proba_b)
            
            # Compute fold metrics
            metrics_a = compute_metrics(y_test, pred_a, proba_a)
            metrics_b = compute_metrics(y_test, pred_b, proba_b)
            all_metrics_a.append(metrics_a)
            all_metrics_b.append(metrics_b)
        
        if len(all_y_true) < 30:
            print(f"    {window_name}: Skipped - insufficient test samples")
            continue
        
        # Convert to arrays
        y_true_arr = np.array(all_y_true)
        proba_a_arr = np.array(all_proba_a)
        proba_b_arr = np.array(all_proba_b)
        
        # Compute losses for DM test
        loss_a = compute_squared_loss(y_true_arr, proba_a_arr)
        loss_b = compute_squared_loss(y_true_arr, proba_b_arr)
        
        # Diebold-Mariano test
        dm_stat, p_value = diebold_mariano_test(loss_a, loss_b)
        
        # Aggregate metrics
        avg_f1_a = np.mean([m['f1'] for m in all_metrics_a])
        avg_f1_b = np.mean([m['f1'] for m in all_metrics_b])
        avg_precision_a = np.mean([m['precision'] for m in all_metrics_a])
        avg_precision_b = np.mean([m['precision'] for m in all_metrics_b])
        avg_recall_a = np.mean([m['recall'] for m in all_metrics_a])
        avg_recall_b = np.mean([m['recall'] for m in all_metrics_b])
        
        # Compute lift
        f1_lift = compute_lift(avg_f1_a, avg_f1_b)
        
        result = {
            'Ticker': ticker,
            'Sector': df_ticker['Sector'].iloc[0],
            'Window': window_name,
            'Window_Days': window_size,
            'N_Folds': n_splits,
            'N_Test_Samples': len(y_true_arr),
            'F1_Baseline': avg_f1_a,
            'F1_Enhanced': avg_f1_b,
            'Lift_%': f1_lift,
            'Precision_Baseline': avg_precision_a,
            'Precision_Enhanced': avg_precision_b,
            'Recall_Baseline': avg_recall_a,
            'Recall_Enhanced': avg_recall_b,
            'DM_Stat': dm_stat,
            'P_Value': p_value,
            'Significant_Alpha?': 'Yes' if is_significant(p_value) else 'No',
        }
        
        results.append(result)
        
        sig_marker = "✓" if is_significant(p_value) else "✗"
        print(f"    {window_name}: F1 Lift={f1_lift:+.1f}%, p={p_value:.3f} {sig_marker}")
    
    return pd.DataFrame(results)


def run_full_tournament(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run tournament across all tickers.
    
    Returns:
        (significance_report, tournament_winners)
    """
    # Get available features
    endo_features = [f for f in ENDOGENOUS_FEATURES if f in df.columns]
    exo_features = [f for f in EXOGENOUS_FEATURES if f in df.columns]
    all_features = endo_features + exo_features
    
    print(f"\nFeatures:")
    print(f"  Endogenous: {len(endo_features)}")
    print(f"  Exogenous: {len(exo_features)}")
    print(f"  Total: {len(all_features)}")
    
    # Get unique tickers (filter by FOCUS_TICKERS if set)
    all_tickers = df['Ticker'].unique()
    if FOCUS_TICKERS:
        tickers = [t for t in FOCUS_TICKERS if t in all_tickers]
        print(f"\nRunning tournament for {len(tickers)} focus tickers (of {len(all_tickers)} total)...")
    else:
        tickers = all_tickers
        print(f"\nRunning tournament for {len(tickers)} tickers...")
    
    all_results = []
    
    for ticker in tqdm(tickers, desc="Tickers"):
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        
        # Drop rows with NaN target
        df_ticker = df_ticker.dropna(subset=['target'])
        
        if len(df_ticker) < MIN_HISTORY[126]:
            print(f"\n  {ticker}: Skipped - insufficient history ({len(df_ticker)} days)")
            continue
        
        print(f"\n  {ticker} ({len(df_ticker)} days):")
        
        ticker_results = run_ticker_tournament(
            df_ticker, ticker, endo_features, all_features
        )
        
        if len(ticker_results) > 0:
            all_results.append(ticker_results)
    
    # Combine results
    if len(all_results) == 0:
        print("No results generated!")
        return pd.DataFrame(), pd.DataFrame()
    
    significance_report = pd.concat(all_results, ignore_index=True)
    
    # Determine tournament winners (best window per ticker)
    tournament_winners = significance_report.loc[
        significance_report.groupby('Ticker')['Lift_%'].idxmax()
    ][['Ticker', 'Sector', 'Window', 'Lift_%', 'Significant_Alpha?']]
    tournament_winners = tournament_winners.rename(columns={'Window': 'Best_Window', 'Lift_%': 'Best_Lift_%'})
    
    return significance_report, tournament_winners


def generate_heatmap(significance_report: pd.DataFrame):
    """
    Generate sector × window heatmap of F1 lift.
    """
    pivot = significance_report.pivot_table(
        index='Sector',
        columns='Window',
        values='Lift_%',
        aggfunc='mean'
    )
    
    # Reorder columns
    window_order = ['Reactionary', 'Tactical', 'Strategic']
    pivot = pivot[[c for c in window_order if c in pivot.columns]]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt='.1f', 
        cmap='RdYlGn', 
        center=0,
        cbar_kws={'label': 'F1 Lift (%)'}
    )
    plt.title('F1 Lift (%) from Exogenous Variables\nby Sector × Window Size', fontsize=14)
    plt.xlabel('Training Window', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    plt.tight_layout()
    
    output_path = RESULTS_DIR / 'sector_window_heatmap.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nHeatmap saved: {output_path}")


def generate_summary_stats(significance_report: pd.DataFrame):
    """
    Print summary statistics.
    """
    print("\n" + "=" * 70)
    print("TOURNAMENT SUMMARY")
    print("=" * 70)
    
    total = len(significance_report)
    significant = (significance_report['Significant_Alpha?'] == 'Yes').sum()
    
    print(f"\nTotal ticker×window combinations: {total}")
    print(f"Statistically significant (p<0.05): {significant} ({100*significant/total:.1f}%)")
    
    print("\nMean F1 Lift by Window:")
    for window in ['Reactionary', 'Tactical', 'Strategic']:
        subset = significance_report[significance_report['Window'] == window]
        if len(subset) > 0:
            mean_lift = subset['Lift_%'].mean()
            sig_pct = (subset['Significant_Alpha?'] == 'Yes').mean() * 100
            print(f"  {window}: {mean_lift:+.1f}% (sig: {sig_pct:.0f}%)")
    
    print("\nMean F1 Lift by Sector:")
    for sector in significance_report['Sector'].unique():
        subset = significance_report[significance_report['Sector'] == sector]
        mean_lift = subset['Lift_%'].mean()
        print(f"  {sector}: {mean_lift:+.1f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("EGX MACRO SIGNIFICANCE STUDY")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Load and prepare data
    print("\n[1] Loading data...")
    df = load_and_prepare_data()
    
    # Phase 2: Engineer features
    print("\n[2] Engineering features...")
    df = engineer_all_features(df)
    
    # Drop rows with NaN features (due to lags)
    initial_rows = len(df)
    df = df.dropna(subset=['target'])
    print(f"  Rows after dropping NaN: {len(df):,} (dropped {initial_rows - len(df):,})")
    
    # Phase 3: Run tournament
    print("\n[3] Running tournament...")
    significance_report, tournament_winners = run_full_tournament(df)
    
    if len(significance_report) == 0:
        print("No results to report!")
        return
    
    # Phase 4: Save results
    print("\n[4] Saving results...")
    
    sig_path = RESULTS_DIR / 'significance_report.csv'
    significance_report.to_csv(sig_path, index=False)
    print(f"  Saved: {sig_path}")
    
    winner_path = RESULTS_DIR / 'tournament_winner.csv'
    tournament_winners.to_csv(winner_path, index=False)
    print(f"  Saved: {winner_path}")
    
    # Phase 5: Generate visualizations
    print("\n[5] Generating visualizations...")
    generate_heatmap(significance_report)
    
    # Summary
    generate_summary_stats(significance_report)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
