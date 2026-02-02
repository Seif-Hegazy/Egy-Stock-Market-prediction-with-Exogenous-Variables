"""
EGX Macro Significance Study - Main Experiment Runner (v2 FIXED)
================================================================
Fixes:
1. Unified correlation filtering (exogenous features, then subset for endogenous)
2. Minimum quality thresholds (samples, F1)
3. Non-overlapping windows
4. Better reporting
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_raw_data, 
    create_endogenous_samples, 
    create_exogenous_samples,
    prepare_datasets,
    MIN_SAMPLES
)
from src.models import train_model, get_percentile_threshold, evaluate_model
from src.validation import diebold_mariano_test, compute_squared_loss, is_significant
from src.feature_selection import remove_correlated_features, analyze_feature_importance

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CORRELATION_THRESHOLD = 0.85
MIN_F1_THRESHOLD = 0.30  # Exclude tickers where even baseline fails


def run_experiment():
    print("=" * 70)
    print("EGX MACRO SIGNIFICANCE STUDY (v2 FIXED)")
    print("Non-overlapping windows | Unified correlation filtering | Quality gates")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1] Loading Raw Data...")
    stocks, macro = load_raw_data()
    print(f"  Stocks: {len(stocks):,} rows")
    print(f"  Macro:  {len(macro):,} rows")
    
    tickers = stocks['Ticker'].unique()
    print(f"\n[2] Running Experiment for {len(tickers)} tickers...")
    
    results = []
    feature_importance_all = []
    skipped = {'insufficient_samples': [], 'low_baseline': []}
    
    for ticker in tqdm(tickers):
        df_ticker = stocks[stocks['Ticker'] == ticker].copy()
        
        # Skip tickers with insufficient raw data
        if len(df_ticker) < 500:
            continue
        
        # --- Generate Samples ---
        samples_exo = create_exogenous_samples(df_ticker, macro)
        samples_endo = create_endogenous_samples(df_ticker, macro)
        
        # Quality Gate 1: Minimum samples
        if len(samples_exo) < MIN_SAMPLES:
            skipped['insufficient_samples'].append(ticker)
            continue
        
        # Prepare datasets
        data_exo = prepare_datasets(samples_exo)
        data_endo = prepare_datasets(samples_endo)
        
        # --- UNIFIED Correlation Filtering ---
        # Step 1: Filter on EXOGENOUS features (full set)
        X_train_exo_filtered, dropped_exo = remove_correlated_features(
            data_exo['X_train'], data_exo['y_train'],
            threshold=CORRELATION_THRESHOLD, verbose=False
        )
        keep_cols_exo = X_train_exo_filtered.columns.tolist()
        
        # Step 2: Apply same filter to exo val/test
        X_val_exo = data_exo['X_val'][keep_cols_exo]
        X_test_exo = data_exo['X_test'][keep_cols_exo]
        
        # Step 3: For ENDOGENOUS, use subset of exo features (non-macro only)
        keep_cols_endo = [c for c in keep_cols_exo if not c.startswith('macro_')]
        X_train_endo = data_endo['X_train'][[c for c in keep_cols_endo if c in data_endo['X_train'].columns]]
        X_val_endo = data_endo['X_val'][[c for c in keep_cols_endo if c in data_endo['X_val'].columns]]
        X_test_endo = data_endo['X_test'][[c for c in keep_cols_endo if c in data_endo['X_test'].columns]]
        
        # Feature importance (first ticker only)
        if len(feature_importance_all) == 0:
            importance = analyze_feature_importance(X_train_exo_filtered, data_exo['y_train'])
            importance['Ticker'] = ticker
            feature_importance_all.append(importance)
        
        # --- Train Models ---
        try:
            model_endo = train_model(X_train_endo, data_endo['y_train'],
                                     X_val_endo, data_endo['y_val'])
            model_exo = train_model(X_train_exo_filtered, data_exo['y_train'],
                                    X_val_exo, data_exo['y_val'])
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")
            continue
        
        # Thresholds
        thresh_endo = get_percentile_threshold(model_endo, X_val_endo, quantile=0.40)
        thresh_exo = get_percentile_threshold(model_exo, X_val_exo, quantile=0.40)
        
        # Evaluate
        y_true = data_endo['y_test']
        metrics_endo = evaluate_model(model_endo, X_test_endo, y_true, thresh_endo)
        metrics_exo = evaluate_model(model_exo, X_test_exo, y_true, thresh_exo)
        
        # Quality Gate 2: Minimum baseline F1
        if metrics_endo['f1'] < MIN_F1_THRESHOLD:
            skipped['low_baseline'].append((ticker, metrics_endo['f1']))
            continue
        
        # Statistical Test
        probs_endo = model_endo.predict_proba(X_test_endo)[:, 1]
        probs_exo = model_exo.predict_proba(X_test_exo)[:, 1]
        
        loss_endo = compute_squared_loss(y_true.values, probs_endo)
        loss_exo = compute_squared_loss(y_true.values, probs_exo)
        
        dm_stat, p_value = diebold_mariano_test(loss_endo, loss_exo)
        
        # Compute lift
        lift = (metrics_exo['f1'] - metrics_endo['f1']) / metrics_endo['f1'] if metrics_endo['f1'] > 0 else 0
        
        res = {
            'Ticker': ticker,
            'Samples': len(samples_exo),
            'Test_Size': len(y_true),
            'Endo_Features': len(X_train_endo.columns),
            'Exo_Features': len(keep_cols_exo),
            
            'Endo_F1': metrics_endo['f1'],
            'Endo_Precision': metrics_endo['precision'],
            'Endo_Recall': metrics_endo['recall'],
            
            'Exo_F1': metrics_exo['f1'],
            'Exo_Precision': metrics_exo['precision'],
            'Exo_Recall': metrics_exo['recall'],
            
            'F1_Lift': lift,
            'Exo_Better': metrics_exo['f1'] > metrics_endo['f1'],
            'DM_Stat': dm_stat,
            'P_Value': p_value,
            'Significant': is_significant(p_value) and lift > 0,  # Only positive significant
        }
        results.append(res)
        
        if res['Significant']:
            print(f"  ✓ {ticker}: Lift={lift*100:.1f}% (F1: {metrics_endo['f1']:.2f}→{metrics_exo['f1']:.2f})")
    
    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'experiment_results_v2.csv', index=False)
    
    # Save Feature Importance
    if feature_importance_all:
        df_importance = pd.concat(feature_importance_all)
        df_importance.to_csv(RESULTS_DIR / 'feature_importance_v2.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (v2 FIXED)")
    print("=" * 70)
    
    print(f"\nQuality Gates Applied:")
    print(f"  Skipped (insufficient samples): {len(skipped['insufficient_samples'])}")
    print(f"  Skipped (low baseline F1<{MIN_F1_THRESHOLD}): {len(skipped['low_baseline'])}")
    
    if len(df_results) > 0:
        print(f"\nAnalyzed Tickers: {len(df_results)}")
        print(f"Avg Endo Features: {df_results['Endo_Features'].mean():.0f}")
        print(f"Avg Exo Features: {df_results['Exo_Features'].mean():.0f}")
        
        sig_better = df_results[df_results['Significant']]
        print(f"\n✓ Exogenous Significantly Better: {len(sig_better)} ({100*len(sig_better)/len(df_results):.1f}%)")
        
        exo_better = df_results[df_results['Exo_Better']]
        print(f"  Exogenous Better (any): {len(exo_better)} ({100*len(exo_better)/len(df_results):.1f}%)")
        
        mean_lift = df_results['F1_Lift'].mean() * 100
        print(f"  Mean F1 Lift: {mean_lift:+.2f}%")
        
        print("\nTop 5 Improvements:")
        top5 = df_results.nlargest(5, 'F1_Lift')[['Ticker', 'Endo_F1', 'Exo_F1', 'F1_Lift', 'Significant']]
        print(top5.to_string(index=False))
    else:
        print("No results - all tickers filtered out")
    
    return df_results


if __name__ == '__main__':
    run_experiment()
