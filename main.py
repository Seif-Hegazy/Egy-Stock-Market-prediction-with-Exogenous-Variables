#!/usr/bin/env python3
"""
EGX Prediction Model v3 - Main Runner (Research Exact)
Implements strictly controlled experiment:
- 5-day rolling window concatenation
- Neutral zone filtering
- Chronological split
- Threshold optimization
- Comparison: Endogenous Only vs. Endogenous + Exogenous
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader_v3 import load_raw_data, construct_rolling_windows, prepare_datasets
from src.models_v3 import train_model, get_percentile_threshold, evaluate_model
from src.validation import diebold_mariano_test, compute_squared_loss, is_significant

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent / 'results' / 'v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment():
    print("=" * 70)
    print("EGX PREDICTION MODEL v3 - RESEARCH FRAMEWORK")
    print("5-day Rolling Window | Neutral Zone Labeling | Fixed Q0.40 Threshold")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1] Loading Raw Data...")
    stocks, macro = load_raw_data()
    print(f"  Stocks: {len(stocks):,} rows")
    print(f"  Macro:  {len(macro):,} rows")
    
    tickers = stocks['Ticker'].unique()
    results = []
    
    print(f"\n[2] Running Experiment for {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers):
        # Prepare data for this ticker
        df_ticker = stocks[stocks['Ticker'] == ticker].copy()
        
        # Construct rolling windows (Week 0 + Week 1 -> Week 2)
        samples = construct_rolling_windows(df_ticker, macro, w=5, margin=0.001)
        
        if len(samples) < 100:  # Minimum samples check
            continue
            
        # Split and Scale
        data = prepare_datasets(samples)
        
        # --- Define Feature Sets ---
        all_cols = data['X_train'].columns.tolist()
        
        # Model A: Endogenous Only (Price + Technical features)
        # Columns that start with 'price_' or 'tech_'
        feat_endo = [c for c in all_cols if c.startswith('price_') or c.startswith('tech_')]
        
        # Model B: Endogenous + Exogenous (All features)
        feat_all = all_cols  # Price + Macro columns
        
        # check if we actually have macro features
        if len(feat_endo) == len(feat_all):
            print(f"  Warning: No macro features found for {ticker}")
            continue
            
        # --- Train Model A (Endogenous) ---
        model_a = train_model(
            data['X_train'][feat_endo], data['y_train'],
            data['X_val'][feat_endo], data['y_val']
        )
        
        # Threshold A (Q0.40)
        thresh_a = get_percentile_threshold(model_a, data['X_val'][feat_endo], quantile=0.40)
        
        # Evaluate A
        metrics_a = evaluate_model(model_a, data['X_test'][feat_endo], data['y_test'], thresh_a)
        
        # --- Train Model B (Enhanced) ---
        model_b = train_model(
            data['X_train'][feat_all], data['y_train'],
            data['X_val'][feat_all], data['y_val']
        )
        
        # Threshold B (Q0.40)
        thresh_b = get_percentile_threshold(model_b, data['X_val'][feat_all], quantile=0.40)
        
        # Evaluate B
        metrics_b = evaluate_model(model_b, data['X_test'][feat_all], data['y_test'], thresh_b)
        
        # --- Statistical Significance (DM Test) ---
        probs_a = model_a.predict_proba(data['X_test'][feat_endo])[:, 1]
        probs_b = model_b.predict_proba(data['X_test'][feat_all])[:, 1]
        y_true = data['y_test'].values
        
        loss_a = compute_squared_loss(y_true, probs_a)
        loss_b = compute_squared_loss(y_true, probs_b)
        dm_stat, p_value = diebold_mariano_test(loss_a, loss_b)
        
        # --- Store Results ---
        res = {
            'Ticker': ticker,
            'Samples': len(samples),
            'Test_Size': len(y_true),
            'Model_A_F1': metrics_a['f1'],
            'Model_B_F1': metrics_b['f1'],
            'Model_A_Precision': metrics_a['precision'],
            'Model_B_Precision': metrics_b['precision'],
            'Model_A_Recall': metrics_a['recall'],
            'Model_B_Recall': metrics_b['recall'],
            'F1_Lift': (metrics_b['f1'] - metrics_a['f1']) / metrics_a['f1'] if metrics_a['f1'] > 0 else 0,
            'Model_A_AUC': metrics_a['auc'],
            'Model_B_AUC': metrics_b['auc'],
            'DM_Stat': dm_stat,
            'P_Value': p_value,
            'Significant': is_significant(p_value),
            'Thresh_A': thresh_a,
            'Thresh_B': thresh_b
        }
        results.append(res)
        
        # Live log significant results
        if is_significant(p_value) and res['F1_Lift'] > 0:
            print(f"  {ticker}: Lift={res['F1_Lift']:.1%} (P: {res['Model_A_Precision']:.2f}->{res['Model_B_Precision']:.2f}, R: {res['Model_A_Recall']:.2f}->{res['Model_B_Recall']:.2f})")
            
    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'v3_results.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("V3 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if len(df_results) > 0:
        sig_count = df_results['Significant'].sum()
        pos_lift = (df_results['F1_Lift'] > 0).sum()
        
        print(f"Total Tickers: {len(df_results)}")
        print(f"Significant Differences: {sig_count} ({sig_count/len(df_results):.1%})")
        print(f"Positive Lift: {pos_lift} ({pos_lift/len(df_results):.1%})")
        print(f"Mean F1 Lift: {df_results['F1_Lift'].mean():.1%}")
        
        print("\nTop 5 Improvers:")
        print(df_results.sort_values('F1_Lift', ascending=False).head(5)[
            ['Ticker', 'Model_A_F1', 'Model_B_F1', 'F1_Lift', 'Significant']
        ])
    else:
        print("No results generated.")


if __name__ == '__main__':
    run_experiment()
