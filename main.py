"""
EGX Macro Significance Study - Main Experiment Runner
======================================================
Compares Endogenous-only model vs Exogenous-enhanced model
using identical feature engineering, windowing, and thresholding.
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

from src.data_loader import (
    load_raw_data, 
    create_endogenous_samples, 
    create_exogenous_samples,
    prepare_datasets
)
from src.models import train_model, get_percentile_threshold, evaluate_model
from src.validation import diebold_mariano_test, compute_squared_loss, is_significant

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment():
    print("=" * 70)
    print("EGX MACRO SIGNIFICANCE STUDY")
    print("Endogenous (Technical) vs Exogenous (Technical + Macro)")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1] Loading Raw Data...")
    stocks, macro = load_raw_data()
    print(f"  Stocks: {len(stocks):,} rows")
    print(f"  Macro:  {len(macro):,} rows")
    
    # Get unique tickers
    tickers = stocks['Ticker'].unique()
    print(f"\n[2] Running Experiment for {len(tickers)} tickers...")
    
    results = []
    
    for ticker in tqdm(tickers):
        df_ticker = stocks[stocks['Ticker'] == ticker].copy()
        
        # Skip tickers with insufficient data
        if len(df_ticker) < 500:
            continue
        
        # --- Pipeline A: Endogenous Only (Technical) ---
        samples_a = create_endogenous_samples(df_ticker, macro)
        if len(samples_a) < 100:
            continue
            
        data_a = prepare_datasets(samples_a)
        
        # --- Pipeline B: Exogenous (Technical + Macro) ---
        samples_b = create_exogenous_samples(df_ticker, macro)
        data_b = prepare_datasets(samples_b)
        
        # Train Models
        model_a = train_model(data_a['X_train'], data_a['y_train'],
                              data_a['X_val'], data_a['y_val'])
        model_b = train_model(data_b['X_train'], data_b['y_train'],
                              data_b['X_val'], data_b['y_val'])
        
        # Get thresholds (fixed percentile)
        thresh_a = get_percentile_threshold(model_a, data_a['X_val'], quantile=0.40)
        thresh_b = get_percentile_threshold(model_b, data_b['X_val'], quantile=0.40)
        
        # Evaluate on Test Set
        y_true = data_a['y_test']  # Same for both (same samples, different features)
        
        metrics_a = evaluate_model(model_a, data_a['X_test'], y_true, thresh_a)
        metrics_b = evaluate_model(model_b, data_b['X_test'], y_true, thresh_b)
        
        # Statistical Test (Diebold-Mariano)
        probs_a = model_a.predict_proba(data_a['X_test'])[:, 1]
        probs_b = model_b.predict_proba(data_b['X_test'])[:, 1]
        
        loss_a = compute_squared_loss(y_true.values, probs_a)
        loss_b = compute_squared_loss(y_true.values, probs_b)
        
        dm_stat, p_value = diebold_mariano_test(loss_a, loss_b)
        
        # Store Results
        res = {
            'Ticker': ticker,
            'Samples': len(samples_a),
            'Test_Size': len(y_true),
            
            # Model A (Endogenous)
            'Endo_F1': metrics_a['f1'],
            'Endo_Precision': metrics_a['precision'],
            'Endo_Recall': metrics_a['recall'],
            'Endo_AUC': metrics_a['auc'],
            
            # Model B (Exogenous)
            'Exo_F1': metrics_b['f1'],
            'Exo_Precision': metrics_b['precision'],
            'Exo_Recall': metrics_b['recall'],
            'Exo_AUC': metrics_b['auc'],
            
            # Comparison
            'F1_Lift': (metrics_b['f1'] - metrics_a['f1']) / metrics_a['f1'] if metrics_a['f1'] > 0 else 0,
            'Exo_Better': metrics_b['f1'] > metrics_a['f1'],
            
            # Statistical Significance
            'DM_Stat': dm_stat,
            'P_Value': p_value,
            'Significant': is_significant(p_value),
        }
        results.append(res)
        
        # Live log significant improvements
        if is_significant(p_value) and res['F1_Lift'] > 0:
            print(f"  {ticker}: Lift={res['F1_Lift']*100:.1f}% "
                  f"(Endo F1={metrics_a['f1']:.2f} -> Exo F1={metrics_b['f1']:.2f})")
    
    # Save Results
    df_results = pd.DataFrame(results)
    output_path = RESULTS_DIR / 'experiment_results.csv'
    df_results.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total Tickers: {len(df_results)}")
    
    sig_better = df_results[(df_results['Significant']) & (df_results['Exo_Better'])]
    print(f"Exogenous Significantly Better: {len(sig_better)} ({100*len(sig_better)/len(df_results):.1f}%)")
    
    mean_lift = df_results['F1_Lift'].mean() * 100
    print(f"Mean F1 Lift: {mean_lift:+.2f}%")
    
    print("\nTop 5 Improvements (Exogenous vs Endogenous):")
    top5 = df_results.nlargest(5, 'F1_Lift')[['Ticker', 'Endo_F1', 'Exo_F1', 'F1_Lift', 'Significant']]
    print(top5.to_string(index=False))
    
    return df_results


if __name__ == '__main__':
    run_experiment()
