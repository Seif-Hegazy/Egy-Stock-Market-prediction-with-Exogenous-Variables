"""
EGX Macro Significance Study - Main Experiment (v3 FINAL)
=========================================================
FIXES:
1. Optimal threshold search (maximize F1 on validation)
2. Global correlation filtering
3. Proper statistical testing
4. Quality gates
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import (
    load_raw_data,
    create_endogenous_samples,
    create_exogenous_samples,
    prepare_datasets,
    MIN_SAMPLES
)
from src.models import train_model, evaluate_model
from src.validation import diebold_mariano_test, compute_squared_loss, is_significant

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MIN_BASELINE_F1 = 0.35


def find_optimal_threshold(model, X_val, y_val) -> float:
    """
    Find threshold that maximizes F1 on validation set.
    This is better than fixed percentile because it adapts to each model.
    """
    probs = model.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0
    
    for thresh in np.arange(0.3, 0.7, 0.02):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh


def run_experiment():
    print("=" * 70)
    print("EGX MACRO SIGNIFICANCE STUDY (v3 FINAL)")
    print("Semi-overlapping | Fixed features | Optimal threshold | No leakage")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading Data...")
    stocks, macro = load_raw_data()
    print(f"  Stocks: {len(stocks):,} rows")
    print(f"  Macro:  {len(macro):,} rows")
    
    tickers = stocks['Ticker'].unique()
    print(f"\n[2] Running for {len(tickers)} tickers...")
    
    results = []
    skipped = {'samples': [], 'baseline': []}
    
    for ticker in tqdm(tickers):
        df_ticker = stocks[stocks['Ticker'] == ticker].copy()
        
        if len(df_ticker) < 500:
            continue
        
        # === Create BOTH pipelines ===
        samples_endo = create_endogenous_samples(df_ticker, macro)
        samples_exo = create_exogenous_samples(df_ticker, macro)
        
        # Quality gate: minimum samples
        if len(samples_endo) < MIN_SAMPLES:
            skipped['samples'].append(ticker)
            continue
        
        # === Prepare datasets (identical split for both) ===
        data_endo = prepare_datasets(samples_endo)
        data_exo = prepare_datasets(samples_exo)
        
        # === Train Endogenous Model ===
        try:
            model_endo = train_model(
                data_endo['X_train'], data_endo['y_train'],
                data_endo['X_val'], data_endo['y_val']
            )
        except Exception as e:
            print(f"  ERROR {ticker} (endo): {e}")
            continue
        
        # Optimal threshold for endo
        thresh_endo = find_optimal_threshold(model_endo, data_endo['X_val'], data_endo['y_val'])
        
        # Evaluate endo
        metrics_endo = evaluate_model(model_endo, data_endo['X_test'], data_endo['y_test'], thresh_endo)
        
        # Quality gate: minimum baseline F1
        if metrics_endo['f1'] < MIN_BASELINE_F1:
            skipped['baseline'].append((ticker, metrics_endo['f1']))
            continue
        
        # === Train Exogenous Model ===
        try:
            model_exo = train_model(
                data_exo['X_train'], data_exo['y_train'],
                data_exo['X_val'], data_exo['y_val']
            )
        except Exception as e:
            print(f"  ERROR {ticker} (exo): {e}")
            continue
        
        # Optimal threshold for exo
        thresh_exo = find_optimal_threshold(model_exo, data_exo['X_val'], data_exo['y_val'])
        
        # Evaluate exo
        metrics_exo = evaluate_model(model_exo, data_exo['X_test'], data_exo['y_test'], thresh_exo)
        
        # === Statistical Test ===
        probs_endo = model_endo.predict_proba(data_endo['X_test'])[:, 1]
        probs_exo = model_exo.predict_proba(data_exo['X_test'])[:, 1]
        
        loss_endo = compute_squared_loss(data_endo['y_test'], probs_endo)
        loss_exo = compute_squared_loss(data_exo['y_test'], probs_exo)
        
        dm_stat, p_value = diebold_mariano_test(loss_endo, loss_exo)
        
        # Compute lift
        lift = (metrics_exo['f1'] - metrics_endo['f1']) / metrics_endo['f1'] if metrics_endo['f1'] > 0 else 0
        
        # Significant only if POSITIVE lift and p < 0.05
        significant = is_significant(p_value) and lift > 0
        
        res = {
            'Ticker': ticker,
            'Samples': len(samples_endo),
            'Train': data_endo['n_train'],
            'Val': data_endo['n_val'],
            'Test': data_endo['n_test'],
            
            'Endo_Features': len(data_endo['feature_names']),
            'Exo_Features': len(data_exo['feature_names']),
            
            'Endo_F1': metrics_endo['f1'],
            'Endo_Precision': metrics_endo['precision'],
            'Endo_Recall': metrics_endo['recall'],
            'Endo_Threshold': thresh_endo,
            
            'Exo_F1': metrics_exo['f1'],
            'Exo_Precision': metrics_exo['precision'],
            'Exo_Recall': metrics_exo['recall'],
            'Exo_Threshold': thresh_exo,
            
            'F1_Lift': lift,
            'Exo_Better': metrics_exo['f1'] > metrics_endo['f1'],
            'DM_Stat': dm_stat,
            'P_Value': p_value,
            'Significant': significant,
        }
        results.append(res)
        
        if significant:
            print(f"  ✓ {ticker}: +{lift*100:.1f}% (F1: {metrics_endo['f1']:.2f}→{metrics_exo['f1']:.2f})")
    
    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'experiment_results_v3.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (v3 FINAL)")
    print("=" * 70)
    
    print(f"\nQuality Gates:")
    print(f"  Skipped (insufficient samples): {len(skipped['samples'])}")
    print(f"  Skipped (low baseline F1<{MIN_BASELINE_F1}): {len(skipped['baseline'])}")
    
    if len(df_results) > 0:
        print(f"\nAnalyzed: {len(df_results)} tickers")
        print(f"Features: {df_results['Endo_Features'].iloc[0]} endo, {df_results['Exo_Features'].iloc[0]} exo")
        
        exo_better = df_results[df_results['Exo_Better']]
        sig = df_results[df_results['Significant']]
        
        print(f"\n✓ Exogenous Better: {len(exo_better)}/{len(df_results)} ({100*len(exo_better)/len(df_results):.0f}%)")
        print(f"✓ Statistically Significant: {len(sig)}/{len(df_results)} ({100*len(sig)/len(df_results):.0f}%)")
        print(f"  Mean Lift: {df_results['F1_Lift'].mean()*100:+.1f}%")
        
        print("\nTop 5 Improvements:")
        top = df_results.nlargest(5, 'F1_Lift')[['Ticker', 'Endo_F1', 'Exo_F1', 'F1_Lift', 'P_Value', 'Significant']]
        print(top.to_string(index=False))
    
    return df_results


if __name__ == '__main__':
    run_experiment()
