"""
EGX Macro Significance Study - Main Experiment (v5 FINAL)
=========================================================
FIXES:
1. Youden's J statistic for optimal threshold (Sensitivity + Specificity - 1)
2. Wider threshold search (0.20-0.80)
3. Less aggressive filtering (min F1 = 0.25)
4. ROC-AUC based probability calibration check
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix
)

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

MIN_BASELINE_F1 = 0.25  # Relaxed to include more tickers


def find_optimal_threshold_youden(model, X_val, y_val) -> float:
    """
    Find threshold using Youden's J statistic.
    J = Sensitivity + Specificity - 1 = TPR - FPR
    
    This is the standard method for finding optimal classification threshold.
    Maximizes the distance from the random guess line on ROC curve.
    """
    probs = model.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_j = -1
    
    # Wide search range
    for thresh in np.arange(0.20, 0.81, 0.01):
        preds = (probs >= thresh).astype(int)
        
        # Need both classes predicted
        if len(np.unique(preds)) == 1:
            continue
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        
        # Sensitivity (TPR) and Specificity (TNR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Youden's J
        j = sensitivity + specificity - 1
        
        if j > best_j:
            best_j = j
            best_thresh = thresh
    
    return best_thresh


def run_experiment():
    print("=" * 70)
    print("EGX MACRO SIGNIFICANCE STUDY (v5 FINAL)")
    print("Youden's J threshold | Wide search | Relaxed filtering")
    print("=" * 70)
    
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
        
        # Create pipelines
        samples_endo = create_endogenous_samples(df_ticker, macro)
        samples_exo = create_exogenous_samples(df_ticker, macro)
        
        if len(samples_endo) < MIN_SAMPLES:
            skipped['samples'].append(ticker)
            continue
        
        # Prepare datasets
        data_endo = prepare_datasets(samples_endo)
        data_exo = prepare_datasets(samples_exo)
        
        train_pos_ratio = data_endo['y_train'].mean()
        
        # === Train Endogenous Model ===
        try:
            model_endo = train_model(
                data_endo['X_train'], data_endo['y_train'],
                data_endo['X_val'], data_endo['y_val']
            )
        except Exception as e:
            print(f"  ERROR {ticker} (endo): {e}")
            continue
        
        # Youden's J threshold
        thresh_endo = find_optimal_threshold_youden(
            model_endo, data_endo['X_val'], data_endo['y_val']
        )
        
        # Evaluate
        metrics_endo = evaluate_model(
            model_endo, data_endo['X_test'], data_endo['y_test'], thresh_endo
        )
        
        # Relaxed quality gate
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
        
        # Youden's J threshold
        thresh_exo = find_optimal_threshold_youden(
            model_exo, data_exo['X_val'], data_exo['y_val']
        )
        
        metrics_exo = evaluate_model(
            model_exo, data_exo['X_test'], data_exo['y_test'], thresh_exo
        )
        
        # === ROC-AUC for probability quality ===
        try:
            auc_endo = roc_auc_score(data_endo['y_test'], 
                                      model_endo.predict_proba(data_endo['X_test'])[:, 1])
            auc_exo = roc_auc_score(data_exo['y_test'],
                                     model_exo.predict_proba(data_exo['X_test'])[:, 1])
        except:
            auc_endo = auc_exo = 0.5
        
        # === Statistical Test ===
        probs_endo = model_endo.predict_proba(data_endo['X_test'])[:, 1]
        probs_exo = model_exo.predict_proba(data_exo['X_test'])[:, 1]
        
        loss_endo = compute_squared_loss(data_endo['y_test'], probs_endo)
        loss_exo = compute_squared_loss(data_exo['y_test'], probs_exo)
        
        dm_stat, p_value = diebold_mariano_test(loss_endo, loss_exo)
        
        # Compute lift
        lift = (metrics_exo['f1'] - metrics_endo['f1']) / metrics_endo['f1'] if metrics_endo['f1'] > 0 else 0
        
        significant = is_significant(p_value) and lift > 0
        
        res = {
            'Ticker': ticker,
            'Samples': len(samples_endo),
            'Train_Pos_Ratio': round(train_pos_ratio, 3),
            'Test': data_endo['n_test'],
            
            'Endo_F1': round(metrics_endo['f1'], 4),
            'Endo_Precision': round(metrics_endo['precision'], 4),
            'Endo_Recall': round(metrics_endo['recall'], 4),
            'Endo_Threshold': round(thresh_endo, 2),
            'Endo_AUC': round(auc_endo, 4),
            
            'Exo_F1': round(metrics_exo['f1'], 4),
            'Exo_Precision': round(metrics_exo['precision'], 4),
            'Exo_Recall': round(metrics_exo['recall'], 4),
            'Exo_Threshold': round(thresh_exo, 2),
            'Exo_AUC': round(auc_exo, 4),
            
            'F1_Lift': round(lift, 4),
            'AUC_Lift': round(auc_exo - auc_endo, 4),
            'Exo_Better': metrics_exo['f1'] > metrics_endo['f1'],
            'DM_Stat': round(dm_stat, 4),
            'P_Value': round(p_value, 6),
            'Significant': significant,
        }
        results.append(res)
        
        # Log
        status = "✓" if significant else " "
        print(f"  {status} {ticker}: Lift={lift*100:+.1f}% "
              f"(AUC:{auc_endo:.2f}→{auc_exo:.2f}, Th:{thresh_endo:.2f}→{thresh_exo:.2f})")
    
    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'experiment_results_v5.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (v5 FINAL)")
    print("=" * 70)
    
    print(f"\nFiltering:")
    print(f"  Skipped (insufficient samples): {len(skipped['samples'])}")
    print(f"  Skipped (low baseline F1<{MIN_BASELINE_F1}): {len(skipped['baseline'])}")
    
    if len(df_results) > 0:
        print(f"\nAnalyzed: {len(df_results)} tickers")
        print(f"Threshold range: {df_results['Endo_Threshold'].min():.2f} - {df_results['Endo_Threshold'].max():.2f}")
        
        exo_better = df_results[df_results['Exo_Better']]
        sig = df_results[df_results['Significant']]
        
        print(f"\n✓ Exogenous Better (F1): {len(exo_better)}/{len(df_results)} ({100*len(exo_better)/len(df_results):.0f}%)")
        print(f"✓ Statistically Significant: {len(sig)}/{len(df_results)} ({100*len(sig)/len(df_results):.0f}%)")
        print(f"  Mean F1 Lift: {df_results['F1_Lift'].mean()*100:+.1f}%")
        print(f"  Mean AUC Lift: {df_results['AUC_Lift'].mean()*100:+.2f}%")
        
        print("\nTop 5 by F1 Lift:")
        cols = ['Ticker', 'Endo_F1', 'Exo_F1', 'F1_Lift', 'Endo_AUC', 'Exo_AUC', 'Significant']
        top = df_results.nlargest(5, 'F1_Lift')[cols]
        print(top.to_string(index=False))
        
        print("\nTop 5 by AUC Lift:")
        top_auc = df_results.nlargest(5, 'AUC_Lift')[cols]
        print(top_auc.to_string(index=False))
    
    return df_results


if __name__ == '__main__':
    run_experiment()
