"""
EGX Macro Significance Study - Main Experiment (v4 FIXED)
=========================================================
FIXES:
1. MCC-based threshold optimization (penalizes all-positive predictions)
2. Balanced threshold search range (0.35-0.65)
3. Reject degenerate solutions (recall=1 or precision=1)
4. Better class weighting
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

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

MIN_BASELINE_F1 = 0.35


def find_optimal_threshold_mcc(model, X_val, y_val) -> float:
    """
    Find threshold that maximizes Matthews Correlation Coefficient (MCC).
    
    MCC is better than F1 because:
    - It accounts for all 4 confusion matrix quadrants
    - Penalizes degenerate solutions (all-positive, all-negative)
    - Range: -1 to +1 (0 = random, 1 = perfect)
    """
    probs = model.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_mcc = -1
    
    # Search in balanced range
    for thresh in np.arange(0.35, 0.66, 0.01):
        preds = (probs >= thresh).astype(int)
        
        # Skip degenerate solutions
        pred_ratio = preds.mean()
        if pred_ratio < 0.1 or pred_ratio > 0.9:
            continue
        
        mcc = matthews_corrcoef(y_val, preds)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh
    
    return best_thresh


def find_optimal_threshold_balanced_f1(model, X_val, y_val) -> float:
    """
    Find threshold using balanced F1 criteria.
    Penalizes solutions where precision or recall is extreme.
    """
    probs = model.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_score = -1
    
    for thresh in np.arange(0.35, 0.66, 0.01):
        preds = (probs >= thresh).astype(int)
        
        # Skip if all same prediction
        if len(np.unique(preds)) == 1:
            continue
        
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        # Penalize extreme precision/recall (want balance)
        balance_penalty = 1.0 - abs(precision - recall)
        
        # Penalize recall = 1 (all positive predictions)
        if recall >= 0.99:
            balance_penalty *= 0.5
        
        score = f1 * balance_penalty
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh


def run_experiment():
    print("=" * 70)
    print("EGX MACRO SIGNIFICANCE STUDY (v4 FIXED)")
    print("MCC threshold | Balanced predictions | No degenerate solutions")
    print("=" * 70)
    
    print("\n[1] Loading Data...")
    stocks, macro = load_raw_data()
    print(f"  Stocks: {len(stocks):,} rows")
    print(f"  Macro:  {len(macro):,} rows")
    
    tickers = stocks['Ticker'].unique()
    print(f"\n[2] Running for {len(tickers)} tickers...")
    
    results = []
    skipped = {'samples': [], 'baseline': [], 'degenerate': []}
    
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
        
        # Check class balance
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
        
        # MCC-based threshold
        thresh_endo = find_optimal_threshold_mcc(model_endo, data_endo['X_val'], data_endo['y_val'])
        
        # Evaluate
        metrics_endo = evaluate_model(model_endo, data_endo['X_test'], data_endo['y_test'], thresh_endo)
        
        # Check for degenerate solution
        if metrics_endo['recall'] >= 0.99 or metrics_endo['recall'] <= 0.01:
            skipped['degenerate'].append((ticker, 'endo', metrics_endo['recall']))
            # Try balanced F1 instead
            thresh_endo = find_optimal_threshold_balanced_f1(model_endo, data_endo['X_val'], data_endo['y_val'])
            metrics_endo = evaluate_model(model_endo, data_endo['X_test'], data_endo['y_test'], thresh_endo)
        
        # Quality gate
        if metrics_endo['f1'] < MIN_BASELINE_F1:
            skipped['baseline'].append((ticker, metrics_endo['f1']))
            continue
        
        # Still degenerate after retry?
        if metrics_endo['recall'] >= 0.99:
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
        
        # MCC-based threshold
        thresh_exo = find_optimal_threshold_mcc(model_exo, data_exo['X_val'], data_exo['y_val'])
        metrics_exo = evaluate_model(model_exo, data_exo['X_test'], data_exo['y_test'], thresh_exo)
        
        # Retry if degenerate
        if metrics_exo['recall'] >= 0.99 or metrics_exo['recall'] <= 0.01:
            thresh_exo = find_optimal_threshold_balanced_f1(model_exo, data_exo['X_val'], data_exo['y_val'])
            metrics_exo = evaluate_model(model_exo, data_exo['X_test'], data_exo['y_test'], thresh_exo)
        
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
            'Train_Pos_Ratio': train_pos_ratio,
            'Test': data_endo['n_test'],
            
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
        
        # Log balanced predictions
        if metrics_endo['recall'] < 0.95 and metrics_exo['recall'] < 0.95:
            status = "✓" if significant else " "
            print(f"  {status} {ticker}: Lift={lift*100:+.1f}% "
                  f"(R:{metrics_endo['recall']:.2f}→{metrics_exo['recall']:.2f})")
    
    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'experiment_results_v4.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (v4 FIXED)")
    print("=" * 70)
    
    print(f"\nQuality Gates:")
    print(f"  Skipped (insufficient samples): {len(skipped['samples'])}")
    print(f"  Skipped (low baseline F1): {len(skipped['baseline'])}")
    print(f"  Skipped (degenerate solutions): {len(skipped['degenerate'])}")
    
    if len(df_results) > 0:
        print(f"\nAnalyzed: {len(df_results)} tickers")
        
        # Check for balanced predictions
        balanced = df_results[(df_results['Endo_Recall'] < 0.95) & (df_results['Endo_Recall'] > 0.05)]
        print(f"Balanced predictions: {len(balanced)}/{len(df_results)}")
        
        if len(balanced) > 0:
            exo_better = balanced[balanced['Exo_Better']]
            sig = balanced[balanced['Significant']]
            
            print(f"\n✓ Exogenous Better: {len(exo_better)}/{len(balanced)} ({100*len(exo_better)/len(balanced):.0f}%)")
            print(f"✓ Statistically Significant: {len(sig)}/{len(balanced)} ({100*len(sig)/len(balanced):.0f}%)")
            print(f"  Mean Lift: {balanced['F1_Lift'].mean()*100:+.1f}%")
            
            print("\nTop 5 (Balanced predictions only):")
            top = balanced.nlargest(5, 'F1_Lift')[['Ticker', 'Endo_F1', 'Endo_Recall', 'Exo_F1', 'Exo_Recall', 'F1_Lift', 'Significant']]
            print(top.to_string(index=False))
    
    return df_results


if __name__ == '__main__':
    run_experiment()
