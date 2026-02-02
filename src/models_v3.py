"""
EGX Prediction Model v3 - Models
Implements CatBoost classifier and F1 threshold optimization.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from typing import Dict, Any, Tuple, Optional

# =============================================================================
# Model Configuration
# =============================================================================

CATBOOST_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    
    # Tree structure
    'depth': 6,
    'l2_leaf_reg': 3.0,
    
    # Learning
    'learning_rate': 0.03,
    'iterations': 1000,
    
    # Early stopping
    'od_type': 'Iter',
    'od_wait': 50,
    
    # Class imbalance
    'auto_class_weights': 'Balanced',
    
    'random_seed': 42,
    'verbose': False,
    'allow_writing_files': False,
}


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> CatBoostClassifier:
    """
    Train CatBoost model with early stopping on validation set.
    """
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model


def get_percentile_threshold(model: CatBoostClassifier, X_val: pd.DataFrame, quantile: float = 0.40) -> float:
    """
    Set threshold at the q-th percentile of validation probabilities.
    Strategy: Fixed Percentile Threshold (Q0.40)
    Ensures ~60% of predictions are classified as UP.
    """
    probs = model.predict_proba(X_val)[:, 1]
    threshold = np.quantile(probs, quantile)
    return threshold


def evaluate_model(model: CatBoostClassifier, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model on test set using optimized threshold.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'f1': f1_score(y_test, preds, zero_division=0),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'auc': roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5,
        'accuracy': (preds == y_test).mean(),
        'threshold': threshold
    }
    
    return metrics
