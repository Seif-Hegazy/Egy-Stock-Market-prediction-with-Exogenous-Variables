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


from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 4,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42
}

from sklearn.ensemble import HistGradientBoostingClassifier

HGB_PARAMS = {
    'learning_rate': 0.05,
    'max_iter': 1000,
    'max_depth': 8,
    'max_leaf_nodes': 31,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 50,
    'random_state': 42,
    'class_weight': 'balanced'
}

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> CatBoostClassifier:
    """Train CatBoost model with early stopping."""
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> RandomForestClassifier:
    """Train Random Forest model."""
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model

def train_hgb(X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> HistGradientBoostingClassifier:
    """
    Train Histogram-based Gradient Boosting (Sklearn's LightGBM).
    Handles interactions well, native NaN support, reliable on Mac.
    """
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    # HGB uses internal validation for early stopping if early_stopping=True
    # We pass the full train set here, or we could concat train+val
    # To keep it comparable, we'll fit on Train and let it split internally
    # or just fit on Train. simpler to just fit.
    model.fit(X_train, y_train)
    return model

# Alias for easy switching in main.py
# Change this to switch models
train_model = train_catboost  # Best performer
# train_model = train_random_forest
# train_model = train_hgb


def get_percentile_threshold(model: Any, X_val: pd.DataFrame, quantile: float = 0.40) -> float:
    """
    Set threshold at the q-th percentile of validation probabilities.
    Strategy: Fixed Percentile Threshold (Q0.40)
    Ensures ~60% of predictions are classified as UP.
    """
    probs = model.predict_proba(X_val)[:, 1]
    threshold = np.quantile(probs, quantile)
    return threshold


def evaluate_model(model: Any, 
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
