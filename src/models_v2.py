"""
EGX Prediction Model v2 - CatBoost Model
CatBoost is a gradient boosting library similar to XGBoost.
Based on Gu, Kelly, Xiu (2020) recommendations for financial ML.

Note: CatBoost doesn't require OpenMP, making it easier to install on macOS.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# Model Configuration
# Based on Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning"
# CatBoost equivalent of XGBoost settings
# =============================================================================

CATBOOST_CONFIG = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    
    # Tree structure - prevent overfitting
    'depth': 4,
    'min_data_in_leaf': 10,
    
    # Regularization
    'l2_leaf_reg': 1.0,
    
    # Learning
    'learning_rate': 0.05,
    'iterations': 300,
    
    # Subsampling
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    
    # Class imbalance
    'auto_class_weights': 'Balanced',
    
    'random_seed': 42,
    'thread_count': -1,
    'verbose': False,
}


# =============================================================================
# Model Factory
# =============================================================================

def create_model(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Create CatBoost classifier pipeline with scaling."""
    cfg = CATBOOST_CONFIG.copy()
    if config:
        cfg.update(config)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(**cfg))
    ])
    
    return pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                config: Optional[Dict[str, Any]] = None) -> Tuple[Pipeline, Dict[str, float]]:
    """Train CatBoost model."""
    cfg = CATBOOST_CONFIG.copy()
    if config:
        cfg.update(config)
    
    pipeline = create_model(cfg)
    
    # Handle NaN values
    X_train_clean = X_train.fillna(0)
    
    pipeline.fit(X_train_clean, y_train)
    
    # Diagnostics
    clf = pipeline.named_steps['clf']
    diagnostics = {
        'n_samples': len(y_train),
        'n_features': X_train.shape[1],
        'class_distribution': dict(pd.Series(y_train).value_counts()),
        'iterations': clf.get_param('iterations'),
    }
    
    return pipeline, diagnostics


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions."""
    X_clean = X.fillna(0)
    return model.predict_proba(X_clean)[:, 1]


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Get class predictions."""
    X_clean = X.fillna(0)
    return model.predict(X_clean)


def get_feature_importance(model: Pipeline, feature_names: list) -> pd.DataFrame:
    """Extract feature importances."""
    clf = model.named_steps['clf']
    importances = clf.get_feature_importance()
    
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df_imp


if __name__ == '__main__':
    print("CatBoost Configuration (XGBoost-equivalent):")
    for k, v in CATBOOST_CONFIG.items():
        print(f"  {k}: {v}")
