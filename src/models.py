"""
EGX Macro Significance Study - Models Module
Implements RandomForest classifier wrapper.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_CONFIG = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 20,
    'min_samples_split': 50,
    'class_weight': 'balanced_subsample',
    'n_jobs': -1,
    'random_state': 42,
    'oob_score': True,
}


# =============================================================================
# Model Factory
# =============================================================================

def create_model(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Create a RandomForest classifier pipeline with scaling.
    
    Args:
        config: Override default MODEL_CONFIG
        
    Returns:
        sklearn Pipeline with scaler and classifier
    """
    cfg = MODEL_CONFIG.copy()
    if config:
        cfg.update(config)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(**cfg))
    ])
    
    return pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                config: Optional[Dict[str, Any]] = None) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a model and return training diagnostics.
    
    Args:
        X_train: Feature matrix
        y_train: Target vector
        config: Model configuration override
        
    Returns:
        (trained_pipeline, diagnostics_dict)
    """
    pipeline = create_model(config)
    
    # Handle NaN values
    X_train_clean = X_train.fillna(0)
    
    pipeline.fit(X_train_clean, y_train)
    
    # Training diagnostics
    clf = pipeline.named_steps['clf']
    diagnostics = {
        'n_samples': len(y_train),
        'n_features': X_train.shape[1],
        'class_distribution': dict(pd.Series(y_train).value_counts()),
        'oob_score': clf.oob_score_ if hasattr(clf, 'oob_score_') else None,
    }
    
    return pipeline, diagnostics


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Get probability predictions.
    
    Args:
        model: Trained pipeline
        X: Feature matrix
        
    Returns:
        Probability of class 1 (UP)
    """
    X_clean = X.fillna(0)
    return model.predict_proba(X_clean)[:, 1]


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Get class predictions.
    
    Args:
        model: Trained pipeline
        X: Feature matrix
        
    Returns:
        Predicted classes (0 or 1)
    """
    X_clean = X.fillna(0)
    return model.predict(X_clean)


def get_feature_importance(model: Pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from trained model.
    
    Args:
        model: Trained pipeline
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances sorted descending
    """
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df_imp


if __name__ == '__main__':
    # Quick test
    print("Model configuration:")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k}: {v}")
