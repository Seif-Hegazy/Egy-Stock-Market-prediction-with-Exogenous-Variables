"""
EGX Macro Significance Study - Source Package
"""

from .data_loader import load_and_prepare_data
from .feature_eng import engineer_all_features, ENDOGENOUS_FEATURES, EXOGENOUS_FEATURES
from .models import create_model, train_model, predict_proba
from .validation import (
    PurgedWalkForwardCV, 
    create_cv_for_window,
    diebold_mariano_test,
    compute_metrics,
    compute_lift,
    check_sufficient_history
)

__all__ = [
    'load_and_prepare_data',
    'engineer_all_features',
    'ENDOGENOUS_FEATURES',
    'EXOGENOUS_FEATURES',
    'create_model',
    'train_model',
    'predict_proba',
    'PurgedWalkForwardCV',
    'create_cv_for_window',
    'diebold_mariano_test',
    'compute_metrics',
    'compute_lift',
    'check_sufficient_history',
]
