"""
EGX Macro Significance Study - Source Package
"""

from .data_loader import (
    load_raw_data,
    create_endogenous_samples,
    create_exogenous_samples,
    construct_samples,
    prepare_datasets,
    add_technical_features,
    TECHNICAL_FEATURES,
    MACRO_FEATURES,
    WINDOW_SIZE,
    NEUTRAL_MARGIN,
)

from .models import (
    train_model,
    train_catboost,
    train_hgb,
    train_random_forest,
    evaluate_model,
    get_percentile_threshold,
)

from .validation import (
    diebold_mariano_test,
    compute_squared_loss,
    is_significant,
)

__all__ = [
    # Data Loading & Feature Engineering
    'load_raw_data',
    'create_endogenous_samples',
    'create_exogenous_samples',
    'construct_samples',
    'prepare_datasets',
    'add_technical_features',
    'TECHNICAL_FEATURES',
    'MACRO_FEATURES',
    'WINDOW_SIZE',
    'NEUTRAL_MARGIN',
    
    # Models
    'train_model',
    'train_catboost',
    'train_hgb',
    'train_random_forest',
    'evaluate_model',
    'get_percentile_threshold',
    
    # Validation
    'diebold_mariano_test',
    'compute_squared_loss',
    'is_significant',
]
