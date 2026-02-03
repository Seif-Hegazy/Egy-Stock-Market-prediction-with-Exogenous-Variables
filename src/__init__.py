"""
EGX Macro Significance Study - Source Package
"""

from .data_loader import (
    load_raw_data,
    create_endogenous_samples,
    create_exogenous_samples,
    prepare_datasets,
    compute_fixed_percentile_threshold,
    MIN_SAMPLES,
    THRESHOLD_PERCENTILE,
)

from .models import (
    train_model,
    train_catboost,
    evaluate_model,
    get_percentile_threshold,
)

from .validation import (
    diebold_mariano_test,
    compute_squared_loss,
    is_significant,
)

__all__ = [
    'load_raw_data',
    'create_endogenous_samples',
    'create_exogenous_samples',
    'prepare_datasets',
    'MIN_SAMPLES',
    'train_model',
    'train_catboost',
    'evaluate_model',
    'get_percentile_threshold',
    'diebold_mariano_test',
    'compute_squared_loss',
    'is_significant',
]
