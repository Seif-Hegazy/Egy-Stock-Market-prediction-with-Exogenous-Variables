"""
EGX Macro Significance Study - Validation Module
Implements Purged Walk-Forward Validation and Diebold-Mariano Test.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, roc_auc_score
)
from typing import List, Tuple, Dict, Generator


# =============================================================================
# Purged Walk-Forward Validation
# =============================================================================

class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation.
    
    Implements time-series splits with purge and embargo gaps
    to prevent information leakage.
    
    Parameters:
        window_size: Training window in trading days
        test_size: Test window in trading days (default: 63 = ~3 months)
        purge: Gap after train to remove label leakage (default: 2)
        embargo: Additional gap to remove feature leakage (default: 2)
    """
    
    def __init__(self, window_size: int, test_size: int = 63, 
                 purge: int = 2, embargo: int = 2):
        self.window_size = window_size
        self.test_size = test_size
        self.purge = purge
        self.embargo = embargo
        self.total_gap = purge + embargo
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.
        
        Args:
            X: Feature DataFrame (must be sorted by time)
            y: Target (unused, for sklearn compatibility)
            groups: Group labels (unused)
            
        Yields:
            (train_indices, test_indices) for each fold
        """
        n = len(X)
        indices = np.arange(n)
        
        # Starting point: need at least window_size + gap + test_size
        min_start = self.window_size + self.total_gap
        
        current_test_start = min_start
        
        while current_test_start + self.test_size <= n:
            # Train: ends at current_test_start - total_gap
            train_end = current_test_start - self.total_gap
            train_start = max(0, train_end - self.window_size)
            
            # Only yield if we have enough training data
            if train_end - train_start >= self.window_size * 0.8:
                train_idx = indices[train_start:train_end]
                test_idx = indices[current_test_start:current_test_start + self.test_size]
                
                yield train_idx, test_idx
            
            # Slide forward by test_size (non-overlapping test sets)
            current_test_start += self.test_size
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, 
                     groups: pd.Series = None) -> int:
        """Get the number of folds."""
        if X is None:
            return 0
        return sum(1 for _ in self.split(X, y, groups))
    
    def __repr__(self):
        return (f"PurgedWalkForwardCV(window={self.window_size}, test={self.test_size}, "
                f"purge={self.purge}, embargo={self.embargo})")


def create_cv_for_window(window_size: int) -> PurgedWalkForwardCV:
    """
    Factory function to create CV splitter for a given window size.
    
    Args:
        window_size: One of 126 (Reactionary), 378 (Tactical), 756 (Strategic)
        
    Returns:
        Configured PurgedWalkForwardCV instance
    """
    return PurgedWalkForwardCV(
        window_size=window_size,
        test_size=63,  # ~3 months
        purge=2,
        embargo=2
    )


# =============================================================================
# Minimum Data Requirements
# =============================================================================

MIN_HISTORY = {
    126: 126 + 63 + 4,   # 193 trading days
    378: 378 + 63 + 4,   # 445 trading days  
    756: 756 + 63 + 4,   # 823 trading days (~3.3 years)
}


def check_sufficient_history(n_samples: int, window_size: int) -> bool:
    """
    Check if a ticker has sufficient history for the given window.
    
    Args:
        n_samples: Number of trading days for the ticker
        window_size: Training window size
        
    Returns:
        True if sufficient, False otherwise
    """
    min_required = MIN_HISTORY.get(window_size, window_size + 67)
    return n_samples >= min_required


# =============================================================================
# Diebold-Mariano Test
# =============================================================================

def compute_squared_loss(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """
    Compute squared error loss for probability predictions.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        
    Returns:
        Array of squared errors
    """
    return (y_true - y_proba) ** 2


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray, 
                          h: int = 1) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests whether Model B's predictions are significantly better than Model A's.
    
    H0: E[L_A] = E[L_B] (equal expected loss)
    H1: E[L_A] ≠ E[L_B] (different expected loss)
    
    Args:
        loss_a: Array of losses from Model A (baseline)
        loss_b: Array of losses from Model B (enhanced)
        h: Forecast horizon (1 for one-step ahead)
    
    Returns:
        (dm_statistic, p_value)
        Positive DM stat means Model B is better (lower loss)
    """
    # Loss differential (positive = A has higher loss = B is better)
    d = loss_a - loss_b
    T = len(d)
    
    if T < 10:
        # Insufficient samples for reliable test
        return np.nan, np.nan
    
    # Mean loss differential
    d_bar = np.mean(d)
    
    # Variance estimation
    # For h=1, use simple variance (no autocorrelation adjustment needed)
    if h == 1:
        gamma_0 = np.var(d, ddof=1)
        var_d_bar = gamma_0 / T
    else:
        # Newey-West HAC variance estimator for h > 1
        gamma_0 = np.var(d, ddof=1)
        var_d_bar = gamma_0
        
        for k in range(1, h):
            if k < T:
                gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0
                var_d_bar += 2 * (1 - k/h) * gamma_k
        
        var_d_bar /= T
    
    # Handle edge case of zero variance
    if var_d_bar <= 0:
        return np.nan, np.nan
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d_bar)
    
    # Two-sided p-value using t-distribution (more conservative for small samples)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1))
    
    return dm_stat, p_value


def is_significant(p_value: float, alpha: float = 0.05) -> bool:
    """
    Check if result is statistically significant.
    
    Args:
        p_value: P-value from DM test
        alpha: Significance level (default 0.05)
        
    Returns:
        True if significant (p < alpha)
    """
    if np.isnan(p_value):
        return False
    return p_value < alpha


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan
    
    return metrics


def compute_lift(baseline_metric: float, enhanced_metric: float) -> float:
    """
    Compute percentage lift from baseline to enhanced.
    
    Args:
        baseline_metric: Baseline model metric
        enhanced_metric: Enhanced model metric
        
    Returns:
        Percentage improvement
    """
    if baseline_metric == 0:
        return np.nan
    return ((enhanced_metric - baseline_metric) / baseline_metric) * 100


if __name__ == '__main__':
    # Test DM test with synthetic data
    np.random.seed(42)
    
    # Simulate: Model B has lower loss
    loss_a = np.random.exponential(0.3, 100)  # Baseline
    loss_b = np.random.exponential(0.25, 100)  # Enhanced (lower loss)
    
    dm_stat, p_value = diebold_mariano_test(loss_a, loss_b)
    
    print("Diebold-Mariano Test:")
    print(f"  DM Statistic: {dm_stat:.4f}")
    print(f"  P-Value: {p_value:.4f}")
    print(f"  Significant at α=0.05: {is_significant(p_value)}")
    
    # Test CV splitter
    print("\nCV Splitter Test:")
    cv = create_cv_for_window(126)
    print(f"  {cv}")
