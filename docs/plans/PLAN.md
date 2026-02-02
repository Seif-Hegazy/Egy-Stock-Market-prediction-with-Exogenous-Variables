# PLAN.md - EGX Macro Significance Architecture

> **Objective**: Prove that macroeconomic variables add statistically significant predictive value to EGX30 stock direction using rigorous scientific methodology.

---

## 1. Data Ingestion Strategy

### 1.1 Data Sources

| Source | Frequency | Path |
|--------|-----------|------|
| Stock Prices | Daily (trading days) | `data/raw/stocks/egx_daily_12y.csv` |
| Macro Data | Daily (calendar days) | `data/raw/economic/egypt_economic_data.csv` |
| Sentiment | Sparse | `data/processed/daily_sentiment_features.csv` |

### 1.2 Merge Logic (No Look-Ahead Bias)

```
Stock Data (high-freq)  ──┐
                          ├──► Merge on Date ──► Forward-Fill Macro ──► Apply Lags
Macro Data (low-freq)   ──┘

CRITICAL: Forward-fill BEFORE applying lags to prevent future information leakage
```

**Implementation**:
```python
def merge_stock_macro(df_stock, df_macro):
    """
    Merge with temporal integrity.
    
    Steps:
    1. Left-join stock dates onto macro (stock dates = trading days only)
    2. Forward-fill NaN macro values (weekends/holidays use last known)
    3. Apply LAG_SCHEMA AFTER merge (ensures t-1 macro is truly from t-1)
    """
    # Ensure date columns are datetime
    df_merged = df_stock.merge(
        df_macro, 
        left_on='Date', 
        right_on='date', 
        how='left'
    )
    
    # Forward-fill macro columns (grouped by ticker to prevent cross-ticker leakage)
    macro_cols = ['usd_sell_rate', 'gold_24k', 'cbe_deposit_rate', 
                  'cbe_lending_rate', 'headline_inflation']
    
    for col in macro_cols:
        df_merged[col] = df_merged.groupby('Ticker')[col].ffill()
    
    return df_merged
```

### 1.3 LAG_SCHEMA Implementation

```python
LAG_SCHEMA = [1, 2, 3, 5]  # Trading days, not calendar days

def apply_lags(df, cols, lags=[1, 2, 3, 5]):
    """
    Apply lags PER TICKER to prevent cross-ticker information leakage.
    
    Example: usd_sell_rate_lag1 = USD/EGP from t-1 trading day
    """
    for col in cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('Ticker')[col].shift(lag)
    return df
```

---

## 2. Adaptive Volatility Flag (Rolling 3σ)

### 2.1 Rationale

Static 18% threshold (near circuit breaker) is too crude. Markets have regime shifts where "extreme" varies.

### 2.2 Definition

$$\text{is\_extreme\_volatility}_t = \begin{cases} 1 & \text{if } |r_t| > 3 \times \sigma_{20,t} \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $r_t = \frac{Close_t - Close_{t-1}}{Close_{t-1}}$ (daily return)
- $\sigma_{20,t}$ = 20-day rolling standard deviation of returns

### 2.3 Implementation

```python
def compute_adaptive_volatility_flag(df):
    """
    Compute per-ticker adaptive volatility flag.
    """
    df = df.sort_values(['Ticker', 'Date'])
    
    # Daily return
    df['return'] = df.groupby('Ticker')['Close'].pct_change()
    
    # 20-day rolling volatility (per ticker)
    df['rolling_std_20'] = df.groupby('Ticker')['return'].transform(
        lambda x: x.rolling(window=20, min_periods=10).std()
    )
    
    # Adaptive extreme flag: |return| > 3 * rolling_std
    df['is_extreme_volatility'] = (
        df['return'].abs() > 3 * df['rolling_std_20']
    ).astype(int)
    
    return df
```

---

## 3. Purged Walk-Forward Validation (CPCV)

### 3.1 The Leakage Problem

Standard TimeSeriesSplit can leak information:
- **Problem 1**: Train data ends at $t$, test starts at $t+1$ → overlapping labels
- **Problem 2**: Lagged features in test may use train-period calculations

### 3.2 Purged Validation Design

```
Timeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│   TRAIN SET      │ PURGE │ EMBARGO │        TEST SET          │
│   (Window Size)  │ (2d)  │  (2d)   │    (Fixed: 63 days)      │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   ↑
            No data used from this gap
```

**Parameters**:
- `PURGE_GAP = 2` trading days (remove label leakage)
- `EMBARGO_GAP = 2` trading days (remove feature leakage from lags)
- `TOTAL_GAP = 4` trading days between train end and test start

### 3.3 Sliding Window Implementation

```python
def purged_walk_forward_split(df, window_size, test_size=63, purge=2, embargo=2):
    """
    Generate train/test indices with purging.
    
    Args:
        df: DataFrame sorted by Date (per ticker)
        window_size: Training window in trading days (126, 378, or 756)
        test_size: Test window (63 days = ~3 months)
        purge: Gap after train to remove label leakage
        embargo: Additional gap to remove feature leakage
        
    Yields:
        (train_indices, test_indices) for each fold
    """
    total_gap = purge + embargo
    n = len(df)
    
    # Starting point: need at least window_size + gap + test_size
    start_test = window_size + total_gap
    
    folds = []
    current_test_start = start_test
    
    while current_test_start + test_size <= n:
        # Train: [current_test_start - total_gap - window_size, current_test_start - total_gap)
        train_end = current_test_start - total_gap
        train_start = train_end - window_size
        
        if train_start < 0:
            break
            
        train_idx = list(range(train_start, train_end))
        test_idx = list(range(current_test_start, current_test_start + test_size))
        
        folds.append((train_idx, test_idx))
        
        # Slide forward by test_size (non-overlapping test sets)
        current_test_start += test_size
    
    return folds
```

---

## 4. Tournament Logic (Grid Search)

### 4.1 Window Size Grid

| Name | Days | Market Interpretation |
|------|------|----------------------|
| **Reactionary** | 126 | ~6 months, captures recent regime |
| **Tactical** | 378 | ~18 months, medium-term patterns |
| **Strategic** | 756 | ~3 years, full market cycles |

### 4.2 Tournament Flow

```
FOR each ticker in EGX35:
    IF len(ticker_data) < 756 + 63 + 4:
        SKIP (insufficient history for Strategic)
    
    FOR each window_size in [126, 378, 756]:
        folds = purged_walk_forward_split(ticker_data, window_size)
        
        FOR each (train_idx, test_idx) in folds:
            # Model A: Endogenous only
            model_a = train_baseline(X_train[ENDO_FEATURES], y_train)
            pred_a = model_a.predict_proba(X_test[ENDO_FEATURES])
            
            # Model B: Endogenous + Exogenous
            model_b = train_enhanced(X_train[ALL_FEATURES], y_train)
            pred_b = model_b.predict_proba(X_test[ALL_FEATURES])
            
            # Store predictions for DM test
            store_results(ticker, window_size, y_test, pred_a, pred_b)
        
        # Aggregate metrics across folds
        compute_f1_lift(ticker, window_size)
    
    # Select best window for this ticker
    best_window = argmax(f1_lift over windows)
    store_tournament_winner(ticker, best_window)
```

### 4.3 Minimum Data Requirements

```python
MIN_HISTORY = {
    126: 126 + 63 + 4,   # 193 trading days
    378: 378 + 63 + 4,   # 445 trading days  
    756: 756 + 63 + 4,   # 823 trading days (~3.3 years)
}
```

---

## 5. Diebold-Mariano Test

### 5.1 Purpose

Test whether Model B's predictions are **statistically significantly better** than Model A's, not just numerically better.

### 5.2 Hypothesis

- $H_0$: $E[L_A] = E[L_B]$ (models have equal expected loss)
- $H_1$: $E[L_A] \neq E[L_B]$ (models differ significantly)

Where $L$ = loss function (we use squared error on predicted probabilities)

### 5.3 DM Statistic

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}}$$

Where:
- $d_t = L_{A,t} - L_{B,t}$ (loss differential at time $t$)
- $\bar{d} = \frac{1}{T}\sum_{t=1}^{T} d_t$ (mean loss differential)
- $\hat{V}(\bar{d})$ = HAC-consistent variance estimator (accounts for autocorrelation)

### 5.4 Implementation

```python
import numpy as np
from scipy import stats

def diebold_mariano_test(loss_a, loss_b, h=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Args:
        loss_a: Array of losses from Model A
        loss_b: Array of losses from Model B
        h: Forecast horizon (1 for one-step ahead)
    
    Returns:
        dm_stat: DM test statistic
        p_value: Two-sided p-value
    """
    d = loss_a - loss_b  # Loss differential
    T = len(d)
    
    # Mean loss differential
    d_bar = np.mean(d)
    
    # Newey-West HAC variance estimator
    # Bandwidth = h-1 for h-step ahead forecasts
    gamma_0 = np.var(d, ddof=1)
    
    # For h=1, no autocorrelation adjustment needed
    if h == 1:
        var_d_bar = gamma_0 / T
    else:
        # Add autocovariance terms
        var_d_bar = gamma_0
        for k in range(1, h):
            gamma_k = np.cov(d[:-k], d[k:])[0, 1]
            var_d_bar += 2 * (1 - k/h) * gamma_k
        var_d_bar /= T
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d_bar)
    
    # Two-sided p-value (t-distribution with T-1 df for small samples)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T-1))
    
    return dm_stat, p_value


def compute_squared_loss(y_true, y_proba):
    """
    Squared error loss for probability predictions.
    """
    return (y_true - y_proba) ** 2
```

### 5.5 Significance Threshold

| p-value | Interpretation | Action |
|---------|----------------|--------|
| < 0.01 | Highly significant | ✅ Mark as significant |
| 0.01 - 0.05 | Significant | ✅ Mark as significant |
| 0.05 - 0.10 | Marginally significant | ⚠️ Flag for review |
| > 0.10 | Not significant | ❌ Alpha not proven |

---

## 6. Feature Engineering Summary

### 6.1 Endogenous Features (Model A Baseline)

```python
ENDOGENOUS_FEATURES = [
    # 10-day flattened lookback (raw OHLCV normalized)
    'open_norm_lag1', 'open_norm_lag2', ..., 'open_norm_lag10',
    'high_norm_lag1', ..., 'high_norm_lag10',
    'low_norm_lag1', ..., 'low_norm_lag10',
    'close_norm_lag1', ..., 'close_norm_lag10',
    'volume_norm_lag1', ..., 'volume_norm_lag10',
    
    # Technical indicators
    'return_1d', 'return_5d', 'return_10d',
    'volatility_5d', 'volatility_10d',
    'rsi_14',
    'volume_ma_ratio_5',
]
```

### 6.2 Exogenous Features (Model B Enhancement)

```python
EXOGENOUS_FEATURES = [
    # USD/EGP (currency risk)
    'usd_sell_rate_lag1', 'usd_sell_rate_lag2', 'usd_sell_rate_lag3', 'usd_sell_rate_lag5',
    'usd_change_1d', 'usd_change_5d',
    
    # Gold (safe haven)
    'gold_24k_lag1', 'gold_24k_lag2', 'gold_24k_lag3', 'gold_24k_lag5',
    'gold_change_1d', 'gold_change_5d',
    
    # Interest rates (monetary policy)
    'cbe_deposit_rate_lag1',
    'cbe_lending_rate_lag1',
    'rate_spread_lag1',  # lending - deposit
    
    # Inflation
    'headline_inflation_lag1',
    
    # Calendar effects
    'holiday_gap',  # Days since last trading day (>3 = extended closure)
    'is_post_holiday',  # Binary: was previous gap > 3 days?
    
    # Volatility regime
    'is_extreme_volatility',  # Adaptive 3σ flag
]
```

---

## 7. Model Specification

### 7.1 Algorithm

```python
from sklearn.ensemble import RandomForestClassifier

MODEL_CONFIG = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 20,
    'class_weight': 'balanced_subsample',
    'n_jobs': -1,
    'random_state': 42,
}

def create_model():
    return RandomForestClassifier(**MODEL_CONFIG)
```

### 7.2 Why RandomForest?

- Handles non-linear interactions (macro × price dynamics)
- `balanced_subsample`: Each tree sees balanced classes
- Feature importance for interpretability
- Robust to outliers (important given volatility spikes)

---

## 8. Output Artifacts

### 8.1 `results/tournament_winner.csv`

```csv
Ticker,Sector,Best_Window,F1_Lift_%,Is_Significant
COMI.CA,Banking,378,12.4,True
TMGH.CA,Real Estate,756,8.7,True
FWRY.CA,Financial Services,126,3.2,False
...
```

### 8.2 `results/significance_report.csv`

```csv
Ticker,Window,F1_Baseline,F1_Enhanced,Lift_%,DM_Stat,P_Value,Significant_Alpha?
COMI.CA,126,0.482,0.523,8.5,2.34,0.021,Yes
COMI.CA,378,0.491,0.552,12.4,3.12,0.002,Yes
COMI.CA,756,0.488,0.541,10.9,2.87,0.005,Yes
...
```

### 8.3 Heatmap Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_sector_window_heatmap(results_df):
    """
    Heatmap: Sectors (rows) × Windows (cols), values = mean F1 Lift %
    """
    pivot = results_df.pivot_table(
        index='Sector',
        columns='Window',
        values='Lift_%',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
    plt.title('F1 Lift (%) from Exogenous Variables by Sector × Window')
    plt.savefig('results/sector_window_heatmap.png', dpi=150)
```

---

## 9. Directory Structure

```
Grad Project/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Merge logic, LAG_SCHEMA
│   ├── feature_eng.py      # Endogenous + Exogenous features
│   ├── models.py           # RandomForest wrapper
│   └── validation.py       # Purged CV + Diebold-Mariano
├── main.py                  # Orchestration
├── results/
│   ├── tournament_winner.csv
│   ├── significance_report.csv
│   └── sector_window_heatmap.png
└── PLAN.md                  # This file
```

---

## 10. Constraint Compliance Checklist

| Constraint | Implementation | Status |
|------------|----------------|--------|
| No random shuffle | `purged_walk_forward_split()` time-based only | ✅ |
| Forward-fill before lag | `merge_stock_macro()` → `apply_lags()` | ✅ |
| Purge gap (2 days) | Built into CV split | ✅ |
| Embargo gap (2 days) | Built into CV split | ✅ |
| p-value > 0.05 = not significant | Explicit check in report | ✅ |
| Skip insufficient history | Min 823 days for Strategic | ✅ |
| Adaptive volatility (3σ) | `compute_adaptive_volatility_flag()` | ✅ |

---

## 11. Execution Order

```
Phase 2 Implementation Sequence:
1. src/data_loader.py  ─── Merge + lags (testable independently)
2. src/feature_eng.py  ─── Feature pipeline (depends on 1)
3. src/models.py       ─── Model wrapper (standalone)
4. src/validation.py   ─── CV + DM test (standalone)
5. main.py             ─── Orchestration (depends on 1-4)
6. Run & generate results
```

---

**AWAITING APPROVAL TO PROCEED TO PHASE 2**
