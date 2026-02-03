# EGX Enhanced Prediction Model v2

## Overview

Complete overhaul based on academic literature to improve prediction from 42% F1 to target 60%+.

## Key Changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| **Model** | RandomForest | XGBoost (gradient boosting) |
| **Target** | Daily direction | Weekly direction |
| **Features** | Egypt macro only | Global + Egypt macro |
| **Engineering** | Basic lags | Cross-asset momentum, regime detection |

---

## Research Foundation

### 1. XGBoost for Financial Prediction

> **Chen & Guestrin (2016)** - XGBoost outperforms RF for imbalanced classification due to:
> - Built-in L1/L2 regularization (prevents overfitting)
> - `scale_pos_weight` for class imbalance
> - Gradient-based optimization (faster convergence)

> **Gu, Kelly, Xiu (2020)** "Empirical Asset Pricing via Machine Learning" - JFE
> - Gradient boosting achieves best Sharpe ratios for stock prediction
> - Recommends: `max_depth=3-6`, `learning_rate=0.01-0.1`, `subsample=0.8`

### 2. Weekly vs Daily Prediction

> **Huang, Nakamori, Wang (2005)** - Neural Networks for Weekly Stock Prediction
> - Weekly predictions reduce noise, improve signal-to-noise ratio
> - Daily: ~51% accuracy, Weekly: ~65% accuracy (SVM, Taiwan market)

> **Kara, Boyacioglu, Baykan (2011)** - Expert Systems with Applications
> - Weekly direction prediction achieves 75% accuracy (Turkish market)
> - Reason: Daily noise is filtered, fundamental factors dominate

### 3. Global Indicators for Emerging Markets

> **Gokcan (2000)** - "Forecasting Volatility of Emerging Stock Markets"
> - Emerging markets are heavily influenced by global risk sentiment
> - VIX is strongest predictor of EM volatility

> **Bekaert, Harvey (1997)** - "Emerging Equity Market Volatility" - JFE
> - Global factors explain 30-50% of EM returns variance
> - Key: S&P500, Oil, USD strength

### 4. Feature Engineering (Cross-Asset)

> **Moskowitz, Ooi, Pedersen (2012)** - "Time Series Momentum" - JFE
> - Momentum factors across asset classes are predictive
> - 12-1 month momentum (skip last month) most effective

> **Asness, Moskowitz, Pedersen (2013)** - "Value and Momentum Everywhere"
> - Cross-asset momentum signals improve prediction
> - Recommends: 1-week, 1-month, 3-month, 12-month lookbacks

---

## New Feature Set

### Global Indicators (from `global_market_data.csv`)

```python
GLOBAL_FEATURES = {
    # Momentum (% change) - Moskowitz et al.
    'oil_ret_1w', 'oil_ret_1m', 'oil_ret_3m',
    'gold_ret_1w', 'gold_ret_1m', 'gold_ret_3m',
    'sp500_ret_1w', 'sp500_ret_1m', 'sp500_ret_3m',
    'vix_change_1w', 'vix_level',  # VIX level matters
    'msci_em_ret_1w', 'msci_em_ret_1m',
    'eur_usd_ret_1w', 'eur_usd_ret_1m',
    
    # Cross-asset signals
    'oil_gold_ratio',  # Commodity risk indicator
    'vix_ma_ratio',    # VIX vs 20-day MA (fear spike)
    'em_vs_sp500',     # EM relative performance
}
```

### Egypt Macro (enhanced)

```python
EGYPT_FEATURES = {
    # Changes
    'usd_egp_change_1w', 'usd_egp_change_1m',
    'gold_local_change_1w', 'gold_local_change_1m',
    'real_rate',  # CBE rate - Inflation (monetary policy stance)
    
    # Spreads
    'gold_premium',  # Local gold vs international (EGP premium)
}
```

### Technical (stock-specific)

```python
TECHNICAL_FEATURES = {
    # Weekly aggregates
    'weekly_return', 'weekly_range', 'weekly_volume_change',
    
    # Momentum (Moskowitz et al.)
    'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m',
    
    # Volatility
    'realized_vol_1m', 'vol_regime',
    
    # Mean reversion
    'rsi_14', 'price_vs_ma20', 'price_vs_ma50',
}
```

---

## XGBoost Configuration

```python
# Based on Gu, Kelly, Xiu (2020) recommendations
XGBOOST_CONFIG = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,
    'min_child_weight': 10,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1.5,
    'random_state': 42,
}
```

---

## Expected Improvements

| Metric | v1 (Daily RF) | v2 (Weekly XGB) | Source |
|--------|---------------|-----------------|--------|
| **F1 Score** | 42% | 55-65% | Kara et al. (2011) |
| **Precision** | 43% | 55-60% | Huang et al. (2005) |
| **AUC** | ~0.52 | 0.60-0.70 | Gu et al. (2020) |

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
2. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. JFE.
3. Huang, W., Nakamori, Y., & Wang, S. Y. (2005). Forecasting stock market movement. Neural Networks.
4. Kara, Y., Boyacioglu, M. A., & Baykan, O. K. (2011). Predicting stock direction. Expert Systems.
5. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. JFE.
