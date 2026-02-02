# EGX Prediction Model v3 - Research Exact

## Overview

Implementation of the rigorous rolling window framework specified by the user's research paper, adapted to include exogenous variables.

## Scientific Framework

### 1. Rolling Window Scheme
- **Window Size ($w$)**: 5 trading days (1 week)
- **Structure**:
  - **Week 0 ($t$ to $t+w-1$)**: Historical prices part 1
  - **Week 1 ($t+w$ to $t+2w-1$)**: Historical prices part 2
  - **Week 2 ($t+2w$ to $t+3w-1$)**: **Future Target**
- **Feature Vector $x_i$**: Concatenation of Week 0 and Week 1 prices (and exogenous vars)
- **Target Vector $y_i$**: Binary direction of Week 2 vs Week 0+1 mean

### 2. Labeling with Neutral Zone
- **Margin ($m$)**: 0.001 (0.1%)
- **Formula**:
  - $P_{past} = \text{mean}(p_{i:i+2w})$
  - $P_{future} = \text{mean}(p_{i+2w:i+3w})$
  - Ratio $R = P_{future} / P_{past}$
  - $y_i = 1$ if $R > 1 + m$
  - $y_i = 0$ if $R < 1 - m$
  - **Discard** if $1 - m \le R \le 1 + m$

### 3. Data Splitting (Strict Chronological)
- **Test Set ($r_{test}$)**: Last 20%
- **Validation Set ($r_{val}$)**: Last 10% of the remaining 80% (approx 8% of total)
- **Training Set**: First ~72%
- **Z-Score Normalization**: Parameters ($\mu, \sigma$) computed on **Training Set ONLY** and applied to Val/Test.

---

## Methodology: Two-Model Comparison

We will train two models for every stock to isolate the impact of macro data:

### Model A: Endogenous Only (Baseline)
- Input: Daily adjusted close prices for Week 0 and Week 1 ($2w = 10$ features)
- Plus computed technical features derived strictly from these 10 days

### Model B: Endogenous + Exogenous (Enhanced)
- Input: Model A features
- **PLUS**: Daily values of global/macro indicators for Week 0 and Week 1 ($2w \times N_{macro}$ features)
- Exogenous vars: Oil, Gold, VIX, S&P500, USD/EGP, CBE Rate

---

## Implementation Strategy

### 1. `src/data_loader_v3.py`
- Strict window generator
- Neutral zone filtering
- Train/Val/Test splitter with Z-score scaling

### 2. `src/feature_eng_v3.py`
- Feature construction: $x_i = \text{concat}(p_{i:i+w}, p_{i+w:i+2w})$
- Feature alignment check (zero look-ahead)

### 3. `main_v3.py`
- Orchestrates the training loop
- Optimizes classification threshold for F1 score on Validation set
- Evaluates on Test set
- Calculates F1 Lift

### 4. `src/models_v3.py`
- CatBoost/XGBoost classifier
- Hyperparameters optimized for this specific data structure

---

## Expected Outcome

A definitive, scientifically rigorous answer to: "Does adding macro data improve weekly direction prediction accuracy beyond the random noise threshold?"
