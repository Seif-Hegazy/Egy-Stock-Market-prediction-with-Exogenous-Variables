# EGX Macro Significance Study

**Evaluating the impact of macroeconomic variables on EGX 100 stock prediction**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment
python main.py
```

## Project Structure

```
├── main.py                    # Main experiment script (v9 FINAL)
├── src/                       # Core source code
│   ├── data_loader.py         # Feature engineering & data prep
│   ├── models.py              # CatBoost training
│   └── validation.py          # Statistical testing (DM test)
├── data/raw/                  # Raw data files
│   ├── stocks/                # EGX stock data (274k rows)
│   └── economic/              # Macro data (CBE, global markets)
├── results/                   # Experiment results
│   └── experiment_results_v9.csv  # Final results
├── models/                    # Saved trained models (152 files)
├── documentation/             # Methodology documentation
│   └── METHODOLOGY.md         # Complete methodology & results
├── scripts/                   # Utility scripts
├── archive/                   # Old versions & deprecated files
└── requirements.txt           # Python dependencies
```

## Key Results

| Metric | Value |
|--------|-------|
| Analyzed Tickers | 76 |
| **Significant Winners** | **18 (24%)** |
| Mean F1 Lift | -0.1% |
| Mean AUC Lift | -0.29% |

## Methodology (v9)

1. **Features:** Log returns (stationary) from OHLCV + macro changes
2. **Split:** 72/8/20 with purge gap (no leakage)
3. **Model:** CatBoost with early stopping
4. **Testing:** Diebold-Mariano with corrected direction

## Documentation

See `documentation/METHODOLOGY.md` for complete methodology, feature engineering details, and results analysis.

## Models

Trained models saved in `models/` directory:
- `{TICKER}_endo.joblib` - Endogenous model (OHLCV only)
- `{TICKER}_exo.joblib` - Exogenous model (OHLCV + Macro)

Each model file includes: model, threshold, normalization params, metrics.
