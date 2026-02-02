# EGX Macro Significance Study

A research framework to test whether macroeconomic variables (Gold, Oil, VIX, USD/EGP) improve stock direction prediction for the Egyptian Stock Exchange (EGX30).

## 🏗️ Repository Structure

```
Grad Project/
├── main.py               # PRIMARY ENTRY POINT - Run the experiment
├── src/                  # Core Source Code
│   ├── data_loader.py    # 5-day rolling window construction
│   ├── models.py         # CatBoost/HGB/RF model implementations
│   └── validation.py     # Statistical testing (Diebold-Mariano)
├── data/
│   └── raw/              # Raw CSV data (stocks, economic, global)
├── results/              # Experiment outputs (CSVs, heatmaps)
├── archive/              # Old experiments and legacy code
├── airflow/              # Automated data collection DAGs
├── services/             # Dashboard and Sentiment API
└── docs/                 # Planning documents
```

## 🧠 Research Framework

**Hypothesis:** "Global/local macroeconomic variables improve weekly stock direction prediction for EGX30 stocks."

**Methodology:**
- **Window**: 5-Day rolling ($W_0 + W_1 \to Target_{W2}$)
- **Model**: CatBoost (Default), HGB, Random Forest
- **Threshold**: Fixed 40th Percentile (Q0.40)
- **Baseline**: Technicals Only (Price + RSI + Volatility + Momentum)
- **Test**: Technicals + Macro (Gold, Oil, VIX, USD, Interest Rates)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment
python3 main.py
```

## 📊 Key Findings

| Ticker | Sector | Lift | Insight |
|--------|--------|------|---------|
| **SAUD.CA** | Construction | +45.8% | Macro cycles drive this sector |
| **ETEL.CA** | Telecom | +10.4% | Import/USD sensitivity |
| **CICH.CA** | Financials | +3.1% | Consistent alpha |

**Conclusion:** Macro data significantly improves prediction for Construction, Telecom, and Financial sectors.

## 📝 License

MIT License
