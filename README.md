# EgySentiment - Egyptian Stock Market Prediction

A comprehensive ML pipeline for Egyptian stock market prediction using sentiment analysis, economic indicators, and historical price data.

## 🏗️ Architecture

```
Grad Project/
├── airflow/              # Airflow DAGs for automated data collection
│   └── dags/
├── data/
│   ├── raw/
│   │   ├── stocks/       # EGX daily prices (35 tickers)
│   │   ├── economic/     # USD/EGP, gold, inflation, interest rates
│   │   └── global/       # S&P500, VIX, Oil, etc.
├── docs/                 # Documentation and Research Plans
├── experiments/          # Archived experimental scripts (v1, v2)
├── main.py               # PRIMARY ENTRY POINT (v3.1 Research Framework)
├── models/               # Trained prediction models
├── results/              # Experiment results
│   ├── v1/               # Initial findings
│   └── v3/               # Final v3.1 Research Results (Robust Winners)
├── src/                  # Core Source Code
│   ├── data_loader_v3.py # Rolling window logic
│   └── models_v3.py      # CatBoost implementation
└── ...
```

## 🧠 Research Framework (v3.1)

The project implements a rigorous hypothesis test: **"Do global/local macroeconomic variables improve weekly stock direction prediction for EGX30 stocks?"**
- **Window**: 5-Day concatenated rolling window ($W_0 + W_1 \to Target_{W2}$).
- **Model**: CatBoost with Fixed 40th Percentile Threshold (Q0.40).
- **Control**: Technicals Only (Price + RSI + Volatility + Momentum).
- **Test**: Technicals + Macro (Gold, Oil, VIX, USD, Interest Rates).

**Key Findings:**
- **Construction (ORAS, SAUD)** and **Banking (EGBE, CICH)** show statistically significant improvement with macro data.
- **Recall** (Trend Capture) is the primary driver of alpha.

### 🚀 Running the Experiment
```bash
python3 main.py
```

## 📊 Data Coverage

| Dataset | Records | Date Range | Source |
|---------|---------|------------|--------|
| Stock Prices | 147,118 | 2000-2026 | yfinance (EGX) |
| Economic Data | 4,760 | 2013-2026 | CBE, various APIs |
| News Articles | 2,100+ | 2024-2026 | RSS feeds, scrapers |

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Groq API key (for sentiment analysis)

### Running with Docker

```bash
# Start all services
docker compose up -d

# Access Airflow UI
open http://localhost:8080  # admin/admin

# Access Streamlit app
open http://localhost:8501
```

### Automated Data Collection

The following DAGs run automatically:
- **Stock ingestion**: Daily at 14:30 UTC
- **Economic data**: Daily at 04:00 UTC
- **Sentiment collection**: Every 4 hours

## 📈 Features

- **35 EGX tickers** with verified ISIN codes
- **Real-time sentiment analysis** using Groq LLM
- **Economic indicators**: USD/EGP, gold prices, CBE interest rates, inflation
- **Automated pipelines** via Apache Airflow
- **Interactive dashboard** via Streamlit

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:
```
GROQ_API_KEY=your_api_key
AIRFLOW__CORE__FERNET_KEY=your_fernet_key
```

## 📝 License

MIT License
