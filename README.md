# EgySentiment - Egyptian Stock Market Prediction

A comprehensive ML pipeline for Egyptian stock market prediction using sentiment analysis, economic indicators, and historical price data.

## 🏗️ Architecture

```
Grad Project/
├── airflow/              # Airflow DAGs for automated data collection
│   └── dags/
│       ├── egx_stock_ingestion.py      # Stock data from yfinance
│       ├── egypt_economic_dag.py       # Economic indicators (CBE, gold, FX)
│       └── sentiment_collection.py     # News sentiment analysis
├── data/
│   ├── raw/
│   │   ├── stocks/       # EGX daily prices (35 tickers, 12+ years)
│   │   ├── economic/     # USD/EGP, gold, inflation, interest rates
│   │   └── news/         # Articles with sentiment scores
│   ├── processed/        # Engineered features
│   └── model_ready/      # Training-ready datasets
├── services/
│   └── sentiment-analysis/   # Groq-powered sentiment pipeline
├── models/               # Trained prediction models
└── notebooks/            # Analysis and experimentation
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
