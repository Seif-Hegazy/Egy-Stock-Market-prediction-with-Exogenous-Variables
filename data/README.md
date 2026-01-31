# Data Directory

## 📁 Structure

### `data/raw/stocks/`
- `egx_daily_12y.csv` - Raw EGX30 daily OHLCV data (12 years)

### `data/raw/economic/`
- `egypt_economic_data.csv` - Macro data (USD, Gold, Gas, Inflation, CBE rates)

### `data/metadata/economic/`
- `data_sources.json` - Data provenance metadata

### `data/processed/news/`
- `daily_sentiment_features.csv` - Aggregated sentiment per ticker/day
- `forecast_features.csv` - Raw news articles with sentiment scores
- `testing_data.jsonl` - Raw news data

### `model_ready/` - Training-Ready Datasets
- `train_ready_data.csv` - General model training data
- `train_ready_logreg.csv` - LogReg-specific with lag features
- `safe_tickers.txt` - Tickers with sufficient data for training
- `results_summary.csv` - Basic model results
- `hybrid_results.csv` - Hybrid model results
- `sharpe_results.csv` - Sharpe-optimized results

## 📊 Data Flow

```
stocks/raw/ ──┬──> data_prep scripts ──> model_ready/
economic/   ──┤
news/       ──┘
```
