# Feature Engineering Context Report
## EgySentiment - Directional Stock Forecasting Model

**Purpose:** Context brief for Gemini to suggest advanced features for EGX stock direction prediction.

---

# System Architecture

The system has 4 main components:

1. **Stock Data Ingestion** — Daily EGX100 prices via yfinance API (Airflow DAG)
2. **Economic Data Pipeline** — CBE rates, gold, USD, inflation, GDP (manual + automated)
3. **News Sentiment Pipeline** — RSS/web scraping → LLM classification → aggregation
4. **Feature Storage** — CSV files for stocks, economic data, and sentiment features

---

# Data Schema

## 1. Price Data (`egx_daily_12y.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `Date` | date | Trading day |
| `Ticker` | string | EGX symbol (e.g., `COMI.CA`) |
| `ISIN` | string | International ID |
| `Sector` | string | Industry sector |
| `Open` | float | Opening price (EGP) |
| `High` | float | Intraday high |
| `Low` | float | Intraday low |
| `Close` | float | Closing price (EGP) |
| `Volume` | int | Trading volume |

**Coverage:** 35 EGX100 tickers, 12+ years history

**Tickers:**
```
COMI.CA, CIEB.CA, HDBK.CA, FAIT.CA, ADIB.CA, SAUD.CA, EGBE.CA, EXPA.CA,
HRHO.CA, EFIH.CA, FWRY.CA, BTFH.CA, CICH.CA, TMGH.CA, PHDC.CA, OCDI.CA,
MASR.CA, HELI.CA, ORAS.CA, EMFD.CA, SWDY.CA, ABUK.CA, MFPC.CA, SKPC.CA,
AMOC.CA, ETEL.CA, EAST.CA, JUFO.CA, EFID.CA, ISPH.CA, CLHO.CA, GBCO.CA,
EKHO.CA, CCAP.CA, ORHD.CA
```

---

## 2. Economic/Exogenous Data (`egypt_economic_data.csv`)

| Column | Type | Source | Fill Method |
|--------|------|--------|-------------|
| `date` | date | - | - |
| `usd_sell_rate` | float | CBE | Daily (no imputation) |
| `gold_24k` | float | Egyptian market | Daily |
| `gasoline_80` | float | Government | Rarely changes |
| `gasoline_92` | float | Government | Rarely changes |
| `gasoline_95` | float | Government | Rarely changes |
| `cbe_deposit_rate` | float | CBE MPC | Forward-fill from announcements |
| `cbe_lending_rate` | float | CBE MPC | Forward-fill from announcements |
| `headline_inflation` | float | CAPMAS | Monthly, forward-filled |
| `gdp_usd_billion` | float | World Bank | Annual, forward-filled |

**Coverage:** 2013-01-02 to 2025-12-12 (4,728 rows)

---

## 3. Sentiment Features (`daily_sentiment_features.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Calendar date |
| `ticker` | string | Stock ticker (without `.CA`) |
| `sector` | string | Sector name |
| `direct_sentiment` | float | Avg sentiment of articles mentioning ticker (-1 to +1) |
| `sector_sentiment` | float | Avg sentiment of sector peer articles (-1 to +1) |
| `direct_count` | int | Articles directly mentioning ticker |
| `sector_count` | int | Sector-related articles |

---

## 4. Raw Sentiment Data (`forecast_features.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Article date |
| `text` | string | Full article text |
| `sentiment` | string | `positive`, `negative`, `neutral` |
| `sentiment_score` | int | `+1`, `0`, `-1` |
| `reasoning` | string | LLM explanation |

---

# Feature Engineering Logic

## Sentiment Score Conversion

```python
def get_sentiment_score(sentiment_str):
    s = sentiment_str.lower()
    if 'positive' in s: return 1
    if 'negative' in s: return -1
    return 0
```

## Sentiment Aggregation (Data Merge)

```python
def aggregate_sentiment_features(jsonl_file):
    """
    Generate stock-day features:
    - direct_sentiment: Avg sentiment of articles mentioning the ticker
    - sector_sentiment: Avg sentiment of articles mentioning sector peers
    """
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['score'] = df['sentiment'].apply(get_sentiment_score)
    df['tickers'] = df['text'].apply(identify_tickers)
    
    for date in dates:
        day_articles = df[df['date'] == date]
        
        for ticker in all_tickers:
            sector = TICKER_TO_SECTOR.get(ticker)
            
            # Direct Sentiment - articles mentioning THIS ticker
            direct_mask = day_articles['tickers'].apply(lambda x: ticker in x)
            direct_df = day_articles[direct_mask]
            direct_sentiment = direct_df['score'].mean() if not direct_df.empty else 0
            
            # Sector Sentiment - articles mentioning sector peers (excluding direct)
            sector_tickers = SECTOR_MAP.get(sector, [])
            other_tickers = [t for t in sector_tickers if t != ticker]
            
            # Check for other tickers or sector name in text
            sector_mask = day_articles.apply(
                lambda row: has_sector_signal(row['tickers'], row['text']) 
                            and (ticker not in row['tickers']), 
                axis=1
            )
            sector_df = day_articles[sector_mask]
            sector_sentiment = sector_df['score'].mean() if not sector_df.empty else 0
```

## Sector Mapping

```python
SECTOR_MAP = {
    "Banks":              ['COMI', 'ADIB', 'CIEB', 'QNBA', 'FAIT'],
    "Real Estate":        ['TMGH', 'EMFD', 'PHDC', 'MASR', 'ORHD'],
    "Basic Resources":    ['EGAL', 'ABUK', 'MFPC', 'SIDPEC', 'ESRS'],
    "Non-Bank Financials":['HRHO', 'BTFH', 'FWRY', 'EFIH', 'CIRA'],
    "Consumer":           ['EAST', 'JUFO', 'ORWE', 'AUTO'],
    "Telecom":            ['ETEL']
}
```

---

# Target Definition (NOT YET IMPLEMENTED)

The codebase does NOT currently define a "Direction" target variable. The infrastructure prepares features, but target generation is pending.

**Proposed Target:**
```python
df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# Direction = 1 if next day's Close > today's Close, else 0
```

---

# Current Features Summary

| Category | Features |
|----------|----------|
| **Price** | Open, High, Low, Close, Volume |
| **Sentiment - Direct** | direct_sentiment, direct_count |
| **Sentiment - Sector** | sector_sentiment, sector_count |
| **Macro** | usd_sell_rate, gold_24k, gasoline_80/92/95 |
| **Monetary** | cbe_deposit_rate, cbe_lending_rate |
| **Inflation** | headline_inflation |
| **GDP** | gdp_usd_billion |

---

# NOT Currently Calculated (Opportunities)

1. **Price Returns:** `daily_return = (Close - Close.shift(1)) / Close.shift(1)`
2. **Moving Averages:** SMA(5), SMA(20), SMA(50), EMA variants
3. **Volatility:** Bollinger Bands, ATR, historical volatility
4. **Momentum:** RSI, MACD, ROC
5. **Volume Features:** Volume MA, Volume ratio, OBV
6. **Lagged Features:** Sentiment lags (t-1, t-2, ...), Price lags
7. **Sector Aggregates:** Sector-level price performance
8. **Economic Deltas:** Changes in USD rate, gold price changes
9. **Cross-asset correlations:** Gold vs EGP, sector rotation signals
10. **Calendar features:** Day of week, month, holiday proximity

---

# Task for Gemini

**Suggest advanced features for a Directional Stock Forecasting Model given:**
1. The data sources above (prices, economic indicators, LLM sentiment)
2. The EGX market context (Egyptian Stock Exchange, emerging market)
3. The goal of predicting next-day direction (binary: up/down)

Consider features that leverage:
- Multi-timeframe analysis
- Cross-stock/sector relationships
- Macro-sentiment interactions
- Egyptian market-specific factors (CBE policy, USD shortage, inflation regime)
