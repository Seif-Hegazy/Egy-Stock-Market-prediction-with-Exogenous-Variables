# EgySentiment EGX Ingestion DAG - Docker Compose Setup Guide

## 1. Volume Mounting Configuration

Add the following volume mounts to your `docker-compose.yml` Airflow service:

```yaml
services:
  airflow-webserver:
    # ... existing config ...
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      # ADD THIS LINE for persistent stock data storage:
      - ./data/stocks:/opt/airflow/data/stocks
      # Optional: If you want to share data with other services
      - ./data:/opt/airflow/data
  
  airflow-scheduler:
    # ... existing config ...
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      # ADD THIS LINE for persistent stock data storage:
      - ./data/stocks:/opt/airflow/data/stocks
      # Optional: If you want to share data with other services
      - ./data:/opt/airflow/data
  
  airflow-worker:
    # ... existing config ...
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      # ADD THIS LINE for persistent stock data storage:
      - ./data/stocks:/opt/airflow/data/stocks
      # Optional: If you want to share data with other services
      - ./data:/opt/airflow/data
```

## 2. Create Data Directory

Before starting Airflow, create the stocks data directory:

```bash
mkdir -p data/stocks
```

## 3. Verify Persistence

After the first backfill run (12 years), you should see these files:

```
data/stocks/
├── anti_gravity_daily.csv       # ~500-1000 MB (12 years × 35 tickers)
└── anti_gravity_metadata.json   # ~2-5 KB (cached ISIN/Sector data)
```

These files will persist across container restarts because they're stored on the host machine.

## 4. Complete docker-compose.yml Example

Here's a minimal working example with volume mounts:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data

  airflow-webserver:
    image: apache/airflow:2.8.0-python3.11
    command: webserver
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./data/stocks:/opt/airflow/data/stocks  # Stock data persistence
    ports:
      - "8080:8080"
    depends_on:
      - postgres

  airflow-scheduler:
    image: apache/airflow:2.8.0-python3.11
    command: scheduler
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
      - ./data/stocks:/opt/airflow/data/stocks  # Stock data persistence
    depends_on:
      - postgres

  airflow-init:
    image: apache/airflow:2.8.0-python3.11
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com \
          --password admin
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    depends_on:
      - postgres

volumes:
  postgres-db-volume:
```

## 5. Installation Steps

```bash
# 1. Navigate to your project directory
cd /Users/seifhegazy/Documents/Grad\ Project

# 2. Create the data directory
mkdir -p data/stocks

# 3. Copy the DAG file (already done)
# The DAG is in: dags/anti_gravity_egx_ingestion.py

# 4. Update docker-compose.yml with volume mounts

# 5. Install dependencies in Airflow
docker compose exec airflow-webserver pip install yfinance==0.2.66 pytz

# 6. Restart Airflow to pick up the DAG
docker compose restart airflow-scheduler airflow-webserver

# 7. Check the DAG is loaded
# Open http://localhost:8080 and look for "anti_gravity_egx_ingestion"
```

## 6. Manual Backfill Trigger (Optional)

If you want to trigger the initial 12-year backfill immediately:

```bash
# Trigger the DAG manually via CLI
docker compose exec airflow-scheduler airflow dags trigger anti_gravity_egx_ingestion

# Or use the Airflow UI:
# 1. Go to http://localhost:8080
# 2. Find "anti_gravity_egx_ingestion"
# 3. Click the "Play" button (Trigger DAG)
```

## 7. Monitoring

```bash
# Watch the logs in real-time
docker compose logs -f airflow-scheduler

# Check the CSV file size (should grow during backfill)
watch -n 5 'ls -lh data/stocks/anti_gravity_daily.csv'
```

## 8. Memory Considerations

The DAG is optimized for low RAM usage:
- Processes **one ticker at a time** (not all 35 in memory)
- Writes immediately to disk
- Clears memory after each ticker

Expected RAM usage: **< 500 MB** (even during 12-year backfill)

## 9. Troubleshooting

**Issue: DAG not appearing in UI**
```bash
# Check for syntax errors
docker compose exec airflow-scheduler python /opt/airflow/dags/anti_gravity_egx_ingestion.py

# Force DAG refresh
docker compose exec airflow-scheduler airflow dags list-import-errors
```

**Issue: Permission errors on data/stocks/**
```bash
# Fix permissions
chmod -R 777 data/stocks
```

**Issue: CSV file too large**
```bash
# The 12-year backfill will create a ~500-1000 MB CSV
# This is expected and normal
# To verify it's working:
tail -n 10 data/stocks/anti_gravity_daily.csv
```
