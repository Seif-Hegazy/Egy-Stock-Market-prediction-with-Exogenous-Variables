# Project Reorganization Migration Guide

## Overview

This guide helps you transition from the old project structure to the new organized layout.

---

## What Changed

### File Locations

| Old Location | New Location | Renamed? |
|--------------|--------------|----------|
| `dags/anti_gravity_egx_ingestion.py` | `airflow/dags/egx_stock_ingestion.py` | ✅ |
| `dags/requirements_dag.txt` | `airflow/requirements.txt` | ✅ |
| `dags/SETUP_INSTRUCTIONS.md` | `airflow/README.md` | ✅ |
| `services/sentiment-analysis/src/app.py` | `src/streamlit_app/app.py` | ❌ |
| `services/sentiment-analysis/src/analytics.py` | `src/streamlit_app/analytics.py` | ❌ |
| `services/sentiment-analysis/src/test_*.py` | `tests/test_*.py` | ❌ |
| `data/stocks/anti_gravity_daily.csv` | `data/raw/stocks/egx_daily_12y.csv` | ✅ |
| `data/stocks/anti_gravity_metadata.json` | `data/metadata/stocks/ticker_info.json` | ✅ |
| `data/news/testing_data.jsonl` | `data/raw/news/articles.jsonl` | ✅ |

### Data File Naming

The following files were renamed for clarity:

- `anti_gravity_daily.csv` → `egx_daily_12y.csv` (more descriptive)
- `anti_gravity_metadata.json` → `ticker_info.json` (clearer purpose)
- `testing_data.jsonl` → `articles.jsonl` (production-ready name)

---

## Migration Steps

### Step 1: Update Docker Compose

Edit `docker-compose.yml` and update volume mounts:

**OLD**:
```yaml
volumes:
  - ./dags:/opt/airflow/dags
  - ./data/stocks:/opt/airflow/data/stocks
```

**NEW**:
```yaml
volumes:
  - ./airflow/dags:/opt/airflow/dags
  - ./data:/opt/airflow/data
  - ./src:/app/src
```

### Step 2: Rename Data Files (Optional)

If you have existing data files, rename them:

```bash
# Stock data
mv data/stocks/anti_gravity_daily.csv data/raw/stocks/egx_daily_12y.csv 2>/dev/null || true

# Metadata
mv data/stocks/anti_gravity_metadata.json data/metadata/stocks/ticker_info.json 2>/dev/null || true

# News data
mv data/news/testing_data.jsonl data/raw/news/articles.jsonl 2>/dev/null || true
```

### Step 3: Restart Services

```bash
# Stop all services
docker compose down

# Restart with new configuration
docker compose up -d

# Verify services are running
docker compose ps
```

### Step 4: Verify DAG Discovery

Check that Airflow can find the reorganized DAG:

```bash
# List all DAGs
docker exec airflow-scheduler airflow dags list

# Should see: egx_stock_ingestion
```

### Step 5: Test Data Pipelines

```bash
# Trigger the stock ingestion DAG
docker exec airflow-scheduler airflow dags trigger egx_stock_ingestion

# Monitor logs
docker compose logs -f airflow-scheduler
```

---

## Import Statement Changes

If you're developing custom Python modules, update imports:

**OLD**:
```python
from services.sentiment_analysis.src.analytics import calculate_granger_causality
```

**NEW**:
```python
from src.streamlit_app.analytics import calculate_granger_causality
```

---

## Backward Compatibility

### Old DAGs Still Work

The old `dags/` directory still exists and is mounted. Old DAGs will continue working.

To fully migrate:
1. Test new DAG (`egx_stock_ingestion`) works
2. Disable old DAG in Airflow UI
3. Delete old `dags/` directory (optional)

### Legacy Service Structure

The `services/` directory is preserved for backward compatibility. It's marked as deprecated in the new README.

---

## Rollback Plan

If you encounter issues, rollback is easy since files were **copied** (not moved):

### Quick Rollback

```bash
# Restore old docker-compose.yml from git
git checkout docker-compose.yml

# Restart with old configuration
docker compose restart
```

### Full Rollback

```bash
# Remove new directories
rm -rf airflow/ src/ tests/ docs/

# Restore old mounts in docker-compose.yml
# Restart services
docker compose up -d
```

---

## New Workflow Examples

### Running Tests

**OLD**:
```bash
docker exec gradproject-streamlit-1 python /app/src/test_analytics.py
```

**NEW**:
```bash
docker exec gradproject-streamlit-1 python /app/tests/test_analytics.py
```

### Viewing Stock Data

**OLD**:
```bash
head data/stocks/anti_gravity_daily.csv
```

**NEW**:
```bash
head data/raw/stocks/egx_daily_12y.csv
```

### Editing Streamlit App

**OLD**:
```bash
vim services/sentiment-analysis/src/app.py
```

**NEW**:
```bash
vim src/streamlit_app/app.py
```

---

## Benefits of New Structure

✅ **Clarity**: Each directory has a single, clear purpose  
✅ **Discoverability**: Easy to find files  
✅ **Scalability**: Room for growth (new apps, new DAGs)  
✅ **Standards**: Follows Python best practices  
✅ **Documentation**: Centralized in `docs/`

---

## Troubleshooting

### Issue: DAG not found

**Symptom**: `egx_stock_ingestion` doesn't appear in Airflow UI

**Solution**:
```bash
# Check volume mount
docker exec airflow-scheduler ls /opt/airflow/dags

# Check for errors
docker exec airflow-scheduler airflow dags list-import-errors
```

### Issue: Data files missing

**Symptom**: CSV file not found after restart

**Solution**:
```bash
# Verify volume mount
docker exec airflow-scheduler ls /opt/airflow/data/stocks/raw

# Check docker-compose.yml has correct path
grep "data:" docker-compose.yml
```

### Issue: Import errors in Python

**Symptom**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure /app/src is in PYTHONPATH
docker exec gradproject-streamlit-1 python -c "import sys; print(sys.path)"

# Add to docker-compose.yml if missing:
environment:
  PYTHONPATH: /app
```

---

## Next Steps

1. ✅ Review the new [README.md](../README.md)
2. ✅ Check [Data Schema Documentation](../docs/data_schema.md)
3. ✅ Read [Airflow Setup Guide](../airflow/README.md)
4. ✅ Test the reorganized structure
5. ✅ Update bookmarks and documentation links

---

## Questions?

If you encounter issues during migration, check:
- Docker logs: `docker compose logs`
- Airflow UI: http://localhost:8080
- Streamlit logs: `docker logs gradproject-streamlit-1`

**Rollback is always safe** - all original files are preserved!
