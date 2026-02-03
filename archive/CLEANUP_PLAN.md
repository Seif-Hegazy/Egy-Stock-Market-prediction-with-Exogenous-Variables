# File Cleanup Plan

## Files to Remove

### 1. Old DAG Directory (Now in airflow/dags/)
```
dags/
├── anti_gravity_egx_ingestion.py    ❌ REMOVE (copied to airflow/dags/egx_stock_ingestion.py)
├── requirements_dag.txt              ❌ REMOVE (copied to airflow/requirements.txt)
└── SETUP_INSTRUCTIONS.md             ❌ REMOVE (copied to airflow/README.md)
```

### 2. Duplicate Test Files (Now in tests/)
```
services/sentiment-analysis/src/
├── test_analytics.py                 ❌ REMOVE (copied to tests/)
└── test_all_tickers.py               ❌ REMOVE (copied to tests/)
```

### 3. System Files (.DS_Store)
```
Total: 10 .DS_Store files             ❌ REMOVE (macOS metadata)
```

### 4. Debug/Utility Scripts (Keep or Remove?)
```
services/sentiment-analysis/src/
├── debug_yfinance.py                 ⚠️ REVIEW (temporary debug script)
├── test_ollama.py                    ⚠️ REVIEW (temporary test)
├── auto_score.py                     ⚠️ REVIEW (utility - may be useful)
├── deduplicate_data.py               ⚠️ REVIEW (utility - may be useful)
├── historical_scraper.py             ⚠️ REVIEW (utility - may be useful)
└── inference_engine.py               ⚠️ REVIEW (utility - may be useful)
```

---

## Removal Strategy

### Phase 1: Safe Removals (No Risk)
1. Remove old `dags/` directory
2. Remove duplicate test files from services/
3. Remove all .DS_Store files

### Phase 2: Review Required
1. Evaluate debug scripts
2. Move useful utilities to src/utils/
3. Remove truly temporary files

---

## Commands to Execute

### Remove Old DAG Directory
```bash
rm -rf dags/
```

### Remove Duplicate Test Files
```bash
rm services/sentiment-analysis/src/test_analytics.py
rm services/sentiment-analysis/src/test_all_tickers.py
```

### Remove .DS_Store Files
```bash
find . -name ".DS_Store" -type f -delete
```

### Remove Debug Scripts (After Review)
```bash
rm services/sentiment-analysis/src/debug_yfinance.py
rm services/sentiment-analysis/src/test_ollama.py
```

---

## Files to Keep

✅ `services/sentiment-analysis/src/app.py` - Original (still in use by Docker)  
✅ `services/sentiment-analysis/src/analytics.py` - Original (still in use by Docker)  
✅ `services/sentiment-analysis/src/data_pipeline.py` - Active utility  
✅ `services/sentiment-analysis/src/libs/` - Runtime dependencies  

---

## Disk Space Savings

- Old DAGs: ~15 KB
- Duplicate tests: ~5 KB
- .DS_Store files: ~60 KB
- Debug scripts: ~20 KB

**Total: ~100 KB** (minimal, but improves organization)
