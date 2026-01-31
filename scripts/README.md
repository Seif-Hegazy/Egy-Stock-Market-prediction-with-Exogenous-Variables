# Scripts Directory

## 📁 Structure

### `training/` - Model Training Scripts
- `train_models.py` - Basic per-ticker LogReg training
- `train_hybrid_model.py` - Hybrid model with 10-day window & T+5 target
- `train_sharpe_thresholds.py` - Sharpe Ratio optimized thresholds

### `data_prep/` - Data Preparation
- `prepare_training_data.py` - Raw data → `train_ready_data.csv`
- `prepare_logreg_data.py` - LogReg-specific with lag features → `train_ready_logreg.csv`
- `clean_features.py` - Feature engineering utilities

### `validation/` - Quality Assurance
- `verify_data_quality.py` - Pre-training data audit
- `verify_final_data.py` - Final dataset verification
- `audit_leakage.py` - Data leakage forensics
- `validate_data_integrity.py` - Raw data validation

### `archive/` - Legacy/One-time Scripts
- Scripts used for initial data fixes (not needed for regular workflow)

## 🚀 Typical Workflow

```bash
# 1. Prepare data
python scripts/data_prep/prepare_logreg_data.py

# 2. Validate data quality
python scripts/validation/verify_data_quality.py

# 3. Train models
python scripts/training/train_sharpe_thresholds.py
```
