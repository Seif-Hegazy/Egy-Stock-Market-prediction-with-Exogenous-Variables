import sys
import os

# --- Runtime Dependency Hotfix ---
# Install packages to a local directory since system/user installs are blocked
LIB_DIR = "/app/src/libs"
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import pandas as pd
import numpy as np
import subprocess
import importlib

# Check if statsmodels is already installed in the local lib directory
STATSMODELS_DIR = os.path.join(LIB_DIR, "statsmodels")
should_install = not os.path.exists(STATSMODELS_DIR)

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    print("✅ Statsmodels imported successfully.")
except ImportError as e:
    print(f"⚠️ Statsmodels import failed: {e}")
    import_error_msg = str(e)
    if os.path.exists(STATSMODELS_DIR):
        print(f"⚠️ Directory {STATSMODELS_DIR} exists but import failed. This might be a version mismatch.")
        should_install = False 
    else:
        should_install = True

if not should_install and 'adfuller' not in locals():
    # Define dummy functions if we decided not to install but import failed
    def grangercausalitytests(*args, **kwargs): raise ImportError(f"Statsmodels import failed: {import_error_msg}. Please restart container.")
    def adfuller(*args, **kwargs): raise ImportError(f"Statsmodels import failed: {import_error_msg}. Please restart container.")

if should_install:
    print(f"📉 Statsmodels not found in {LIB_DIR}. Installing... (This may take 5+ minutes)")
    try:
        os.makedirs(LIB_DIR, exist_ok=True)
        # Install with --no-deps for statsmodels to avoid messing up numpy, 
        # but we need scipy and patsy.
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--target=" + LIB_DIR, 
            "statsmodels==0.14.0", "numpy<2", "scipy<1.13", "patsy"
        ])
        importlib.invalidate_caches()
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
        print("✅ Statsmodels installed successfully.")
    except Exception as e:
        install_error_msg = str(e)
        print(f"❌ Failed to install statsmodels: {install_error_msg}")
        # Define dummy functions to prevent crash if install fails
        def grangercausalitytests(*args, **kwargs): raise ImportError(f"statsmodels installation failed: {install_error_msg}")
        def adfuller(*args, **kwargs): raise ImportError(f"statsmodels installation failed: {install_error_msg}")

from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_stationarity(series):
    """
    Checks stationarity using Augmented Dickey-Fuller test.
    Returns (is_stationary, p_value).
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < 0.05, p_value

def make_stationary(series):
    """
    Applies differencing until the series is stationary.
    Returns the stationary series.
    """
    is_stationary, p_value = check_stationarity(series)
    
    if is_stationary:
        return series
    
    # Apply first difference
    diff_series = series.diff().dropna()
    is_stationary, p_value = check_stationarity(diff_series)
    
    if is_stationary:
        return diff_series
        
    # Apply second difference if needed
    diff_series = diff_series.diff().dropna()
    return diff_series

def calculate_granger_causality(news_df, stock_df, max_lag=5):
    """
    Performs Granger Causality test to see if News Sentiment predicts Stock Price.
    
    Args:
        news_df (pd.DataFrame): Must contain 'date' and 'sentiment_score'.
        stock_df (pd.DataFrame): Must contain 'Date' and 'Close'.
        max_lag (int): Maximum lags to test (days).
        
    Returns:
        dict: {
            'min_p_value': float,
            'is_significant': bool,
            'best_lag': int,
            'details': dict
        }
    """
    try:
        # 1. Daily Aggregation of News
        # Ensure date format consistency
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        daily_news = news_df.groupby('date')['sentiment_score'].mean().reset_index()
        daily_news.set_index('date', inplace=True)
        
        # 2. Prepare Stock Data
        stock_df = stock_df.copy()
        if 'Date' in stock_df.columns:
            stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
        else:
            stock_df['date'] = pd.to_datetime(stock_df.index).dt.date
            
        stock_df.set_index('date', inplace=True)
        # We want to predict price *movement* or *returns*, not raw price usually, 
        # but Granger test handles raw if stationary. Let's use Close price.
        daily_stock = stock_df[['Close']].sort_index()
        
        # 3. Data Alignment
        # Inner join to get matching days
        merged_df = daily_news.join(daily_stock, how='inner').dropna()
        print(f"DEBUG: merged_df length: {len(merged_df)}")
        
        # Dynamic Lag Adjustment
        # Statsmodels Granger test requires roughly 3*lag + 2 observations
        required_obs = max_lag * 3 + 2
        
        if len(merged_df) < required_obs:
            # Try to reduce lag
            # We subtract 4 instead of 2: 2 for the formula, and 2 extra for potential differencing (d=2)
            new_lag = (len(merged_df) - 4) // 3
            
            if new_lag < 1:
                return {"error": f"Not enough overlapping data points. Found {len(merged_df)}, need at least 5."}
            
            print(f"⚠️ Data sparse (n={len(merged_df)}). Reducing lag from {max_lag} to {new_lag}")
            max_lag = new_lag
            # Add a warning to be returned later if needed, or just proceed


        # 4. Stationarity Check & Transformation
        # Granger Causality requires stationary time series
        news_series = make_stationary(merged_df['sentiment_score'])
        stock_series = make_stationary(merged_df['Close'])
        
        # Re-align after differencing (which drops rows)
        aligned_data = pd.concat([stock_series, news_series], axis=1).dropna()
        aligned_data.columns = ['Stock', 'News'] # Target first (Stock), Predictor second (News)
        
        print(f"DEBUG: aligned_data length: {len(aligned_data)}, max_lag: {max_lag}")

        # Heuristic: statsmodels needs roughly 3x lags + buffer
        required_obs = 3 * max_lag + 2
        if len(aligned_data) < required_obs:
             return {"error": f"Not enough data. Found {len(aligned_data)} points, need at least {required_obs} for lag {max_lag}."}

        # 5. Granger Causality Test
        # statsmodels expects a 2D array with columns [target, predictor]
        # We are testing if 'News' causes 'Stock'
        try:
            test_result = grangercausalitytests(aligned_data, maxlag=max_lag, verbose=False)
        except ValueError as ve:
            # Catch "Insufficient observations" error from statsmodels
            return {"error": f"Statistical test failed: {str(ve)}. Try reducing the lag or selecting a stock with more data."}
        
        min_p_value = 1.0
        best_lag = -1
        
        # Extract p-values for each lag
        # test_result format: {lag: ({test_stats}, {params})}
        # We usually look at the SSR based F test or Chi2 test. Let's use ssr_chi2test.
        
        for lag, result in test_result.items():
            # result[0] is the dictionary of tests
            # 'ssr_chi2test' is a tuple: (statistic, p-value, df)
            p_val = result[0]['ssr_chi2test'][1]
            if p_val < min_p_value:
                min_p_value = p_val
                best_lag = lag
                
        return {
            "min_p_value": min_p_value,
            "is_significant": min_p_value < 0.05,
            "best_lag": best_lag,
            "sample_size": len(aligned_data)
        }

    except Exception as e:
        logger.error(f"Granger Test Failed: {e}")
        return {"error": str(e)}
