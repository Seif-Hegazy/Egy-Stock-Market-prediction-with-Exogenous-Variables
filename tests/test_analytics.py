import sys
import os

# --- Runtime Dependency Hotfix ---
LIB_DIR = "/app/src/libs"
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import pandas as pd
import numpy as np
from analytics import calculate_granger_causality

def test_granger_causality():
    print("--- Testing Granger Causality ---")
    
    # 1. Create Synthetic Data (Causal Relationship)
    # News happens at t, Stock moves at t+1
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Random news sentiment
    np.random.seed(42)
    news_sentiment = np.random.normal(0, 1, 100)
    
    # Stock price follows news with 1 day lag + noise
    stock_price = np.zeros(100)
    stock_price[0] = 100
    for i in range(1, 100):
        # Price change depends on yesterday's news
        stock_price[i] = stock_price[i-1] + (news_sentiment[i-1] * 2) + np.random.normal(0, 0.5)
        
    news_df = pd.DataFrame({'date': dates, 'sentiment_score': news_sentiment})
    stock_df = pd.DataFrame({'Date': dates, 'Close': stock_price})
    
    print("Running test on Causal Data (Lag 1)...")
    result = calculate_granger_causality(news_df, stock_df, max_lag=3)
    print(f"Result: {result}")
    
    if result.get('is_significant') and result.get('best_lag') == 1:
        print("✅ SUCCESS: Detected causality at Lag 1")
    else:
        print("❌ FAILURE: Did not detect expected causality")
        
    # 2. Create Non-Causal Data
    print("\nRunning test on Random Data (No Causality)...")
    random_stock = pd.DataFrame({'Date': dates, 'Close': np.random.normal(100, 10, 100)})
    result_random = calculate_granger_causality(news_df, random_stock, max_lag=3)
    print(f"Result: {result_random}")
    
    if not result_random.get('is_significant'):
        print("✅ SUCCESS: Correctly identified no causality")
    else:
        print("❌ FAILURE: False positive detected")

if __name__ == "__main__":
    test_granger_causality()
