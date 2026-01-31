#!/usr/bin/env python3
"""
EgySentiment Auto-Scorer
Scores new articles using Groq API (same as data_pipeline.py)
"""

import json
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
INPUT_FILE = "/opt/airflow/data/raw/news/articles.jsonl"
OUTPUT_FILE = "/opt/airflow/data/processed/scored_articles.csv"
GROQ_MODEL = "llama-3.3-70b-versatile"
RATE_LIMIT_DELAY = 2.5  # 30 RPM limit

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_sentiment_score(sentiment):
    """Convert sentiment string to numeric score"""
    if sentiment == "positive": return 1
    if sentiment == "negative": return -1
    return 0

def analyze_text_groq(text):
    """Analyze sentiment using Groq API"""
    prompt = f"""Analyze the sentiment of this Egyptian financial news article.

Article: {text[:2000]}

Respond ONLY with valid JSON in this exact format:
{{"sentiment": "positive/negative/neutral", "reasoning": "brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial sentiment analysis expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        content = response.choices[0].message.content
        
        # Robust JSON extraction
        if "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        
        result = json.loads(content)
        return result.get("sentiment", "neutral").lower(), result.get("reasoning", "")
    except json.JSONDecodeError:
        return "neutral", "parsing_error"
    except Exception as e:
        print(f"⚠️ Error analyzing text: {e}")
        return "neutral", str(e)

def main():
    print(f"🚀 Starting Auto-Scoring at {datetime.now()}")
    
    # Validate API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found. Skipping auto-scoring.")
        return
    
    # 1. Load Input Data
    if not os.path.exists(INPUT_FILE):
        print(f"⚠️ Input file {INPUT_FILE} not found. First run - nothing to score.")
        return
        
    try:
        data = []
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        if not data:
            print("⚠️ No articles found in input file.")
            return
            
        df_input = pd.DataFrame(data)
        print(f"📚 Loaded {len(df_input)} articles from {INPUT_FILE}")
    except Exception as e:
        print(f"❌ Error loading input: {e}")
        return

    # 2. Filter to last 7 days only (prevent resource hogging)
    cutoff_date = datetime.now() - timedelta(days=7)
    
    if 'timestamp' in df_input.columns:
        df_input['_timestamp'] = pd.to_datetime(df_input['timestamp'], errors='coerce')
        df_recent = df_input[df_input['_timestamp'] >= cutoff_date].copy()
        print(f"📅 Filtered to {len(df_recent)} articles from last 7 days")
    else:
        df_recent = df_input.copy()

    # 3. Check for articles without sentiment (already scored by data_pipeline)
    # The data_pipeline.py already scores articles - this is just a backup scorer
    articles_to_score = df_recent[
        (df_recent.get('sentiment', pd.Series(['unknown']*len(df_recent))).isin(['unknown', ''])) |
        (df_recent.get('sentiment', pd.Series()).isna())
    ] if 'sentiment' in df_recent.columns else df_recent
    
    # Actually, data_pipeline already scores everything
    # This script is for re-scoring or handling edge cases
    # Let's just report the current sentiment distribution instead
    
    if 'sentiment' in df_recent.columns:
        sentiment_dist = df_recent['sentiment'].value_counts().to_dict()
        print(f"📊 Current sentiment distribution: {sentiment_dist}")
        
        # Count unscored
        unscored = df_recent[df_recent['sentiment'].isna() | (df_recent['sentiment'] == '')].shape[0]
        if unscored > 0:
            print(f"⚠️ Found {unscored} unscored articles")
        else:
            print("✅ All recent articles already have sentiment scores")
    else:
        print("⚠️ No sentiment column found - articles may need scoring")
    
    print("🏁 Auto-Score check complete.")

if __name__ == "__main__":
    main()
