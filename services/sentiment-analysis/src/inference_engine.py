import os
import json
from groq import Groq

def analyze_aspect_sentiment(text):
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) on financial text.
    
    Aspects:
    1. Financial Fundamentals (Earnings, Revenue, Dividends)
    2. Macro/Regulatory (Interest rates, Central Bank, Legal)
    3. Market Sentiment (Volume, Speculation, Momentum)
    4. Geopolitics/External (Supply chain, Regional stability)
    
    Returns:
        dict: JSON object with scores (-1.0 to 1.0) and reasoning for each aspect.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"error": "GROQ_API_KEY not found"}
            
        client = Groq(api_key=api_key)
        
        prompt = f"""Analyze the provided financial text and extract sentiment scores for the following specific aspects.
        
        Text: {text[:3000]}
        
        For each aspect, provide:
        - "score": A float between -1.0 (Very Negative) and 1.0 (Very Positive). Use 0.0 if neutral or not mentioned.
        - "reasoning": A comprehensive explanation (approx 80 words) citing specific quantitative data, names, and causal links from the text.
        
        Aspects to Analyze:
        1. **Financial Fundamentals**: Earnings, revenue, profit margins, dividends, balance sheet health.
        2. **Macro/Regulatory**: Interest rates, inflation, Central Bank (CBE) decisions, government policies, legal rulings.
        3. **Market Sentiment**: Trading volume, investor confidence, speculation, trend momentum, technical indicators.
        4. **Geopolitics/External**: Regional stability, supply chain disruptions, foreign exchange availability, global market influence.
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "general_sentiment": "Positive/Negative/Neutral",
            "general_reasoning": "Overall summary of the sentiment...",
            "graph_explanation": "Interpret the balance of the aspects (e.g., 'Strong fundamentals are offset by negative macro factors').",
            "fundamentals": {{ "score": 0.0, "reasoning": "..." }},
            "macro": {{ "score": 0.0, "reasoning": "..." }},
            "sentiment": {{ "score": 0.0, "reasoning": "..." }},
            "geopolitics": {{ "score": 0.0, "reasoning": "..." }}
        }}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior financial analyst specializing in the Egyptian market (EGX). You provide precise, nuanced aspect-based sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        return json.loads(content)
        
    except Exception as e:
        return {"error": str(e)}
