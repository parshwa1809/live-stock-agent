import requests
import json
import logging
import time
from datetime import datetime
from threading import Lock
from typing import List, Dict
from pathlib import Path
import os 
import re # Import regex for better word matching

logger = logging.getLogger("sentiment_signals")

# --- Define file path for news data ---
try:
    from phase2_pipeline import cfg
    DATA_PATH = Path(cfg.DATA_DIR)
except ImportError:
    DATA_PATH = Path("./data")

NEWS_JSON_FILE = DATA_PATH / "news_headlines.json"
NEWS_JSON_FILE_TMP = DATA_PATH / "news_headlines.json.tmp"


# --- Global Cache and Lock ---
SENTIMENT_CACHE: Dict[str, dict] = {}
CACHE_LOCK = Lock()

# --- Simple Sentiment Logic (Placeholder for stability) ---
def analyze_sentiment(title: str) -> float:
    """
    Simple rule-based sentiment score based on keywords.
    """
    title = title.lower()
    positive_words = ["rises", "beats", "strong", "gains", "upgrade", "success", "achieves"]
    negative_words = ["falls", "misses", "weak", "losses", "downgrade", "struggles", "cuts"]
    
    score = 0
    for word in positive_words:
        if word in title:
            score += 1
    for word in negative_words:
        if word in title:
            score -= 1
            
    return score 

def fetch_news_and_analyze(tickers: List[str], api_url: str, api_key: str):
    """
    Fetches *financially-relevant* news headlines (from all domains), 
    saves them atomically, and updates the sentiment cache.
    """
    if not api_key or api_key == "YOUR_NEWS_API_KEY":
        logger.warning("Sentiment: Skipping fetch. NEWS_API_KEY not configured. Please update .env.")
        return

    # --- UPDATED: Smart Querying ---
    # 1. Create a keyword filter for *financial* news
    financial_keywords = "(stock OR earnings OR finance OR market OR analyst OR dividend OR shares OR equity OR financial)"
    # 2. Create the query for all tickers
    ticker_query = f"({' OR '.join(tickers)})"
    
    # 3. Combine them
    smart_query = f"{ticker_query} AND {financial_keywords}"
    
    # --- FIX: Removed the restrictive domain filter to broaden the search ---
    # smart_domains = "reuters.com,bloomberg.com,marketwatch.com,wsj.com,finance.yahoo.com,cnbc.com"
    # --- END FIX ---
    
    headers = {'X-Api-Key': api_key}
    params = {
        'q': smart_query,
        # 'domains': smart_domains, # <-- This filter is now removed
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100 # <-- FIX: Changed from 20 to 100
    }
    
    logger.info(f"Sentiment: Fetching relevant financial news with query: {smart_query}...")
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'error':
             logger.error(f"Sentiment API Error: {data.get('message')}")
             return

        new_cache = {}
        articles = data.get('articles', [])
        
        # --- Atomic Write to prevent race conditions ---
        if articles:
            try:
                # 1. Write to the temporary file first
                with open(NEWS_JSON_FILE_TMP, 'w', encoding='utf-8') as f:
                    json.dump(articles, f, indent=2)
                
                # 2. Atomically rename/move the file
                os.replace(NEWS_JSON_FILE_TMP, NEWS_JSON_FILE)
                
                logger.info(f"Sentiment: Successfully saved {len(articles)} headlines to {NEWS_JSON_FILE}")
            except Exception as e:
                logger.error(f"Sentiment: Failed to save news JSON file: {e}")
        # --- END Atomic Write ---

        if articles:
            for article in articles:
                title = article.get('title', '')
                
                # Find the primary ticker this article is about
                target_ticker = None
                for t in tickers:
                    # --- NEW: Use regex for whole-word matching (e.g., "TSLA" but not "TSLA-WANNABE") ---
                    if re.search(r'\b' + re.escape(t) + r'\b', title, re.IGNORECASE):
                        target_ticker = t
                        break
                
                if target_ticker:
                    score = analyze_sentiment(title)
                    
                    if target_ticker not in new_cache:
                        new_cache[target_ticker] = {
                            'total_score': 0,
                            'count': 0,
                            'last_updated': datetime.now().isoformat()
                        }
                    
                    new_cache[target_ticker]['total_score'] += score
                    new_cache[target_ticker]['count'] += 1

            for t, entry in new_cache.items():
                if entry['count'] > 0:
                    entry['average_score'] = entry['total_score'] / entry['count']
                else:
                    entry['average_score'] = 0.0

            with CACHE_LOCK:
                SENTIMENT_CACHE.update(new_cache)
            logger.info(f"Sentiment: Successfully processed news for {len(new_cache)} tickers.")
        else:
            logger.info("Sentiment: No relevant financial news found in this cycle.")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Sentiment API Call failed: {e}")
        

def compute_signals(ticker: str) -> tuple[list[str], dict]:
    """
    Generates alerts based on the latest data in the SENTIMENT_CACHE.
    
    Returns:
        A tuple containing:
        1. alerts (list[str]): A list of human-readable alert strings.
        2. indicators (dict): An empty dictionary (for API consistency).
    """
    alerts = []
    indicators = {} # <-- NEW: Empty dict for consistent return signature
    
    with CACHE_LOCK:
        data = SENTIMENT_CACHE.get(ticker)
        
    if data and data['count'] > 0:
        score = data['average_score']
        
        if score > 0.5:
            alerts.append(f"Strong Positive News: Average sentiment score ({score:.2f}).")
        elif score < -0.5:
            alerts.append(f"Strong Negative News: Average sentiment score ({score:.2f}). Caution advised.")
        elif data['count'] >= 5: 
            alerts.append(f"High News Volume: {data['count']} recent headlines detected.")

    return alerts, indicators
