import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TICKERS = os.getenv("TICKERS", "AAPL,AMZN,GOOGL,MSFT,TSLA").split(",")
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 300)) 
    
    CONTEXT_TICKERS = ["SPY", "QQQ", "^VIX"] 
    # Ensure all required tickers are fetched
    ALL_FETCH_TICKERS = list(set(TICKERS) | set(CONTEXT_TICKERS))
    
    TICKER_BATCH_SIZE = int(os.getenv("TICKER_BATCH_SIZE", 8)) 

    CSV_RETENTION_DAYS = int(os.getenv("CSV_RETENTION_DAYS", 30))
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "phi3:mini")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY") 
    NEWS_API_URL = "https://newsapi.org/v2/everything" 
    NEWS_REFRESH_INTERVAL = int(os.getenv("NEWS_REFRESH_INTERVAL", 600))
    
    EMERGENCY_COOLDOWN = int(os.getenv("EMERGENCY_COOLDOWN", 1800))
    
    PROXIES = [] 

cfg = Config()