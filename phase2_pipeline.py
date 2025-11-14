#!/usr/bin/env python3
"""
phase2_pipeline.py — Phase 2 live stock pipeline (Fixed)

Features:
- Creates separate files for historical (30d/30min) and live (1d/1min) data
- Historical data is fetched once with extended delay to prevent rate limiting
- Live data is updated every 60 seconds
- No data mixing, which fixes all chart bugs
"""

import os
import time
import threading
import logging
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import signal

# -----------------------
# Config
# -----------------------
load_dotenv()

class Config:
    TICKERS = os.getenv("TICKERS", "AAPL,AMZN,GOOGL,MSFT,TSLA").split(",")
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 60))  # seconds between cycles
    # Back to 30 days for historical data
    CSV_RETENTION_DAYS = int(os.getenv("CSV_RETENTION_DAYS", 30))
    LIVE_MEMORY_MINUTES = int(os.getenv("LIVE_MEMORY_MINUTES", 1440)) # 1 full day
    ALERT_GAP_THRESHOLD = float(os.getenv("ALERT_GAP_THRESHOLD", 0.02))
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "phi3:mini")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

cfg = Config()
Path(cfg.DATA_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------
# Logging & shutdown
# -----------------------
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
)
logger = logging.getLogger("phase2_pipeline")
STOP_EVENT = threading.Event()

def _shutdown(signum, frame):
    logger.info("Shutdown signal received, stopping pipeline...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# -----------------------
# Ollama LLM interface (synchronous for Dash)
# -----------------------
class LLMInterface:
    def __init__(self, model_name: str = cfg.OLLAMA_MODEL_NAME, url: str = cfg.OLLAMA_API_URL):
        self.model_name = model_name
        self.url = url

    def generate(self, prompt: str, timeout: int = 60) -> str:
        """
        Synchronous call; returns final text (or error string).
        Compatible with Dash callback usage.
        (FIXED: Increased robustness/timeout for Ollama requests)
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False # Crucial for synchronous call
            }
            # Increased requests timeout for safety
            resp = requests.post(self.url, json=payload, timeout=timeout) 
            
            # --- IMPROVED ERROR LOGGING ---
            if resp.status_code != 200:
                error_text = resp.text.strip()
                logger.error(f"Ollama HTTP Error {resp.status_code}: {error_text}") 
                return f"[Ollama Error {resp.status_code}] {error_text}"
            # --- END IMPROVED ERROR LOGGING ---

            # Try to parse the single expected JSON object
            try:
                data = resp.json()
            except Exception as e:
                # Log the parsing failure if the response wasn't clean JSON
                logger.error(f"Ollama JSON Parsing Error: {e}. Raw response: {resp.text[:100]}...")
                return resp.text or "[Ollama returned non-json response]"

            # Try common response keys
            for key in ("response", "completion", "output", "text"):
                if key in data:
                    return str(data.get(key) or "").strip()
            
            # If no known key is found, return the full JSON structure
            return json.dumps(data)
        except Exception as e:
            return f"[Ollama Exception] {e}"

llm = LLMInterface()

# -----------------------
# Column flattening & canonicalization
# -----------------------
def flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{a}_{b}" if (isinstance(b, str) and b) else str(a)
            for a,b in df.columns
        ]

    # clean stringified tuple columns like "('Close','AAPL')"
    new_cols = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith("(") and "," in c and c.endswith(")"):
            parts = [p.strip().strip("'\" ") for p in c.strip("()").split(",")]
            if len(parts) >= 2:
                new_cols.append(f"{parts[0]}_{parts[1]}")
                continue
        new_cols.append(c)
    df.columns = new_cols

    # remove exact duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    try:
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    except Exception:
        df.index = pd.to_datetime(df.index, errors='coerce')
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
        except Exception:
            pass

    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    for base in base_cols:
        suff = f"{base}_{ticker}"
        if suff in df.columns and base not in df.columns:
            df[base] = pd.to_numeric(df[suff], errors='coerce')
        elif base in df.columns and suff not in df.columns:
            df[suff] = pd.to_numeric(df[base], errors='coerce')

    for c in list(df.columns):
        if any(c.startswith(b) for b in base_cols):
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.loc[:, ~df.columns.duplicated()]
    return df

# -----------------------
# CSV loader helper
# -----------------------
def load_csv(path: str, ticker: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        try:
            df = pd.read_csv(path, index_col=0, header=[0,1])
        except Exception as e:
            logger.warning(f"Failed to read CSV {path}: {e}")
            return pd.DataFrame()

    df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    df = flatten_columns(df, ticker)
    return df

# -----------------------
# Data manager (Handles HISTORICAL 30-day data)
# -----------------------
class DataManager:
    def __init__(self, tickers, data_dir):
        self.tickers = list(tickers)
        self.data_dir = Path(data_dir)
        self.historical = {}  # ticker -> DataFrame

    def path_for_hist(self, ticker: str) -> Path:
        return self.data_dir / f"{ticker}_hist.csv"

    def load_or_fetch_hist(self, ticker: str) -> pd.DataFrame:
        p = self.path_for_hist(ticker)
        if p.exists():
            df = load_csv(str(p), ticker)
            self.historical[ticker] = df
            logger.info(f"Loaded {ticker} HISTORICAL: {len(df)} rows")
            return df
        return self.fetch_historical(ticker)

    def fetch_historical(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        end = end or datetime.now(timezone.utc)
        start = start or (end - timedelta(days=cfg.CSV_RETENTION_DAYS))
        logger.info(f"Fetching 30-MINUTE historical for {ticker} from {start.date()} to {end.date()}")
        # Fetch 30-day, 30-minute data
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                         interval="30m", progress=False)
        
        if df is None or df.empty:
            logger.warning(f"No historical data for {ticker}")
            return pd.DataFrame()
            
        df.index.name = "Datetime"
        df = flatten_columns(df, ticker)
        try:
            df.to_csv(self.path_for_hist(ticker))
            logger.info(f"Saved {ticker} HISTORICAL CSV ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Failed to save CSV for {ticker}: {e}")
        self.historical[ticker] = df
        return df

data_mgr = DataManager(cfg.TICKERS, cfg.DATA_DIR)

# -----------------------
# LiveTracker (Handles LIVE 1-day/1-min data)
# -----------------------
class LiveTracker:
    def __init__(self, tickers):
        self.tickers = list(tickers)
        self._buffers = {t: pd.DataFrame() for t in self.tickers}
        self.lock = threading.Lock()

    def update_one(self, ticker: str):
        """
        Fetch all 1-minute data for the current day.
        """
        try:
            # Fetch 1-day, 1-minute data
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if df is None or df.empty:
                logger.debug(f"No LIVE data for {ticker} (market may be closed)")
                return
                
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df = flatten_columns(df, ticker)
            
            with self.lock:
                self._buffers[ticker] = df
                
        except Exception as e:
            logger.debug(f"Live update failed for {ticker}: {e}")

    def update_all(self):
        # FIX: Added a 20-second delay for robust live data fetching
        LIVE_FETCH_DELAY = 20 # seconds
        for t in self.tickers:
            self.update_one(t)
            if not STOP_EVENT.is_set():
                logger.debug(f"Pausing for {LIVE_FETCH_DELAY}s after fetching live data for {t}...")
                time.sleep(LIVE_FETCH_DELAY) # Added delay here

    def get_buffer(self, ticker: str) -> pd.DataFrame:
        with self.lock:
            buf = self._buffers.get(ticker)
            return buf.copy() if buf is not None else pd.DataFrame()

    def save_buffer(self, ticker: str, data_dir: Path):
        """Saves the current live buffer to a _live.csv file."""
        p = Path(data_dir) / f"{ticker}_live.csv"
        buf = self.get_buffer(ticker)
        if not buf.empty:
            try:
                buf.to_csv(p)
            except Exception as e:
                logger.warning(f"Failed to save LIVE CSV for {ticker}: {e}")

live_tracker = LiveTracker(cfg.TICKERS)

# -----------------------
# Simple AlertEngine (relies on canonical "Close" and "Volume")
# -----------------------
class AlertEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def _series(self, df: pd.DataFrame, ticker: str, name: str) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if name in df.columns:
            s = df[name]
            return s.dropna() if hasattr(s, "dropna") else pd.Series(dtype=float)
        suff = f"{name}_{ticker}"
        if suff in df.columns:
            s = df[suff]
            return s.dropna() if hasattr(s, "dropna") else pd.Series(dtype=float)
        for c in df.columns:
            if isinstance(c, str) and c.startswith(name):
                s = df[c]
                return s.dropna() if hasattr(s, "dropna") else pd.Series(dtype=float)
        return pd.Series(dtype=float)

    def evaluate(self, ticker: str, df: pd.DataFrame):
        df = flatten_columns(df, ticker)
        alerts = []
        close = self._series(df, ticker, "Close")
        vol = self._series(df, ticker, "Volume")
        
        # trend (good for 30-min or 1-min data)
        if len(close) >= 20:
            ma5 = close.rolling(5, min_periods=5).mean().iloc[-1]
            ma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
            if ma5 > ma20:
                # *** NEW FRIENDLY NAME ***
                alerts.append("Price Trending Up Recently")
            else:
                # *** NEW FRIENDLY NAME ***
                alerts.append("Price Trending Down Recently")
                
        # volatility (good for 1-min data)
        if len(close) >= 2:
            pct = abs(close.iloc[-1] / close.iloc[-2] - 1)
            if pct > self.cfg.ALERT_GAP_THRESHOLD:
                # *** NEW FRIENDLY NAME ***
                alerts.append("Sudden Price Swing")
                
        # volume spike (good for 1-min data)
        if len(vol) >= 10:
            latest = float(vol.iloc[-1])
            # Use a 10-period avg for 1-min data
            avg = float(vol.rolling(10).mean().iloc[-1])
            if avg > 0 and latest > avg * 2.0: # 2x spike
                # *** NEW FRIENDLY NAME ***
                alerts.append("Unusual Trading Activity")
        return alerts

alert_engine = AlertEngine(cfg)

# -----------------------
# Pipeline orchestration
# -----------------------
def initial_setup():
    """Load or fetch historical 30-day data for all tickers."""
    # FIX 1: Increased delay to 30 seconds for robust rate limit prevention
    HISTORICAL_FETCH_DELAY = 30 # seconds 
    for t in cfg.TICKERS:
        try:
            data_mgr.load_or_fetch_hist(t)
            if not STOP_EVENT.is_set():
                logger.info(f"Pausing for {HISTORICAL_FETCH_DELAY}s before fetching next historical data.")
                time.sleep(HISTORICAL_FETCH_DELAY) # Added/Increased delay
        except Exception as e:
            logger.warning(f"Initial HISTORICAL load failed for {t}: {e}")

def process_cycle():
    """Single pipeline cycle: update live 1-min buffers, save to _live.csv"""
    
    # 1. Fetch 1-day/1-min data for all tickers into memory (with 20s delay between each)
    live_tracker.update_all()
    
    for t in cfg.TICKERS:
        # 2. Save the live data from memory to `_live.csv`
        live_tracker.save_buffer(t, cfg.DATA_DIR)
        
        # 3. (Optional) Run alerts on the live data
        live_df = live_tracker.get_buffer(t)
        if not live_df.empty:
            alerts = alert_engine.evaluate(t, live_df)
            if alerts:
                logger.info(f"{t} Live Alerts: {alerts}")
        else:
            logger.debug(f"No live data for {t}, skipping alerts.")

def run_forever():
    # 1. Get 30-day/30-min data *once*
    initial_setup()
    
    # FIX 2 (FINAL): Increased cooldown to 180 seconds (3 minutes) to ensure API limit resets
    COOLDOWN_SECONDS = 180 
    logger.info(f"Initial historical setup complete. Waiting {COOLDOWN_SECONDS}s before starting live loop...")
    for _ in range(COOLDOWN_SECONDS):
        if STOP_EVENT.is_set():
            break
        time.sleep(1)
    
    logger.info("Starting Phase 2 pipeline loop (updating 1-min data)...")
    while not STOP_EVENT.is_set():
        try:
            # 2. Every 60s, get 1-day/1-min data
            process_cycle()
        except Exception as e:
            logger.exception(f"Error in process cycle: {e}")
        
        for _ in range(max(1, int(cfg.REFRESH_INTERVAL))):
            if STOP_EVENT.is_set():
                break
            time.sleep(1)
            
    logger.info("Pipeline stopped")

if __name__ == "__main__":
    run_forever()