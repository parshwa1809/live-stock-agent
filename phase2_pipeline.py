#!/usr/bin/env python3
"""
phase2_pipeline.py — Phase 2 live stock pipeline (Final Stable Version)

Features:
- Creates separate files for historical (30d/30min) and live (1d/1min) data
- Historical data is bypassed at startup to prevent rate limit crash
- **FIXED: Live data is updated every 5 minutes using a SINGLE BULK API CALL.**
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
    # --- FIXED: Refresh interval set to 300 seconds (5 minutes) for stability ---
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 300))  # seconds between cycles
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

    def generate(self, prompt: str, timeout: int = 120) -> str:
        """
        Synchronous call; returns final text (or error string).
        Compatible with Dash callback usage.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False # Crucial for synchronous call
            }
            resp = requests.post(self.url, json=payload, timeout=timeout) 
            
            if resp.status_code != 200:
                error_text = resp.text.strip()
                logger.error(f"Ollama HTTP Error {resp.status_code}: {error_text}") 
                return f"[Ollama Error {resp.status_code}] {error_text}"

            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"Ollama JSON Parsing Error: {e}. Raw response: {resp.text[:100]}...")
                return resp.text or "[Ollama returned non-json response]"

            for key in ("response", "completion", "output", "text"):
                if key in data:
                    return str(data.get(key) or "").strip()
            
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
        # Handle MultiIndex DataFrame from bulk fetch (e.g., Close, Open for AAPL, MSFT)
        # We only flatten here if it's the expected structure (like for live data split later)
        # For historical data, this flattening might be necessary to fix old file structures
        new_cols = []
        for col_level1, col_level2 in df.columns:
            # If the index is (Attribute, Ticker), we want 'Attribute_Ticker'
            if col_level2.upper() in cfg.TICKERS:
                new_cols.append(f"{col_level1}_{col_level2}")
            # If the index is (Attribute, ''), we just want 'Attribute' (for single ticker fetch)
            elif not col_level2:
                new_cols.append(col_level1)
            else:
                new_cols.append(f"{col_level1}_{col_level2}")

        df.columns = new_cols
    
    # Clean up stringified tuple columns like "('Close','AAPL')" - this is a legacy fix
    # We leave this in case old CSVs have this format, but the bulk fetch fixes this.
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

    # Ensure canonical columns exist for single ticker access
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
        # Note: If called, it will run the single-ticker fetch below, which is slow/risky
        return self.fetch_historical(ticker)

    def fetch_historical(self, ticker: str, start=None, end=None) -> pd.DataFrame:
        # This function is retained but is what we recommend avoiding for multiple tickers.
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
# LiveTracker (Handles LIVE 1-day/1-min data) - MODIFIED FOR BULK FETCH
# -----------------------
class LiveTracker:
    def __init__(self, tickers):
        self.tickers = list(tickers)
        self._buffers = {t: pd.DataFrame() for t in self.tickers}
        self.lock = threading.Lock()

    def update_all(self):
        """
        Fetch all 1-minute data for ALL tickers in a single bulk API call.
        """
        logger.info("Starting BULK live data fetch for all tickers.")
        try:
            # --- CRITICAL FIX: Single bulk call for all tickers ---
            # Fetch 1-day, 1-minute data for all tickers
            df_bulk = yf.download(self.tickers, period="1d", interval="1m", progress=False)
            
            if df_bulk is None or df_bulk.empty:
                logger.debug("No BULK LIVE data (market may be closed or rate limit error)")
                return
            
            # Ensure the index is a datetime index
            df_bulk.index = pd.to_datetime(df_bulk.index, utc=True, errors='coerce')
            
            with self.lock:
                for t in self.tickers:
                    # 1. Slice the MultiIndex DataFrame for the specific ticker
                    # The MultiIndex is structured as (Attribute, Ticker)
                    # Use df_bulk.columns.get_level_values(1) to check which tickers are present
                    
                    if t in df_bulk.columns.get_level_values(1):
                        # Extract all attributes (Open, High, Low, etc.) for the current ticker (t)
                        df_ticker = df_bulk.xs(t, level=1, axis=1)
                        
                        # 2. Rename columns to canonical format (e.g., 'Close' not 'Close_AAPL') for the buffer
                        df_ticker.columns.name = None # Remove the name of the column level
                        df_ticker.columns = [c.capitalize() for c in df_ticker.columns]
                        
                        # 3. Store the cleaned single-ticker DataFrame in the buffer
                        self._buffers[t] = df_ticker
                        logger.debug(f"Successfully updated live buffer for {t} ({len(df_ticker)} rows).")
                    else:
                        logger.warning(f"Ticker {t} not found in the bulk download result.")
                        
        except Exception as e:
            # This will catch a single YFRateLimitError for the entire batch, which is much cleaner
            logger.error(f"Bulk live update failed: {e}")

    def get_buffer(self, ticker: str) -> pd.DataFrame:
        with self.lock:
            buf = self._buffers.get(ticker)
            # Need to re-flatten the columns on output to satisfy the AlertEngine/Flatten func 
            # and ensure it has the expected 'Close_TICKER' format if needed.
            df_copy = buf.copy() if buf is not None else pd.DataFrame()
            return flatten_columns(df_copy, ticker)


    def save_buffer(self, ticker: str, data_dir: Path):
        """Saves the current live buffer to a _live.csv file."""
        p = Path(data_dir) / f"{ticker}_live.csv"
        buf = self.get_buffer(ticker)
        
        # Ensure we are saving the canonical format with Ticker suffix for easy reading later
        df_to_save = buf.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_to_save.columns:
                df_to_save[f"{col}_{ticker}"] = df_to_save[col]
                del df_to_save[col]

        if not df_to_save.empty:
            try:
                df_to_save.to_csv(p)
                logger.info(f"Saved LIVE CSV for {ticker}")
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
        # Check for both canonical 'Close' and suffixed 'Close_TICKER'
        for col_name in [name, f"{name}_{ticker}"]:
            if col_name in df.columns:
                s = df[col_name]
                return pd.to_numeric(s, errors='coerce').dropna()
                
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
    # This is still here in case you want to uncomment and run it separately.
    HISTORICAL_FETCH_DELAY = 60 # seconds 
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
    
    # 1. Fetch 1-day/1-min data using a single bulk call
    live_tracker.update_all()
    
    for t in cfg.TICKERS:
        # 2. Save the live data from memory to `_live.csv`
        # This step is still done individually because we need to save one file per ticker
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
    # --- FINAL FIX: COMMENTING OUT initial_setup to bypass perpetual rate limit failures ---
    # initial_setup() 
    
    # FIX 2 (FINAL): Increased cooldown to 300 seconds (5 minutes)
    COOLDOWN_SECONDS = 300 
    logger.info(f"Skipping historical setup. Waiting {COOLDOWN_SECONDS}s before starting live loop...")
    for _ in range(COOLDOWN_SECONDS):
        if STOP_EVENT.is_set():
            break
        time.sleep(1)
    
    # --- LOG UPDATED TO REFLECT 5 MINUTE CYCLE ---
    logger.info("Starting Phase 2 pipeline loop (updating 5-min interval)...")
    while not STOP_EVENT.is_set():
        try:
            # 2. Every 5 minutes, get 1-day/1-min data
            process_cycle()
        except Exception as e:
            logger.exception(f"Error in process cycle: {e}")
        
        # Wait for the full 5-minute refresh interval
        for _ in range(max(1, int(cfg.REFRESH_INTERVAL))):
            if STOP_EVENT.is_set():
                break
            time.sleep(1)
            
    logger.info("Pipeline stopped")

if __name__ == "__main__":
    run_forever()