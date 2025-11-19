#!/usr/bin/env python3
"""
phase2_pipeline.py — Phase 2 live stock pipeline (FINAL STABLE VERSION)

Features:
- **DUAL-INTERVAL FETCH:** This pipeline is now fully self-healing and stable.
- **LIVE (5-Min):** Runs every 5 minutes (300s). Performs a **SINGLE API CALL**
  for all tickers to strictly limit request frequency. Uses period="1d" for
  correct yfinance syntax.
- **HISTORICAL (30-Min):** Every 3rd run (i.e., every 15 mins), it also
  runs a "Smart Incremental Backfill" to heal the `_hist.csv` (30-min data).
- **ADAPTIVE COOLDOWN:** If the single live fetch fails due to a rate limit, the
  pipeline stalls the next cycle for 30 minutes to guarantee the API limit resets.
"""

import os 
import time
import threading
import logging
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import signal
import math
import requests.exceptions 
import random 

# --- Configuration ---
load_dotenv()

class Config:
    TICKERS = os.getenv("TICKERS", "AAPL,AMZN,GOOGL,MSFT,TSLA").split(",")
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    # --- Live loop runs every 5 minutes (300s) for the strict single-call requirement ---
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 300)) 
    
    CONTEXT_TICKERS = ["SPY", "QQQ", "^VIX"] 
    ALL_FETCH_TICKERS = list(set(TICKERS) | set(CONTEXT_TICKERS))
    
    # Batch size is for the *historical* fetch in run_all.py
    TICKER_BATCH_SIZE = int(os.getenv("TICKER_BATCH_SIZE", 8)) 

    CSV_RETENTION_DAYS = int(os.getenv("CSV_RETENTION_DAYS", 30))
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "phi3:mini")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY") 
    NEWS_API_URL = "https://newsapi.org/v2/everything" 
    NEWS_REFRESH_INTERVAL = int(os.getenv("NEWS_REFRESH_INTERVAL", 600))
    
    # --- Rate Limit Defense ---
    EMERGENCY_COOLDOWN = int(os.getenv("EMERGENCY_COOLDOWN", 1800)) # 1800 seconds = 30 minutes
    
    # --- Proxy Configuration (Removed for single-call safety) ---
    PROXIES = [] # List is empty to enforce fetching from local IP only


cfg = Config()

# --- *** FIX 1: Imports moved here to prevent circular dependency *** ---
import signals.technical as technical
import signals.volume as volume
import signals.market_context as market_context
import signals.calendar as calendar
import signals.sentiment as sentiment 
# --- END FIX ---

# --- POTENTIAL BUG FIX: Use the configured DATA_DIR ---
DATA_DIR_PATH = Path(cfg.DATA_DIR)
# --- END FIX ---
DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
# --- NEW: Define the alerts output file ---
ALERTS_JSON_FILE = DATA_DIR_PATH / "live_alerts.json"

# --- FIX: Moved RTH definitions here to be globally available BEFORE instantiation ---
RTH_START = '14:30'
RTH_END = '21:00'
# --- END FIX ---

# -----------------------
# Logging & shutdown 
# -----------------------
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
)
logger = logging.getLogger("phase2_pipeline")
STOP_EVENT = threading.Event()
# --- NEW GLOBAL: Track next wait time for adaptive cooldown ---
NEXT_WAIT_TIME = cfg.REFRESH_INTERVAL

def _shutdown(signum, frame):
    logger.info("Shutdown signal received, stopping pipeline...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# -----------------------
# Ollama LLM interface (Omitted for brevity)
# -----------------------
class LLMInterface:
    def __init__(self, model_name: str = cfg.OLLAMA_MODEL_NAME, url: str = cfg.OLLAMA_API_URL):
        self.model_name = model_name
        self.url = url
    def generate(self, prompt: str, timeout: int = 120) -> str:
        try:
            payload = { "model": self.model_name, "prompt": prompt, "stream": False }
            resp = requests.post(self.url, json=payload, timeout=timeout) 
            resp.raise_for_status()
            data = resp.json()
            for key in ("response", "completion", "output", "text"):
                if key in data:
                    return str(data.get(key) or "").strip()
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Ollama Error: {e}")
            return f"[Ollama Exception] {e}"
llm = LLMInterface()


# -----------------------
# Column flattening & canonicalization (Unchanged)
# -----------------------
def flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        all_tickers = cfg.ALL_FETCH_TICKERS
        new_cols = []
        for col_level1, col_level2 in df.columns:
            # Handle cases where level 2 is the ticker
            if col_level2 and col_level2.upper() in all_tickers:
                new_cols.append(f"{col_level1}_{col_level2}")
            # Handle cases where level 1 is the ticker (from group_by='ticker')
            elif col_level1 and col_level1.upper() in all_tickers:
                 new_cols.append(f"{col_level2}_{col_level1}")
            elif not col_level2:
                new_cols.append(col_level1)
            else:
                new_cols.append(f"{col_level1}_{col_level2}")
        
        # This logic is for the standard yf.download format (cols = [('Open', 'AAPL'), ('Close', 'AAPL')])
        if not new_cols and df.columns.nlevels == 2:
             # Standard format
             new_cols = [f"{L1}_{L2}" for L1, L2 in df.columns]

        df.columns = new_cols
    
    df = df.loc[:, ~df.columns.duplicated()]

    try:
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
    except Exception:
        df.index = pd.to_datetime(df.index, errors='coerce')
        if df.index.tz is None:
            try: df.index = df.index.tz_localize('UTC')
            except Exception: pass

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
# CSV loader/saver helpers (Unchanged)
# -----------------------
def load_csv(ticker: str, file_suffix: str) -> pd.DataFrame:
    """Generic CSV loader."""
    path = DATA_DIR_PATH / f"{ticker}{file_suffix}"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        df.index = df.index.tz_convert('UTC')
        df = flatten_columns(df, ticker)
        return df
    except Exception as e:
        logger.warning(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()

# --- THIS IS THE UPDATED ATOMIC FUNCTION (Unchanged) ---
def save_csv(df: pd.DataFrame, ticker: str, file_suffix: str):
    """Saves a CSV file with suffixed columns ATOMICALLY."""
    path = DATA_DIR_PATH / f"{ticker}{file_suffix}"
    path_tmp = DATA_DIR_PATH / f"{ticker}{file_suffix}.tmp"
    
    df_to_save = df.copy()
    
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    
    for col in base_cols:
        if col not in df_to_save.columns:
            if col == 'Volume':
                df_to_save['Volume'] = 0 
            elif 'Close' in df_to_save.columns: 
                df_to_save[col] = df_to_save['Close'] 
            else:
                continue 

    for col in base_cols:
        if col in df_to_save.columns and f"{col}_{ticker}" not in df_to_save.columns:
            df_to_save[f"{col}_{ticker}"] = df_to_save[col]
            if col in df_to_save.columns:
                del df_to_save[col]
    try:
        df_to_save.to_csv(path_tmp) 
        os.replace(path_tmp, path)
        logger.info(f"Saved {file_suffix} CSV for {ticker} ({len(df_to_save)} rows total)")
    except Exception as e:
        logger.warning(f"Failed to save {file_suffix} CSV for {ticker}: {e}")
        if path_tmp.exists():
            try:
                os.remove(path_tmp)
            except Exception as e_clean:
                logger.error(f"Failed to clean up {path_tmp}: {e_clean}")

# -----------------------
# LiveTracker (MODIFIED for Single API Call)
# -----------------------
class LiveTracker:
    def __init__(self, tickers):
        self.all_tickers = cfg.ALL_FETCH_TICKERS 
        self._hist_buffers = {t: pd.DataFrame() for t in self.all_tickers}
        self._live_buffers = {t: pd.DataFrame() for t in self.all_tickers}
        self.lock = threading.Lock()
        
        self.load_all_buffers()

    def load_all_buffers(self):
        """Loads all _hist and _live files into memory on startup."""
        logger.info("Loading existing data into memory buffers...")
        with self.lock:
            for t in self.all_tickers:
                self._hist_buffers[t] = load_csv(t, "_hist.csv")
                self._live_buffers[t] = load_csv(t, "_live.csv")

    def get_last_timestamp(self, ticker: str, file_suffix: str) -> datetime | None:
        """Finds the last timestamp for a ticker in the specified buffer."""
        buffer = self._hist_buffers if file_suffix == "_hist.csv" else self._live_buffers
        with self.lock:
            df = buffer.get(ticker)
            if df is not None and not df.empty:
                return df.index.max()
        return None
        
    def get_hist_buffer(self, ticker: str) -> pd.DataFrame:
        with self.lock:
            buf = self._hist_buffers.get(ticker)
            return buf.copy() if buf is not None else pd.DataFrame()

    def get_live_buffer(self, ticker: str) -> pd.DataFrame:
        with self.lock:
            buf = self._live_buffers.get(ticker)
            return buf.copy() if buf is not None else pd.DataFrame()

    # --- UPDATED FUNCTION FOR SINGLE API CALL ---
    def update_live_buffers(self):
        """
        Fetches 1 day of 5-MINUTE data for ALL tickers using a single API call,
        reducing high volume risk and strictly limiting request frequency.
        """
        logger.info(f"Starting SINGLE API CALL live fetch for ALL {len(self.all_tickers)} tickers...")
        
        # NEW: Perform ONE bulk download for ALL tickers
        try:
            # FIX: Using period="1d" for correct yfinance syntax and relying on 5-min cooldown
            df_bulk = yf.download(self.all_tickers, period="1d", interval="5m", progress=False, group_by='ticker')
        except requests.exceptions.RequestException as e:
            logger.error(f"FATAL API RATE LIMIT FAILURE during single live fetch: {e}")
            raise # Crash the cycle to trigger the adaptive cooldown
        except Exception as e:
            logger.error(f"General error during single live fetch: {e}")
            raise # Crash the cycle

        if df_bulk is None or df_bulk.empty:
            logger.warning("No live data returned (market may be closed or data volume too low).")
            return

        df_bulk.index = pd.to_datetime(df_bulk.index, utc=True, errors='coerce')
        
        with self.lock:
            for t in self.all_tickers:
                df_ticker = pd.DataFrame()
                
                # Check if we got a MultiIndex (standard for >1 tickers)
                if isinstance(df_bulk.columns, pd.MultiIndex):
                    if t in df_bulk.columns:
                        df_ticker = df_bulk[t].copy()
                    else:
                        logger.warning(f"Ticker {t} not found in bulk download result.")
                        continue
                # If we got a simple DataFrame (e.g., if ALL_FETCH_TICKERS only had one item)
                elif len(self.all_tickers) == 1:
                    df_ticker = df_bulk.copy()
                else:
                     logger.warning(f"Unexpected DataFrame structure for {t} after bulk fetch.")
                     continue
                
                # Clean and save the individual ticker data
                df_ticker.columns.name = None
                df_ticker.columns = [c.capitalize() for c in df_ticker.columns]
                
                if not df_ticker.empty:
                    self._live_buffers[t] = df_ticker 
                    logger.debug(f"Updated live (5-min) buffer for {t} ({len(df_ticker)} rows).")
                    
                    if t in cfg.TICKERS:
                        save_csv(df_ticker, t, "_live.csv")

        logger.info("Completed single-call live fetch.")

    # ... (update_historical_buffers is unchanged) ...
    def update_historical_buffers(self):
        """
        Runs the "Smart Incremental Backfill" for all 30-min _hist.csv files.
        """
        logger.info("Starting Smart Historical Backfill (30-min data)...")
        end_date = datetime.now(timezone.utc)
        
        for ticker in self.all_tickers:
            if STOP_EVENT.is_set(): return
            
            df_hist = self.get_hist_buffer(ticker)
            start_date = None
            
            if df_hist.empty:
                start_date = end_date - timedelta(days=cfg.CSV_RETENTION_DAYS)
                logger.info(f"No 30-min data for {ticker}. Fetching full {cfg.CSV_RETENTION_DAYS}-day history.")
            else:
                last_ts = df_hist.index.max()
                # Check if the last data is older than 30-mins ago
                if last_ts < end_date - timedelta(minutes=30):
                    start_date = last_ts + timedelta(minutes=30)
                    logger.info(f"Fetching new 30-min data for {ticker} since {start_date}...")
                else:
                    logger.info(f"30-min data for {ticker} is up-to-date. Skipping.")
            
            if start_date:
                try:
                    # --- THIS IS THE STABLE 30-MINUTE CALL ---
                    df_chunk = yf.download(
                        ticker, 
                        start=start_date.strftime("%Y-%m-%d"), 
                        end=end_date.strftime("%Y-%m-%d"),
                        interval="30m", 
                        progress=False
                    )
                    
                    if not df_chunk.empty:
                        df_chunk.index.name = "Datetime"
                        df_chunk = flatten_columns(df_chunk, ticker)
                        
                        with self.lock:
                            # --- APPEND the new 30-min chunk ---
                            df_combined = pd.concat([df_hist, df_chunk])
                            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                            df_combined = df_combined.sort_index()
                            
                            # Prune data older than 30 days
                            cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.CSV_RETENTION_DAYS)
                            df_combined = df_combined[df_combined.index >= cutoff]
                            
                            self._hist_buffers[ticker] = df_combined
                            logger.info(f"Appended {len(df_chunk)} new 30-min rows for {ticker}.")
                            save_csv(df_combined, ticker, "_hist.csv")
                    else:
                        logger.info(f"No new 30-min data found for {ticker}.")
                        
                    if not STOP_EVENT.is_set():
                        time.sleep(5) # 5-second safety delay between each *historical* fetch

                except Exception as e:
                    logger.error(f"Failed to backfill 30-min data for {ticker}: {e}")
                    
# -----------------------
# Modular Alert Engine (Unchanged)
# -----------------------
class AlertEngine:
    def __init__(self, cfg, live_tracker):
        self.cfg = cfg
        self.live_tracker = live_tracker

    def evaluate_live_alerts(self, ticker: str) -> tuple[list[str], dict]:
        # --- Get 5-MINUTE data for live alerts ---
        df = self.live_tracker.get_live_buffer(ticker)
        if df.empty or len(df) < 21:
            return [], {} 
        
        df_rth = df.between_time(RTH_START, RTH_END)
        if df_rth.empty or len(df_rth) < 21:
            logger.debug(f"Not enough RTH 5-min data for {ticker} to run alerts.")
            return [], {} 

        all_alerts = []
        all_indicators = {}
        
        try:
            alerts, indicators = technical.compute_signals(df_rth, ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
            alerts, indicators = volume.compute_signals(df_rth, ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
            alerts, indicators = calendar.compute_signals(df_rth, ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)

            alerts, indicators = market_context.compute_signals(df_rth, ticker, self.live_tracker, "get_live_buffer")
            all_alerts.extend(alerts)
            all_indicators.update(indicators)

            alerts, indicators = sentiment.compute_signals(ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
        except Exception as e:
            logger.error(f"Error computing signals for {ticker} (live): {e}")
            
        return all_alerts, all_indicators
        
    def evaluate_historical_alerts(self, ticker: str) -> tuple[list[str], dict]:
        # --- Get 30-MINUTE data for historical alerts ---
        df = self.live_tracker.get_hist_buffer(ticker)
        if df.empty or len(df) < 21:
            return [], {} 
        
        all_alerts = []
        all_indicators = {}

        try:
            alerts, indicators = technical.compute_signals(df, ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)

            alerts, indicators = volume.compute_signals(df, ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)

            alerts, indicators = calendar.compute_signals(df, ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
            alerts, indicators = market_context.compute_signals(df, ticker, self.live_tracker, "get_hist_buffer") 
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
        except Exception as e:
            logger.error(f"Error computing signals for {ticker} (historical): {e}")

        return all_alerts, all_indicators

# --- Instances must be created after class definitions ---
live_tracker = LiveTracker(cfg.TICKERS)
alert_engine = AlertEngine(cfg, live_tracker) 

# -----------------------
# MODIFIED: News/Sentiment Fetcher (Unchanged)
# -----------------------
def news_fetcher_task():
    """ Runs news fetch serially. """
    logger.info(f"Sentiment: Starting news/sentiment fetch...")
    try:
        sentiment.fetch_news_and_analyze(cfg.TICKERS, cfg.NEWS_API_URL, cfg.NEWS_API_KEY)
    except Exception as e:
        logger.error(f"Error in news_fetcher_task: {e}")
    logger.info("Sentiment: News fetch complete.")

# -----------------------
# Pipeline orchestration (MODIFIED for Adaptive Cooldown)
# -----------------------
def process_cycle(run_count: int):
    """
    Single pipeline cycle (every N seconds):
    1. Fetches news.
    2. Fetches 1-day/5-min data (triggers adaptive cooldown on failure).
    3. Runs alerts on the new data AND SAVES THEM.
    """
    global NEXT_WAIT_TIME
    
    logger.info(f"Starting new cycle (Run #{run_count})... Next wait: {NEXT_WAIT_TIME}s")
    
    # 1. Run news fetch (Low risk, run first)
    news_fetcher_task()
    
    live_fetch_succeeded = True
    
    # 2. Fetch 5-min live data (High risk, must handle exceptions to trigger stall)
    try:
        live_tracker.update_live_buffers()
    except requests.exceptions.RequestException as e:
        logger.error(f"FATAL API RATE LIMIT FAILURE during single live fetch. Aborting cycle: {e}")
        live_fetch_succeeded = False 
    except Exception as e:
        logger.error(f"Unexpected error during single live fetch. Aborting cycle: {e}")
        live_fetch_succeeded = False 

    # 3. Check for Failure and ADJUST THE NEXT WAIT TIME
    if not live_fetch_succeeded:
        # If API failed, increase the cooldown for the next cycle
        if NEXT_WAIT_TIME != cfg.EMERGENCY_COOLDOWN:
            logger.critical(f"RATE LIMIT DETECTED. Setting NEXT_WAIT_TIME to EMERGENCY_COOLDOWN ({cfg.EMERGENCY_COOLDOWN}s).")
            NEXT_WAIT_TIME = cfg.EMERGENCY_COOLDOWN
            # We skip the rest of the cycle since data is stale/missing
            return 
    else:
        # If successful, reset back to the normal 5-minute interval
        if NEXT_WAIT_TIME != cfg.REFRESH_INTERVAL:
            logger.info("Live fetch SUCCESS. Resetting NEXT_WAIT_TIME to normal interval (5 min).")
            NEXT_WAIT_TIME = cfg.REFRESH_INTERVAL

    # 4. Run remaining operations (Historical Backfill and Alerts)
    
    # Fetch 30-min historical data (every 3rd run, i.e., ~15 mins now)
    if run_count % 3 == 0:
        live_tracker.update_historical_buffers()
    else:
        logger.info("Skipping 30-min historical backfill this cycle.")
    
    # Run alerts and SAVE them to JSON
    all_alerts = {}
    for t in cfg.TICKERS:
        if STOP_EVENT.is_set():
            break
            
        live_alerts, live_indicators = alert_engine.evaluate_live_alerts(t)
        hist_alerts, hist_indicators = alert_engine.evaluate_historical_alerts(t)
        
        all_alerts[t] = {
            "live": live_alerts,
            "live_indicators": live_indicators,
            "historical": hist_alerts,
            "historical_indicators": hist_indicators
        }
        
        if live_alerts:
            logger.info(f"{t} Modular Live (5-Min) Alerts: {live_alerts}")

    # Atomic write for the alerts file
    if not STOP_EVENT.is_set() and all_alerts:
        try:
            ALERTS_JSON_FILE_TMP = DATA_DIR_PATH / "live_alerts.json.tmp"
            with open(ALERTS_JSON_FILE_TMP, 'w', encoding='utf-8') as f:
                json.dump(all_alerts, f, indent=2)
            os.replace(ALERTS_JSON_FILE_TMP, ALERTS_JSON_FILE)
            logger.info(f"Successfully saved alerts for {len(all_alerts)} tickers to {ALERTS_JSON_FILE}")
        except Exception as e:
            logger.error(f"Failed to save alerts JSON file: {e}")
    
    logger.info(f"Completed cycle. Next run in {NEXT_WAIT_TIME}s.")


def run_forever():
    
    logger.info("Running initial data backfill on startup...")
    try:
        # 1. Run low-risk actions first
        news_fetcher_task()
        live_tracker.update_historical_buffers()
        
        # --- CRITICAL FIX: The immediate live_tracker.update_live_buffers() is REMOVED ---
        # The first live fetch MUST occur inside the main loop after the initial wait period.

    except Exception as e:
        logger.exception(f"Error in initial data fetch: {e}")

    logger.info(f"Starting Phase 2 pipeline loop (updating {cfg.REFRESH_INTERVAL // 60}-min interval)...")
    run_count = 1
    
    while not STOP_EVENT.is_set():
        cycle_start_time = time.time()
        
        # 1. PROCESS
        try:
            # We removed the immediate live fetch from startup, so the first process_cycle
            # will handle the first live fetch after the run_all.py cooldown.
            process_cycle(run_count)
        except Exception as e:
            logger.exception(f"Error in process cycle: {e}")
        
        run_count += 1
        
        # 2. WAIT
        if STOP_EVENT.is_set():
            break
            
        wait_time = NEXT_WAIT_TIME
        
        logger.info(f"Waiting for {wait_time}s until next live (5-min) fetch...")
        for _ in range(wait_time):
            if STOP_EVENT.is_set():
                break
            time.sleep(1)
            
    logger.info("Pipeline stopped")

if __name__ == "__main__":
    run_forever()