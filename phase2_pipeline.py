#!/usr/bin/env python3
"""
phase2_pipeline.py — Phase 2 live stock pipeline (FINAL STABLE VERSION)

Features:
- **FILE STRUCTURE (NEW):** Data is stored in a single unified file (<TICKER>.csv)
  containing 30 days of 5-minute bars.
- **LIVE (5-Min):** Runs every 5 minutes (300s). Performs a **SINGLE API CALL**
  for all tickers, requesting period="1d" for correct syntax.
- **STARTUP OPTIMIZATION (NEW):** Runs initial alert/RAG processing immediately on
  startup so the dashboard is fully populated with historical context.
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
import pandas as pd
import yfinance as yf
import signal
import math
import requests.exceptions 
import random 

# --- Configuration FIX: Import Config from centralized file ---
from config import cfg

# --- FIX: Imports moved here to prevent circular dependency ---
import signals.technical as technical
import signals.volume as volume
import signals.market_context as market_context
import signals.calendar as calendar
import signals.sentiment as sentiment 
# --- END FIX ---

# --- Use the configured DATA_DIR ---
DATA_DIR_PATH = Path(cfg.DATA_DIR)
# --- END FIX ---
DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
ALERTS_JSON_FILE = DATA_DIR_PATH / "live_alerts.json"

# --- RTH definitions remain for local use ---
RTH_START = '14:30'
RTH_END = '21:00'

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
# Ollama LLM interface (Defined after cfg import)
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
# CSV loader/saver helpers (MODIFIED for Unified File)
# -----------------------
def load_unified_csv(ticker: str) -> pd.DataFrame:
    """Loads the single unified <TICKER>.csv file."""
    path = DATA_DIR_PATH / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame() 
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
        df.index = df.index.tz_convert('UTC')
        df = flatten_columns(df, ticker)
        return df
    except Exception as e:
        logger.warning(f"Failed to read unified CSV {path}: {e}")
        return pd.DataFrame()

def save_unified_csv(df: pd.DataFrame, ticker: str):
    """Saves the unified <TICKER>.csv file ATOMICALLY."""
    path = DATA_DIR_PATH / f"{ticker}.csv"
    path_tmp = DATA_DIR_PATH / f"{ticker}.csv.tmp"
    
    df_to_save = df.copy()
    
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    
    # Ensure all base columns exist for the canonicalization below
    for col in base_cols:
        if col not in df_to_save.columns:
            # Attempt to set Volume to 0, or skip
            if col == 'Volume':
                df_to_save['Volume'] = 0 
            else:
                continue 

    # Ensure suffixed columns are saved
    for col in base_cols:
        if col in df_to_save.columns and f"{col}_{ticker}" not in df_to_save.columns:
            df_to_save[f"{col}_{ticker}"] = df_to_save[col]
            # Delete the base column to prevent duplication, trusting the suffixed column
            if col in df_to_save.columns:
                del df_to_save[col]

    try:
        # Write to temp file
        df_to_save.to_csv(path_tmp) 
        # Atomically replace
        os.replace(path_tmp, path)
        logger.info(f"Saved unified CSV for {ticker} ({len(df_to_save)} rows total)")
    except Exception as e:
        logger.warning(f"Failed to save unified CSV for {ticker}: {e}")
        # Clean up the tmp file on failure
        if path_tmp.exists():
            try:
                os.remove(path_tmp)
            except Exception as e_clean:
                logger.error(f"Failed to clean up {path_tmp}: {e_clean}")
                
# -----------------------
# LiveTracker (MODIFIED for Incremental Update)
# -----------------------
class LiveTracker:
    # FIX: Removed the unused 'tickers' argument for code hygiene
    def __init__(self): 
        self.all_tickers = cfg.ALL_FETCH_TICKERS 
        # Buffers now store the single unified DataFrame per ticker
        self._unified_buffers = {t: pd.DataFrame() for t in self.all_tickers}
        self.lock = threading.Lock()
        
        self.load_all_buffers()

    def load_all_buffers(self):
        """Loads the single unified file into memory buffers on startup."""
        logger.info("Loading existing unified data into memory buffers...")
        with self.lock:
            for t in self.all_tickers:
                self._unified_buffers[t] = load_unified_csv(t)

    def get_last_timestamp(self, ticker: str) -> datetime | None:
        """Finds the last timestamp for a ticker in the unified buffer."""
        with self.lock:
            df = self._unified_buffers.get(ticker)
            if df is not None and not df.empty:
                return df.index.max()
        return None
        
    def get_unified_buffer(self, ticker: str) -> pd.DataFrame:
        """Returns a copy of the single unified DataFrame."""
        with self.lock:
            buf = self._unified_buffers.get(ticker)
            # IMPORTANT: The unified buffer is used for *both* live and historical analysis now.
            return buf.copy() if buf is not None else pd.DataFrame()

    # --- UPDATED FUNCTION FOR INCREMENTAL SINGLE API CALL ---
    def update_live_buffers(self):
        """
        Fetches the missing 5-minute bars since the last recorded timestamp for ALL tickers
        using a single API call, and appends the data to the unified file.
        """
        logger.info(f"Starting SINGLE API CALL incremental update for ALL {len(self.all_tickers)} tickers...")
        
        end_date = datetime.now(timezone.utc)
        earliest_start_time = end_date - timedelta(days=cfg.CSV_RETENTION_DAYS)
        
        try:
            # FIX: Using period="1d" for correct yfinance syntax and single bulk fetch
            df_bulk = yf.download(cfg.ALL_FETCH_TICKERS, period="1d", interval="5m", progress=False, group_by='ticker')
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
                df_new = pd.DataFrame()
                
                # --- Extract and Clean New Data ---
                if isinstance(df_bulk.columns, pd.MultiIndex) and t in df_bulk.columns:
                    df_new = df_bulk[t].copy()
                elif len(cfg.ALL_FETCH_TICKERS) == 1:
                    df_new = df_bulk.copy()
                else:
                    continue

                if df_new.empty:
                    continue

                # Get the existing buffer and its last timestamp
                df_existing = self._unified_buffers.get(t, pd.DataFrame())
                last_ts = df_existing.index.max() if not df_existing.empty else None

                # --- Combine Data ---
                df_new.index.name = "Datetime"
                df_new = flatten_columns(df_new, t)
                
                # Filter new data to only include bars *newer* than the last timestamp
                if last_ts is not None:
                    # Filter out old data that is already in the existing file
                    df_new = df_new[df_new.index > last_ts] 
                    # If the data is fully up to date, df_new might now be empty
                    if df_new.empty:
                        logger.debug(f"Unified data for {t} is already up to date.")
                        continue
                
                # 1. Combine new and existing data
                df_combined = pd.concat([df_existing, df_new])
                # 2. Drop duplicates (keeping the latest entry if any overlap occurs)
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                # 3. Sort by index
                df_combined = df_combined.sort_index()
                
                # 4. Prune old data (e.g., older than 30 days)
                cutoff = end_date - timedelta(days=cfg.CSV_RETENTION_DAYS)
                df_combined = df_combined[df_combined.index >= cutoff]

                # --- Update and Save ---
                self._unified_buffers[t] = df_combined
                save_unified_csv(df_combined, t)

        logger.info("Completed unified data fetch and save.")

# -----------------------
# Modular Alert Engine (MODIFIED to use Unified Buffer)
# -----------------------
class AlertEngine:
    def __init__(self, cfg, live_tracker):
        self.cfg = cfg
        self.live_tracker = live_tracker

    def evaluate_live_alerts(self, ticker: str) -> tuple[list[str], dict]:
        # --- Get UNIFIED 5-MINUTE data for live alerts ---
        df = self.live_tracker.get_unified_buffer(ticker)
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

            alerts, indicators = market_context.compute_signals(df_rth, ticker, self.live_tracker, "get_unified_buffer")
            all_alerts.extend(alerts)
            all_indicators.update(indicators)

            alerts, indicators = sentiment.compute_signals(ticker)
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
        except Exception as e:
            logger.error(f"Error computing signals for {ticker} (live): {e}")
            
        return all_alerts, all_indicators
        
    def evaluate_historical_alerts(self, ticker: str) -> tuple[list[str], dict]:
        # --- Historical alerts now use UNIFIED 5-MINUTE data, which must be resampled ---
        df_unified = self.live_tracker.get_unified_buffer(ticker)
        if df_unified.empty or len(df_unified) < 21:
            return [], {} 

        # --- NEW: Resample 5-min data to 30-min for consistent historical analysis ---
        df = df_unified.resample('30min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna(subset=['Close'])
        
        if df.empty or len(df) < 21:
             logger.debug(f"Not enough resampled 30-min data for {ticker} to run historical alerts.")
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
            
            alerts, indicators = market_context.compute_signals(df, ticker, self.live_tracker, "get_unified_buffer")
            all_alerts.extend(alerts)
            all_indicators.update(indicators)
            
        except Exception as e:
            logger.error(f"Error computing signals for {ticker} (historical): {e}")

        return alerts, indicators

# --- Instances (Now placeholders, initialized in run_forever) ---
live_tracker = None
alert_engine = None 

# -----------------------
# MODIFIED: News/Sentiment Fetcher
# -----------------------
def news_fetcher_task():
    """ Runs news fetch serially. """
    logger.info(f"Sentiment: Starting news/sentiment fetch...")
    try:
        # --- MODIFIED CALL: Pass DATA_DIR explicitly ---
        sentiment.fetch_news_and_analyze(
            cfg.TICKERS, 
            cfg.NEWS_API_URL, 
            cfg.NEWS_API_KEY,
            cfg.DATA_DIR # <-- PASS DATA_DIR
        )
    except Exception as e:
        logger.error(f"Error in news_fetcher_task: {e}")
    logger.info("Sentiment: News fetch complete.")

# --- NEW HELPER FUNCTION FOR STARTUP PROCESSING ---
def process_initial_alerts_and_rag():
    """
    Runs the alert engine on the initial historical data to populate the
    live_alerts.json file immediately on startup.
    """
    logger.info("Running initial historical alert processing for dashboard...")
    all_alerts = {}
    
    # Must use global alert_engine, which is initialized in run_forever()
    if alert_engine is None:
        logger.error("AlertEngine not initialized. Cannot run initial processing.")
        return

    for t in cfg.TICKERS:
        if STOP_EVENT.is_set():
            break
            
        # NOTE: Live alerts will be empty as data is not RTH recent, but we run 
        # the historical alerts on the resampled data.
        hist_alerts, hist_indicators = alert_engine.evaluate_historical_alerts(t)
        
        # We need a clean structure for the dashboard to read
        all_alerts[t] = {
            "live": [], # Set live alerts to empty list
            "live_indicators": {}, # Set live indicators to empty dict
            "historical": hist_alerts,
            "historical_indicators": hist_indicators
        }
        
    # Atomic write for the alerts file
    if not STOP_EVENT.is_set() and all_alerts:
        try:
            ALERTS_JSON_FILE_TMP = DATA_DIR_PATH / "live_alerts.json.tmp"
            with open(ALERTS_JSON_FILE_TMP, 'w', encoding='utf-8') as f:
                json.dump(all_alerts, f, indent=2)
            os.replace(ALERTS_JSON_FILE_TMP, ALERTS_JSON_FILE)
            logger.info(f"Successfully saved initial alerts for dashboard.")
        except Exception as e:
            logger.error(f"Failed to save alerts JSON file: {e}")
# --- END NEW HELPER ---


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

    # 4. Run remaining operations (Alerts)
    
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
    global live_tracker, alert_engine 
    
    # --- CRITICAL FIX: INSTANTIATE OBJECTS HERE ---
    # LiveTracker now takes NO ARGUMENT, relying solely on config.cfg
    logger.info("Initializing LiveTracker and AlertEngine...")
    live_tracker = LiveTracker() # FIXED: Removed cfg.TICKERS argument
    alert_engine = AlertEngine(cfg, live_tracker)
    # --- END CRITICAL FIX ---

    logger.info("Running initial data backfill on startup...")
    try:
        # 1. Run low-risk actions first (News fetch and historical buffer load)
        news_fetcher_task()
        
        # 2. NEW STEP: Process alerts on existing historical data immediately
        # This populates the dashboard with charts and historical alerts/RAG context.
        process_initial_alerts_and_rag()
        
    except Exception as e:
        logger.exception(f"Error in initial data fetch: {e}")

    logger.info(f"Starting Phase 2 pipeline loop (updating {cfg.REFRESH_INTERVAL // 60}-min interval)...")
    run_count = 1
    
    while not STOP_EVENT.is_set():
        cycle_start_time = time.time()
        
        # 1. PROCESS
        try:
            # The first process_cycle handles the first live fetch after startup
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
