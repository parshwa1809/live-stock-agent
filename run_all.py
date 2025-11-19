#!/usr/bin/env python3
"""
run_all.py
Master launcher for Live Stock Agent Phase 2.

Features:
- **One-Time Setup (NEW):** Safely fetches 30 days of 5-minute data for all
  tickers SERIALLY, with a 5-minute cooldown between *each* ticker.
- **Service Launch:** Starts the three main services in parallel.
- **File Structure:** Merges historical and live data into a single unified file (<TICKER>.csv).
"""

import threading
import subprocess
import signal
import logging
import sys
from pathlib import Path
from time import sleep
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import math

# --- Import key components from the pipeline itself ---
import yfinance as yf
import pandas as pd
# We import Config from the pipeline module
from phase2_pipeline import Config 

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
)
logger = logging.getLogger("run_all")

STOP_EVENT = threading.Event()
threads = []
# --- Throttling Configuration ---
HISTORICAL_DELAY_SECONDS = 300 # 5-minute safe delay between EACH ticker fetch

# -----------------------
# Graceful shutdown handler
# -----------------------
def shutdown_handler(signum, frame):
    logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# =====================================================
# SECTION 1: ONE-TIME INITIAL DATA FETCH (ULTRA-SAFE)
# =====================================================
def fetch_and_save_initial_data(ticker: str, cfg: Config, data_dir: Path):
    """
    Fetches 30-day, 5-minute initial data for a single ticker and saves it to <ticker>.csv.
    """
    end = datetime.now(timezone.utc)
    # Fetch exactly 30 days of data (e.g., today minus 30 days)
    start = end - timedelta(days=cfg.CSV_RETENTION_DAYS) 
    
    logger.info(f"[Setup] Fetching 30-DAY, 5-MINUTE data for TICKER: {ticker}...")
    
    try:
        # Fetch the single ticker data
        df_ticker = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                             interval="5m", progress=False, group_by='ticker')
        
        if df_ticker is None or df_ticker.empty:
            logger.warning(f"[Setup] No initial data returned for {ticker}")
            return

        # --- NEW ROBUSTNESS FIX: Handle MultiIndex returned by YFinance for single ticker ---
        if isinstance(df_ticker.columns, pd.MultiIndex):
            # Drop the outer level (which is usually the redundant ticker name)
            df_ticker.columns = df_ticker.columns.droplevel(0)
            
        # 1. Normalize columns to Title Case (e.g., 'close' -> 'Close')
        df_ticker.columns = df_ticker.columns.str.capitalize()
        # 2. Drop rows with no price (This should now work because 'Close' is capitalized)
        df_ticker = df_ticker.dropna(subset=['Close']) 
        # --- END FIX ---
            
        if df_ticker.empty:
            logger.warning(f"[Setup] No usable data found for {ticker} after cleaning.")
            return

        df_ticker.index.name = "Datetime"
        
        # Canonicalize and save the unified file
        df_to_save = df_ticker.copy()
        
        # Ensure base columns are saved with suffixes for safety/consistency, 
        # using the convention <Col>_<TICKER>
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            # Check for the canonicalized column name
            if col in df_to_save.columns and f"{col}_{ticker}" not in df_to_save.columns:
                df_to_save[f"{col}_{ticker}"] = df_to_save[col]
        
        # IMPORTANT: The single file is now named <TICKER>.csv
        path = data_dir / f"{ticker}.csv"
        df_to_save.to_csv(path)
        logger.info(f"[Setup] ✅ Successfully saved {ticker}.csv ({len(df_to_save)} rows) at 5-min interval.")

    except Exception as e:
        logger.error(f"[Setup] ❌ Failed to fetch initial data for {ticker}: {e}")

def run_initial_data_fetch():
    """
    Checks if unified data files exist. If not, fetches them in ultra-safe serial mode.
    """
    logger.info("="*50)
    logger.info("Checking for required unified data files...")
    
    # Need to load config first
    load_dotenv()
    cfg = Config()
    data_dir = Path(cfg.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Use ALL_FETCH_TICKERS to ensure context tickers are included
    all_tickers = cfg.ALL_FETCH_TICKERS 
    tickers_to_fetch = []
    
    for ticker in all_tickers:
        # IMPORTANT: Check for the single, unified file: <TICKER>.csv
        path = data_dir / f"{ticker}.csv"
        # Also check for the old structure for backwards compatibility during transition
        if not path.exists() and not (data_dir / f"{ticker}_hist.csv").exists():
            tickers_to_fetch.append(ticker)
            
    if not tickers_to_fetch:
        logger.info("All unified data files already exist. Skipping setup.")
        logger.info("="*50)
        return
        
    logger.warning(f"Missing unified data for: {tickers_to_fetch}")
    
    num_tickers = len(tickers_to_fetch)
    
    logger.info(f"Starting one-time serial fetch for {num_tickers} tickers (5-minute delay between each).")
    
    for i, ticker in enumerate(tickers_to_fetch):
        if STOP_EVENT.is_set():
            logger.info("[Setup] Shutdown signal received during fetch. Aborting.")
            return
            
        logger.info(f"[Setup] Processing Ticker {i+1}/{num_tickers}: {ticker}...")
        fetch_and_save_initial_data(ticker, cfg, data_dir)
        
        # Wait 5 minutes between each ticker, but not after the last one
        if i < num_tickers - 1:
            logger.info(f"[Setup] Waiting {HISTORICAL_DELAY_SECONDS}s ({HISTORICAL_DELAY_SECONDS // 60} min) for rate limit safety...")
            for _ in range(HISTORICAL_DELAY_SECONDS):
                if STOP_EVENT.is_set():
                    return
                sleep(1)
            
    logger.info("Initial data fetch complete.")
    logger.info("="*50)


# =====================================================
# SECTION 2: MAIN SERVICE LAUNCHER (THREADED)
# =====================================================
def run_script(script_path: Path):
    """Run a Python script in a subprocess until STOP_EVENT is triggered."""
    try:
        logger.info(f"▶ Starting {script_path.name} ...")
        # Use sys.executable to ensure script runs in the same venv
        proc = subprocess.Popen([sys.executable, str(script_path)])
        logger.info(f"{script_path.name} running with PID {proc.pid}")

        # Monitor the process
        while not STOP_EVENT.is_set():
            retcode = proc.poll()
            if retcode is not None:
                logger.warning(f"⚠ {script_path.name} exited with code {retcode}")
                break
            sleep(1)

        # Stop process if still active
        if proc.poll() is None:
            logger.info(f"🛑 Terminating {script_path.name} ...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {script_path.name} ...")
                proc.kill()

    except Exception as e:
        logger.exception(f"❌ Error running {script_path.name}: {e}")

def start_thread(script_path: Path):
    t = threading.Thread(target=run_script, args=(script_path,), daemon=True, name=f"{script_path.name}_Thread")
    t.start()
    threads.append(t)
    return t

# =====================================================
# Main launcher
# =====================================================
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    logger.info(f"Project root: {project_root}")

    # --- STEP 1: Run initial data check/fetch first ---
    try:
        run_initial_data_fetch() 
    except Exception as e:
        logger.error(f"FATAL: Initial data fetch failed: {e}")
        sys.exit(1)
        
    if STOP_EVENT.is_set():
        logger.info("Shutdown requested during setup. Exiting.")
        sys.exit(0)

    # --- STEP 2: Launch all three main services ---
    scripts = [
        project_root / "phase2_pipeline.py",
        project_root / "build_vector_index.py",
        project_root / "live_dashboard.py"
    ]

    logger.info("🚀 Launching Phase 2 services...")

    for script in scripts:
        if script.exists():
            start_thread(script)
            sleep(2) # Stagger the startup to prevent initial resource clash
        else:
            logger.error(f"❌ Script not found: {script}")

    # --- STEP 3: Wait for shutdown signal ---
    try:
        while not STOP_EVENT.is_set():
            sleep(1)
    except KeyboardInterrupt:
        logger.info("🧭 KeyboardInterrupt detected, shutting down...")
        STOP_EVENT.set()

    logger.info("⏳ Waiting for threads to exit...")
    for t in threads:
        t.join(timeout=3)

    logger.info("✅ All services stopped. Exiting cleanly.")