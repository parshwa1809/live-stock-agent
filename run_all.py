#!/usr/bin/env python3
"""
run_all.py
Master launcher for Live Stock Agent Phase 2.

Features:
- **One-Time Setup:** On first run, safely fetches historical data
  in stable, scalable batches.
- **Service Launch:** Starts the three main services in parallel.
- **Graceful Shutdown:** Handles Ctrl+C to stop all services.

- **BUG FIX (v2.1):** The 'fetch_and_save_historical_batch' function
  now correctly handles batches of size 1.
- **CRITICAL FIX (v2.3):** Added a 'Final Cooldown' after the last
  historical batch to prevent an immediate rate-limit crash when
  launching the live data pipeline.
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
# We import Config and flatten_columns from the pipeline module
from phase2_pipeline import Config, flatten_columns

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
# --- Scalable Batching Configuration (used by historical fetch logic) ---
# --- FINAL FIX: Set default batch size to 2 for ultra-safe fetching ---
TICKER_BATCH_SIZE = 2
HISTORICAL_BATCH_DELAY_SECONDS = 300 # 5-minute safe delay for YFinance

# -----------------------
# Graceful shutdown handler
# -----------------------
def shutdown_handler(signum, frame):
    logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# =====================================================
# SECTION 1: ONE-TIME HISTORICAL DATA FETCH (SCALABLE)
# =====================================================
def fetch_and_save_historical_batch(tickers: list, cfg: Config, data_dir: Path):
    """
    Fetches 30-day, 30-minute historical data for a *batch* of tickers
    and saves their individual _hist.csv files.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.CSV_RETENTION_DAYS)
    
    logger.info(f"[Setup] Fetching 30-MINUTE historical for BATCH: {tickers}...")
    
    try:
        # Perform one bulk download for the entire batch (interval="30m")
        # We use group_by='ticker' for a clean DataFrame structure
        df_bulk = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                             interval="30m", progress=False, group_by='ticker')
        
        if df_bulk is None or df_bulk.empty:
            logger.warning(f"[Setup] No historical data returned for batch {tickers}")
            return

        for ticker in tickers:
            df_ticker = pd.DataFrame() # Create an empty DataFrame
            
            # --- BUG FIX: Handle single-ticker vs multi-ticker yfinance response ---
            # Check if we got a MultiIndex (standard for >1 tickers)
            if isinstance(df_bulk.columns, pd.MultiIndex):
                if ticker in df_bulk.columns:
                    df_ticker = df_bulk[ticker].copy()
                else:
                    logger.warning(f"[Setup] No data for {ticker} in multi-batch result.")
                    continue
            # Check if we got a simple DataFrame (standard for 1 ticker)
            elif len(tickers) == 1:
                # yfinance returns a simple df, not multi-index, for one ticker
                df_ticker = df_bulk.copy()
            else:
                 logger.warning(f"[Setup] Unexpected DataFrame structure for {ticker}.")
                 continue
            # --- END BUG FIX ---
            
            df_ticker = df_ticker.dropna(subset=['Close']) # Drop rows with no price
            
            if df_ticker.empty:
                logger.warning(f"[Setup] No historical data found for {ticker} after cleaning.")
                continue

            df_ticker.index.name = "Datetime"
            # Flatten columns logic is not needed here as group_by='ticker'
            # gives a simple [Open, High, Low, Close, Volume] structure
            
            df_to_save = df_ticker.copy()
            # We must save with suffixed columns for consistency
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df_to_save.columns and f"{col}_{ticker}" not in df_to_save.columns:
                    df_to_save[f"{col}_{ticker}"] = df_to_save[col]
                    if col in df_to_save.columns:
                        del df_to_save[col]
            
            path = data_dir / f"{ticker}_hist.csv"
            df_to_save.to_csv(path)
            logger.info(f"[Setup] ✅ Successfully saved {ticker} HISTORICAL CSV ({len(df_to_save)} rows)")

    except Exception as e:
        logger.error(f"[Setup] ❌ Failed to fetch historical batch {tickers}: {e}")

def run_initial_historical_fetch():
    """
    Checks if historical data exists. If not, fetches it in safe batches.
    """
    logger.info("="*50)
    logger.info("Checking for required historical data files...")
    
    # Need to load config first
    load_dotenv()
    cfg = Config()
    data_dir = Path(cfg.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    all_tickers = cfg.ALL_FETCH_TICKERS
    tickers_to_fetch = []
    
    for ticker in all_tickers:
        path = data_dir / f"{ticker}_hist.csv"
        if not path.exists():
            tickers_to_fetch.append(ticker)
            
    if not tickers_to_fetch:
        logger.info("All historical data files already exist. Skipping setup.")
        logger.info("="*50)
        return
        
    logger.warning(f"Missing historical data for: {tickers_to_fetch}")
    
    # --- FIX: Use the local variable, not the config one ---
    batch_size = TICKER_BATCH_SIZE      # <--- NEW
    
    num_batches = math.ceil(len(tickers_to_fetch) / batch_size)
    batches = [tickers_to_fetch[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    
    logger.info(f"Starting one-time fetch in {num_batches} batches (size={batch_size}) with a {HISTORICAL_BATCH_DELAY_SECONDS // 60}-minute delay between batches.")
    
    for i, batch in enumerate(batches):
        if STOP_EVENT.is_set():
            logger.info("[Setup] Shutdown signal received during fetch. Aborting.")
            return
            
        logger.info(f"[Setup] Processing Batch {i+1}/{num_batches}...")
        fetch_and_save_historical_batch(batch, cfg, data_dir)
        
        # Wait between batches (Existing logic)
        if i < len(batches) - 1:
            logger.info(f"[Setup] Waiting {HISTORICAL_BATCH_DELAY_SECONDS}s ({HISTORICAL_BATCH_DELAY_SECONDS // 60} min) for rate limit safety...")
            for _ in range(HISTORICAL_BATCH_DELAY_SECONDS):
                if STOP_EVENT.is_set():
                    return
                sleep(1)

    # --- NEW: THE "FINAL COOLDOWN" FIX ---
    # This prevents the pipeline from starting immediately after the last batch
    # and triggering a rate limit.
    if not STOP_EVENT.is_set():
        logger.info(f"[Setup] Final Batch Complete. Waiting {HISTORICAL_BATCH_DELAY_SECONDS}s (Final Cooldown) before launching services...")
        for _ in range(HISTORICAL_BATCH_DELAY_SECONDS):
            if STOP_EVENT.is_set(): return
            sleep(1)
    # --- END NEW FIX ---
            
    logger.info("Historical data fetch complete.")
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

    # --- STEP 1: Run historical data check/fetch first ---
    try:
        run_initial_historical_fetch()
    except Exception as e:
        logger.error(f"FATAL: Historical data fetch failed: {e}")
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