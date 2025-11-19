#!/usr/bin/env python3
"""
run_all.py
Master launcher for Live Stock Agent Phase 2.

Features:
- **One-Time Setup (NEW):** Safely fetches 30 days of 5-minute data for all
  tickers SERIALLY, with a 5-minute cooldown between *each* ticker.
- **Service Launch:** Starts the three main services in parallel.
- **File Structure:** Merges historical and live data into a single unified file (<TICKER>.csv).
- **CRITICAL FIX:** Implements a strict retry loop to ensure all tickers download successfully
  before launching the dashboard.
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
# FIX: Import Config from the new centralized file
from config import cfg, Config 

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
        path_tmp = data_dir / f"{ticker}.csv.tmp"
        
        # Atomic write
        df_to_save.to_csv(path_tmp) 
        os.replace(path_tmp, path)
        
        logger.info(f"[Setup] ✅ Successfully saved {ticker}.csv ({len(df_to_save)} rows) at 5-min interval.")

    except Exception as e:
        logger.error(f"[Setup] ❌ Failed to fetch initial data for {ticker}: {e}")

def run_initial_data_fetch():
    """
    STRICT BLOCKING MODE: Fetches all historical data and retries failed tickers
    indefinitely (up to MAX_RETRIES) before starting the main services.
    """
    logger.info("="*50)
    logger.info("🔒 STRICT SETUP: Checking and downloading all 30-day historical data...")
    
    data_dir = Path(cfg.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    all_tickers = cfg.ALL_FETCH_TICKERS 
    
    MAX_RETRIES = 5  # Total passes allowed for all tickers
    retry_delay = HISTORICAL_DELAY_SECONDS # 5 minutes wait between individual fetches
    attempt_count = 0
    
    while True:
        # Re-evaluate missing tickers every loop
        missing_tickers = []
        for ticker in all_tickers:
            path = data_dir / f"{ticker}.csv"
            if not path.exists():
                missing_tickers.append(ticker)
        
        # ✅ EXIT CONDITION: All files exist
        if not missing_tickers:
            logger.info("✅ All unified data files successfully secured!")
            logger.info("="*50)
            return # <--- This allows the main script to continue to launch services

        # ❌ FAILURE CONDITION
        attempt_count += 1
        if attempt_count > MAX_RETRIES:
            logger.critical("❌ FATAL: Could not download all data after maximum retries.")
            logger.critical("   Cannot safely start Dashboard. Exiting.")
            sys.exit(1)
        
        logger.warning(f"⚠️ Missing historical data for: {missing_tickers}. Starting Retry Round {attempt_count}/{MAX_RETRIES}...")
        
        # Fetch missing tickers with a delay between them
        for i, ticker in enumerate(missing_tickers):
            if STOP_EVENT.is_set(): return
            
            logger.info(f"[Retry Round {attempt_count}] Fetching {ticker}...")
            fetch_and_save_initial_data(ticker, cfg, data_dir)

            # Rate limit wait (Skip if it's the last ticker in this specific list)
            if i < len(missing_tickers) - 1:
                logger.info(f"⏳ Waiting {retry_delay}s before next ticker...")
                for _ in range(retry_delay):
                    if STOP_EVENT.is_set(): return
                    sleep(1)
        
        # Cooldown between entire rounds if any tickers failed
        if missing_tickers:
            # We must wait 5 minutes before restarting the loop/check
            logger.info(f"⏳ Waiting {retry_delay}s before checking status for next retry round...")
            sleep(retry_delay)


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

    # --- NEW STEP: Final Mandatory Cooldown Before Live Start ---
    FINAL_COOLDOWN = 300 # 5 minutes
    logger.info(f"🎯 Historical setup complete. Waiting {FINAL_COOLDOWN}s before first LIVE burst...")
    for _ in range(FINAL_COOLDOWN):
        if STOP_EVENT.is_set(): break
        sleep(1)

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