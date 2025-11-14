#!/usr/bin/env python3
"""
run_all.py
Master launcher for Live Stock Agent Phase 2.

Features:
- Starts historical data pipeline (phase2_pipeline.py)
- Starts the RAG index builder (build_vector_index.py)
- Launches the live dashboard (live_dashboard.py)
- Graceful shutdown on Ctrl+C
"""

import threading
import subprocess
import signal
import logging
import sys
from pathlib import Path
from time import sleep

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

# -----------------------
# Graceful shutdown handler
# -----------------------
def shutdown_handler(signum, frame):
    logger.info(f"Signal {signum} received. Initiating graceful shutdown...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# =====================================================
# Thread runner
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

# -----------------------
# Thread starter
# -----------------------
def start_thread(script_path: Path):
    t = threading.Thread(target=run_script, args=(script_path,), daemon=True)
    t.start()
    threads.append(t)
    return t

# =====================================================
# Main launcher
# =====================================================
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    logger.info(f"Project root: {project_root}")

    # --- NOW LAUNCHES ALL THREE SCRIPTS ---
    scripts = [
        project_root / "phase2_pipeline.py",
        project_root / "build_vector_index.py", # <-- ADDED RAG BUILDER
        project_root / "live_dashboard.py"
    ]

    logger.info("🚀 Launching Phase 2 services...")

    for script in scripts:
        if script.exists():
            start_thread(script)
            sleep(2) # Stagger the startup to prevent initial resource clash
        else:
            logger.error(f"❌ Script not found: {script}")

    try:
        while not STOP_EVENT.is_set():
            sleep(1)
    except KeyboardInterrupt:
        logger.info("🧭 KeyboardInterrupt detected, shutting down...")

    logger.info("⏳ Waiting for threads to exit...")
    for t in threads:
        t.join(timeout=3)

    logger.info("✅ All services stopped. Exiting cleanly.")