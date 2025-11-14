import pandas as pd
import pandas_ta as ta
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import time 
import signal 
import sys
import threading # <-- ADDED THIS IMPORT

# --- Configuration ---
load_dotenv()
DATA_DIR = Path("./data")
TICKERS = os.getenv("TICKERS", "AAPL,AMZN,GOOGL,MSFT,TSLA").split(",")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_FILE = DATA_DIR / "stock_index.faiss"
TEXTS_FILE = DATA_DIR / "stock_texts.json"
# Run this script on its own 15-minute loop
REBUILD_INTERVAL_SECONDS = 900 # 15 minutes

# Define Regular Trading Hours (RTH) in UTC (14:30 - 21:00)
RTH_START = '14:30'
RTH_END = '21:00'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
logger = logging.getLogger("build_vector_index")

# --- Graceful shutdown handler ---
STOP_EVENT = threading.Event()
def _shutdown(signum, frame):
    logger.info("Shutdown signal received, stopping RAG indexer...")
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)
# ------------------------------------

def load_data(ticker: str, file_suffix: str) -> pd.DataFrame:
    """Loads and validates a single CSV file."""
    path = DATA_DIR / f"{ticker}{file_suffix}"
    if not path.exists():
        logger.warning(f"RAG: {path} not found, skipping.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC')
        
        # Ensure 'Close' column exists after loading
        if 'Close' not in df.columns and f'Close_{ticker}' in df.columns:
             df['Close'] = df[f'Close_{ticker}']
        if 'Volume' not in df.columns and f'Volume_{ticker}' in df.columns:
             df['Volume'] = df[f'Volume_{ticker}']
             
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

def create_single_smart_chunk(row: pd.Series, ticker: str, period_name: str) -> str:
    """
    Creates one single "smart chunk" that contains:
    1. Simple Facts (for simple questions)
    2. TA Analysis (for smart questions)
    3. Engaging Hooks (for proactive chat)
    """
    
    time_str = row.name.strftime('%Y-%m-%d') if period_name == "Daily" else row.name.strftime('%Y-%m-%d %H:%M %Z')

    # --- 1. The Simple Facts ---
    fact_text = (
        f"Fact for {ticker} ({period_name} data for {time_str}): "
        f"The closing price was ${row['Close']:.2f} and the total volume was {int(row['Volume']):,}."
    )

    # --- 2. The Smart Analysis ---
    analysis_text = ""
    hook_text = ""

    # Trend Analysis
    if row['MA_5'] > row['MA_20']:
        trend_analysis = "it was in a bullish (upward) trend, as the short-term 5-period average was above the 20-period average."
        trend_hook = "This is often a positive sign for short-term price action."
    else:
        trend_analysis = "it was in a bearish (downward) trend, as the short-term 5-period average was below the 20-period average."
        trend_hook = "This is often a negative sign for short-term price action."

    # Momentum Analysis
    rsi = row['RSI_14']
    if rsi > 70:
        momentum_analysis = f"Momentum was in overbought territory (RSI {rsi:.0f})."
        momentum_hook = f"This can suggest the stock is 'overheated' and might be due for a pullback."
    elif rsi < 30:
        momentum_analysis = f"Momentum was in oversold territory (RSI {rsi:.0f})."
        momentum_hook = f"This can suggest the stock is 'running cold' and might be due for a bounce."
    else:
        momentum_analysis = f"Momentum was neutral (RSI {rsi:.0f})."
        momentum_hook = "" # No hook for neutral

    analysis_text = f"Analysis: {trend_analysis} {momentum_analysis}"

    # --- 3. The Engaging Hook (Your Request) ---
    hook_text_combined = f"{trend_hook} {momentum_hook}".strip()
    if hook_text_combined:
        hook_text = (
            f"Insight: {hook_text_combined} "
            f"**Would you like me to explain what this combination of trend and momentum means for the stock?**"
        )
    
    # Combine all three parts into one chunk
    return f"{fact_text}\n{analysis_text}\n{hook_text}".strip()


def analyze_and_get_chunks(df: pd.DataFrame, ticker: str, period_name: str) -> list:
    """
    Analyzes data and returns a list of "smart chunks".
    """
    texts = []
    if df.empty or 'Close' not in df.columns or len(df) < 21:
        # Need at least 20 periods for the MA, 14 for RSI
        logger.debug(f"Skipping {ticker} {period_name}, not enough data for TA (need 21, have {len(df)})")
        return texts
        
    try:
        # Calculate Technical Indicators
        # Use .ta. prefix for pandas-ta
        df['MA_5'] = df.ta.sma(length=5, close='Close')
        df['MA_20'] = df.ta.sma(length=20, close='Close')
        df['RSI_14'] = df.ta.rsi(length=14, close='Close')
    except Exception as e:
        logger.error(f"Error calculating TA for {ticker}: {e}")
        return texts
        
    df = df.dropna() # Remove rows where TA indicators couldn't be calculated
    if df.empty:
        logger.debug(f"Skipping {ticker} {period_name}, all rows dropped after TA.")
        return texts

    # Create the single smart chunk for each row
    for _, row in df.iterrows():
        if row['Volume'] > 0: # Only analyze periods with trading
            texts.append(create_single_smart_chunk(row, ticker, period_name))
            
    return texts


def build_index():
    logger.info("RAG: Starting RAG index build with 'Smart Chunks'...")
    
    all_text_chunks = []
    
    # 1. Load data
    for ticker in TICKERS:
        hist_df = load_data(ticker, "_hist.csv")
        live_df = load_data(ticker, "_live.csv")
        
        # --- 2. Resample and Analyze ---
        
        # Resample historical 30-min data to Daily for analysis
        if not hist_df.empty:
            hist_daily_df = hist_df.resample('D').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna() # Drop non-trading days
            
            hist_chunks = analyze_and_get_chunks(hist_daily_df, ticker, "Daily")
            all_text_chunks.extend(hist_chunks)

        # Filter live 1-min data and resample to 30-min for analysis
        if not live_df.empty:
            live_rth_df = live_df.between_time(RTH_START, RTH_END)
            if not live_rth_df.empty:
                live_30min_df = live_rth_df.resample('30min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna() # Drop intervals with no trades
                
                live_chunks = analyze_and_get_chunks(live_30min_df, ticker, "30-Minute")
                all_text_chunks.extend(live_chunks)

    if not all_text_chunks:
        logger.warning("RAG: No data found to index. Exiting build.")
        return

    # 3. Load the embedding model
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error(f"RAG: Failed to load SentenceTransformer model. Is '{EMBEDDING_MODEL_NAME}' correct? Error: {e}")
        return

    # 4. Create embeddings (vectors)
    logger.info(f"RAG: Creating {len(all_text_chunks)} 'smart' embeddings...")
    embeddings = model.encode(all_text_chunks, show_progress_bar=False)
    
    # 5. Build the FAISS index
    dimension = embeddings.shape[1]  # Get the size of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # 6. Save the index and the text chunks
    faiss.write_index(index, str(INDEX_FILE))
    with open(TEXTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_text_chunks, f)
        
    logger.info(f"RAG: Successfully built and saved 'smart' index with {len(all_text_chunks)} chunks.")

def run_forever():
    """
    Main loop to run the index builder every N seconds.
    """
    logger.info(f"Starting RAG indexer loop (rebuilds every {REBUILD_INTERVAL_SECONDS}s)...")
    while not STOP_EVENT.is_set():
        try:
            build_index()
        except Exception as e:
            logger.error(f"Error in build_index loop: {e}")
        
        # Wait for the interval, but check for stop event every second
        for _ in range(REBUILD_INTERVAL_SECONDS):
            if STOP_EVENT.is_set():
                break
            time.sleep(1)
            
    logger.info("RAG indexer stopped.")

if __name__ == "__main__":
    run_forever()