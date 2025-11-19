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
import threading

# --- Configuration ---
load_dotenv()
# --- FIX: Import config directly from the pipeline ---
from phase2_pipeline import cfg 

DATA_DIR = Path(cfg.DATA_DIR)
TICKERS = cfg.TICKERS
CONTEXT_TICKERS = cfg.CONTEXT_TICKERS
ALL_RAG_TICKERS = cfg.ALL_FETCH_TICKERS 
# --- END FIX ---

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

INDEX_FILE = DATA_DIR / "stock_index.faiss"
TEXTS_FILE = DATA_DIR / "stock_texts.json"
INDEX_FILE_TMP = DATA_DIR / "stock_index.faiss.tmp"
TEXTS_FILE_TMP = DATA_DIR / "stock_texts.json.tmp"
NEWS_JSON_FILE = DATA_DIR / "news_headlines.json"
REBUILD_INTERVAL_SECONDS = 900 # 15 minutes

RTH_START = '14:30'
RTH_END = '21:00'

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
    """
    Loads and validates a single CSV file, restoring all base columns
    from their suffixed versions (e.g., Open_AAPL -> Open).
    """
    path = DATA_DIR / f"{ticker}{file_suffix}"
    if not path.exists():
        # --- Handle VIX historical file fallback ---
        if ticker == "^VIX" and file_suffix == "_hist.csv":
            path_fallback = DATA_DIR / "^VIX_hist.csv" 
            if path_fallback.exists():
                logger.warning("RAG: Found old '^VIX_hist.csv'. Using it.")
                path = path_fallback 
            else:
                logger.warning(f"RAG: {ticker}{file_suffix} not found, skipping.")
                return pd.DataFrame()
        else:
             logger.warning(f"RAG: {ticker}{file_suffix} not found, skipping.")
             return pd.DataFrame()
             
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC')
        
        # --- FIX: Loop through ALL base columns to restore them from suffixed versions ---
        base_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in base_cols:
            suffixed_col = f"{col}_{ticker}"
            # If the simple column "Open" doesn't exist, but "Open_AAPL" does, copy it over
            if col not in df.columns and suffixed_col in df.columns:
                df[col] = pd.to_numeric(df[suffixed_col], errors='coerce')
        
        # Final check to ensure we have what we need
        if 'Close' not in df.columns:
             logger.debug(f"RAG: Close price column missing for {ticker}.")
             return pd.DataFrame() 

        # Drop any rows where 'Close' may have been coerced to NaN
        df = df.dropna(subset=['Close']) 
             
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

def load_news_chunks() -> list:
    """
    Loads news headlines from the JSON file and formats them as text chunks
    for the vector index.
    """
    texts = []
    if not NEWS_JSON_FILE.exists():
        logger.warning(f"RAG: News file not found at {NEWS_JSON_FILE}. Skipping news indexing.")
        return texts
        
    try:
        if NEWS_JSON_FILE.stat().st_size == 0:
            logger.warning(f"RAG: {NEWS_JSON_FILE} is empty, possibly during a write. Skipping news load this cycle.")
            return []
            
        with open(NEWS_JSON_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            
        for article in articles:
            title = article.get('title', '')
            source = article.get('source', {}).get('name', 'Unknown')
            published_at = article.get('publishedAt', 'unknown time')
            
            target_ticker = next((t for t in TICKERS if t.lower() in title.lower()), None)
            
            if target_ticker:
                chunk = (
                    f"Fact for {target_ticker} (News Headline from {source} at {published_at}): "
                    f"\"{title}\""
                )
                texts.append(chunk)
                
        logger.info(f"RAG: Loaded and formatted {len(texts)} news headline chunks.")
        return texts
        
    except json.JSONDecodeError as e:
        logger.warning(f"RAG: Failed to decode {NEWS_JSON_FILE}. It might have been read during a write. Skipping cycle. Error: {e}")
        return []
    except Exception as e:
        logger.error(f"RAG: Failed to read or process {NEWS_JSON_FILE}: {e}")
        return []

def create_single_smart_chunk(row: pd.Series, ticker: str, period_name: str) -> str:
    """Creates one single "smart chunk" for stock price data."""
    
    time_str = row.name.strftime('%Y-%m-%d') if period_name == "Daily" else row.name.strftime('%Y-%m-%d %H:%M %Z')

    volume_str = f"The total volume was {int(row['Volume']):,}" if 'Volume' in row and not pd.isna(row['Volume']) and row['Volume'] > 0 else "Volume data is not applicable or zero."
    
    fact_text = (
        f"Fact for {ticker} ({period_name} data for {time_str}): "
        f"The closing price was ${row['Close']:.2f}. {volume_str}"
    )

    analysis_text = ""
    hook_text = ""
    
    ta_cols = ['MA_5', 'MA_20', 'RSI_14']
    ta_available = all(col in row and not pd.isna(row[col]) for col in ta_cols)
    
    if ticker != '^VIX' and ta_available:
        if row['MA_5'] > row['MA_20']:
            trend_analysis = "it was in a bullish (upward) trend, as the short-term 5-period average was above the 20-period average."
            trend_hook = "This is often a positive sign for short-term price action."
        else:
            trend_analysis = "it was in a bearish (downward) trend, as the short-term 5-period average was below the 20-period average."
            trend_hook = "This is often a negative sign for short-term price action."

        rsi = row['RSI_14']
        if rsi > 70:
            momentum_analysis = f"Momentum was in overbought territory (RSI {rsi:.0f})."
            momentum_hook = f"This can suggest the stock is 'overheated' and might be due for a pullback."
        elif rsi < 30:
            momentum_analysis = f"Momentum was in oversold territory (RSI {rsi:.0f})."
            momentum_hook = f"This can suggest the stock is 'running cold' and might be due for a bounce."
        else:
            momentum_analysis = f"Momentum was neutral (RSI {rsi:.0f})."
            momentum_hook = ""

        analysis_text = f"Analysis: {trend_analysis} {momentum_analysis}"
        hook_text_combined = f"{trend_hook} {momentum_hook}".strip()
        if hook_text_combined:
            hook_text = (
                f"Insight/Hook: {hook_text_combined} "
                f"Would you like me to explain what this combination of trend and momentum means for the stock?"
            )
    else:
        if ticker == '^VVIX':
            analysis_text = "Analysis: VIX analysis focuses on volatility and risk sentiment, not standard price trends."
            hook_text = "Insight/Hook: The VIX level is critical for overall market risk assessment. Would you like to know what the current VIX level implies for market stability?"
        elif ticker not in TICKERS:
             analysis_text = f"Analysis: This is market index data, used for contextual analysis of primary stocks."
             hook_text = ""

    return f"{fact_text}\n{analysis_text}\n{hook_text}".strip()


def analyze_and_get_chunks(df: pd.DataFrame, ticker: str, period_name: str) -> list:
    """Analyzes data and returns a list of "smart chunks"."""
    texts = []
    if df.empty or 'Close' not in df.columns:
        logger.debug(f"Skipping {ticker} {period_name}, DataFrame empty or no 'Close' column.")
        return texts

    # --- Only run TA if we have enough data AND it's not VIX ---
    if len(df) >= 21 and ticker != '^VIX':
        required_cols = ['Close', 'High', 'Low', 'Open']
        if not all(col in df.columns for col in required_cols):
             logger.warning(f"Skipping TA for {ticker} {period_name}: Missing one or more required columns (Open, High, Low) after aggregation.")
        else:
            try:
                df['MA_5'] = df.ta.sma(length=5, close='Close')
                df['MA_20'] = df.ta.sma(length=20, close='Close')
                df['RSI_14'] = df.ta.rsi(length=14, close='Close')
            except Exception as e:
                logger.error(f"Error calculating TA for {ticker}: {e}")
            
    df = df.dropna(subset=['Close']) 
    if df.empty:
        logger.debug(f"Skipping {ticker} {period_name}, all rows dropped after cleaning.")
        return texts

    for _, row in df.iterrows():
        if 'Volume' not in row or pd.isna(row['Volume']) or row['Volume'] > 0 or ticker in CONTEXT_TICKERS: 
            texts.append(create_single_smart_chunk(row, ticker, period_name))
            
    return texts


def build_index():
    logger.info("RAG: Starting RAG index build with 'Smart Chunks'...")
    
    all_text_chunks = []
    
    # 1. Load data for ALL RAG TICKERS (Primary Stocks + Context Tickers)
    for ticker in ALL_RAG_TICKERS:
        # Load 30-MIN HISTORICAL data
        hist_df = load_data(ticker, "_hist.csv")
        if not hist_df.empty:
            # Resample to Daily for high-level summary
            hist_daily_df = hist_df.resample('D').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna(subset=['Close']) 
            
            hist_chunks = analyze_and_get_chunks(hist_daily_df, ticker, "Daily (from 30-min)")
            all_text_chunks.extend(hist_chunks)
        
        # Load 5-MIN LIVE data
        live_df = load_data(ticker, "_live.csv")
        if not live_df.empty:
            live_rth_df = live_df.between_time(RTH_START, RTH_END)
            if not live_rth_df.empty:
                # --- Resample 5-min data to 30-min for consistent RAG analysis ---
                live_30min_df = live_rth_df.resample('30min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna(subset=['Close']) 
                
                # NOTE: The RAG chunk is correctly labeled "30-Minute (from 5-min)" for accuracy
                live_chunks = analyze_and_get_chunks(live_30min_df, ticker, "30-Minute (from 5-min)")
                all_text_chunks.extend(live_chunks)

    # 2. Load news headlines and add them to the chunks
    all_text_chunks.extend(load_news_chunks())

    if not all_text_chunks:
        logger.warning("RAG: No data found to index. Skipping index build until data is available.")
        return

    # 3. Load the embedding model
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error(f"RAG: Failed to load SentenceTransformer model. Is '{EMBEDDING_MODEL_NAME}' correct? Error: {e}")
        return

    # 4. Create embeddings (vectors) and build FAISS index
    logger.info(f"RAG: Creating {len(all_text_chunks)} 'smart' embeddings...")
    embeddings = model.encode(all_text_chunks, show_progress_bar=False)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # 5. Save the index and the text chunks ATOMICALLY
    try:
        # Step 1: Write to temporary files
        faiss.write_index(index, str(INDEX_FILE_TMP))
        with open(TEXTS_FILE_TMP, 'w', encoding='utf-8') as f:
            json.dump(all_text_chunks, f)
            
        # Step 2: Atomically replace the old files with the new ones
        os.replace(INDEX_FILE_TMP, INDEX_FILE)
        os.replace(TEXTS_FILE_TMP, TEXTS_FILE)
        
        logger.info(f"RAG: Successfully built and saved 'smart' index with {len(all_text_chunks)} chunks.")
    except Exception as e:
        logger.error(f"RAG: Failed to save index or text files atomically: {e}")
        # Clean up .tmp files if they exist
        try:
            if INDEX_FILE_TMP.exists():
                os.remove(INDEX_FILE_TMP)
            if TEXTS_FILE_TMP.exists():
                os.remove(TEXTS_FILE_TMP)
        except OSError as oe:
            logger.error(f"RAG: Error cleaning up temp files: {oe}")


def run_forever():
    """Main loop to run the index builder every N seconds."""
    logger.info(f"Starting RAG indexer loop (rebuilds every {REBUILD_INTERVAL_SECONDS}s)...")
    while not STOP_EVENT.is_set():
        try:
            build_index()
        except Exception as e:
            logger.error(f"Error in build_index loop: {e}")
        
        for _ in range(REBUILD_INTERVAL_SECONDS):
            if STOP_EVENT.is_set():
                break
            time.sleep(1)
            
    logger.info("RAG indexer stopped.")

if __name__ == "__main__":
    run_forever()