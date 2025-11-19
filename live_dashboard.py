#!/usr/bin/env python3
"""
Unified Live Stock Dashboard — Phase 2 (FINAL STABLE VERSION)

Features:
- **STABLE 5-MIN DATA:** All "Live" charts and alerts are now driven by the
  stable 5-minute data (`_live.csv`) fetched by the pipeline.
- **30-MIN HISTORICAL:** "Historical" charts still use the 30-minute
  `_hist.csv` file for long-term context.
- **ALL ALERTS & NEWS:** All alert types (Live 5-Min, Hist 30-Min, News)
  are fully functional and displayed in the UI.

- **BUG FIX (v2.1):** Removed all global variables and locks (e.g., 
  dash_live_tracker, alert_lock, alert_history lists).
- **STATE MANAGEMENT (v2.1):** All data is now shared between callbacks 
  using dcc.Store components (e.g., 'hist-data-store', 
  'live-data-store', 'hist-alerts-store', 'live-alerts-store'). 
  This makes the dashboard stateless and process-safe for production
  servers like Gunicorn.
- **BUG FIX (v2.2):** The RAG index now automatically reloads from disk
  if it detects a newer version, preventing stale chat responses.
- **OPTIMIZATION (v2.2):** Technical indicators (RSI, MACD) are no longer
  calculated in the dashboard. They are read directly from the
  'live_alerts.json' file produced by the pipeline, centralizing logic.
"""

import os
import logging
import multiprocessing 
from datetime import datetime
import diskcache
from dash import Dash, dcc, html, Output, Input, State, callback
from dash import DiskcacheManager 
import plotly.graph_objs as go
import pandas as pd
import numpy as np 
import json
from pathlib import Path 
import faiss
from sentence_transformers import SentenceTransformer
# --- OPTIMIZATION: Removed pandas_ta import ---
# import pandas_ta as ta 
# --- FIX: Removed threading import ---

# --- CRITICAL CHANGE: Import cfg, llm. All signal modules are REMOVED ---
from phase2_pipeline import cfg, llm 
# --- FIX: Removed all signal imports (technical, volume, etc.) ---

# Set up the cache manager for background callbacks
cache = diskcache.Cache("./cache", timeout=120) 
long_callback_manager = DiskcacheManager(cache) 

DATA_PATH = Path(cfg.DATA_DIR) # Use DATA_DIR from imported cfg
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("live_dashboard")

# Pass the correct argument for new Dash versions
app = Dash(__name__, background_callback_manager=long_callback_manager) 
server = app.server

# --- FIX: All global alert/data variables and locks have been REMOVED ---
# (e.g., hist_alert_history, live_alert_history, alert_lock, etc.)

# Define Regular Trading Hours (RTH) in UTC
RTH_START = '14:30'
RTH_END = '21:00'

# =====================================================
# RAG: Load models and index globally for performance 
# =====================================================
RAG_INDEX_FILE = DATA_PATH / "stock_index.faiss"
RAG_TEXTS_FILE = DATA_PATH / "stock_texts.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
NEWS_JSON_FILE = DATA_PATH / "news_headlines.json"
ALERTS_JSON_FILE = DATA_PATH / "live_alerts.json"

# --- FIX: Load model globally ONCE at startup ---
logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' globally...")
try:
    embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not load SentenceTransformer model. RAG will fail. Error: {e}")
    embedding_model_cache = None
# --- END FIX ---

# --- NEW FIX: Load RAG index and texts globally ONCE at startup ---
logger.info("Loading RAG index and text files into memory...")
try:
    if RAG_INDEX_FILE.exists() and RAG_TEXTS_FILE.exists():
        rag_index_cache = faiss.read_index(str(RAG_INDEX_FILE))
        with open(RAG_TEXTS_FILE, 'r', encoding='utf-8') as f:
            rag_texts_cache = json.load(f)
        # --- BUG FIX: Add global variable to track load time ---
        rag_index_load_time = os.path.getmtime(RAG_INDEX_FILE)
        logger.info(f"Successfully loaded RAG index and {len(rag_texts_cache)} text chunks.")
    else:
        rag_index_cache = None
        rag_texts_cache = None
        # --- BUG FIX: Initialize load time to 0 ---
        rag_index_load_time = 0.0
        logger.warning("RAG index or text file not found. Chat will have no context.")
except Exception as e:
    logger.error(f"FATAL: Could not load RAG files. RAG will fail. Error: {e}")
    rag_index_cache = None
    rag_texts_cache = None
    rag_index_load_time = 0.0
# --- END NEW FIX ---


# --- THIS IS THE UPDATED SELF-HEALING FUNCTION ---
def search_rag_index(query: str, k=3) -> str:
    """
    Embeds a query, searches the FAISS index, and returns the top k text chunks.
    --- SELF-HEALING: If the index is not in memory, or if the file on disk
    is newer, it will (re)load it from disk on-the-fly.
    """
    global embedding_model_cache, rag_index_cache, rag_texts_cache, rag_index_load_time # <-- Added rag_index_load_time
    
    if embedding_model_cache is None:
        logger.error("Embedding model is not loaded. Check startup logs.")
        return "Error: Embedding model is not loaded."
        
    # --- NEW SELF-HEALING LOGIC ---
    try:
        # Check if files exist NOW
        if RAG_INDEX_FILE.exists() and RAG_TEXTS_FILE.exists():
            current_mtime = os.path.getmtime(RAG_INDEX_FILE)
            
            # --- THIS IS THE KEY CHANGE ---
            if rag_index_cache is None or current_mtime > rag_index_load_time:
                logger.warning("RAG index is stale (or not loaded). Reloading from disk...")
                
                # Load them into the global cache
                rag_index_cache = faiss.read_index(str(RAG_INDEX_FILE))
                with open(RAG_TEXTS_FILE, 'r', encoding='utf-8') as f:
                    rag_texts_cache = json.load(f)
                
                # Update the load time
                rag_index_load_time = current_mtime
                
                logger.info(f"RAG index reloaded successfully ({len(rag_texts_cache)} chunks).")
        else:
            # Files still don't exist, tell user to wait
            logger.warning("RAG files still not found on disk.")
            return "No data in RAG index. Please wait for pipeline to run."
            
    except Exception as e:
        logger.error(f"Failed to load RAG index from disk: {e}")
        # Clear cache on partial failure
        rag_index_cache = None
        rag_texts_cache = None
        rag_index_load_time = 0.0 # Reset load time
        return f"Error loading RAG index: {e}"
    # --- END NEW LOGIC ---
        
    # Proceed with search (cache is now guaranteed to be loaded if files exist)
    try:
        query_vector = embedding_model_cache.encode([query]).astype('float32')
        distances, indices = rag_index_cache.search(query_vector, k)
        results = [rag_texts_cache[i] for i in indices[0]]
        return "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"Error during RAG search: {e}")
        return "Error searching RAG index."
# --- END UPDATED FUNCTION ---


# =====================================================
# LOAD CSV & GET COLUMN HELPERS
# =====================================================
def load_data(ticker: str, file_suffix: str) -> pd.DataFrame:
    # --- This function is now stable and reads both files ---
    path = DATA_PATH / f"{ticker}{file_suffix}"
    if not path.exists():
        if ticker in cfg.TICKERS:
            logger.warning(f"{ticker}{file_suffix} not found at {path}")
        return pd.DataFrame()
    try:
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            df = pd.read_csv(path, index_col=0, header=[0,1])
            
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index = df.index.tz_convert('UTC')
        
        # Ensure base columns are available from suffixed columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            suff = f"{col}_{ticker}"
            if suff in df.columns and col not in df.columns:
                 df[col] = df[suff]
        
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

def load_news_headlines() -> list:
    """Safely loads and parses the news_headlines.json file."""
    if not NEWS_JSON_FILE.exists():
        return []
    
    try:
        if NEWS_JSON_FILE.stat().st_size == 0:
            logger.warning(f"Dashboard: {NEWS_JSON_FILE} is empty, skipping load this cycle.")
            return []
            
        with open(NEWS_JSON_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        return articles
        
    except json.JSONDecodeError as e:
        logger.warning(f"Dashboard: Failed to decode {NEWS_JSON_FILE}. It might have been read during a write. Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Dashboard: Failed to read {NEWS_JSON_FILE}: {e}")
        return []

# --- NEW: Helper function to load alerts from the pipeline ---
def load_alerts() -> dict:
    """Safely loads alerts from the JSON file."""
    if not ALERTS_JSON_FILE.exists():
        logger.warning(f"Dashboard: {ALERTS_JSON_FILE} not found. Waiting for pipeline.")
        return {}
    try:
        if ALERTS_JSON_FILE.stat().st_size == 0:
            logger.warning(f"Dashboard: {ALERTS_JSON_FILE} is empty, skipping load this cycle.")
            return {}
            
        with open(ALERTS_JSON_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Dashboard: Failed to decode {ALERTS_JSON_FILE}. It might have been read during a write. Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Dashboard: Failed to read {ALERTS_JSON_FILE}: {e}")
        return {}
# --- END NEW FUNCTION ---

def get_col(df, ticker, col, fill_na=None):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(dtype=float)
        
    if s.empty:
        return pd.Series(dtype=float)

    if fill_na is not None:
        return s.fillna(fill_na)
    else:
        return s.dropna()

# =====================================================
# Dashboard-Side Alert Engine (REMOVED)
# =====================================================
# --- FIX: Removed the entire DashboardLiveTracker class and instance ---


# =====================================================
# LAYOUT (Vertically Compressed and Aligned)
# =====================================================
app.layout = html.Div([
    dcc.Store(id='chat-history-store', storage_type='memory', data=[]),
    
    # --- FIX: Add dcc.Store components to hold all shared data ---
    # This replaces the need for global variables and locks.
    # Data is stored as JSON-serializable dictionaries.
    dcc.Store(id='hist-data-store', storage_type='memory'),
    dcc.Store(id='live-data-store', storage_type='memory'),
    dcc.Store(id='hist-alerts-store', storage_type='memory'),
    dcc.Store(id='live-alerts-store', storage_type='memory'),
    # --- END FIX ---

    
    html.Div([
        html.H1("📈 Live Stock Dashboard", style={"textAlign": "center", "marginBottom": "20px"}),
        
        html.Div(id="ticker-cards", style={
            "display": "flex",
            "flexWrap": "wrap", 
            "justifyContent": "space-evenly", 
            "gap": "15px",
            "padding": "10px",
            "marginBottom": "20px"
        }),
        html.Hr(),
        
        html.Div([
            html.Label("Select Ticker:", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": t, "value": t} for t in cfg.TICKERS],
                value=cfg.TICKERS[0],
                clearable=False
            )
        ], style={"width": "300px", "margin": "0 auto 30px auto"}), 
        
        # --- Indicators are now 5-MIN ---
        html.H3("Key Live Indicators (5-Min)", style={"textAlign": "center"}),
        html.Div(id="key-indicators-panel", style={
            "display": "flex",
            "justifyContent": "space-evenly",
            "padding": "15px",
            "marginBottom": "30px",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "backgroundColor": "#f9f9f9"
        }),
        
        # Chart and Alert Columns
        html.Div([
            # --- Historical Column ---
            html.Div([
                html.H3("Historical Price (30-Day, 30-Min)", style={"textAlign": "center"}),
                dcc.Graph(id="price-chart-hist", style={'height': '350px'}), 
                html.H3("Historical Volume (30-Day, Daily)", style={"textAlign": "center"}),
                dcc.Graph(id="volume-chart-hist", style={'height': '250px'}), 
                html.H3("Historical Alerts (30-Min)"),
                html.Div(id="alerts-panel-hist", style={"whiteSpace": "pre-wrap", "height": "150px", "overflowY": "scroll", "border": "1px solid #ccc", "padding": "10px", "backgroundColor": "#fff"}),
            ], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
            
            # --- Live Column (Now "Recent 5-Min Data") ---
            html.Div([
                html.H3("Live Price (RTH, Last 30 Candles)", style={"textAlign": "center"}),
                dcc.Graph(id="price-chart-live", style={'height': '350px'}),
                html.H3("Live Volume (RTH, Last 30 Candles)", style={"textAlign": "center"}),
                dcc.Graph(id="volume-chart-live", style={'height': '250px'}),
                html.H3("Live Intraday Alerts (5-Min RTH)"),
                html.Div(id="alerts-panel-live", style={"whiteSpace": "pre-wrap", "height": "150px", "overflowY": "scroll", "border": "1px solid #ccc", "padding": "10px", "backgroundColor": "#fff"}),
                
                html.H3("Recent News Headlines", style={"textAlign": "center", "marginTop": "20px"}),
                html.Div(id="news-headlines-panel", style={"height": "150px", "overflowY": "scroll", "border": "1px solid #ccc", "padding": "10px", "backgroundColor": "#fff"}),
            ], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
            
        ], style={"display": "flex", "justifyContent": "space-between", "marginTop": "20px"}),

        # LLM Chat
        html.H2("💬 RAG-Enabled Chat", style={"marginTop": "40px"}),
        html.Div(id="llm-response", style={"whiteSpace": "pre-wrap", "marginTop": "10px", "height": "300px", "overflowY": "scroll", "border": "1px solid #ccc", "padding": "10px", "backgroundColor": "#fff"}),
        dcc.Textarea(id="llm-prompt", placeholder="Ask about 5-min data, 30-min data, or news...", style={"width": "100%", "height": 60, "marginTop": "10px"}),
        html.Button("Send", id="llm-send", n_clicks=0, style={"padding": "10px 20px", "backgroundColor": "#007bff", "color": "white", "border": "none", "borderRadius": "5px", "cursor": "pointer", "marginTop": "10px"}),
        
        
        dcc.Interval(id="interval", interval=600 * 1000, n_intervals=0),

        # --- FIX: Removed 'alerts-panel-live-hidden-trigger' ---
    
    ], style={"maxWidth": "1200px", "margin": "20px auto", "padding": "0 20px"}) 
])


# =====================================================
# CALLBACK: Ticker Cards & Global Data Store Update (MASTER CALLBACK)
# =====================================================
# --- FIX: This is now the master callback that loads ALL data from disk ---
# It runs every 'interval' and populates the dcc.Store components.
# All other callbacks are consumers of these stores.
# ---
@callback(
    Output("ticker-cards", "children"),
    Output("hist-data-store", "data"),
    Output("live-data-store", "data"),
    Output("hist-alerts-store", "data"),
    Output("live-alerts-store", "data"),
    Input("interval", "n_intervals")
)
def update_cards_and_global_stores(n):
    logger.info("Master data load: Reading all CSVs and alert JSONs from disk...")
    cards = []
    
    # --- 1. Load ALL data into serializable dictionaries ---
    hist_data_store = {}
    live_data_store = {}
    
    for t in cfg.ALL_FETCH_TICKERS:
        df_hist = load_data(t, "_hist.csv")
        df_live = load_data(t, "_live.csv")
        
        if not df_hist.empty:
            # Convert DataFrame to JSON serializable format
            hist_data_store[t] = df_hist.to_json(orient='split', date_format='iso')
        if not df_live.empty:
            live_data_store[t] = df_live.to_json(orient='split', date_format='iso')
    
    # --- 2. Load alerts from the JSON file ---
    all_alerts = load_alerts()
    hist_alerts_store = {}
    live_alerts_store = {}
    
    # --- 3. Process Ticker Cards (and populate alert stores) ---
    for t in cfg.TICKERS:
        # Re-constitute the live DataFrame from the JSON store data
        df_live = pd.DataFrame()
        if t in live_data_store:
            try:
                df_live = pd.read_json(live_data_store[t], orient='split', convert_dates=True)
                df_live.index = pd.to_datetime(df_live.index, utc=True)
            except Exception as e:
                logger.error(f"Failed to read live data from store for {t}: {e}")

        if not df_live.empty:
            df_rth = df_live.between_time(RTH_START, RTH_END)
        else:
            df_rth = pd.DataFrame()
        
        close = get_col(df_rth, t, "Close")
        vol = get_col(df_rth, t, "Volume", fill_na=0)
        
        # Get alerts from the loaded dictionary
        ticker_alerts = all_alerts.get(t, {})
        new_live_alerts = ticker_alerts.get("live", [])
        new_hist_alerts = ticker_alerts.get("historical", [])
        
        # --- OPTIMIZATION: Also get the indicators from the alert blob ---
        new_live_indicators = ticker_alerts.get("live_indicators", {})
        new_hist_indicators = ticker_alerts.get("historical_indicators", {})

        # --- Populate the alert stores for other callbacks ---
        # --- Store alerts AND indicators together ---
        live_alerts_store[t] = {
            "alerts": new_live_alerts,
            "indicators": new_live_indicators
        }
        hist_alerts_store[t] = {
            "alerts": new_hist_alerts,
            "indicators": new_hist_indicators
        }
        # --- END OPTIMIZATION ---
        
        if close.empty:
            price, total_volume, alert_text = "N/A", "N/A", "Market Closed / No Data"
        else:
            price = round(float(close.iloc[-1]), 2)
            total_volume = int(vol.sum()) 
            
            # Get the latest alert for the card
            alert_text = new_live_alerts[0] if new_live_alerts else "None"
            
        cards.append(html.Div([
            html.H3(t, style={"fontSize": "1.2em", "margin": "0 0 5px 0"}),
            html.P(f"Price: {price}", style={"fontSize": "1.1em", "fontWeight": "bold"}),
            html.P(f"RTH Volume: {total_volume}", style={"fontSize": "0.9em", "color": "#555"}), 
            html.P(f"Alert: {alert_text}", style={"fontSize": "0.8em", "color": "#999"})
        ],
        style={"border": "1px solid #ddd", "padding": "10px", "borderRadius": "5px", "minWidth": "200px", "flex": "1 1 300px", "backgroundColor": "#fff", "boxShadow": "2px 2px 5px rgba(0,0,0,0.05)"}))
        
    logger.info("Master data load complete. Updating stores.")
    return cards, hist_data_store, live_data_store, hist_alerts_store, live_alerts_store


# =====================================================
# CALLBACK: Historical Charts (Now reads from dcc.Store)
# =====================================================
@callback(
    Output("price-chart-hist", "figure"),
    Output("volume-chart-hist", "figure"), 
    Output("alerts-panel-hist", "children"),
    Input("ticker-dropdown", "value"),
    Input("hist-data-store", "data"),    # <-- FIX: Input from store
    Input("hist-alerts-store", "data") # <-- FIX: Input from store
)
def update_hist_charts(ticker, hist_data_json, hist_alerts_dict):
    logger.info(f"Updating historical charts for {ticker} from dcc.Store...")
    
    # --- 1. Re-constitute DataFrame from the store ---
    df_hist = pd.DataFrame()
    if hist_data_json and ticker in hist_data_json:
        try:
            df_hist = pd.read_json(hist_data_json[ticker], orient='split', convert_dates=True)
            df_hist.index = pd.to_datetime(df_hist.index, utc=True)
        except Exception as e:
            logger.error(f"Failed to read hist data from store for {ticker}: {e}")

    # --- 2. Get alerts from the store ---
    alerts_list = []
    if hist_alerts_dict and ticker in hist_alerts_dict:
        # --- OPTIMIZATION: Get alerts from the new store structure ---
        alerts_list = hist_alerts_dict[ticker].get("alerts", [])
    
    alerts_text = "\n".join([f"Historical - {a}" for a in alerts_list[-20:]]) \
                  if alerts_list else "No historical alerts found."

    close = get_col(df_hist, ticker, "Close")

    price_fig = go.Figure()
    if not close.empty:
        price_fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
        price_fig.update_layout(title=f"{ticker} 30-Day Price (30-Min)", xaxis_title="Datetime", yaxis_title="Price", template="plotly_white")
    else:
        price_fig.update_layout(title=f"{ticker} 30-Day Price (30-Min)", template="plotly_white", annotations=[{"text": "No Historical Data Available", "showarrow": False}])


    vol_fig = go.Figure()
    vol = get_col(df_hist, ticker, "Volume", fill_na=0)
    if not vol.empty:
        daily_vol = vol.resample('D').sum()
        daily_vol = daily_vol[daily_vol > 0]
        if not daily_vol.empty:
            vol_fig.add_trace(go.Bar(x=daily_vol.index, y=daily_vol, name=f"Volume ({ticker})"))
            vol_fig.update_layout(
                title=f"{ticker} 30-Day Daily Volume",
                xaxis_title="Date",
                yaxis_title="Total Volume",
                template="plotly_white"
            )
        else:
            vol_fig.update_layout(title=f"{ticker} 30-Day Daily Volume", template="plotly_white", annotations=[{"text": "No Historical Volume Available", "showarrow": False}])
    else:
        vol_fig.update_layout(title=f"{ticker} 30-Day Daily Volume", template="plotly_white", annotations=[{"text": "No Historical Volume Available", "showarrow": False}])


    return price_fig, vol_fig, alerts_text


# =====================================================
# CALLBACK: "Live" Charts (Now "Recent 5-Min", reads from dcc.Store)
# =====================================================
# --- RACE CONDITION FIX: Removed Input("interval", "n_intervals") ---
@callback(
    Output("price-chart-live", "figure"),
    Output("volume-chart-live", "figure"),
    Output("alerts-panel-live", "children"), 
    Output("key-indicators-panel", "children"), 
    Output("news-headlines-panel", "children"),
    Input("ticker-dropdown", "value"),
    Input("live-data-store", "data"),     # <-- FIX: Input from store
    Input("live-alerts-store", "data")  # <-- FIX: Input from store
)
def update_live_charts(ticker, live_data_json, live_alerts_dict): # <-- FIX: Removed n_intervals
    logger.info(f"Updating live (5-min) charts for {ticker} from dcc.Store...")
    
    # --- 1. LOAD NEWS HEADLINES (Still reads from disk, which is fine) ---
    all_articles = load_news_headlines()
    ticker_articles = [
        a for a in all_articles 
        if a.get('title') and ticker.lower() in a.get('title','').lower()
    ]
    
    news_elements = []
    if not ticker_articles:
        news_elements = [html.P("No recent news headlines found for this ticker.", style={"color": "#999", "padding": "10px"})]
    else:
        for article in ticker_articles[:10]: # Show top 10
            news_elements.append(
                html.Div([
                    html.A(
                        article.get('title', 'No Title'), 
                        href=article.get('url', '#'), 
                        target="_blank",
                        style={"fontWeight": "bold", "textDecoration": "none", "fontSize": "0.9em"}
                    ),
                    html.P(
                        f"{article.get('source', {}).get('name', 'Unknown Source')} - {article.get('publishedAt', '')}",
                        style={"fontSize": "0.8em", "color": "#777", "margin": "2px 0 10px 0"}
                    )
                ], style={"borderBottom": "1px solid #eee", "paddingBottom": "5px", "marginBottom": "5px"})
            )
    # --- END NEWS LOGIC ---

    # --- 2. Re-constitute DataFrame from the store ---
    df_live = pd.DataFrame()
    if live_data_json and ticker in live_data_json:
        try:
            df_live = pd.read_json(live_data_json[ticker], orient='split', convert_dates=True)
            df_live.index = pd.to_datetime(df_live.index, utc=True)
        except Exception as e:
            logger.error(f"Failed to read live data from store for {ticker}: {e}")

    if df_live.empty or len(df_live) < 21:
        empty_fig = go.Figure().update_layout(template="plotly_white", title="Not enough data for live charts/indicators (need 21 periods).")
        return empty_fig, empty_fig, "Waiting for pipeline data...", html.P("Not enough data to compute key indicators.", style={"textAlign": "center"}), news_elements

    df_rth = df_live.between_time(RTH_START, RTH_END)
    
    if df_rth.empty or len(df_rth) < 21:
        # Chart will show RTH filters are applied, but not enough data yet
        empty_fig = go.Figure().update_layout(template="plotly_white", title="Market Closed or No RTH Data (need 21 data points).")
        return empty_fig, empty_fig, "Market Closed. No RTH alerts.", html.P("Market Closed or No RTH Data.", style={"textAlign": "center"}), news_elements

    # --- 3. Get alerts and indicators from the store ---
    alerts_list = []
    indicators = {}
    if live_alerts_dict and ticker in live_alerts_dict:
        # --- OPTIMIZATION: Get alerts AND indicators from the new store structure ---
        alerts_list = live_alerts_dict[ticker].get("alerts", [])
        indicators = live_alerts_dict[ticker].get("indicators", {})

    # Format alerts with a timestamp (optional, but nice)
    now_str = datetime.now().strftime('%H:%M:%S')
    alerts_text = "\n".join([f"{now_str} - {a}" for a in alerts_list[-20:]]) \
                  if alerts_list else "No RTH alerts yet"

    # --- 4. CALCULATE KEY INDICATORS (on 5-min RTH data) ---
    # --- OPTIMIZATION: All TA calculation is removed ---
    
    # --- OPTIMIZATION: Get values from the 'indicators' dict ---
    rsi = indicators.get('RSI_14', np.nan)
    macd_hist = indicators.get('MACDh_12_26_9', np.nan)
    bb_width_pct = indicators.get('BBB_20_2.0', np.nan)
    
    # --- VWAP (Volume-Weighted Average Price) ---
    # --- OPTIMIZATION: VWAP calculation is REMOVED ---
    
    # --- OPTIMIZATION (NEW): Get VWAP and Close from indicators dict ---
    vwap = indicators.get('VWAP', np.nan)
    latest_close = indicators.get('Close', np.nan)
    # --- END VWAP OPTIMIZATION ---
    
    def format_indicator(name, value, unit="", color_logic=None):
        if pd.isna(value):
            return html.Div([html.P(name), html.P("N/A", style={"color": "#999"})], style={"textAlign": "center"})
        display_value = f"{value:.2f}{unit}"
        color = "#333" 
        if name == "RSI":
            display_value = f"{value:.1f}"
            if value >= 70: color = "red"
            elif value <= 30: color = "green"
        elif name == "MACD Hist.":
            if value > 0.01: color = "green"
            elif value < -0.01: color = "red"
        elif name == "VWAP Diff.":
            display_value = f"{value * 100:.2f}%"
            if value > 0: color = "green"
            elif value < 0: color = "red"
        elif name == "BBand Width %":
            display_value = f"{value:.2f}%"
        return html.Div([
            html.P(name, style={"fontWeight": "bold", "margin": "0"}), 
            html.P(display_value, style={"color": color, "fontSize": "1.4em", "margin": "5px 0"})
        ], style={"textAlign": "center"})

    vwap_diff = (latest_close - vwap) / vwap if vwap > 0 and not pd.isna(latest_close) else 0
    
    indicator_elements = [
        format_indicator("RSI (14)", rsi),
        format_indicator("MACD Hist.", macd_hist),
        format_indicator("BBand Width %", bb_width_pct),
        format_indicator("VWAP Diff.", vwap_diff)
    ]
    
    # --- 5. CHART UPDATES (Show last 30 5-min candles) ---
    df_chart_data = df_rth.tail(30).copy() 
    close = get_col(df_chart_data, ticker, "Close")
    vol_series = get_col(df_chart_data, ticker, "Volume", fill_na=0) 

    price_fig = go.Figure()
    if not close.empty:
        price_fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
        price_fig.update_layout(title=f"{ticker} Live Price (RTH, Last 30 Candles)", xaxis_title="Time", yaxis_title="Price", template="plotly_white")

    vol_fig = go.Figure()
    chart_title = f"{ticker} Live Volume (RTH, Last 30 Candles)"
    if not vol_series.empty:
        vol_fig.add_trace(go.Bar(x=vol_series.index, y=vol_series, name=f"Volume ({ticker})"))
        vol_fig.update_layout(
            title=chart_title,
            xaxis_title="Time",
            yaxis_title="Volume",
            yaxis=dict(type="linear", autorange=True),
            template="plotly_white"
        )
    
    return price_fig, vol_fig, alerts_text, indicator_elements, news_elements


# =====================================================
# CHAT LOGIC (Includes New Rule 5 Synthesis)
# =====================================================
@callback(
    Output("llm-response", "children"),
    Input("chat-history-store", "data")
)
def update_chat_display(chat_history):
    if not chat_history:
        return []
    
    messages = []
    for msg in chat_history:
        if msg.get("role") == "user":
            messages.append(html.P(f"You: {msg.get('content')}", style={"font-weight": "bold"}))
        elif msg.get("role") == "assistant":
            messages.append(dcc.Markdown(f"Analyst: {msg.get('content')}", style={"whiteSpace": "pre-wrap", "margin-top": "5px", "margin-bottom": "15px"}))
    return messages


# --- IMMUTABILITY FIX: This callback now creates new lists instead of mutating state ---
@callback(
    Output("chat-history-store", "data"), 
    Output("llm-prompt", "value"),        
    Input("llm-send", "n_clicks"),
    State("llm-prompt", "value"),
    State("chat-history-store", "data"),
    background=True, # <--- FIX: Enable background callback
    prevent_initial_call=True
)
def chat_llm(n_clicks, prompt, chat_history): 
    if not prompt:
        return chat_history, ""

    # --- FIX: Create a new list for the user prompt ---
    new_history = chat_history + [{"role": "user", "content": prompt}]

    # --- FIX: Use the new_history list to build the prompt ---
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in new_history])

    retrieved_context = search_rag_index(prompt, k=3)
    
    payload = f"""
System: You are a professional, friendly, and highly efficient financial analyst for a stock dashboard.
Your goal is to provide clear, direct, and non-redundant answers to the investor.

RULES:
1.  **START DIRECTLY:** Begin your response immediately with the direct answer to the user's latest question. Do not include greetings (e.g., "Hello," "Welcome").
2.  **STAY CONCISE:** Be professional and concise. Focus only on the *new* analysis or facts needed to answer the current question. Avoid repeating facts or analysis from the previous turn unless absolutely necessary for clarity.
3.  **SOURCE ONLY:** Answer based *only* on the provided context AND the conversation history. Do not use any outside knowledge. If the answer is not in the context, say "I do not have that specific information in my data files."
4.  **USE HOOKS:** If the context contains an "Insight/Hook" (a question asking the user if they want more detail), you **must include that full hook question** as the very last sentence of your response to encourage follow-up. Do not embed it in conversational filler.
5.  **SYNTHESIZE SIGNALS:** When providing analysis, you MUST explicitly synthesize data from the major signal categories (Technical, Volume, Market Context, Sentiment, Calendar) found in the context. Relate these findings to broader financial concepts (e.g., liquidity, risk sentiment) to add depth.
6.  **INTERPRET YES:** If the user's latest input is a simple affirmation (e.g., "Yes," "Go on," "Explain that"), interpret it as a direct request to **elaborate on the LAST hook question** you provided.
7.  **CLARITY ON DATES:** Always explicitly state the **ticker and the date** when citing closing prices or technical analysis results from the context.
8.  **ACTIONABLE TONE:** Maintain a tone that is **confident, helpful, and objective**. Never use phrases that express doubt or uncertainty (e.g., "I think," "maybe," "I hope").
9.  **RISK DISCLAIMER:** If offering analysis that suggests market movement (bullish/bearish), add a final, brief warning that all technical analysis involves risk and should not be considered financial advice.
10. **DATA INTEGRITY:** If the context is ambiguous, vague, or contradictory, state that the current data is inconclusive for a definitive judgment, but offer the known facts.
11. **AVOID REPETITION:** Never repeat the exact same hook question if the user has already acknowledged it in the previous turn (e.g., by saying "yes"). If a new hook is not available, offer a general prompt for the next question.
12. **FOCUS SHIFT:** If the user shifts the topic (e.g., from AAPL to GOOGL), drop the previous thread of questioning and address the new query immediately.
13. **VOLUME CONTEXT:** When discussing volume, frame it in terms of liquidity or investor interest relative to typical trading activity, rather than just listing the number.
14. **TREND TIMELINE:** When describing a trend (bullish/bearish), specify if it is a short-term (last 5-20 periods) or long-term trend based on the MA analysis in the context.
15. **NO GENERIC CHAT:** Do not engage in conversational chitchat or respond to greetings/farewells unless they are part of a substantive question.
16. **STRICT CONTEXT FILTER:** The primary focus of the current discussion is: [IDENTIFY THE TICKER FROM HISTORY]. Use this focus to strictly prioritize and filter context before generating the answer. Ignore data related to other tickers unless specifically asked for comparison.
17. **IF HOOK ANSWERED:** If the user affirms the hook (Rule 6) and you have provided the elaboration, ask a new, generic follow-up question (e.g., "What other ticker would you like to review?") to conclude the specific thread.

Conversation History:
---
{history_str}
---

Context from data files (for the user's *latest* question):
---
{retrieved_context}
---

Helpful Analyst Answer (for the latest User question only):
"""
    
    try:
        resp = llm.generate(payload)
        # --- FIX: Create a final new list for the assistant's response ---
        final_history = new_history + [{"role": "assistant", "content": resp}]
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        # --- FIX: Create a final new list for the error message ---
        final_history = new_history + [{"role": "assistant", "content": "Sorry, I encountered an error. Please try again."}]
    
    # --- FIX: Return the new, final list ---
    return final_history, ""


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    logger.info("Starting dashboard at http://127.0.0.1:8050")
    app.run(debug=False)