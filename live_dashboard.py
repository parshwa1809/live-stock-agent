#!/usr/bin/env python3
"""
Unified Live Stock Dashboard — Phase 2
(FIXED: RAG-enabled LLM Chat with "Smart Hooks", engaging prompts, and Chat History)
(FINAL FIX: LLM chat callback forced to run on main thread to bypass macOS/MPS/fork crash, and system prompt is now strict and concise)
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
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path 

# Import only the non-instance items from the pipeline
from phase2_pipeline import cfg, alert_engine, llm, flatten_columns

# Set up the cache manager for background callbacks
# --- FINAL FIX: Removed executor argument due to incompatibility ---
cache = diskcache.Cache("./cache", timeout=120) 
long_callback_manager = DiskcacheManager(cache) 
# --- END FINAL FIX ---


DATA_PATH = Path("./data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("live_dashboard")

# Pass the correct argument for new Dash versions
app = Dash(__name__, background_callback_manager=long_callback_manager) 
server = app.server

# Stores the text of alerts for display
hist_alert_history = {t: [] for t in cfg.TICKERS} 
live_alert_history = {t: [] for t in cfg.TICKERS}
# Stores the *set* of last known alerts to prevent duplicates
last_known_hist_alerts = {t: set() for t in cfg.TICKERS}
last_known_live_alerts = {t: set() for t in cfg.TICKERS}

# Define Regular Trading Hours (RTH) in UTC
RTH_START = '14:30'
RTH_END = '21:00'

# =====================================================
# RAG: Load models and index globally for performance
# =====================================================
RAG_INDEX_FILE = DATA_PATH / "stock_index.faiss"
RAG_TEXTS_FILE = DATA_PATH / "stock_texts.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- RAG FIX: MODEL LOADING MOVED TO FUNCTION ---
embedding_model_cache = None 
# =====================================================

# =====================================================
# RAG: Helper function to search the index - MODEL LOADING MOVED HERE
# =====================================================
def search_rag_index(query: str, k=3) -> str: # k=3 for concise context
    """
    Embeds a query, searches the FAISS index, and returns the top k text chunks.
    """
    global embedding_model_cache
    
    # --- RAG FIX 2: LAZY LOAD MODEL ---
    # Load the model only if it hasn't been cached yet (First run in worker thread)
    if embedding_model_cache is None:
        try:
            logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' in worker thread...")
            embedding_model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"FATAL: Could not load SentenceTransformer model. RAG will fail. Error: {e}")
            return "Error: Embedding model is not loaded."
    # --- END LAZY LOAD ---
        
    if not RAG_INDEX_FILE.exists() or not RAG_TEXTS_FILE.exists():
        logger.warning("FAISS index or text file not found. Returning no context.")
        return "No data in RAG index. Please wait for pipeline to run."
        
    try:
        # 1. Load index and texts
        index = faiss.read_index(str(RAG_INDEX_FILE))
        with open(RAG_TEXTS_FILE, 'r') as f:
            texts = json.load(f)
            
        # 2. Embed the query
        query_vector = embedding_model_cache.encode([query]).astype('float32')
        
        # 3. Search the index
        distances, indices = index.search(query_vector, k)
        
        # 4. Retrieve the text chunks
        results = [texts[i] for i in indices[0]]
        
        # 5. Format as context
        return "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"Error during RAG search: {e}")
        return "Error searching RAG index."

# =====================================================
# LOAD CSV (Now a generic loader)
# =====================================================
def load_data(ticker: str, file_suffix: str) -> pd.DataFrame:
    path = DATA_PATH / f"{ticker}{file_suffix}"
    if not path.exists():
        logger.warning(f"{ticker}{file_suffix} not found at {path}")
        return pd.DataFrame()
    try:
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            df = pd.read_csv(path, index_col=0, header=[0,1])
            
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index = df.index.tz_convert('UTC') # Ensure index is UTC
        df = flatten_columns(df, ticker)
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()


# =====================================================
# SAFE COLUMN GETTER
# =====================================================
def get_col(df, ticker, col, fill_na=None):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    suff = f"{col}_{ticker}"
    if suff in df.columns:
        s = pd.to_numeric(df[suff], errors="coerce")
    elif col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(dtype=float)
        for c in df.columns:
            if isinstance(c, str) and c.startswith(col):
                s = pd.to_numeric(df[c], errors="coerce")
                break
    
    if s.empty:
        return pd.Series(dtype=float)

    if fill_na is not None:
        return s.fillna(fill_na)
    else:
        return s.dropna()


# =====================================================
# LAYOUT (4 charts) - REVERTED TO ORIGINAL STRUCTURE
# =====================================================
app.layout = html.Div([
    # --- ADDED: Session storage for chat history ---
    dcc.Store(id='chat-history-store', storage_type='memory', data=[]),
    
    html.H1("📈 Live Stock Dashboard", style={"textAlign": "center"}),
    
    # REVERTED: Back to simple div structure
    html.Div(id="ticker-cards", style={"display": "flex", "justifyContent": "space-around"}),
    html.Hr(),
    
    html.Div([
        html.Label("Select Ticker:"),
        dcc.Dropdown(
            id="ticker-dropdown",
            options=[{"label": t, "value": t} for t in cfg.TICKERS],
            value=cfg.TICKERS[0],
            clearable=False
        )
    ], style={"width": "250px", "margin": "auto"}),
    
    html.Div([
        # --- Historical Column ---
        html.Div([
            html.H3("Historical Price (30-Day, 30-Min)", style={"textAlign": "center"}),
            dcc.Graph(id="price-chart-hist"),
            html.H3("Historical Volume (30-Day, Daily)", style={"textAlign": "center"}),
            dcc.Graph(id="volume-chart-hist"), 
            html.H3("Historical Alerts"),
            html.Div(id="alerts-panel-hist", style={"whiteSpace": "pre-wrap", "height": "150px", "overflowY": "scroll", "border": "1px solid #ccc"}),
        ], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
        
        # --- Live Column ---
        html.Div([
            html.H3("Live Price (RTH, Last 30 Mins)", style={"textAlign": "center"}),
            dcc.Graph(id="price-chart-live"),
            html.H3("Live Volume (RTH, Last 30 Mins)", style={"textAlign": "center"}),
            dcc.Graph(id="volume-chart-live"),
            html.H3("Live Intraday Alerts (RTH)"),
            html.Div(id="alerts-panel-live", style={"whiteSpace": "pre-wrap", "height": "150px", "overflowY": "scroll", "border": "1px solid #ccc"}),
        ], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
        
    ], style={"marginTop": "20px"}),

    # LLM Chat
    html.H2("💬 RAG-Enabled Chat"),
    # --- FIXED SCROLLING: Keep the fixed height for the history panel ---
    html.Div(id="llm-response", style={"whiteSpace": "pre-wrap", "marginTop": "10px", "height": "300px", "overflowY": "scroll", "border": "1px solid #ccc"}),
    dcc.Textarea(id="llm-prompt", placeholder="Ask about any stock's live or historical data...", style={"width": "100%", "height": 60}),
    html.Button("Send", id="llm-send", n_clicks=0),
    
    
    # Interval to refresh live data (60s)
    dcc.Interval(id="interval", interval=cfg.REFRESH_INTERVAL * 1000, n_intervals=0)
])


# =====================================================
# CALLBACK: Ticker Cards & Global Alert Update - REVERTED STYLING
# =====================================================
@callback(
    Output("ticker-cards", "children"),
    Output("alerts-panel-live-hidden-trigger", "children"), 
    Input("interval", "n_intervals")
)
def update_cards_and_global_alerts(n):
    cards = []
    
    for t in cfg.TICKERS:
        df = load_data(t, "_live.csv")
        
        if not df.empty:
            df_rth = df.between_time(RTH_START, RTH_END)
        else:
            df_rth = pd.DataFrame()
        
        close = get_col(df_rth, t, "Close")
        vol = get_col(df_rth, t, "Volume", fill_na=0)
        
        new_alerts = alert_engine.evaluate(t, df_rth)
        new_alerts_set = set(new_alerts)
        if new_alerts_set and new_alerts_set != last_known_live_alerts[t]:
            new_alerts_to_add = [a for a in new_alerts if a not in last_known_live_alerts[t]]
            if new_alerts_to_add:
                live_alert_history[t].extend([
                    f"{datetime.now().strftime('%H:%M:%S')} - {a}" for a in new_alerts_to_add
                ])
            last_known_live_alerts[t] = new_alerts_set

        if close.empty:
            price, total_volume, alert_text = "N/A", "N/A", "Market Closed"
        else:
            price = round(float(close.iloc[-1]), 2)
            total_volume = int(vol.sum()) 
            alert_text = live_alert_history[t][-1] if live_alert_history[t] else "None"
            
        # REVERTED: Back to simple Div styling
        cards.append(html.Div([
            html.H3(t),
            html.P(f"Current RTH Price: {price}"),
            html.P(f"Total RTH Volume: {total_volume:,}"),
            html.P(f"Last RTH Alert: {alert_text.split(' - ')[-1]}")
        ],
        style={"border": "1px solid #ccc", "padding": "10px", "borderRadius": "5px"}))
        
    return cards, ""


# =====================================================
# CALLBACK: Historical Charts - REVERTED STYLING
# =====================================================
@callback(
    Output("price-chart-hist", "figure"),
    Output("volume-chart-hist", "figure"), 
    Output("alerts-panel-hist", "children"),
    Input("ticker-dropdown", "value")
)
def update_hist_charts(ticker):
    df = load_data(ticker, "_hist.csv")
    close = get_col(df, ticker, "Close")

    new_alerts = alert_engine.evaluate(ticker, df)
    new_alerts_set = set(new_alerts)
    if new_alerts_set and new_alerts_set != last_known_hist_alerts[ticker]:
        new_alerts_to_add = [a for a in new_alerts if a not in last_known_hist_alerts[ticker]]
        if new_alerts_to_add:
            hist_alert_history[ticker].extend([
                f"{datetime.now().strftime('%H:%M:%S')} - {a}" for a in new_alerts_to_add
            ])
        last_known_hist_alerts[ticker] = new_alerts_set
    
    alerts_text = "\n".join(hist_alert_history[ticker][-20:]) if hist_alert_history[ticker] else "No alerts yet"

    # REVERTED: Removed Plotly theme settings
    price_fig = go.Figure()
    if not close.empty:
        price_fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
        price_fig.update_layout(title=f"{ticker} 30-Day Price", xaxis_title="Datetime", yaxis_title="Price")

    vol_fig = go.Figure()
    vol = get_col(df, ticker, "Volume", fill_na=0)
    if not vol.empty:
        daily_vol = vol.resample('D').sum()
        daily_vol = daily_vol[daily_vol > 0] 
        vol_fig.add_trace(go.Bar(x=daily_vol.index, y=daily_vol, name=f"Volume ({ticker})"))
        vol_fig.update_layout(
            title=f"{ticker} 30-Day Daily Volume",
            xaxis_title="Date",
            yaxis_title="Total Volume"
        )

    return price_fig, vol_fig, alerts_text


# =====================================================
# CALLBACK: Live Charts - REVERTED STYLING
# =====================================================
@callback(
    Output("price-chart-live", "figure"),
    Output("volume-chart-live", "figure"),
    Output("alerts-panel-live", "children"), 
    Input("ticker-dropdown", "value"),
    Input("interval", "n_intervals"),
    Input("alerts-panel-live-hidden-trigger", "children") 
)
def update_live_charts(ticker, n, alert_trigger):
    df = load_data(ticker, "_live.csv")

    if not df.empty:
        df_rth = df.between_time(RTH_START, RTH_END)
    else:
        df_rth = pd.DataFrame()

    alerts_text = "\n".join(live_alert_history[ticker][-20:]) if live_alert_history[ticker] else "No RTH alerts yet"

    df_chart_data = df_rth.tail(30).copy()
    
    close = get_col(df_chart_data, ticker, "Close")
    vol_series = get_col(df_chart_data, ticker, "Volume", fill_na=0) 

    # REVERTED: Removed Plotly theme settings
    price_fig = go.Figure()
    if not close.empty:
        price_fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
        price_fig.update_layout(title=f"{ticker} Live Price", xaxis_title="Time", yaxis_title="Price")

    vol_fig = go.Figure()
    chart_title = f"{ticker} Live Volume"
    
    if not vol_series.empty:
        real_vol = vol_series[vol_series > 0]
        if not real_vol.empty:
            q95 = real_vol.quantile(0.95)
            # Use a robust threshold (e.g., 3x 95th percentile)
            threshold = q95 * 3
            vol_filtered = vol_series.where(vol_series < threshold, np.nan)
        else:
            vol_filtered = vol_series 
            
        vol_fig.add_trace(go.Bar(x=vol_filtered.index, y=vol_filtered, name=f"Volume ({ticker})"))
        
        vol_fig.update_layout(
            title=chart_title,
            xaxis_title="Time",
            yaxis_title="Volume",
            yaxis=dict(type="linear", autorange=True) 
        )
    else:
        logger.warning(f"Live Volume series for {ticker} is empty, chart will be blank.")

    return price_fig, vol_fig, alerts_text


# =====================================================
# CALLBACK: NEW - Renders the chat history - ADJUSTED STYLE BACK TO ORIGINAL
# =====================================================
@callback(
    Output("llm-response", "children"),
    Input("chat-history-store", "data")
)
def update_chat_display(chat_history):
    if not chat_history:
        return []
    
    # Format the history into HTML components (Reverted to simple styling)
    messages = []
    for msg in chat_history:
        if msg.get("role") == "user":
            messages.append(html.P(f"You: {msg.get('content')}", style={'font-weight': 'bold'}))
        elif msg.get("role") == "assistant":
            # Using dcc.Markdown for LLM response formatting (functionality retained)
            messages.append(dcc.Markdown(f"Analyst: {msg.get('content')}", style={'whiteSpace': 'pre-wrap', 'margin-top': '5px', 'margin-bottom': '15px'}))
    return messages


# =====================================================
# CALLBACK: RAG-ENABLED LLM CHAT (with History) - FORCED TO MAIN THREAD
# =====================================================
@callback(
    Output("chat-history-store", "data"), # <-- Stores chat history
    Output("llm-prompt", "value"),         # <-- Output to clear the prompt
    Input("llm-send", "n_clicks"),
    State("llm-prompt", "value"),
    State("chat-history-store", "data"), # <-- Gets current history
    # CRITICAL FIX: REMOVED background=True AND running ARGUMENTS
    prevent_initial_call=True
)
def chat_llm(n_clicks, prompt, chat_history): 
    # Check for empty prompt before processing
    if not prompt:
        return chat_history, "" # Return history as is, clear the prompt input

    # 1. Add user's new message to history
    chat_history.append({"role": "user", "content": prompt})

    # 2. Format history for the LLM
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # 3. RETRIEVE: Search RAG index for context
    retrieved_context = search_rag_index(prompt, k=3)
    
    # 4. AUGMENT: Build the prompt (Prompt Engineering)
    payload = f"""
System: You are a professional, friendly, and highly efficient financial analyst for a stock dashboard.
Your goal is to provide clear, direct, and non-redundant answers to the investor.

RULES:
1.  **START DIRECTLY:** Begin your response immediately with the direct answer to the user's latest question. Do not include greetings (e.g., "Hello," "Welcome").
2.  **STAY CONCISE:** Be professional and concise. Focus only on the *new* analysis or facts needed to answer the current question. Avoid repeating facts or analysis from the previous turn unless absolutely necessary for clarity.
3.  **SOURCE ONLY:** Answer based *only* on the provided context AND the conversation history. Do not use any outside knowledge. If the answer is not in the context, say "I do not have that specific information in my data files."
4.  **USE HOOKS:** If the context contains an "Insight/Hook" (a question asking the user if they want more detail), you **must include that full hook question** as the very last sentence of your response to encourage follow-up. Do not embed it in conversational filler.
5.  **ADD DEPTH:** When providing analysis or facts, briefly relate the data point (trend, momentum, volume) to a broader financial concept or investment implication. This adds intelligence and context.
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
16. **IF HOOK ANSWERED:** If the user affirms the hook (Rule 6) and you have provided the elaboration, ask a new, generic follow-up question (e.g., "What other ticker would you like to review?") to conclude the specific thread.
17. **STRICT CONTEXT FILTER:** The primary focus of the current discussion is: [IDENTIFY THE TICKET FROM HISTORY]. Use this focus to strictly prioritize and filter context before generating the answer. Ignore data related to other tickers unless specifically asked for comparison.

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
    
    # 5. GENERATE: Call the LLM
    try:
        resp = llm.generate(payload)
        # 6. Add assistant's response to history
        chat_history.append({"role": "assistant", "content": resp})
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
    
    # 7. Return the *updated* full history AND clear the prompt input field
    return chat_history, ""


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    # Add a hidden div to act as a trigger for alert refreshes
    app.layout.children.append(html.Div(id="alerts-panel-live-hidden-trigger", style={"display": "none"}))
    logger.info("Starting dashboard at http://127.0.0.1:8050")
    app.run(debug=False)