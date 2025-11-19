#!/usr/bin/env python3
import pandas as pd
import logging

logger = logging.getLogger("volume_signals")

# --- Configuration ---
VOLUME_SPIKE_FACTOR = 2.5 # 2.5x average volume is considered a spike

def compute_signals(df: pd.DataFrame, ticker: str) -> tuple[list[str], dict]:
    """
    Generates alerts based on volume patterns (Spikes, VWAP, Divergence).
    
    Args:
        df: The DataFrame for the current ticker.
        ticker: The stock ticker symbol.
        
    Returns:
        A tuple containing:
        1. alerts (list[str]): A list of human-readable alert strings.
        2. indicators (dict): A dictionary of the latest raw indicator values.
    """
    alerts = []
    indicators = {} # <-- NEW: Populated at the end
    
    # Need 10 periods for the rolling average and VWAP calculation to start
    if len(df) < 10: 
        return alerts, indicators # <-- Return both

    df = df.copy() 
    
    # 1. Calculate VWAP (Volume-Weighted Average Price)
    # This is calculated cumulatively from the start of the dataframe.
    df['PV'] = df['Close'] * df['Volume']

    # --- FIX: Ensure VWAP resets daily ---
    # Group by the date portion of the index, then calculate cumsum *within* each group.
    # This correctly handles both intraday (1-day) and historical (30-day) data.
    df['CumPV'] = df.groupby(df.index.date)['PV'].transform('cumsum')
    df['CumV'] = df.groupby(df.index.date)['Volume'].transform('cumsum')
    # --- END FIX ---
    
    df['VWAP'] = df['CumPV'] / df['CumV']
    
    # 2. Calculate Rolling Volume Average
    df['Vol_Avg_10'] = df['Volume'].rolling(window=10).mean()
    
    # Focus on the latest available data point
    latest = df.iloc[-1]
    
    # --- 1. Volume Spike Detection ---
    volume = latest['Volume']
    vol_avg = latest['Vol_Avg_10']
    
    if vol_avg > 0 and volume > vol_avg * VOLUME_SPIKE_FACTOR:
        alerts.append(f"Volume Spike: {volume/vol_avg:.1f}x 10-period average. Potential event detected.")

    # --- 2. VWAP Crossover ---
    if not pd.isna(latest['VWAP']):
        close_price = latest['Close']
        vwap = latest['VWAP']
        
        # Check current price relation to the average institutional price (VWAP)
        if close_price > vwap:
            alerts.append("Price Above VWAP: Suggests strong intraday buying pressure.")
        elif close_price < vwap:
            alerts.append("Price Below VWAP: Suggests weakness or intraday selling pressure.")

    # --- 3. Price-Volume Divergence (Trend Exhaustion) ---
    if len(df) >= 2:
        # Check if price moved up in the last period, but volume dropped significantly
        price_up = latest['Close'] > df.iloc[-2]['Close']
        volume_down = latest['Volume'] < df.iloc[-2]['Volume'] * 0.5 # Volume dropped by 50%
        
        # Alert if the upward price movement is not supported by volume
        if price_up and volume_down:
            alerts.append("Weak Volume on Uptrend: Potential trend exhaustion/reversal risk.")

    # --- NEW: Populate indicators dict ---
    try:
        if not pd.isna(latest['VWAP']):
            indicators['VWAP'] = latest['VWAP']
        if not pd.isna(latest['Close']):
            indicators['Close'] = latest['Close']
    except Exception as e:
        logger.warning(f"Failed to populate volume indicators dict for {ticker}: {e}")

    return alerts, indicators
