import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger("technical_signals")

# --- Configuration ---
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

def compute_signals(df: pd.DataFrame, ticker: str) -> tuple[list[str], dict]:
    """
    Generates alerts based on standard technical indicators (RSI, MACD, Bollinger Bands).
    
    Args:
        df: The DataFrame for the current ticker.
        ticker: The stock ticker symbol.
        
    Returns:
        A tuple containing:
        1. alerts (list[str]): A list of human-readable alert strings.
        2. indicators (dict): A dictionary of the latest raw indicator values.
    """
    alerts = []
    indicators = {} # <-- NEW: Dictionary to hold raw values
    
    # Need at least 21 periods for reliable 20-period BBands/SMA/MACD calculations
    if len(df) < 21:
        return alerts, indicators # <-- Return both
    
    # 1. Calculate indicators (using pandas-ta for vectorized efficiency)
    try:
        df = df.copy() # Operate on a copy to ensure immutability
        
        # RSI (14 periods) - Measures momentum
        df.ta.rsi(close='Close', length=14, append=True)
        # MACD (12, 26, 9) - Measures trend following and momentum
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        # Bollinger Bands (20 periods, 2 standard deviations) - Measures volatility
        df.ta.bbands(close='Close', length=20, std=2, append=True) 
        
    except Exception as e:
        logger.error(f"Error calculating TA for {ticker}: {e}")
        return alerts, indicators # <-- Return both

    # Focus on the latest available data point for signaling
    latest = df.iloc[-1]
    
    # --- 1. RSI (Momentum Alerts) ---
    rsi = latest['RSI_14']
    if rsi >= RSI_OVERBOUGHT:
        alerts.append(f"RSI Overbought ({rsi:.1f}): Potential short-term correction risk.")
    elif rsi <= RSI_OVERSOLD:
        alerts.append(f"RSI Oversold ({rsi:.1f}): Potential for a short-term bounce.")

    # --- 2. MACD Crossover (Trend Reversal/Continuation) ---
    macd_line = latest['MACD_12_26_9']
    signal_line = latest['MACDs_12_26_9']
    
    # Need previous period data to detect a "cross" event
    prev_macd = df.iloc[-2]['MACD_12_26_9']
    prev_signal = df.iloc[-2]['MACDs_12_26_9']
    
    # Bullish Cross: MACD line crosses ABOVE Signal line
    if (macd_line > signal_line and prev_macd <= prev_signal):
        alerts.append("MACD Bullish Crossover: Short-term momentum shifting upward.")
    # Bearish Cross: MACD line crosses BELOW Signal line
    elif (macd_line < signal_line and prev_macd >= prev_signal):
        alerts.append("MACD Bearish Crossover: Short-term momentum shifting downward.")

    # --- 3. Bollinger Bands (Volatility and Breakout) ---
    lower_band = latest['BBL_20_2.0']
    upper_band = latest['BBU_20_2.0']
    close_price = latest['Close']
    
    if close_price > upper_band:
        alerts.append(f"Bollinger Breakout: Price {close_price:.2f} above Upper Band - high volatility detected.")
    elif close_price < lower_band:
        alerts.append(f"Bollinger Breakout: Price {close_price:.2f} below Lower Band - high volatility detected.")

    # --- NEW: Populate the indicators dictionary ---
    try:
        indicators = {
            'RSI_14': latest.get('RSI_14', None),
            'MACD_12_26_9': latest.get('MACD_12_26_9', None),
            'MACDs_12_26_9': latest.get('MACDs_12_26_9', None),
            'MACDh_12_26_9': latest.get('MACDh_12_26_9', None),
            'BBL_20_2.0': latest.get('BBL_20_2.0', None),
            'BBU_20_2.0': latest.get('BBU_20_2.0', None),
            'BBB_20_2.0': latest.get('BBB_20_2.0', None), # Bollinger Band Width
            'BBP_20_2.0': latest.get('BBP_20_2.0', None)  # Bollinger Band Percentage
        }
    except Exception as e:
        logger.warning(f"Failed to populate indicators dict for {ticker}: {e}")
        
    return alerts, indicators