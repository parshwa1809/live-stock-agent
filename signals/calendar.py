import pandas as pd
from datetime import datetime, time, timedelta
import logging

logger = logging.getLogger("calendar_signals")

# Define Regular Trading Hours (RTH) in UTC
RTH_OPEN = time(14, 30, 0) # 9:30 AM EST
RTH_CLOSE = time(21, 0, 0) # 4:00 PM EST

def compute_signals(df: pd.DataFrame, ticker: str) -> tuple[list[str], dict]:
    """
    Generates alerts based on time-of-day, day-of-week, and known cyclical patterns.
    
    Args:
        df: The DataFrame for the current ticker.
        ticker: The stock ticker symbol.
        
    Returns:
        A tuple containing:
        1. alerts (list[str]): A list of human-readable alert strings.
        2. indicators (dict): An empty dictionary (for API consistency).
    """
    alerts = []
    indicators = {} # <-- NEW: Empty dict for consistent return signature
    
    if df.empty:
        return alerts, indicators # <-- Return both

    # Use the timestamp of the latest data point
    latest_timestamp = df.index[-1]
    latest_time = latest_timestamp.time()
    latest_day = latest_timestamp.weekday() # Monday is 0, Sunday is 6
    
    # 1. Time-of-Day Volatility (First and Last 30 Minutes of RTH)
    
    # Calculate the end time of the opening volatility period
    open_vol_end = (datetime.combine(datetime.min, RTH_OPEN) + timedelta(minutes=30)).time()
    
    # Calculate the start time of the closing volatility period
    close_vol_start = (datetime.combine(datetime.min, RTH_CLOSE) - timedelta(minutes=30)).time()
    
    if RTH_OPEN <= latest_time <= open_vol_end:
        alerts.append("Market Open Volatility: Expect high liquidity and price swings.")
    elif close_vol_start <= latest_time <= RTH_CLOSE:
        alerts.append("Approaching Market Close: Volatility typically increases in the final 30 minutes.")

    # 2. Day of Week Context
    if latest_day == 0: # Monday
        alerts.append("Monday Trading Context: Watch for gap fill or weekend news reaction.")
    elif latest_day == 4: # Friday
        alerts.append("Friday Trading Context: Potential for squaring positions or low liquidity near close.")

    return alerts, indicators