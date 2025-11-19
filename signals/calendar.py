import pandas as pd
from datetime import datetime, time, timedelta
import logging
import pytz # <-- NEW IMPORT

logger = logging.getLogger("calendar_signals")

# Define timezone and RTH in local time (EST/EDT)
EASTERN_TZ = pytz.timezone('America/New_York')

RTH_OPEN = time(9, 30, 0) 
RTH_CLOSE = time(16, 0, 0) 

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
    indicators = {} 
    
    if df.empty:
        return alerts, indicators 

    # Use the timestamp of the latest data point
    latest_timestamp_utc = df.index[-1]
    
    # CRITICAL FIX: Convert to Eastern Time for RTH checks
    latest_timestamp_local = latest_timestamp_utc.tz_convert(EASTERN_TZ)
    latest_time = latest_timestamp_local.time()
    latest_day = latest_timestamp_local.weekday() 
    
    # 1. Time-of-Day Volatility (First and Last 30 Minutes of RTH)
    
    # Calculate the end time of the opening volatility period
    open_vol_end = (datetime.combine(datetime.min, RTH_OPEN, tzinfo=EASTERN_TZ) + timedelta(minutes=30)).time()
    
    # Calculate the start time of the closing volatility period
    close_vol_start = (datetime.combine(datetime.min, RTH_CLOSE, tzinfo=EASTERN_TZ) - timedelta(minutes=30)).time()
    
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