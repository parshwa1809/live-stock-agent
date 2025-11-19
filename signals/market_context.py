import pandas as pd
import logging

logger = logging.getLogger("market_context_signals")

# --- THIS IS THE CORRECTED FUNCTION ---
def compute_signals(df: pd.DataFrame, ticker: str, live_tracker, buffer_method_name: str) -> tuple[list[str], dict]:
    """
    Generates alerts by comparing the stock's performance to market indices (SPY and VIX).
    
    Args:
        df: The DataFrame for the current primary ticker.
        ticker: The stock ticker symbol.
        live_tracker: The LiveTracker instance.
        buffer_method_name: The string name of the method to call (e.g., "get_hist_buffer").
        
    Returns:
        A tuple containing:
        1. alerts (list[str]): A list of human-readable alert strings.
        2. indicators (dict): An empty dictionary (for API consistency).
    """
    alerts = []
    indicators = {} # <-- NEW: Empty dict for consistent return signature
    
    try:
        # Dynamically get the correct buffer-getter function
        get_buffer_func = getattr(live_tracker, buffer_method_name)
    except AttributeError:
        logger.error(f"MarketContext: LiveTracker has no method named '{buffer_method_name}'")
        return alerts, indicators # <-- Return both

    # Retrieve the context data buffers using the correct method
    spy_df = get_buffer_func("SPY")
    vix_df = get_buffer_func("^VIX") # <-- Corrected from 'VIX' to '^VIX'
    
    # Need at least 15 periods for VIX analysis and 5 for relative strength
    if df.empty or spy_df.empty or vix_df.empty or len(df) < 15:
        return alerts, indicators # <-- Return both
    
    # Align indices to ensure we are comparing the same time periods
    common_index = df.index.intersection(spy_df.index)
    if len(common_index) < 5:
        return alerts, indicators # Not enough overlapping data to compare

    df_compare = df.loc[common_index].tail(5)
    spy_compare = spy_df.loc[common_index].tail(5)
    
    if len(df_compare) < 2 or len(spy_compare) < 2:
        return alerts, indicators # Need at least 2 points to calculate return

    # 1. Relative Strength vs. SPY (S&P 500)
    try:
        stock_ret = (df_compare['Close'].iloc[-1] / df_compare['Close'].iloc[0]) - 1
        spy_ret = (spy_compare['Close'].iloc[-1] / spy_compare['Close'].iloc[0]) - 1
        relative_strength = stock_ret - spy_ret
        
        if relative_strength > 0.005: 
            alerts.append(f"Strong Relative Strength: Outperforming SPY (+{relative_strength*100:.2f}%) recently.")
        elif relative_strength < -0.005:
            alerts.append(f"Weak Relative Strength: Underperforming SPY ({relative_strength*100:.2f}%) recently.")
    except Exception as e:
        logger.warning(f"Failed to calculate Relative Strength for {ticker}: {e}")


    # 2. Risk Sentiment via VIX (Volatility Index)
    # Align VIX data
    vix_common_index = df.index.intersection(vix_df.index)
    if len(vix_common_index) < 15:
        return alerts, indicators # <-- Return both

    vix_compare = vix_df.loc[vix_common_index].tail(15)
    if len(vix_compare) == 15:
        try:
            vix_change = (vix_compare['Close'].iloc[-1] / vix_compare['Close'].iloc[0]) - 1
            
            if vix_change > 0.03: 
                alerts.append(f"Risk Sentiment Alert: VIX rising quickly (+{vix_change*100:.1f}%). Increased market fear.")
            elif vix_change < -0.03:
                alerts.append(f"Risk Sentiment Alert: VIX falling quickly ({vix_change*100:.1f}%). Decreased market fear.")
        except Exception as e:
            logger.warning(f"Failed to calculate VIX change: {e}")

    return alerts, indicators