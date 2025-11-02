import pandas as pd
from scipy.signal import savgol_filter

def _pct_to_window(n, pct):
    return max(3, int(round(n * pct)))

def smooth_sentiment(df: pd.DataFrame, smooth_method="sma", smooth_params=None) -> pd.DataFrame:
    if smooth_params is None:
        smooth_params = {}
    n = len(df)
    s = df["sentiment"]
    if smooth_method == "sma":
        win = _pct_to_window(n, smooth_params.get("win_per", 0.1))
        sm = s.rolling(win, center=True, min_periods=1).mean()
    elif smooth_method == "es":
        span = _pct_to_window(n, smooth_params.get("span_per", 0.1))
        sm = s.ewm(span=span, adjust=False, min_periods=1).mean()
    elif smooth_method in ("s-g", "sg"):
        win = _pct_to_window(n, smooth_params.get("win_per", 0.1))
        if win % 2 == 0: win += 1
        poly = smooth_params.get("poly_order", 2)
        if win <= poly: win = poly + 1
        arr = savgol_filter(s.to_numpy(), window_length=win, polyorder=poly)
        sm = pd.Series(arr, index=s.index)
    else:
        raise ValueError(f"Unknown smoothing: {smooth_method}")
    df = df.copy()
    df["sentiment_smoothed"] = sm
    df["smooth_method"] = smooth_method
    df["smooth_params"] = [smooth_params] * len(df)
    return df
