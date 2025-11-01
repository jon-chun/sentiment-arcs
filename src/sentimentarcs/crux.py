import pandas as pd
try:
    from scipy.signal import find_peaks
    _SCIPY = True
except ImportError:
    _SCIPY = False

def detect_cruxes(df: pd.DataFrame, **params) -> list[int]:
    if not _SCIPY:
        raise RuntimeError("scipy is required for crux detection")
    s = df["sentiment_norm"]
    peaks, _ = find_peaks(s.to_numpy(), **params)
    return peaks.tolist()

def extract_crux_text(df: pd.DataFrame, crux_indices: list[int], half_win: int = 5) -> pd.DataFrame:
    max_i = len(df) - 1
    rows = []
    for idx in crux_indices:
        start = max(0, idx - half_win)
        end = min(max_i, idx + half_win)
        for i in range(start, end + 1):
            txt = df.loc[i, "text"]
            if i == idx:
                txt = txt.upper()
            rows.append({"line_no": int(df.loc[i, "line_no"]), "text": txt, "crux_center": int(df.loc[idx, "line_no"])})
    return pd.DataFrame(rows)
