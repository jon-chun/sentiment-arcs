import pandas as pd

def normalize_sentiment(df: pd.DataFrame, col="sentiment_smoothed") -> pd.DataFrame:
    vals = df[col]
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        norm = vals * 0
    else:
        norm = 2 * (vals - vmin) / (vmax - vmin) - 1
    df = df.copy()
    df["sentiment_norm"] = norm
    return df
