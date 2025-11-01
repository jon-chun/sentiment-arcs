import os
import pandas as pd

def _sent_vader(lines):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    return [sid.polarity_scores(t)["compound"] for t in lines]

def _sent_textblob(lines):
    from textblob import TextBlob
    return [TextBlob(t).sentiment.polarity for t in lines]

def _sent_hf(lines, model_name):
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model=model_name)
    out = []
    for t in lines:
        r = clf(t)[0]
        label = r["label"].upper()
        score = float(r["score"])
        sign = 1.0 if "POS" in label or "5" in label or "4" in label else -1.0
        out.append(sign * score)
    return out

def get_sentiment_text(lines_ls: list[str], model_name: str) -> pd.DataFrame:
    if model_name == "vader":
        scores = _sent_vader(lines_ls)
    elif model_name == "textblob":
        scores = _sent_textblob(lines_ls)
    elif model_name == "distilbert":
        scores = _sent_hf(lines_ls, "distilbert-base-uncased-finetuned-sst-2-english")
    elif model_name == "nlptown":
        scores = _sent_hf(lines_ls, "nlptown/bert-base-multilingual-uncased-sentiment")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    df = pd.DataFrame({
        "line_no": list(range(len(lines_ls))),
        "text": lines_ls,
        "sentiment": scores,
        "model_name": model_name,
    })
    df.name = f"{model_name}_df"
    return df
