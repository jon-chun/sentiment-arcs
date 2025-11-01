from pathlib import Path
from datetime import datetime
import pandas as pd

def save_sentiments(df, csv_path, txt_path, *, model_name, smooth_algo, smooth_params):
    csv_path = Path(csv_path); txt_path = Path(txt_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    now = datetime.utcnow().isoformat()
    meta = f"model={model_name}\nsmooth={smooth_algo}\nparams={smooth_params}\nexported={now}\nrows={len(df)}\n"
    txt_path.write_text(meta, encoding="utf-8")

def save_crux_text(crux_df, csv_path, txt_path, *, model_name, smooth_algo, smooth_params):
    csv_path = Path(csv_path); txt_path = Path(txt_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    crux_df.to_csv(csv_path, index=False)
    now = datetime.utcnow().isoformat()
    meta = f"model={model_name}\nsmooth={smooth_algo}\nparams={smooth_params}\nexported={now}\nrows={len(crux_df)}\n"
    txt_path.write_text(meta, encoding="utf-8")
