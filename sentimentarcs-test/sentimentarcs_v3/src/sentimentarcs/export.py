from pathlib import Path
from datetime import datetime

def save_sentiments(df, csv_path, txt_path, *, model_name, smooth_algo, smooth_params):
    csv_path = Path(csv_path); txt_path = Path(txt_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    now = datetime.utcnow().isoformat()
    txt = (
        f"model={model_name}\n"
        f"smooth={smooth_algo}\n"
        f"params={smooth_params}\n"
        f"exported={now}\n"
        f"rows={len(df)}\n"
    )
    txt_path.write_text(txt, encoding="utf-8")

def save_crux_text(crux_df, csv_path, txt_path, *, model_name, smooth_algo, smooth_params):
    csv_path = Path(csv_path); txt_path = Path(txt_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    crux_df.to_csv(csv_path, index=False)
    now = datetime.utcnow().isoformat()
    txt = (
        f"model={model_name}\n"
        f"smooth={smooth_algo}\n"
        f"params={smooth_params}\n"
        f"exported={now}\n"
        f"rows={len(crux_df)}\n"
    )
    txt_path.write_text(txt, encoding="utf-8")
