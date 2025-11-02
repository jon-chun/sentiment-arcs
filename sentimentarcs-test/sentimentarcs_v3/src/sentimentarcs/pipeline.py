from pathlib import Path
from .settings import load_settings
from .filenames import parse_input_filename, build_plot_filename, build_ts_filename, build_crux_filename
from .cleaning import clean_text
from .segmentation import segment_text
from .sentiment import get_sentiment_text
from .smoothing import smooth_sentiment
from .normalize import normalize_sentiment
from .crux import detect_cruxes, extract_crux_text
from .plotting import plot_sentiments
from .export import save_sentiments, save_crux_text

def run_pipeline(
    input_path,
    model_name="vader",
    clean_tech=None,
    smooth_method=None,
    smooth_params=None,
    crux_params=None,
):
    cfg = load_settings()
    meta = parse_input_filename(input_path)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = Path(input_path).read_text(encoding="utf-8")
    text_clean = clean_text(raw_text)

    if clean_tech is None:
        clean_tech = cfg.get("text_defaults", {}).get("clean_tech", "pysbd")
    segments = segment_text(text_clean, clean_tech)

    df = get_sentiment_text(segments, model_name)

    if smooth_method is None:
        smooth_method = cfg.get("smoothing_defaults", {}).get("method", "sma")
    if smooth_params is None:
        sd = cfg.get("smoothing_defaults", {})
        if smooth_method == "sma":
            smooth_params = {"win_per": sd.get("win_per", 0.1)}
        elif smooth_method == "es":
            smooth_params = {"span_per": sd.get("span_per", 0.1)}
        else:
            smooth_params = {
                "win_per": sd.get("sg_win_per", 0.1),
                "poly_order": sd.get("sg_poly_order", 2),
            }
    df = smooth_sentiment(df, smooth_method, smooth_params)
    df = normalize_sentiment(df)

    # --- FIXED PART STARTS HERE ---
    if crux_params is None:
        crux_params = cfg.get("crux_defaults", {}).copy()
    else:
        crux_params = crux_params.copy()

    # pull out extraction-only param
    half_win = crux_params.pop(
        "half_win",
        cfg.get("crux_defaults", {}).get("half_win", 5),
    )

    crux_indices = detect_cruxes(df, **crux_params)
    crux_df = extract_crux_text(df, crux_indices, half_win=half_win)
    # --- FIXED PART ENDS HERE ---

    plot_path = build_plot_filename(
        out_dir,
        meta["text_genre"],
        meta["title"],
        meta["author"],
        model_name,
        smooth_method,
    )
    ts_csv = build_ts_filename(
        out_dir,
        meta["text_genre"],
        meta["title"],
        meta["author"],
        model_name,
        smooth_method,
        "csv",
    )
    ts_txt = build_ts_filename(
        out_dir,
        meta["text_genre"],
        meta["title"],
        meta["author"],
        model_name,
        smooth_method,
        "txt",
    )
    crux_csv = build_crux_filename(
        out_dir,
        meta["text_genre"],
        meta["title"],
        meta["author"],
        model_name,
        smooth_method,
        "csv",
    )
    crux_txt = build_crux_filename(
        out_dir,
        meta["text_genre"],
        meta["title"],
        meta["author"],
        model_name,
        smooth_method,
        "txt",
    )

    title_txt = cfg.get("plot_defaults", {}).get("title_template").format(
        title=meta["title"], author=meta["author"], model_name=model_name
    )
    subtitle_txt = cfg.get("plot_defaults", {}).get("subtitle_template").format(
        smooth_algo=smooth_method, smooth_params=smooth_params
    )
    plot_sentiments(df, output_path=plot_path, title=title_txt, subtitle=subtitle_txt)

    save_sentiments(
        df,
        ts_csv,
        ts_txt,
        model_name=model_name,
        smooth_algo=smooth_method,
        smooth_params=smooth_params,
    )
    save_crux_text(
        crux_df,
        crux_csv,
        crux_txt,
        model_name=model_name,
        smooth_algo=smooth_method,
        smooth_params=smooth_params,
    )

    return {
        "df": df,
        "crux_df": crux_df,
        "paths": {
            "plot": plot_path,
            "ts_csv": ts_csv,
            "crux_csv": crux_csv,
            "crux_txt": crux_txt,
        },
    }
