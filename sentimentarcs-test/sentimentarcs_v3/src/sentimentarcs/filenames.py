import re
from pathlib import Path

FILENAME_RE = re.compile(r"^(?P<genre>[^_]+)_(?P<title>[^_]+)_(?P<author>[^.]+)\.txt$")

class FilenameError(ValueError):
    pass

def parse_input_filename(path: str | Path) -> dict:
    name = Path(path).name
    m = FILENAME_RE.match(name)
    if not m:
        raise FilenameError("Expected 'novel_title_author.txt'-style filename")
    d = m.groupdict()
    return {"text_genre": d["genre"], "title": d["title"], "author": d["author"]}

def build_plot_filename(out_dir, genre, title, author, model, smooth, ext="png"):
    return Path(out_dir) / f"sentimentarcs_plot_{genre}_{title}-{author}_{model}_{smooth}.{ext}"

def build_ts_filename(out_dir, genre, title, author, model, smooth, ext="csv"):
    return Path(out_dir) / f"sentimentarcs_ts_{genre}_{title}-{author}_{model}_{smooth}.{ext}"

def build_crux_filename(out_dir, genre, title, author, model, smooth, ext="csv"):
    return Path(out_dir) / f"sentimentarcs_cruxes_{genre}_{title}-{author}_{model}_{smooth}.{ext}"
