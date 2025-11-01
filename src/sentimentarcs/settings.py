from pathlib import Path
import os, yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"

def load_settings(path: str | os.PathLike = CONFIG_PATH):
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
