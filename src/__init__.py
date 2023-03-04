from pathlib import Path
import os

BASE = (Path(__file__) / ".." / "..").resolve()
DATA_ROOT = BASE / "data"
DATA_SOURCES = DATA_ROOT / "data_sources"
PROCED_DATA = DATA_ROOT / "processed_data"
CACHE_DATA = DATA_ROOT / "cache_data"
MODELS = BASE / "models"
