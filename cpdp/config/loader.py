# cpdp/config/loader.py
from typing import Any, Dict
import pathlib

import yaml


def load_config(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return config
