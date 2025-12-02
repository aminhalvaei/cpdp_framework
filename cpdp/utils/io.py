# cpdp/utils/io.py
import json
import pathlib
from typing import Dict, Any


def ensure_dir(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
