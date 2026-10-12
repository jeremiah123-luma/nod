"""Loads config.json. Every tunable number lives there, not in the code."""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        return json.load(f)
