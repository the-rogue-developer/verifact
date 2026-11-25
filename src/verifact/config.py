"""YAML config loading with simple command-line overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. Overrides look like ``train.epochs=1``."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.split("."), yaml.safe_load(value))
    return cfg


def _set_nested(cfg: dict, keys: list[str], value) -> None:
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
