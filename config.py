"""
Konfiguráció — Pydantic modell + YAML betöltés.

Használat:
    config = Config.from_yaml("config.yaml")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator


# ── Timeframe segédek ────────────────────────────────────────────────────────

TF_PATTERN = re.compile(r"^(\d+)([mhd])$")


def tf_to_minutes(tf: str) -> int:
    """Timeframe stringből percek száma.  '1m'->1, '4h'->240, '1d'->1440."""
    m = TF_PATTERN.match(tf)
    if not m:
        raise ValueError(f"Érvénytelen timeframe formátum: '{tf}'")
    val, unit = int(m.group(1)), m.group(2)
    return val * {"m": 1, "h": 60, "d": 1440}[unit]


def tf_to_millis(tf: str) -> int:
    return tf_to_minutes(tf) * 60_000


# ── Nested config modellek ───────────────────────────────────────────────────


class DataConfig(BaseModel):
    days_back: int = 7
    cache_dir: str = "data/cache"


class KalmanConfig(BaseModel):
    q: float = 1e-8
    sigma2_1m: Optional[float] = None
    h_mode: Literal["continuous", "discrete"] = "discrete"
    r_mode: Literal["full", "diagonal"] = "full"
    P0_scale: float = 100.0


class TrendConfig(BaseModel):
    w_mu: float = 0.50
    w_mu_dot: float = 0.35
    w_mu_ddot: float = 0.15
    rolling_window: int = 120


class VisualizationConfig(BaseModel):
    format: Literal["html", "png", "both"] = "html"
    theme: str = "plotly_dark"
    width: int = 1920
    height: int = 1080
    output_dir: str = "output"


# ── Fő Config ────────────────────────────────────────────────────────────────


class Config(BaseModel):
    symbol: str = "BTC/USDT"
    exchange: str = "binance"
    timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

    data: DataConfig = DataConfig()
    kalman: KalmanConfig = KalmanConfig()
    trend: TrendConfig = TrendConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, v: list[str]) -> list[str]:
        minutes = [tf_to_minutes(tf) for tf in v]
        if minutes != sorted(minutes):
            raise ValueError(f"Timeframe-ek nem növekvő sorrendben: {v}")
        if len(set(minutes)) != len(minutes):
            raise ValueError(f"Duplikált timeframe: {v}")
        return v

    @property
    def tf_minutes(self) -> dict[str, int]:
        """{'1m': 1, '5m': 5, ...}"""
        return {tf: tf_to_minutes(tf) for tf in self.timeframes}

    @property
    def base_tf(self) -> str:
        return self.timeframes[0]

    @property
    def base_minutes(self) -> int:
        return tf_to_minutes(self.base_tf)

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config fájl nem található: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
