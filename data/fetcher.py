"""
Adat letöltés — Binance OHLCV (ccxt) + parquet cache + log return számítás.

Használat:
    config = Config.from_yaml()
    df_1m = fetch_or_load(config)
    returns = compute_log_returns(df_1m, config)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from config import Config, tf_to_minutes

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Paginált OHLCV letöltés ccxt-vel."""
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Ismeretlen exchange: {exchange_id}")

    exchange = exchange_class({"enableRateLimit": True})
    tf_ms = tf_to_minutes(timeframe) * 60_000

    rows: list[list] = []
    cursor = since_ms

    try:
        while cursor <= until_ms:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=limit,
            )
            if not batch:
                break

            rows.extend(batch)
            last_ts = int(batch[-1][0])
            next_cursor = last_ts + tf_ms

            if next_cursor <= cursor:
                next_cursor = cursor + tf_ms
            cursor = next_cursor

            if len(batch) < limit:
                break

            time.sleep(exchange.rateLimit / 1000.0)
            logger.info(f"  Letöltve: {len(rows)} sor eddig...")
    finally:
        if hasattr(exchange, "close"):
            exchange.close()

    if not rows:
        raise RuntimeError("Nem érkezett OHLCV adat.")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp")
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= until_ms)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df


def fetch_or_load(config: Config) -> pd.DataFrame:
    """1m OHLCV adat: cache-ből betölti vagy letölti."""
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - config.data.days_back * 24 * 3600 * 1000

    safe_symbol = config.symbol.replace("/", "")
    cache_file = cache_dir / f"{safe_symbol}_1m_{config.data.days_back}d.parquet"

    if cache_file.exists():
        logger.info(f"Cache betöltés: {cache_file}")
        df = pd.read_parquet(cache_file)
        # Ellenőrzés: elég friss-e (max 2 óra régi)
        if len(df) > 0:
            last_ts = df.index[-1].timestamp() * 1000
            age_hours = (now_ms - last_ts) / 3_600_000
            if age_hours < 2:
                logger.info(f"  Cache friss ({age_hours:.1f}h régi), {len(df)} sor")
                return df
            logger.info(f"  Cache elavult ({age_hours:.1f}h régi), újratöltés...")

    logger.info(f"Letöltés: {config.symbol} 1m, {config.data.days_back} nap...")
    df = fetch_ohlcv(
        exchange_id=config.exchange,
        symbol=config.symbol,
        timeframe="1m",
        since_ms=since_ms,
        until_ms=now_ms,
    )
    logger.info(f"  Letöltve: {len(df)} sor")

    df.to_parquet(cache_file)
    logger.info(f"  Cache mentve: {cache_file}")
    return df


def compute_log_returns(df_1m: pd.DataFrame, config: Config) -> dict[str, pd.Series]:
    """
    1m close-ból log hozamok minden konfigurált TF-re.

    Returns:
        dict: {'1m': Series, '5m': Series, ...}
        Minden Series az 1m index-szel, NaN ahol a TF mérés nem elérhető.
    """
    log_price = np.log(df_1m["close"])
    returns: dict[str, pd.Series] = {}

    for tf_label, n_min in config.tf_minutes.items():
        # Log hozam: log(P_t) - log(P_{t - n_min})
        ret = log_price - log_price.shift(n_min)
        # Csak ott legyen érték, ahol a TF ténylegesen frissül
        # (percindex mod n_min == 0)
        mask = pd.Series(range(len(ret)), index=ret.index) % n_min != 0
        ret = ret.copy()
        ret.loc[mask] = np.nan
        returns[tf_label] = ret

    return returns


def estimate_sigma2_1m(returns_1m: pd.Series) -> float:
    """1 perces log hozam varianciájának becslése."""
    clean = returns_1m.dropna()
    return float(clean.var())
