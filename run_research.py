"""
Multi-TF Kalman Filter — Kutatási futtató script.

Minden vizualizáció generálása egyetlen futtatással:
    python run_research.py

Használat:
    python run_research.py                     # alapértelmezett config.yaml
    python run_research.py --config my.yaml    # egyedi config
    python run_research.py --days 3            # override days_back
    python run_research.py --q 1e-7            # override q paraméter
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Projekt root a path-ra ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from data.fetcher import compute_log_returns, estimate_sigma2_1m, fetch_or_load
from kalman.filter import MultiTFKalmanFilter
from kalman.smoother import rts_smooth, smoothed_to_df
from signals import compute_anomaly_flags, compute_predictions, compute_trend_score

from visualizations.viz_states import StatesPlot
from visualizations.viz_returns import ReturnsPlot
from visualizations.viz_gain import GainPlot
from visualizations.viz_innovation import InnovationPlot
from visualizations.viz_covariance import CovariancePlot
from visualizations.viz_prediction import PredictionPlot
from visualizations.viz_trend import TrendDashboardPlot
from visualizations.viz_sensitivity import SensitivityPlot
from visualizations.viz_h_compare import HComparePlot
from visualizations.viz_smoother import SmootherPlot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_research")


def build_filter(config: Config, sigma2_1m: float) -> MultiTFKalmanFilter:
    """Szűrő létrehozása a config alapján."""
    return MultiTFKalmanFilter(
        tf_minutes=config.tf_minutes,
        q=config.kalman.q,
        sigma2_1m=sigma2_1m,
        h_mode=config.kalman.h_mode,
        r_mode=config.kalman.r_mode,
        P0_scale=config.kalman.P0_scale,
        dt=1.0,
    )


def run_filter_with_mode(
    config: Config,
    returns: dict[str, pd.Series],
    sigma2_1m: float,
    h_mode: str,
) -> tuple[MultiTFKalmanFilter, pd.DataFrame]:
    """Szűrő futtatás adott H móddal."""
    kf = MultiTFKalmanFilter(
        tf_minutes=config.tf_minutes,
        q=config.kalman.q,
        sigma2_1m=sigma2_1m,
        h_mode=h_mode,
        r_mode=config.kalman.r_mode,
        P0_scale=config.kalman.P0_scale,
    )
    kf.run(returns)
    base_tf = min(config.tf_minutes, key=lambda k: config.tf_minutes[k])
    idx = returns[base_tf].index
    states_df = kf.get_states_df(idx)
    return kf, states_df


def main():
    parser = argparse.ArgumentParser(description="Multi-TF Kalman Filter kutatás")
    parser.add_argument("--config", default="config.yaml", help="Config YAML fájl")
    parser.add_argument("--days", type=int, default=None, help="Override days_back")
    parser.add_argument("--q", type=float, default=None, help="Override q paraméter")
    args = parser.parse_args()

    # ── 1. Config betöltés ──────────────────────────────────
    config_path = PROJECT_ROOT / args.config
    config = Config.from_yaml(config_path)
    if args.days:
        config.data.days_back = args.days
    if args.q:
        config.kalman.q = args.q

    logger.info(f"Config: {config.symbol}, TF-ek: {config.timeframes}, "
                f"q={config.kalman.q:.2e}, {config.data.days_back} nap")

    # ── 2. Adat letöltés / cache ────────────────────────────
    t0 = time.time()
    df_1m = fetch_or_load(config)
    logger.info(f"Adat kész: {len(df_1m)} sor ({time.time() - t0:.1f}s)")

    # ── 3. Log hozamok ──────────────────────────────────────
    returns = compute_log_returns(df_1m, config)
    for tf, ret in returns.items():
        valid = ret.dropna()
        logger.info(f"  {tf}: {len(valid)} valid mérés")

    # ── 4. σ²_1m becslés ────────────────────────────────────
    sigma2_1m = config.kalman.sigma2_1m
    if sigma2_1m is None:
        sigma2_1m = estimate_sigma2_1m(returns["1m"])
        logger.info(f"σ²_1m automatikus becslés: {sigma2_1m:.2e}")
    else:
        logger.info(f"σ²_1m config-ból: {sigma2_1m:.2e}")

    # ── 5. Fő Kalman szűrő futtatás ────────────────────────
    t0 = time.time()
    kf = build_filter(config, sigma2_1m)
    kf.run(returns)
    logger.info(f"Szűrő kész ({time.time() - t0:.1f}s)")

    base_tf = config.base_tf
    idx = returns[base_tf].index
    states_df = kf.get_states_df(idx)
    price = df_1m["close"]

    # ── 6. RTS simítás ──────────────────────────────────────
    t0 = time.time()
    smoothed = rts_smooth(kf.history, kf.F)
    smooth_df = smoothed_to_df(smoothed, idx)
    logger.info(f"RTS simítás kész ({time.time() - t0:.1f}s)")

    # ── 6b. Burn-in levágás (a P konvergenciáig torzított az output) ──
    burn_in = min(50, len(states_df) // 10)
    states_df = states_df.iloc[burn_in:]
    smooth_df = smooth_df.iloc[burn_in:]
    price = price.loc[price.index.isin(states_df.index)]
    returns = {tf: ret.loc[ret.index.isin(states_df.index)] for tf, ret in returns.items()}
    # A history-t is szűkítjük a gain vizualizációhoz
    kf_history_plot = kf.history[burn_in:]
    idx = states_df.index
    logger.info(f"Burn-in levágva: első {burn_in} lépés kihagyva")

    # ── 7. Jelzések ─────────────────────────────────────────
    trend_df = compute_trend_score(
        states_df,
        w_mu=config.trend.w_mu,
        w_mu_dot=config.trend.w_mu_dot,
        w_mu_ddot=config.trend.w_mu_ddot,
        rolling_window=config.trend.rolling_window,
    )
    anomaly_flags = compute_anomaly_flags(states_df)
    predictions = compute_predictions(
        states_df,
        horizons_minutes=[5, 15, 60],
        h_mode=config.kalman.h_mode,
    )
    logger.info(f"Jelzések kész. Anomáliák: {anomaly_flags.sum()}")

    # ── 8. Vizualizációk generálása ─────────────────────────
    logger.info("=" * 60)
    logger.info("VIZUALIZÁCIÓK GENERÁLÁSA")
    logger.info("=" * 60)

    # VIZ-1: Szűrt állapotok
    logger.info("[1/10] Szűrt állapotok + ár...")
    StatesPlot(config, price).generate(states_df)

    # VIZ-2: Nyers vs szűrt hozamok
    logger.info("[2/10] Nyers vs szűrt hozamok...")
    ReturnsPlot(config, price).generate(
        states_df, returns, config.tf_minutes, config.kalman.h_mode,
    )

    # VIZ-3: Kalman gain dinamika
    logger.info("[3/10] Kalman gain dinamika...")
    GainPlot(config, price).generate(kf_history_plot)

    # VIZ-4: Innováció + anomália
    logger.info("[4/10] Innováció + anomália...")
    InnovationPlot(config, price).generate(states_df, anomaly_flags)

    # VIZ-5: P kovariancia evolúció
    logger.info("[5/10] P kovariancia evolúció...")
    CovariancePlot(config, price).generate(states_df)

    # VIZ-6: Predikció pontosság
    logger.info("[6/10] Predikció pontosság...")
    PredictionPlot(config, price).generate(
        states_df, returns, predictions, config.tf_minutes,
    )

    # VIZ-7: Trend score dashboard
    logger.info("[7/10] Trend score dashboard...")
    TrendDashboardPlot(config, price).generate(trend_df)

    # VIZ-8: q paraméter érzékenység
    logger.info("[8/10] q paraméter érzékenység...")
    q_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    q_results: dict[float, pd.DataFrame] = {}
    for q_val in q_values:
        kf_q = MultiTFKalmanFilter(
            tf_minutes=config.tf_minutes,
            q=q_val,
            sigma2_1m=sigma2_1m,
            h_mode=config.kalman.h_mode,
            r_mode=config.kalman.r_mode,
            P0_scale=config.kalman.P0_scale,
        )
        kf_q.run(returns, progress_interval=0)
        q_results[q_val] = kf_q.get_states_df(idx)
    SensitivityPlot(config, price).generate(q_results)

    # VIZ-9: H mátrix összehasonlítás
    logger.info("[9/10] H mátrix összehasonlítás...")
    _, cont_df = run_filter_with_mode(config, returns, sigma2_1m, "continuous")
    _, disc_df = run_filter_with_mode(config, returns, sigma2_1m, "discrete")
    HComparePlot(config, price).generate(cont_df, disc_df)

    # VIZ-10: RTS simító vs online
    logger.info("[10/10] RTS simító vs online...")
    SmootherPlot(config, price).generate(states_df, smooth_df)

    # ── Összefoglalás ───────────────────────────────────────
    output_dir = Path(config.visualization.output_dir)
    html_files = sorted(output_dir.glob("*.html"))
    logger.info("=" * 60)
    logger.info(f"KÉSZ! {len(html_files)} vizualizáció generálva:")
    for f in html_files:
        logger.info(f"  {f.name}")
    logger.info(f"Mappa: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
