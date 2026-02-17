"""
Kereskedési jelzések — trend score, predikció, anomália detekció.

Ref: KALMAN_LOG_MULTI_TF.md 6. fejezet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from kalman.matrices import build_H_matrix


def compute_trend_score(
    states_df: pd.DataFrame,
    w_mu: float = 0.50,
    w_mu_dot: float = 0.35,
    w_mu_ddot: float = 0.15,
    rolling_window: int = 120,
) -> pd.DataFrame:
    """
    Trend-erő kompozit jel.

    trend_score = w₁·(μ̂/σ_μ) + w₂·(μ̂̇/σ_μ̇) + w₃·(μ̂̈/σ_μ̈)

    Returns:
        DataFrame: trend_score + normalizált komponensek
    """
    mu = states_df["mu_hat"]
    mu_dot = states_df["mu_dot_hat"]
    mu_ddot = states_df["mu_ddot_hat"]

    sigma_mu = mu.rolling(rolling_window, min_periods=20).std()
    sigma_mu_dot = mu_dot.rolling(rolling_window, min_periods=20).std()
    sigma_mu_ddot = mu_ddot.rolling(rolling_window, min_periods=20).std()

    # Normalizált komponensek (NaN-safe)
    norm_mu = mu / sigma_mu.replace(0, np.nan)
    norm_mu_dot = mu_dot / sigma_mu_dot.replace(0, np.nan)
    norm_mu_ddot = mu_ddot / sigma_mu_ddot.replace(0, np.nan)

    trend_score = w_mu * norm_mu + w_mu_dot * norm_mu_dot + w_mu_ddot * norm_mu_ddot

    return pd.DataFrame({
        "trend_score": trend_score,
        "norm_mu": norm_mu,
        "norm_mu_dot": norm_mu_dot,
        "norm_mu_ddot": norm_mu_ddot,
    }, index=states_df.index)


def compute_predictions(
    states_df: pd.DataFrame,
    horizons_minutes: list[int],
    h_mode: str = "discrete",
) -> dict[int, pd.DataFrame]:
    """
    Prediktív hozambecslés tetszőleges horizontokra.

    r̂_{t→t+τ} = μ̂·τ + μ̂̇·½τ² + μ̂̈·⅙τ³

    Returns:
        {horizon_minutes: DataFrame with 'predicted', 'ci_lower', 'ci_upper'}
    """
    results = {}

    for tau in horizons_minutes:
        H_tau = build_H_matrix([tau], h_mode)  # [1x3]

        preds = []
        ci_widths = []

        for i in range(len(states_df)):
            x_hat = np.array([
                [states_df["mu_hat"].iloc[i]],
                [states_df["mu_dot_hat"].iloc[i]],
                [states_df["mu_ddot_hat"].iloc[i]],
            ])
            pred = float((H_tau @ x_hat).item())
            preds.append(pred)

            P = np.array([
                [states_df["P00"].iloc[i], 0, 0],
                [0, states_df["P11"].iloc[i], 0],
                [0, 0, states_df["P22"].iloc[i]],
            ])
            var = float((H_tau @ P @ H_tau.T).item())
            ci_widths.append(1.96 * np.sqrt(max(var, 0)))

        pred_series = pd.Series(preds, index=states_df.index, name="predicted")
        ci_series = pd.Series(ci_widths, index=states_df.index, name="ci_width")

        results[tau] = pd.DataFrame({
            "predicted": pred_series,
            "ci_lower": pred_series - ci_series,
            "ci_upper": pred_series + ci_series,
        })

    return results


def compute_anomaly_flags(
    states_df: pd.DataFrame,
    significance: float = 0.05,
) -> pd.Series:
    """
    Innováció-alapú anomália detekció.

    d_k > χ²(p, 1-α) küszöb → anomália

    Returns:
        Boolean Series: True = anomália az adott lépésben
    """
    mahal = states_df["mahalanobis"]
    n_active = states_df["n_active_tfs"]

    flags = pd.Series(False, index=states_df.index)

    for n_tf in n_active.unique():
        if n_tf < 1:
            continue
        threshold = stats.chi2.ppf(1 - significance, df=n_tf)
        mask = n_active == n_tf
        flags.loc[mask] = mahal.loc[mask] > threshold

    return flags
