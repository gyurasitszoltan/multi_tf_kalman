"""
Rauch–Tung–Striebel (RTS) simító — backward pass.

Ref: KALMAN_LOG_MULTI_TF.md 6.8 fejezet
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .filter import KalmanState


@dataclass
class SmoothedState:
    """Simított állapot egy időlépéshez."""

    x: np.ndarray   # [3x1] simított állapot
    P: np.ndarray   # [3x3] simított kovariancia


def rts_smooth(
    history: list[KalmanState],
    F: np.ndarray,
) -> list[SmoothedState]:
    """
    RTS backward pass.

    A szűrt (forward-only) becsléseket visszamenőleg finomítja
    a jövőbeli mérések figyelembevételével.

    Args:
        history: a filter.run() által gyűjtött KalmanState lista
        F: állapotátmeneti mátrix

    Returns:
        SmoothedState lista (azonos indexeléssel mint a history)
    """
    N = len(history)
    if N == 0:
        return []

    # Inicializálás: az utolsó lépés simított = szűrt
    smoothed = [SmoothedState(x=np.zeros((3, 1)), P=np.zeros((3, 3)))] * N
    smoothed[N - 1] = SmoothedState(
        x=history[N - 1].x.copy(),
        P=history[N - 1].P.copy(),
    )

    # Visszafelé haladva
    for k in range(N - 2, -1, -1):
        P_k = history[k].P            # szűrt P_k|k
        P_kp1_pred = history[k + 1].P_pred  # predikált P_{k+1|k}

        # Smoother gain
        try:
            P_pred_inv = np.linalg.inv(P_kp1_pred)
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_kp1_pred)

        C_k = P_k @ F.T @ P_pred_inv

        # Simított állapot
        x_s = history[k].x + C_k @ (smoothed[k + 1].x - history[k + 1].x_pred)
        P_s = P_k + C_k @ (smoothed[k + 1].P - P_kp1_pred) @ C_k.T

        # Szimmetrizálás
        P_s = (P_s + P_s.T) / 2.0

        smoothed[k] = SmoothedState(x=x_s, P=P_s)

    return smoothed


def smoothed_to_df(
    smoothed: list[SmoothedState],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """SmoothedState lista → DataFrame."""
    records = []
    for s in smoothed:
        records.append({
            "mu_smooth": s.x[0, 0],
            "mu_dot_smooth": s.x[1, 0],
            "mu_ddot_smooth": s.x[2, 0],
            "P00_smooth": s.P[0, 0],
            "P11_smooth": s.P[1, 1],
            "P22_smooth": s.P[2, 2],
        })
    df = pd.DataFrame(records)
    if len(df) == len(index):
        df.index = index
    return df
