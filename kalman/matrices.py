"""
Kalman-szűrő mátrix építők — F, H, Q, R.

Ref: KALMAN_LOG_MULTI_TF.md 4. fejezet
"""

from __future__ import annotations

import numpy as np


# ── F: Állapotátmeneti mátrix (konstans gyorsulás modell) ────────────────────


def build_F(dt: float = 1.0) -> np.ndarray:
    """
    3x3 állapotátmeneti mátrix.

    x_{k+1} = F · x_k
    [μ, μ̇, μ̈] — szint, momentum, gyorsulás
    """
    return np.array([
        [1.0, dt, 0.5 * dt**2],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ])


# ── H: Megfigyelési mátrix ───────────────────────────────────────────────────


def build_H_continuous(tau: float) -> np.ndarray:
    """
    Folytonos integrálos H sor: H_τ = [τ, ½τ², ⅙τ³]

    Ref: KALMAN_LOG_MULTI_TF.md 4.2
    """
    return np.array([[tau, 0.5 * tau**2, tau**3 / 6.0]])


def build_H_discrete(n: int) -> np.ndarray:
    """
    Diszkrét aggregációs H sor: H_n = [n, -n(n-1)/2, n(n-1)(2n-1)/12]

    Ref: KALMAN_LOG_MULTI_TF.md 4.3
    """
    return np.array([[
        float(n),
        -n * (n - 1) / 2.0,
        n * (n - 1) * (2 * n - 1) / 12.0,
    ]])


def build_H_matrix(active_tf_minutes: list[int], mode: str) -> np.ndarray:
    """
    Teljes H mátrix az aktív TF-ekre.

    Args:
        active_tf_minutes: aktív TF-ek percben [1, 5, 15, ...]
        mode: "continuous" vagy "discrete"

    Returns:
        [k x 3] mátrix, ahol k = aktív TF-ek száma
    """
    builder = build_H_continuous if mode == "continuous" else build_H_discrete
    rows = [builder(n) for n in active_tf_minutes]
    return np.vstack(rows)


# ── R: Mérési zaj kovariancia ────────────────────────────────────────────────


def build_R_full(active_tf_minutes: list[int], sigma2_1m: float) -> np.ndarray:
    """
    Nem-diagonális R mátrix: R_ij = min(n_i, n_j) · σ²_1m

    Az átfedő TF-ek kovarianciáját modellezi (a nagy TF
    „tartalmazza" a kicsit → korrelált zaj).

    Ref: KALMAN_LOG_MULTI_TF.md 4.4
    """
    k = len(active_tf_minutes)
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            R[i, j] = min(active_tf_minutes[i], active_tf_minutes[j]) * sigma2_1m
    return R


def build_R_diagonal(active_tf_minutes: list[int], sigma2_1m: float) -> np.ndarray:
    """Egyszerűsített diagonális R: R_ii = n_i · σ²_1m"""
    return np.diag([n * sigma2_1m for n in active_tf_minutes])


def build_R_matrix(
    active_tf_minutes: list[int], sigma2_1m: float, mode: str,
) -> np.ndarray:
    """R mátrix építő dispatcher."""
    if mode == "full":
        return build_R_full(active_tf_minutes, sigma2_1m)
    return build_R_diagonal(active_tf_minutes, sigma2_1m)


# ── Q: Folyamatzaj kovariancia ───────────────────────────────────────────────


def build_Q(q: float, dt: float = 1.0) -> np.ndarray:
    """
    Piecewise constant acceleration folyamatzaj.

    q: gyorsulás spektrális sűrűsége (egyetlen hangolási paraméter)

    Ref: KALMAN_LOG_MULTI_TF.md 4.5
    """
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    dt5 = dt**5
    return q * np.array([
        [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
        [dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0],
        [dt3 / 6.0,  dt2 / 2.0, dt],
    ])
