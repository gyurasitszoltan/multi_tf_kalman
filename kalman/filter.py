"""
Multi-Timeframe Kalman Filter — a szűrő core implementációja.

Ref: KALMAN_LOG_MULTI_TF.md 5. fejezet (teljes algoritmus)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .matrices import build_F, build_H_matrix, build_Q, build_R_matrix


@dataclass
class KalmanState:
    """Egy időlépés teljes állapota (history-hoz)."""

    x: np.ndarray                          # [3x1] szűrt állapot
    P: np.ndarray                          # [3x3] kovariancia
    x_pred: np.ndarray                     # [3x1] predikált állapot (update előtt)
    P_pred: np.ndarray                     # [3x3] predikált kovariancia
    innovation: Optional[np.ndarray]       # [kx1] innováció vektor
    S: Optional[np.ndarray]                # [kxk] innováció kovariancia
    K: Optional[np.ndarray]                # [3xk] Kalman gain
    mahalanobis: float                     # χ² anomália metrika
    active_tf_minutes: list[int]           # mely TF-ek frissültek
    step_idx: int = 0


class MultiTFKalmanFilter:
    """
    Multi-timeframe Kalman-szűrő BTC log hozamokhoz.

    Szekvenciális frissítés: az alap TF-en (1m) fut,
    és minden percben csak az éppen elérhető TF-ek méréseivel frissít.
    """

    def __init__(
        self,
        tf_minutes: dict[str, int],
        q: float,
        sigma2_1m: float,
        h_mode: str = "discrete",
        r_mode: str = "full",
        P0_scale: float = 100.0,
        dt: float = 1.0,
    ):
        self.tf_minutes = tf_minutes
        self.all_tf_values = sorted(tf_minutes.values())
        self.q = q
        self.sigma2_1m = sigma2_1m
        self.h_mode = h_mode
        self.r_mode = r_mode
        self.dt = dt

        # Konstans mátrixok
        self.F = build_F(dt)
        self.Q = build_Q(q, dt)

        # Állapot inicializálás
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * P0_scale

        self.history: list[KalmanState] = []

    def _get_active_tfs(self, step_idx: int) -> list[int]:
        """Mely TF-ek frissülnek az adott lépésben."""
        return [n for n in self.all_tf_values if step_idx % n == 0]

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predikciós lépés. Returns: (x_pred, P_pred)."""
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
        active_tfs: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Korrekciós lépés.

        Returns:
            (x_updated, P_updated, innovation, S, K, mahalanobis)
        """
        H = build_H_matrix(active_tfs, self.h_mode)
        R = build_R_matrix(active_tfs, self.sigma2_1m, self.r_mode)

        # Innováció
        innovation = z - H @ x_pred  # [kx1]

        # Innováció kovariancia
        S = H @ P_pred @ H.T + R  # [kxk]

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pred @ H.T @ S_inv  # [3xk]

        # Állapot update (Joseph-forma a numerikus stabilitásért)
        I_KH = np.eye(3) - K @ H
        x_upd = x_pred + K @ innovation
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        # Mahalanobis-távolság (χ² anomália)
        mahal = float((innovation.T @ S_inv @ innovation).item())

        return x_upd, P_upd, innovation, S, K, mahal

    def _stabilize_P(self) -> None:
        """P mátrix pozitív definitség biztosítása."""
        self.P = (self.P + self.P.T) / 2.0
        eigvals = np.linalg.eigvalsh(self.P)
        if eigvals.min() < 1e-12:
            self.P += np.eye(3) * 1e-12

    def step(self, step_idx: int, measurements: dict[str, float]) -> KalmanState:
        """
        Egy teljes lépés: predict + update (ha van mérés).

        Args:
            step_idx: hányadik perces lépés (0-tól)
            measurements: {'1m': 0.001, '5m': 0.005, ...} — az elérhető mérések
        """
        # Predikció
        x_pred, P_pred = self.predict()

        # Aktív TF-ek meghatározása
        active_tfs = self._get_active_tfs(step_idx)

        # Mérésvektor összeállítása (csak ami ténylegesen rendelkezésre áll)
        tf_minutes_to_label = {v: k for k, v in self.tf_minutes.items()}
        available = []
        z_vals = []
        for n in active_tfs:
            label = tf_minutes_to_label.get(n)
            if label and label in measurements and np.isfinite(measurements[label]):
                available.append(n)
                z_vals.append(measurements[label])

        if available:
            z = np.array(z_vals).reshape(-1, 1)
            x_upd, P_upd, innov, S, K, mahal = self.update(
                x_pred, P_pred, z, available,
            )
            self.x = x_upd
            self.P = P_upd
        else:
            # Nincs mérés → csak predikció
            self.x = x_pred
            self.P = P_pred
            innov, S, K, mahal = None, None, None, 0.0

        self._stabilize_P()

        state = KalmanState(
            x=self.x.copy(),
            P=self.P.copy(),
            x_pred=x_pred.copy(),
            P_pred=P_pred.copy(),
            innovation=innov.copy() if innov is not None else None,
            S=S.copy() if S is not None else None,
            K=K.copy() if K is not None else None,
            mahalanobis=mahal,
            active_tf_minutes=available if available else [],
            step_idx=step_idx,
        )
        self.history.append(state)
        return state

    def run(
        self,
        returns: dict[str, pd.Series],
        progress_interval: int = 1000,
    ) -> list[KalmanState]:
        """
        A szűrő futtatása az összes adaton.

        Args:
            returns: compute_log_returns() outputja
            progress_interval: hány lépésenként logoljon
        """
        import logging
        logger = logging.getLogger(__name__)

        base_tf = min(self.tf_minutes, key=lambda k: self.tf_minutes[k])
        base_series = returns[base_tf]
        n_steps = len(base_series)

        logger.info(f"Szűrő futtatás: {n_steps} lépés")

        for i in range(n_steps):
            # Összegyűjtjük az elérhető méréseket
            meas: dict[str, float] = {}
            for tf_label, series in returns.items():
                val = series.iloc[i]
                if np.isfinite(val):
                    meas[tf_label] = val

            self.step(i, meas)

            if progress_interval and (i + 1) % progress_interval == 0:
                logger.info(f"  {i + 1}/{n_steps} lépés kész")

        logger.info(f"Szűrő kész: {len(self.history)} állapot")
        return self.history

    def get_states_df(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """History → DataFrame a vizualizációkhoz."""
        records = []
        for st in self.history:
            rec = {
                "mu_hat": st.x[0, 0],
                "mu_dot_hat": st.x[1, 0],
                "mu_ddot_hat": st.x[2, 0],
                "P00": st.P[0, 0],
                "P11": st.P[1, 1],
                "P22": st.P[2, 2],
                "mahalanobis": st.mahalanobis,
                "n_active_tfs": len(st.active_tf_minutes),
            }
            records.append(rec)

        df = pd.DataFrame(records)
        if len(df) == len(index):
            df.index = index
        return df
