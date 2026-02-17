"""
Nyers vs szűrt hozamok vizualizáció — TF-enként.

6 subplot (1m, 5m, 15m, 1h, 4h, 1d): nyers hozam (szürke) vs
H_τ @ x̂ rekonstruált szűrt hozam (kék).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from kalman.matrices import build_H_matrix
from visualizations.base import BasePlot

logger = logging.getLogger(__name__)


class ReturnsPlot(BasePlot):
    """Nyers vs szűrt hozamok vizualizáció timeframe-enként."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(
        self,
        states_df: pd.DataFrame,
        returns: dict[str, pd.Series],
        tf_minutes: dict[str, int],
        h_mode: str,
    ) -> Path:
        """
        Nyers vs rekonstruált szűrt hozamok ábrázolása TF-enként.

        Args:
            states_df: szűrt állapotok (mu_hat, mu_dot_hat, mu_ddot_hat, ...)
            returns: nyers log hozam sorozatok TF-enként {'1m': Series, ...}
            tf_minutes: TF → percek mapping {'1m': 1, '5m': 5, ...}
            h_mode: 'continuous' vagy 'discrete' — H mátrix mód

        Returns:
            Az elmentett fájl útvonala.
        """
        # TF-ek rendezése percek szerint
        sorted_tfs = sorted(tf_minutes.keys(), key=lambda t: tf_minutes[t])
        n_tfs = len(sorted_tfs)

        fig = make_subplots(
            rows=n_tfs, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f"{tf} ({tf_minutes[tf]} perc)" for tf in sorted_tfs],
        )

        # ── Állapotvektorok mátrixba (vektorizált H @ x̂ számoláshoz) ────
        # x_hat: [N, 3] mátrix
        x_hat = states_df[["mu_hat", "mu_dot_hat", "mu_ddot_hat"]].values  # [N, 3]
        idx = states_df.index

        for row_i, tf_label in enumerate(sorted_tfs, start=1):
            n_minutes = tf_minutes[tf_label]

            # H sor vektor ehhez a TF-hez: [1, 3]
            H_tf = build_H_matrix([n_minutes], h_mode)  # [1, 3]

            # Rekonstruált hozam: H_tf @ x̂ᵀ — vektorizálva
            # x_hat: [N, 3], H_tf.T: [3, 1] → eredmény: [N, 1]
            reconstructed = (x_hat @ H_tf.T).flatten()  # [N]

            # Nyers hozam
            if tf_label in returns:
                raw_series = returns[tf_label]
                # Reindex a states_df indexéhez (közös tengely)
                raw_aligned = raw_series.reindex(idx)
                raw_vals = raw_aligned.values

                # Csak ott mutatjuk a rekonstruált értéket, ahol a nyers nem NaN
                valid_mask = np.isfinite(raw_vals)
                valid_idx = idx[valid_mask]
                raw_valid = raw_vals[valid_mask]
                recon_valid = reconstructed[valid_mask]

                # Nyers hozam — szürke
                fig.add_trace(
                    go.Scatter(
                        x=valid_idx, y=raw_valid,
                        name=f"{tf_label} nyers",
                        line=dict(color="rgba(150,150,150,0.5)", width=0.8),
                        hovertemplate="%{y:.6f}",
                        showlegend=(row_i == 1),
                        legendgroup="raw",
                    ),
                    row=row_i, col=1,
                )

                # Rekonstruált (szűrt) hozam — kék
                fig.add_trace(
                    go.Scatter(
                        x=valid_idx, y=recon_valid,
                        name=f"{tf_label} szűrt (Hx̂)",
                        line=dict(color="#1f77b4", width=1.2),
                        hovertemplate="%{y:.6f}",
                        showlegend=(row_i == 1),
                        legendgroup="filtered",
                    ),
                    row=row_i, col=1,
                )
            else:
                logger.warning(f"  Nincs nyers hozam adat: {tf_label}")

            fig.update_yaxes(title_text="log hozam", row=row_i, col=1)

        # ── Layout ──────────────────────────────────────────────────────
        fig.update_xaxes(title_text="Idő", row=n_tfs, col=1)
        self.apply_layout(fig, title="Nyers vs szűrt hozamok", height=2000)

        return self.save(fig, "returns_comparison")
