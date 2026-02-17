"""
Kalman-nyereség dinamika vizualizáció.

Felső: K mátrix Frobenius norma időben, TF határokkal.
Alsó: állapotkomponensenkénti összesített abszolút gain.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from visualizations.base import BasePlot

if TYPE_CHECKING:
    from kalman.filter import KalmanState

logger = logging.getLogger(__name__)

# TF szín- és címke mapping
TF_COLORS = {
    1: ("#636EFA", "1m"),
    5: ("#EF553B", "5m"),
    15: ("#00CC96", "15m"),
    60: ("#AB63FA", "1h"),
    240: ("#FFA15A", "4h"),
    1440: ("#19D3F3", "1d"),
}


class GainPlot(BasePlot):
    """Kalman-nyereség (K mátrix) dinamika vizualizáció."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(self, history: list) -> Path:
        """
        Kalman gain dinamika ábrázolása.

        Args:
            history: KalmanState objektumok listája.
                     Minden elemnek van: .K (np.ndarray [3xk] vagy None),
                     .active_tf_minutes (list[int]), .step_idx (int)

        Returns:
            Az elmentett fájl útvonala.
        """
        # ── Adatok előkészítése ──────────────────────────────────────────
        steps = []
        frob_norms = []
        gain_mu = []       # sum(|K[0,:]|)
        gain_mu_dot = []   # sum(|K[1,:]|)
        gain_mu_ddot = []  # sum(|K[2,:]|)
        active_tfs_list = []

        for state in history:
            steps.append(state.step_idx)
            active_tfs_list.append(state.active_tf_minutes)

            if state.K is not None:
                K = state.K
                frob_norms.append(np.linalg.norm(K, "fro"))
                gain_mu.append(np.sum(np.abs(K[0, :])))
                gain_mu_dot.append(np.sum(np.abs(K[1, :])))
                gain_mu_ddot.append(np.sum(np.abs(K[2, :])))
            else:
                frob_norms.append(0.0)
                gain_mu.append(0.0)
                gain_mu_dot.append(0.0)
                gain_mu_ddot.append(0.0)

        steps = np.array(steps)
        frob_norms = np.array(frob_norms)
        gain_mu = np.array(gain_mu)
        gain_mu_dot = np.array(gain_mu_dot)
        gain_mu_ddot = np.array(gain_mu_ddot)

        # ── TF határok megkeresése (első előfordulás) ────────────────────
        # Azon lépések, ahol először jelenik meg egy nagyobb TF
        tf_first_seen: dict[int, int] = {}
        for i, atfs in enumerate(active_tfs_list):
            for tf_min in atfs:
                if tf_min not in tf_first_seen:
                    tf_first_seen[tf_min] = steps[i]

        # ── Subplots ────────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=[
                "‖K‖_F — Kalman gain Frobenius norma",
                "Állapotkomponensenkénti abszolút gain (Σ|K[i,:]|)",
            ],
        )

        # ── Felső: Frobenius norma ──────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=steps, y=frob_norms,
                name="‖K‖_F",
                line=dict(color="#1f77b4", width=1.0),
                hovertemplate="lépés: %{x}<br>‖K‖_F: %{y:.6f}",
            ),
            row=1, col=1,
        )

        # TF határok: függőleges szaggatott vonalak
        for tf_min, first_step in sorted(tf_first_seen.items()):
            color, label = TF_COLORS.get(tf_min, ("#FFFFFF", f"{tf_min}m"))
            # Csak a > 1m TF-ek határait jelöljük (1m mindig aktív)
            if tf_min > 1:
                fig.add_vline(
                    x=first_step, row=1, col=1,
                    line=dict(color=color, width=1.5, dash="dash"),
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_color=color,
                )

        fig.update_yaxes(title_text="‖K‖_F", row=1, col=1)

        # ── Alsó: állapotkomponensenkénti gain ──────────────────────────
        fig.add_trace(
            go.Scatter(
                x=steps, y=gain_mu,
                name="Σ|K[μ,:]|",
                line=dict(color="#1f77b4", width=1.2),
                hovertemplate="lépés: %{x}<br>Σ|K[μ,:]|: %{y:.6f}",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=steps, y=gain_mu_dot,
                name="Σ|K[μ̇,:]|",
                line=dict(color="#2ca02c", width=1.2),
                hovertemplate="lépés: %{x}<br>Σ|K[μ̇,:]|: %{y:.6f}",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=steps, y=gain_mu_ddot,
                name="Σ|K[μ̈,:]|",
                line=dict(color="#ff7f0e", width=1.2),
                hovertemplate="lépés: %{x}<br>Σ|K[μ̈,:]|: %{y:.6f}",
            ),
            row=2, col=1,
        )

        # TF határok az alsó subploton is
        for tf_min, first_step in sorted(tf_first_seen.items()):
            color, label = TF_COLORS.get(tf_min, ("#FFFFFF", f"{tf_min}m"))
            if tf_min > 1:
                fig.add_vline(
                    x=first_step, row=2, col=1,
                    line=dict(color=color, width=1.0, dash="dot"),
                )

        fig.update_yaxes(title_text="Σ|K[i,:]|", row=2, col=1)
        fig.update_xaxes(title_text="Lépés index", row=2, col=1)

        # ── Layout ──────────────────────────────────────────────────────
        self.apply_layout(fig, title="Kalman-nyereség dinamika", height=900)

        return self.save(fig, "kalman_gain_dynamics")
