"""
Szűrt állapotok vizualizáció — μ̂, μ̂̇, μ̂̈ + BTC ár.

4 subplot: ár overlay, mu_hat ± σ, mu_dot_hat, mu_ddot_hat.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from visualizations.base import BasePlot

logger = logging.getLogger(__name__)


class StatesPlot(BasePlot):
    """Szűrt állapotok + BTC ár vizualizáció."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(self, states_df: pd.DataFrame) -> Path:
        """
        Szűrt állapotok ábrázolása 4 subploton.

        Args:
            states_df: DataFrame oszlopok: mu_hat, mu_dot_hat, mu_ddot_hat,
                       P00, P11, P22, mahalanobis, n_active_tfs
                       DatetimeIndex-szel.

        Returns:
            Az elmentett fájl útvonala.
        """
        idx = states_df.index

        # ── Szórás számítás ──────────────────────────────────────────────
        sigma_mu = np.sqrt(states_df["P00"].values)
        sigma_mu_dot = np.sqrt(states_df["P11"].values)
        sigma_mu_ddot = np.sqrt(states_df["P22"].values)

        # ── Subplots létrehozása ─────────────────────────────────────────
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.25, 0.25, 0.25, 0.25],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ],
            subplot_titles=[
                "BTC ár + μ̂ (szűrt log hozam)",
                "μ̂ — szűrt pillanatnyi log hozam ± 1σ",
                "μ̂̇ — momentum",
                "μ̂̈ — gyorsulás",
            ],
        )

        # ── (1) BTC ár + mu_hat ──────────────────────────────────────────
        # Ár a másodlagos y-tengelyen
        self.add_price_trace(fig, row=1, col=1, secondary_y=True, opacity=0.5)

        # mu_hat az elsődleges y-tengelyen
        fig.add_trace(
            go.Scatter(
                x=idx, y=states_df["mu_hat"],
                name="μ̂",
                line=dict(color="#1f77b4", width=1.2),
                hovertemplate="%{y:.6f}",
            ),
            row=1, col=1, secondary_y=False,
        )

        fig.update_yaxes(title_text="μ̂", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="BTC ár (USD)", row=1, col=1, secondary_y=True)

        # ── (2) mu_hat ± 1σ sáv ─────────────────────────────────────────
        upper_mu = states_df["mu_hat"].values + sigma_mu
        lower_mu = states_df["mu_hat"].values - sigma_mu

        fig.add_trace(
            go.Scatter(
                x=idx, y=upper_mu,
                name="+1σ (μ̂)",
                line=dict(width=0),
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=lower_mu,
                name="-1σ (μ̂)",
                line=dict(width=0),
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.2)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=states_df["mu_hat"],
                name="μ̂ ± 1σ",
                line=dict(color="#1f77b4", width=1.2),
                hovertemplate="%{y:.6f}",
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="μ̂", row=2, col=1)

        # ── (3) mu_dot_hat — momentum (zöld/piros) ──────────────────────
        mu_dot = states_df["mu_dot_hat"].values
        # Pozitív: zöld, negatív: piros fill-lel
        mu_dot_pos = np.where(mu_dot >= 0, mu_dot, 0.0)
        mu_dot_neg = np.where(mu_dot < 0, mu_dot, 0.0)

        fig.add_trace(
            go.Scatter(
                x=idx, y=mu_dot_pos,
                name="μ̂̇ ≥ 0",
                fill="tozeroy",
                fillcolor="rgba(0,200,83,0.3)",
                line=dict(color="rgba(0,200,83,0.8)", width=0.8),
                hovertemplate="%{y:.8f}",
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=mu_dot_neg,
                name="μ̂̇ < 0",
                fill="tozeroy",
                fillcolor="rgba(255,82,82,0.3)",
                line=dict(color="rgba(255,82,82,0.8)", width=0.8),
                hovertemplate="%{y:.8f}",
            ),
            row=3, col=1,
        )
        fig.update_yaxes(title_text="μ̂̇", row=3, col=1)

        # ── (4) mu_ddot_hat — gyorsulás (zöld/piros) ────────────────────
        mu_ddot = states_df["mu_ddot_hat"].values
        mu_ddot_pos = np.where(mu_ddot >= 0, mu_ddot, 0.0)
        mu_ddot_neg = np.where(mu_ddot < 0, mu_ddot, 0.0)

        fig.add_trace(
            go.Scatter(
                x=idx, y=mu_ddot_pos,
                name="μ̂̈ ≥ 0",
                fill="tozeroy",
                fillcolor="rgba(0,200,83,0.3)",
                line=dict(color="rgba(0,200,83,0.8)", width=0.8),
                hovertemplate="%{y:.10f}",
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=mu_ddot_neg,
                name="μ̂̈ < 0",
                fill="tozeroy",
                fillcolor="rgba(255,82,82,0.3)",
                line=dict(color="rgba(255,82,82,0.8)", width=0.8),
                hovertemplate="%{y:.10f}",
            ),
            row=4, col=1,
        )
        fig.update_yaxes(title_text="μ̂̈", row=4, col=1)

        # ── Layout ──────────────────────────────────────────────────────
        fig.update_xaxes(title_text="Idő", row=4, col=1)
        self.apply_layout(fig, title="Szűrt állapotok — μ̂, μ̂̇, μ̂̈", height=1400)

        return self.save(fig, "filtered_states")
