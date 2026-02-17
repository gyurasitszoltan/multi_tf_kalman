"""
Trend score dashboard vizualizáció.

Felső subplot: BTC ár + trend_score háttérszínezés (zöld/piros).
Alsó subplot: normalizált komponensek stacked area chart.
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


class TrendDashboardPlot(BasePlot):
    """Trend score dashboard — ár háttérszínezéssel + komponens area chart."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    @staticmethod
    def _build_colored_bands(
        trend_score: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Trend score-ból pozitív/negatív sávokat épít scatter fill-hez.
        Sokkal gyorsabb, mint shape-ek ezrei.

        Returns:
            (positive_band, negative_band): Series, ahol NaN a nem-aktív zónák.
        """
        pos = trend_score.copy()
        neg = trend_score.copy()
        pos[trend_score <= 0] = 0.0
        neg[trend_score >= 0] = 0.0
        return pos, neg

    def generate(self, trend_df: pd.DataFrame) -> Path:
        """
        Trend score dashboard generálása.

        Args:
            trend_df: DataFrame trend_score, norm_mu, norm_mu_dot, norm_mu_ddot
                      oszlopokkal, DatetimeIndex-szel.

        Returns:
            Path az elmentett fájlhoz.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=[
                "BTC ár + Trend score háttér",
                "Normalizált komponensek (stacked)",
            ],
        )

        idx = trend_df.index
        trend_score = trend_df["trend_score"]

        # ── Felső subplot ────────────────────────────────────────────────

        # Trend score vonal (elsődleges y tengely)
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=trend_score.values,
                name="Trend score",
                line=dict(color="#FFA15A", width=1.5),
                hovertemplate="Trend: %{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
            secondary_y=False,
        )

        # BTC ár (másodlagos y tengely)
        # Közös indexre szűrjük az ár adatokat
        price_aligned = self.price.reindex(idx, method="ffill")
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=price_aligned.values,
                name="BTC ár",
                line=dict(color="rgba(180,180,180,0.7)", width=1.2),
                hovertemplate="%{y:,.0f} USD<extra></extra>",
            ),
            row=1, col=1,
            secondary_y=True,
        )

        fig.update_yaxes(title_text="Trend score", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="BTC ár (USD)", row=1, col=1, secondary_y=True)

        # Trend háttérszínezés scatter fill-lel (gyors, shape-ek helyett)
        pos_band, neg_band = self._build_colored_bands(trend_score)
        fig.add_trace(
            go.Scatter(
                x=idx, y=pos_band.values,
                fill="tozeroy",
                fillcolor="rgba(0,200,100,0.15)",
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=neg_band.values,
                fill="tozeroy",
                fillcolor="rgba(220,50,50,0.15)",
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1, secondary_y=False,
        )

        # Nulla referencia vonal a trend score-hoz
        fig.add_hline(
            y=0, line=dict(color="rgba(255,255,255,0.3)", width=0.8, dash="dot"),
            row=1, col=1,
        )

        # ── Alsó subplot: stacked area chart ─────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=trend_df["norm_mu"].values,
                name="norm_μ",
                line=dict(color="#636EFA", width=0),
                fill="tozeroy",
                fillcolor="rgba(99,110,250,0.4)",
                stackgroup="components",
                hovertemplate="norm_μ: %{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=trend_df["norm_mu_dot"].values,
                name="norm_μ̇",
                line=dict(color="#EF553B", width=0),
                fill="tonexty",
                fillcolor="rgba(239,85,59,0.4)",
                stackgroup="components",
                hovertemplate="norm_μ̇: %{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=trend_df["norm_mu_ddot"].values,
                name="norm_μ̈",
                line=dict(color="#00CC96", width=0),
                fill="tonexty",
                fillcolor="rgba(0,204,150,0.4)",
                stackgroup="components",
                hovertemplate="norm_μ̈: %{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="Normalizált érték", row=2, col=1)
        fig.update_xaxes(title_text="Idő", row=2, col=1)

        # ── Layout alkalmazása és mentés ─────────────────────────────────
        self.apply_layout(fig, "Trend score dashboard", height=1000)

        return self.save(fig, "trend_dashboard")
