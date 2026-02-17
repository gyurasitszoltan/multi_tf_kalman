"""
P kovariancia mátrix evolúció vizualizáció.

Felső subplot: P diagonális elemek (log skála)
Alsó subplot: összesített konfidencia = 1/trace(P)
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


class CovariancePlot(BasePlot):
    """P mátrix evolúció — diagonális elemek és trace-alapú konfidencia."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(self, states_df: pd.DataFrame) -> Path:
        """
        P mátrix vizualizáció generálása.

        Args:
            states_df: DataFrame P00, P11, P22 oszlopokkal, DatetimeIndex-szel.

        Returns:
            Path az elmentett fájlhoz.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[
                "P diagonális elemek (log skála)",
                "Összesített konfidencia — 1 / trace(P)",
            ],
        )

        idx = states_df.index
        p00 = states_df["P00"]
        p11 = states_df["P11"]
        p22 = states_df["P22"]

        # ── Felső: P diagonális elemek (log skála) ───────────────────────
        fig.add_trace(
            go.Scatter(
                x=idx, y=p00,
                name="P₀₀ (μ)",
                line=dict(color="#636EFA", width=1.5),
                hovertemplate="P₀₀: %{y:.2e}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=p11,
                name="P₁₁ (μ̇)",
                line=dict(color="#EF553B", width=1.5),
                hovertemplate="P₁₁: %{y:.2e}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=idx, y=p22,
                name="P₂₂ (μ̈)",
                line=dict(color="#00CC96", width=1.5),
                hovertemplate="P₂₂: %{y:.2e}<extra></extra>",
            ),
            row=1, col=1,
        )

        # Log skála a felső subplot y tengelyén
        fig.update_yaxes(type="log", title_text="P érték (log)", row=1, col=1)

        # ── Alsó: összesített konfidencia ────────────────────────────────
        trace_p = p00 + p11 + p22
        # Nullával osztás elleni védelem
        confidence = 1.0 / trace_p.replace(0, np.nan)

        fig.add_trace(
            go.Scatter(
                x=idx, y=confidence,
                name="Konfidencia (1/trace(P))",
                line=dict(color="#FFA15A", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,161,90,0.15)",
                hovertemplate="Konfidencia: %{y:.4g}<extra></extra>",
            ),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="1 / trace(P)", row=2, col=1)
        fig.update_xaxes(title_text="Idő", row=2, col=1)

        # ── Layout alkalmazása és mentés ─────────────────────────────────
        self.apply_layout(fig, "P kovariancia mátrix evolúció", height=900)

        return self.save(fig, "covariance_evolution")
