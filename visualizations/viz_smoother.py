"""
Vizualizáció #10 — RTS simító vs online szűrő.

Felso: mu_hat online (szaggatott kék) vs mu_smooth RTS (folytonos narancs).
Kozepso: mu_dot_hat vs mu_dot_smooth.
Also: P00 online vs P00_smooth (simított P mindig <= online P).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from visualizations.base import BasePlot

logger = logging.getLogger(__name__)


class SmootherPlot(BasePlot):
    """RTS simító vs online szűrő összehasonlítás vizualizáció."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(self, states_df: pd.DataFrame, smoothed_df: pd.DataFrame) -> Path:
        """
        RTS simító vs online szűrő ábra generálása.

        Args:
            states_df:   Online szűrő eredmények.
                         Oszlopok: mu_hat, mu_dot_hat, mu_ddot_hat,
                                   P00, P11, P22
                         DatetimeIndex.
            smoothed_df: RTS simított eredmények.
                         Oszlopok: mu_smooth, mu_dot_smooth, mu_ddot_smooth,
                                   P00_smooth, P11_smooth, P22_smooth
                         DatetimeIndex.

        Returns:
            Path az elmentett fájlhoz.
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "Szint (mu): online szűrő vs RTS simított",
                "Sebesség (mu_dot): online szűrő vs RTS simított",
                "Szint variancia (P00): online vs simított",
            ),
        )

        # --- Felső subplot: mu_hat vs mu_smooth ---
        fig.add_trace(
            go.Scatter(
                x=states_df.index,
                y=states_df["mu_hat"],
                name="Online mu_hat",
                line=dict(color="rgb(65, 105, 225)", width=1.5, dash="dash"),
                hovertemplate="Online<br>mu_hat: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=smoothed_df.index,
                y=smoothed_df["mu_smooth"],
                name="RTS mu_smooth",
                line=dict(color="rgb(255, 165, 0)", width=1.8),
                hovertemplate="RTS<br>mu_smooth: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

        fig.update_yaxes(title_text="mu (szint)", row=1, col=1)

        # --- Középső subplot: mu_dot_hat vs mu_dot_smooth ---
        fig.add_trace(
            go.Scatter(
                x=states_df.index,
                y=states_df["mu_dot_hat"],
                name="Online mu_dot",
                line=dict(color="rgb(65, 105, 225)", width=1.5, dash="dash"),
                hovertemplate="Online<br>mu_dot: %{y:.6f}<extra></extra>",
                legendgroup="mu_dot",
            ),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=smoothed_df.index,
                y=smoothed_df["mu_dot_smooth"],
                name="RTS mu_dot_smooth",
                line=dict(color="rgb(255, 165, 0)", width=1.8),
                hovertemplate="RTS<br>mu_dot_smooth: %{y:.6f}<extra></extra>",
                legendgroup="mu_dot",
            ),
            row=2, col=1,
        )

        # Nulla referencia a sebesség subploton
        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=0.6, row=2, col=1)

        fig.update_yaxes(title_text="mu_dot (sebesség)", row=2, col=1)

        # --- Alsó subplot: P00 online vs P00_smooth ---
        # Simított variancia mindig <= online variancia (RTS tulajdonság)
        fig.add_trace(
            go.Scatter(
                x=states_df.index,
                y=states_df["P00"],
                name="Online P00",
                line=dict(color="rgb(65, 105, 225)", width=1.5, dash="dash"),
                hovertemplate="Online<br>P00: %{y:.6e}<extra></extra>",
                legendgroup="P00",
            ),
            row=3, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=smoothed_df.index,
                y=smoothed_df["P00_smooth"],
                name="RTS P00_smooth",
                line=dict(color="rgb(255, 165, 0)", width=1.8),
                hovertemplate="RTS<br>P00_smooth: %{y:.6e}<extra></extra>",
                legendgroup="P00",
            ),
            row=3, col=1,
        )

        # A kettő közötti terület kitöltése — vizualizálja a variancia csökkenést
        # Közös index metszetén számolunk
        common_idx = states_df.index.intersection(smoothed_df.index)
        if len(common_idx) > 0:
            p00_online = states_df.loc[common_idx, "P00"]
            p00_smooth = smoothed_df.loc[common_idx, "P00_smooth"]

            fig.add_trace(
                go.Scatter(
                    x=common_idx,
                    y=p00_online,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3, col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=common_idx,
                    y=p00_smooth,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255, 165, 0, 0.12)",
                    name="Variancia csökkenés",
                    hoverinfo="skip",
                    legendgroup="P00",
                ),
                row=3, col=1,
            )

        fig.update_yaxes(title_text="P00 (szint variancia)", row=3, col=1)
        fig.update_xaxes(title_text="Idő", row=3, col=1)

        # --- Layout + mentés ---
        self.apply_layout(
            fig,
            title="Online szűrő vs RTS simított",
            height=1100,
        )

        return self.save(fig, "smoother_rts")
