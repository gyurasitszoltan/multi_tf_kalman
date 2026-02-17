"""
Vizualizáció #9 — Folytonos vs diszkrét H mátrix összehasonlítás.

Felso: mu_hat continuous vs discrete.
Kozepso: kulonbseg (continuous - discrete).
Also: innovacio (mahalanobis) mindkét módra.
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


class HComparePlot(BasePlot):
    """Folytonos vs diszkrét H mátrix összehasonlítás vizualizáció."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(self, continuous_df: pd.DataFrame, discrete_df: pd.DataFrame) -> Path:
        """
        H mátrix összehasonlítás ábra generálása.

        Args:
            continuous_df: Folytonos H mátrix eredmények.
                           Oszlopok: mu_hat, mu_dot_hat, mu_ddot_hat,
                                     P00, P11, P22, mahalanobis
                           DatetimeIndex.
            discrete_df:   Diszkrét H mátrix eredmények.
                           Ugyanazok az oszlopok és index.

        Returns:
            Path az elmentett fájlhoz.
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "mu_hat: folytonos vs diszkrét",
                "Különbség (folytonos - diszkrét)",
                "Mahalanobis-távolság (innovációs variancia)",
            ),
        )

        # --- Felső subplot: mu_hat összehasonlítás ---
        fig.add_trace(
            go.Scatter(
                x=continuous_df.index,
                y=continuous_df["mu_hat"],
                name="Folytonos H",
                line=dict(color="rgb(65, 105, 225)", width=1.5),  # royal blue
                hovertemplate="Folytonos<br>mu_hat: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=discrete_df.index,
                y=discrete_df["mu_hat"],
                name="Diszkrét H",
                line=dict(color="rgb(255, 165, 0)", width=1.5),  # narancs
                hovertemplate="Diszkrét<br>mu_hat: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

        fig.update_yaxes(title_text="mu_hat", row=1, col=1)

        # --- Középső subplot: különbség ---
        # Közös index metszete (ha eltérne)
        common_idx = continuous_df.index.intersection(discrete_df.index)
        diff = continuous_df.loc[common_idx, "mu_hat"] - discrete_df.loc[common_idx, "mu_hat"]

        fig.add_trace(
            go.Scatter(
                x=common_idx,
                y=diff,
                name="Különbség (foly. - diskr.)",
                line=dict(color="rgb(50, 205, 50)", width=1.2),  # lime zöld
                hovertemplate="Diff: %{y:,.4f}<extra></extra>",
                fill="tozeroy",
                fillcolor="rgba(50, 205, 50, 0.15)",
            ),
            row=2, col=1,
        )

        # Nulla referencia vonal
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8, row=2, col=1)

        fig.update_yaxes(title_text="mu_hat különbség", row=2, col=1)

        # --- Alsó subplot: Mahalanobis-távolság ---
        fig.add_trace(
            go.Scatter(
                x=continuous_df.index,
                y=continuous_df["mahalanobis"],
                name="Mahalanobis (folytonos)",
                line=dict(color="rgb(65, 105, 225)", width=1.2),
                hovertemplate="Folytonos<br>Mahalanobis: %{y:.3f}<extra></extra>",
                legendgroup="mahal",
            ),
            row=3, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=discrete_df.index,
                y=discrete_df["mahalanobis"],
                name="Mahalanobis (diszkrét)",
                line=dict(color="rgb(255, 165, 0)", width=1.2),
                hovertemplate="Diszkrét<br>Mahalanobis: %{y:.3f}<extra></extra>",
                legendgroup="mahal",
            ),
            row=3, col=1,
        )

        fig.update_yaxes(title_text="Mahalanobis-távolság", row=3, col=1)
        fig.update_xaxes(title_text="Idő", row=3, col=1)

        # --- Layout + mentés ---
        self.apply_layout(
            fig,
            title="H mátrix összehasonlítás (folytonos vs diszkrét)",
            height=1100,
        )

        return self.save(fig, "h_compare")
