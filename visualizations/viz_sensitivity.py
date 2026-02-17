"""
Vizualizáció #8 — q paraméter érzékenység.

Felso subplot: BTC ár + szurt mu_hat kulonbozo q értékekre (szinezve).
Also subplot: empirikus innovacio variancia arany (ratio ~ 1 => jol hangolt q).
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


class SensitivityPlot(BasePlot):
    """q paraméter érzékenységi vizualizáció."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(self, q_results: dict[float, pd.DataFrame]) -> Path:
        """
        q érzékenységi ábra generálása.

        Args:
            q_results: {q_value: states_df} ahol states_df oszlopai:
                       mu_hat, P00, P11, P22, mahalanobis, n_active_tfs
                       DatetimeIndex-szel.

        Returns:
            Path az elmentett fájlhoz.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Szűrt mu_hat különböző q értékekre",
                "Empirikus innovációs variancia arány (≈1 ideális)",
            ),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        # --- Színskála a q értékekhez ---
        sorted_qs = sorted(q_results.keys())
        n_q = len(sorted_qs)

        # Plasma-szerű szinpaletta: lila -> sarga
        plasma_colors = [
            "rgb(13, 8, 135)",    # sötét lila
            "rgb(126, 3, 168)",   # lila
            "rgb(204, 71, 120)",  # pink
            "rgb(248, 149, 64)",  # narancs
            "rgb(240, 249, 33)",  # sárga
            "rgb(60, 190, 113)",  # zöld
            "rgb(56, 116, 215)",  # kék
        ]

        def get_color(idx: int, total: int) -> str:
            """Szín interpolálás a palettából."""
            if total == 1:
                return plasma_colors[0]
            frac = idx / (total - 1)
            pos = frac * (len(plasma_colors) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(plasma_colors) - 1)
            return plasma_colors[lo] if lo == hi else plasma_colors[lo]

        # Pontos szín kiszámítása: egyenletes elosztás a palettán
        def pick_color(idx: int, total: int) -> str:
            """Szín kiválasztás az indexhez."""
            if total <= len(plasma_colors):
                # Kevesebb q mint szín: egyenletes mintavétel
                step = max(1, (len(plasma_colors) - 1) / max(total - 1, 1))
                ci = int(round(idx * step))
                ci = min(ci, len(plasma_colors) - 1)
                return plasma_colors[ci]
            else:
                # Több q mint szín: ciklikus
                return plasma_colors[idx % len(plasma_colors)]

        # --- Felső subplot: BTC ár + mu_hat vonalak ---
        # BTC ár szürke háttér trace (secondary_y)
        self.add_price_trace(fig, row=1, col=1, secondary_y=True, opacity=0.3)

        for i, q_val in enumerate(sorted_qs):
            df = q_results[q_val]
            color = pick_color(i, n_q)

            # q megjelenítés: tudományos jelölés
            q_label = f"q={q_val:.1e}"

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["mu_hat"],
                    name=q_label,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{q_label}<br>mu_hat: %{{y:,.2f}}<extra></extra>",
                    legendgroup=q_label,
                ),
                row=1, col=1,
                secondary_y=False,
            )

        fig.update_yaxes(title_text="mu_hat", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="BTC ár (USD)", row=1, col=1, secondary_y=True)

        # --- Alsó subplot: innovációs variancia arány ---
        # Empirikus: mahalanobis.rolling(100).mean() / n_active_tfs
        # Ha jól hangolt: ratio ≈ 1 (mert E[d_k^2] ~ chi2(p), tehát E[d_k^2]/p ≈ 1)
        for i, q_val in enumerate(sorted_qs):
            df = q_results[q_val]
            color = pick_color(i, n_q)
            q_label = f"q={q_val:.1e}"

            n_active = df["n_active_tfs"].replace(0, np.nan)
            empirical_ratio = df["mahalanobis"].rolling(100, min_periods=10).mean() / n_active

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=empirical_ratio,
                    name=f"{q_label} ratio",
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{q_label}<br>ratio: %{{y:.3f}}<extra></extra>",
                    legendgroup=q_label,
                    showlegend=False,
                ),
                row=2, col=1,
            )

        # Referencia vonal: ratio = 1 (ideális)
        # Az első df indexét használjuk referenciaként
        first_df = q_results[sorted_qs[0]]
        fig.add_trace(
            go.Scatter(
                x=[first_df.index[0], first_df.index[-1]],
                y=[1.0, 1.0],
                name="Ideális (ratio=1)",
                line=dict(color="white", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=True,
            ),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="Innováció variancia arány", row=2, col=1)
        fig.update_xaxes(title_text="Idő", row=2, col=1)

        # --- Layout + mentés ---
        self.apply_layout(fig, title="q paraméter érzékenység", height=1000)

        return self.save(fig, "sensitivity_q")
