"""
Vizualizáció alap osztály — BasePlot.

Egységes plotly layout, export, ár overlay.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config

logger = logging.getLogger(__name__)


class BasePlot:
    """Minden vizualizáció ebből öröklődik."""

    def __init__(self, config: Config, price_series: pd.Series):
        self.config = config
        self.viz = config.visualization
        self.price = price_series
        self.output_dir = Path(self.viz.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def apply_layout(
        self,
        fig: go.Figure,
        title: str,
        height: Optional[int] = None,
    ) -> None:
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            template=self.viz.theme,
            width=self.viz.width,
            height=height or self.viz.height,
            legend=dict(x=0, y=1.02, orientation="h"),
            hovermode="x unified",
            margin=dict(l=60, r=60, t=80, b=40),
        )

    def save(self, fig: go.Figure, filename: str) -> Path:
        fmt = self.viz.format

        if fmt in ("html", "both"):
            path = self.output_dir / f"{filename}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            logger.info(f"  Mentve: {path}")

        if fmt in ("png", "both"):
            path_png = self.output_dir / f"{filename}.png"
            try:
                fig.write_image(str(path_png), width=self.viz.width, height=self.viz.height, scale=2)
                logger.info(f"  Mentve: {path_png}")
            except Exception as e:
                logger.warning(f"  PNG export hiba: {e}")

        return self.output_dir / f"{filename}.html"

    def add_price_trace(
        self,
        fig: go.Figure,
        row: int = 1,
        col: int = 1,
        secondary_y: bool = True,
        opacity: float = 0.3,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=self.price.index,
                y=self.price.values,
                name="BTC ár",
                line=dict(color=f"rgba(180,180,180,{opacity})", width=1),
                hovertemplate="%{y:,.0f} USD",
                showlegend=True,
            ),
            row=row, col=col,
            secondary_y=secondary_y,
        )
