"""
Predikció pontosság vizualizáció.

3x2 grid: 3 horizont (5m, 15m, 60m) × 2 nézet (scatter + idősoros).
Metrikák: RMSE, MAE, hit rate (előjel-egyezés %).
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

# Horizontok percben, és a hozzájuk tartozó címkék
HORIZONS = [5, 15, 60]
HORIZON_LABELS = {5: "5 perc", 15: "15 perc", 60: "1 óra"}


class PredictionPlot(BasePlot):
    """Predikció pontosság — scatter és idősoros nézet horizontonként."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def _find_tf_label(self, tau: int, tf_minutes: dict[str, int]) -> str | None:
        """Megkeresi a tau percnek megfelelő timeframe címkét."""
        for label, minutes in tf_minutes.items():
            if minutes == tau:
                return label
        return None

    def _compute_metrics(
        self, predicted: pd.Series, actual: pd.Series
    ) -> dict[str, float]:
        """RMSE, MAE és hit rate kiszámítása."""
        # Közös index
        common = predicted.index.intersection(actual.index)
        pred = predicted.loc[common].dropna()
        act = actual.loc[common].dropna()
        common = pred.index.intersection(act.index)
        pred = pred.loc[common]
        act = act.loc[common]

        if len(pred) == 0:
            return {"rmse": np.nan, "mae": np.nan, "hit_rate": np.nan}

        errors = pred - act
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        # Hit rate: előjel-egyezés százaléka
        sign_match = (np.sign(pred) == np.sign(act))
        hit_rate = float(sign_match.mean() * 100)

        return {"rmse": rmse, "mae": mae, "hit_rate": hit_rate}

    def generate(
        self,
        states_df: pd.DataFrame,
        returns: dict[str, pd.Series],
        predictions: dict[int, pd.DataFrame],
        tf_minutes: dict[str, int],
    ) -> Path:
        """
        Predikció pontosság vizualizáció generálása.

        Args:
            states_df: Kalman szűrő állapotok.
            returns: Nyers hozamok timeframe-enként {'1m': Series, '5m': Series, ...}.
            predictions: {5: DataFrame(predicted, ci_lower, ci_upper), 15: ..., 60: ...}.
            tf_minutes: {'1m': 1, '5m': 5, ...}.

        Returns:
            Path az elmentett fájlhoz.
        """
        subplot_titles = []
        for tau in HORIZONS:
            label = HORIZON_LABELS.get(tau, f"{tau}m")
            subplot_titles.append(f"Predicted vs Actual — {label}")
            subplot_titles.append(f"Idősoros — {label}")

        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=False,
            vertical_spacing=0.07,
            horizontal_spacing=0.08,
            subplot_titles=subplot_titles,
        )

        colors = {5: "#636EFA", 15: "#EF553B", 60: "#00CC96"}

        for row_idx, tau in enumerate(HORIZONS, start=1):
            pred_df = predictions.get(tau)
            if pred_df is None:
                logger.warning(f"Nincs predikció a {tau} perces horizonthoz.")
                continue

            # Aktuális hozamok keresése
            tf_label = self._find_tf_label(tau, tf_minutes)
            if tf_label is None or tf_label not in returns:
                logger.warning(
                    f"Nem található hozam adat a {tau} perces horizonthoz "
                    f"(keresett tf: {tf_label})."
                )
                continue

            actual = returns[tf_label]
            color = colors.get(tau, "#AB63FA")

            # Közös index a metrikákhoz és ábrázoláshoz
            common_idx = pred_df.index.intersection(actual.index)
            pred_vals = pred_df["predicted"].loc[common_idx].dropna()
            actual_vals = actual.loc[common_idx].dropna()
            common_idx = pred_vals.index.intersection(actual_vals.index)
            pred_vals = pred_vals.loc[common_idx]
            actual_vals = actual_vals.loc[common_idx]

            # ── Bal oszlop: scatter (predicted vs actual) ────────────────
            fig.add_trace(
                go.Scatter(
                    x=actual_vals,
                    y=pred_vals,
                    mode="markers",
                    name=f"pred vs actual ({tau}m)",
                    marker=dict(color=color, size=3, opacity=0.5),
                    showlegend=False,
                    hovertemplate=(
                        "Actual: %{x:.6f}<br>Predicted: %{y:.6f}<extra></extra>"
                    ),
                ),
                row=row_idx, col=1,
            )

            # 45-fokos referencia vonal (fehér szaggatott)
            if len(actual_vals) > 0:
                all_vals = pd.concat([actual_vals, pred_vals])
                vmin = float(all_vals.min())
                vmax = float(all_vals.max())
                fig.add_trace(
                    go.Scatter(
                        x=[vmin, vmax],
                        y=[vmin, vmax],
                        mode="lines",
                        name="45°",
                        line=dict(color="white", width=1, dash="dash"),
                        showlegend=False,
                    ),
                    row=row_idx, col=1,
                )

            # Metrikák annotálása a scatter subplot-ra
            metrics = self._compute_metrics(pred_vals, actual_vals)
            annotation_text = (
                f"RMSE: {metrics['rmse']:.6f}<br>"
                f"MAE: {metrics['mae']:.6f}<br>"
                f"Hit rate: {metrics['hit_rate']:.1f}%"
            )
            # Az x/y tengelyek sorszáma a 3x2 grid-ben (bal oszlop)
            ax_idx = (row_idx - 1) * 2 + 1
            x_ref = "x domain" if ax_idx == 1 else f"x{ax_idx} domain"
            y_ref = "y domain" if ax_idx == 1 else f"y{ax_idx} domain"
            fig.add_annotation(
                text=annotation_text,
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=11, color="white"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1,
                xref=x_ref,
                yref=y_ref,
            )

            fig.update_xaxes(title_text="Actual", row=row_idx, col=1)
            fig.update_yaxes(title_text="Predicted", row=row_idx, col=1)

            # ── Jobb oszlop: idősoros (predicted + CI + actual) ──────────
            pred_full = pred_df["predicted"]
            ci_lower = pred_df["ci_lower"]
            ci_upper = pred_df["ci_upper"]

            # 95% CI sáv — felső határ
            fig.add_trace(
                go.Scatter(
                    x=ci_upper.index,
                    y=ci_upper.values,
                    mode="lines",
                    name=f"95% CI felső ({tau}m)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx, col=2,
            )

            # 95% CI sáv — alsó határ (tonefill az előzőhöz)
            fig.add_trace(
                go.Scatter(
                    x=ci_lower.index,
                    y=ci_lower.values,
                    mode="lines",
                    name=f"95% CI ({tau}m)",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx, col=2,
            )

            # Predikció vonal
            fig.add_trace(
                go.Scatter(
                    x=pred_full.index,
                    y=pred_full.values,
                    mode="lines",
                    name=f"Predikció ({tau}m)",
                    line=dict(color=color, width=1.5),
                    hovertemplate="Pred: %{y:.6f}<extra></extra>",
                ),
                row=row_idx, col=2,
            )

            # Aktuális hozam pontok
            actual_plot = actual.loc[actual.index.isin(pred_full.index)]
            fig.add_trace(
                go.Scatter(
                    x=actual_plot.index,
                    y=actual_plot.values,
                    mode="markers",
                    name=f"Tényleges ({tau}m)",
                    marker=dict(color="white", size=2, opacity=0.6),
                    hovertemplate="Actual: %{y:.6f}<extra></extra>",
                ),
                row=row_idx, col=2,
            )

            fig.update_yaxes(title_text="Hozam", row=row_idx, col=2)

        # ── Layout alkalmazása és mentés ─────────────────────────────────
        self.apply_layout(fig, "Predikció pontosság (5m, 15m, 1h)", height=1200)

        return self.save(fig, "prediction_accuracy")
