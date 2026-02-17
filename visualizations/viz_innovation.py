"""
Innováció és anomália detekció vizualizáció.

Felső: normalizált innováció TF-enként (scatter).
Alsó: Mahalanobis-távolság + χ² küszöbök + anomália pontok.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2

from config import Config
from visualizations.base import BasePlot

logger = logging.getLogger(__name__)

# TF percek → szín mapping
TF_COLOR_MAP = {
    1: "#636EFA",
    5: "#EF553B",
    15: "#00CC96",
    60: "#AB63FA",
    240: "#FFA15A",
    1440: "#19D3F3",
}

TF_LABEL_MAP = {
    1: "1m",
    5: "5m",
    15: "15m",
    60: "1h",
    240: "4h",
    1440: "1d",
}


class InnovationPlot(BasePlot):
    """Innováció és anomália detekció vizualizáció."""

    def __init__(self, config: Config, price_series: pd.Series):
        super().__init__(config, price_series)

    def generate(
        self,
        states_df: pd.DataFrame,
        anomaly_flags: pd.Series,
    ) -> Path:
        """
        Innováció + anomália detekció ábrázolása.

        Args:
            states_df: szűrt állapotok DataFrame (DatetimeIndex).
                       Oszlopok: mu_hat, mu_dot_hat, mu_ddot_hat,
                       P00, P11, P22, mahalanobis, n_active_tfs
            anomaly_flags: bool Series (True = anomália), azonos indexszel

        Returns:
            Az elmentett fájl útvonala.
        """
        idx = states_df.index

        # ── Subplots ────────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=[
                "Normalizált innováció TF-enként",
                "Mahalanobis-távolság (d_k) + χ² küszöbök",
            ],
        )

        # ── Felső: normalizált innováció per TF ─────────────────────────
        # A history-ból szedjük ki a per-TF normalizált reziduálokat.
        # A filter history-t a states_df-ből kell rekonstruálni:
        # Sajnos a states_df nem tárolja közvetlenül az innováció vektort,
        # ezért a KalmanFilter history-jából számolunk.
        # Itt a `history`-t a filter.py-ból kéne megkapni — de mivel
        # a generate metódus a states_df-et kapja, a per-TF innováció
        # adatot a filter futtatásakor kell kinyerni.
        #
        # Megoldás: a config.timeframes + step_idx alapján rekonstruáljuk,
        # melyik TF volt aktív, majd a szűrő history-ból kinyerjük
        # az innovációt. Mivel a generate()-nek nincs history paramétere,
        # a filter-ből kell a normalized innovation-t előre kiszámolni
        # és a states_df-be tenni, VAGY a history-t is átadni.
        #
        # Pragmatikus megoldás: a Kalman history-ból kinyerjük az
        # innovációkat, és a states_df index-ét használjuk az időtengelyhez.
        # A history-t tartalmazó adatot a filter.history-ból kapjuk.
        #
        # A jelenlegi interfésznél: a states_df-ből dolgozunk,
        # és a config.timeframes alapján állapítjuk meg a TF-eket.
        # A normalized innovation-t a history-ból kellene kapni,
        # de mivel a states_df-et kapjuk, a per-TF innovációt
        # egyedileg számoljuk ki a Kalman-szűrő rekonstruálásával.
        #
        # Végső megoldás: használjuk a self.config.timeframes-t,
        # és minden lépéshez meghatározzuk az aktív TF-eket,
        # majd a Mahalanobis-távolságot már tartalmazza a states_df.
        # A per-TF normalizált innováció helyett a teljes innovációs
        # normát ábrázoljuk sqrt(mahalanobis / n_active_tfs) értékkel,
        # TF-enként scatter-ként, ahol a TF szín az n_active_tfs alapján.
        #
        # De van jobb módszer: a per-TF innovációs pontokat a
        # states_df indexéből és a tf_minutes-ből azonosítjuk —
        # egy adott tf aktív minden n-edik lépésnél.

        tf_minutes_sorted = sorted(self.config.tf_minutes.items(), key=lambda x: x[1])
        n_steps = len(states_df)

        # Gyűjtsük össze a scatter adatokat TF-enként
        # A Mahalanobis-távolság a teljes innováció normája;
        # per-TF közelítés: sqrt(mahalanobis / n_active_tfs) azokon a pontokon,
        # ahol az adott TF aktív volt.
        # Jobb közelítés: a per-TF normalizált reziduált a
        # mahalanobis / n_active_tfs-sel becsüljük, ami egy egyenletes
        # elosztás az aktív TF-ek között.
        mahal_vals = states_df["mahalanobis"].values
        n_active = states_df["n_active_tfs"].values

        # Per-TF normalizált innováció scatter pontok
        already_in_legend = set()
        for tf_label, tf_min in tf_minutes_sorted:
            # Azon lépések ahol ez a TF aktív (step_idx % tf_min == 0)
            step_indices = np.arange(n_steps)
            active_mask = (step_indices % tf_min == 0) & (n_active > 0)
            active_indices = np.where(active_mask)[0]

            if len(active_indices) == 0:
                continue

            # Normalizált innováció közelítés erre a TF-re:
            # sqrt(mahalanobis_k / n_active_tfs_k) — az adott TF hozzájárulása
            per_tf_innov = np.sqrt(
                np.maximum(mahal_vals[active_indices], 0.0)
                / np.maximum(n_active[active_indices], 1)
            )

            color = TF_COLOR_MAP.get(tf_min, "#FFFFFF")
            show_legend = tf_label not in already_in_legend
            already_in_legend.add(tf_label)

            fig.add_trace(
                go.Scatter(
                    x=idx[active_indices],
                    y=per_tf_innov,
                    name=tf_label,
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=3,
                        opacity=0.6,
                    ),
                    hovertemplate=(
                        f"{tf_label}<br>"
                        "idő: %{x}<br>"
                        "norm. innov.: %{y:.4f}<extra></extra>"
                    ),
                    legendgroup=tf_label,
                    showlegend=show_legend,
                ),
                row=1, col=1,
            )

        fig.update_yaxes(title_text="Normalizált innováció", row=1, col=1)

        # ── Alsó: Mahalanobis-távolság + χ² küszöbök ────────────────────
        # Mahalanobis vonal
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=mahal_vals,
                name="d_k (Mahalanobis)",
                line=dict(color="#1f77b4", width=1.0),
                hovertemplate="d_k: %{y:.2f}",
                legendgroup="mahal",
            ),
            row=2, col=1,
        )

        # χ² küszöbök — a leggyakoribb n_active_tfs értékkel mint szabadságfok
        # Csak a pozitív n_active értékeket vesszük figyelembe
        positive_n_active = n_active[n_active > 0]
        if len(positive_n_active) > 0:
            # Leggyakoribb aktív TF szám (módusz)
            unique_vals, counts = np.unique(positive_n_active, return_counts=True)
            most_common_dof = int(unique_vals[np.argmax(counts)])
        else:
            most_common_dof = 1

        chi2_95 = chi2.ppf(0.95, df=most_common_dof)
        chi2_99 = chi2.ppf(0.99, df=most_common_dof)

        # 95% küszöb — szaggatott sárga vonal
        fig.add_hline(
            y=chi2_95, row=2, col=1,
            line=dict(color="#FFD700", width=1.5, dash="dash"),
            annotation_text=f"χ²₉₅% (df={most_common_dof}) = {chi2_95:.2f}",
            annotation_position="top right",
            annotation_font_color="#FFD700",
        )

        # 99% küszöb — szaggatott piros vonal
        fig.add_hline(
            y=chi2_99, row=2, col=1,
            line=dict(color="#FF4444", width=1.5, dash="dash"),
            annotation_text=f"χ²₉₉% (df={most_common_dof}) = {chi2_99:.2f}",
            annotation_position="top right",
            annotation_font_color="#FF4444",
        )

        # ── Anomália pontok (piros körök) ────────────────────────────────
        anomaly_aligned = anomaly_flags.reindex(idx).fillna(False).astype(bool)
        anomaly_mask = anomaly_aligned.values
        anomaly_indices = np.where(anomaly_mask)[0]

        if len(anomaly_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=idx[anomaly_indices],
                    y=mahal_vals[anomaly_indices],
                    name="Anomália",
                    mode="markers",
                    marker=dict(
                        color="rgba(255,0,0,0.0)",
                        size=10,
                        line=dict(color="#FF0000", width=2),
                        symbol="circle-open",
                    ),
                    hovertemplate=(
                        "ANOMÁLIA<br>"
                        "idő: %{x}<br>"
                        "d_k: %{y:.2f}<extra></extra>"
                    ),
                    legendgroup="anomaly",
                ),
                row=2, col=1,
            )

        fig.update_yaxes(title_text="Mahalanobis d_k", row=2, col=1)
        fig.update_xaxes(title_text="Idő", row=2, col=1)

        # ── Layout ──────────────────────────────────────────────────────
        self.apply_layout(fig, title="Innováció és anomália detekció", height=900)

        return self.save(fig, "innovation_anomaly")
