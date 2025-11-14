"""Plotting nook for quick sanity checks (because tables only tell half the story)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from .config import FIGURES_DIR

plt.style.use("seaborn-v0_8-darkgrid")


def _format_ax(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Period End")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _highlight_distress(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df[df["distress_flag"] == 1].iterrows():
        ax.axvspan(
            row["period_end"] - pd.Timedelta(days=20),
            row["period_end"] + pd.Timedelta(days=20),
            color="red",
            alpha=0.15,
        )


def plot_ratio_trends(
    df: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """Plot multi-line ratio trends with distress shading."""

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in columns:
        if column in df.columns:
            ax.plot(df["period_end"], df[column], marker="o", label=column)
    _highlight_distress(ax, df)
    ax.legend(loc="best")
    _format_ax(ax, title, ylabel)
    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Sequence[str],
    filename: str,
    title: str | None = None,
) -> Path | None:
    """Plot a correlation heatmap for selected numeric columns."""

    available_cols = [col for col in columns if col in df.columns]
    if len(available_cols) < 2:
        return None

    data = df[available_cols].dropna()
    if data.empty:
        return None

    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    annotate = len(available_cols) <= 12
    width = max(6, len(available_cols) * 0.6)
    height = max(4, len(available_cols) * 0.5)

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annotate,
        fmt=".2f",
        cmap="RdBu",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8} if annotate else None,
    )
    ax.set_title(title or "Feature Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_calibration_curve(model_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Path | None:
    """Plot a reliability diagram for a classifier."""

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None

    bins = min(5, max(2, y_true.size // 4))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration - {model_name}")
    ax.legend()

    output_path = FIGURES_DIR / f"calibration_{model_name}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

