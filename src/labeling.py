"""Hand-rolled distress labels so we remember why each quarter felt painful."""

from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from .config import DISTRESS_EVENTS, PROCESSED_DATA_DIR


def _event_flag(period_end: pd.Series, start: str, end: str) -> pd.Series:
    """Basic window flag. Nothing fancy—just guardrails for the news overrides."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    return (period_end >= start_dt) & (period_end <= end_dt)


def label_distress_periods(
    enriched_features: pd.DataFrame,
    *,
    drawdown_threshold: float = -0.35,
    altman_threshold: float = 1.8,
    liquidity_threshold: float = 1.0,
    margin_threshold: float = -0.05,
) -> pd.DataFrame:
    """Combine news, market, and fundamental triggers into a binary label."""

    # TODO: attach confidence scores per source once we have better metadata.

    df = enriched_features.copy()
    df["period_end"] = pd.to_datetime(df["period_end"])

    news_flags = []
    for event in DISTRESS_EVENTS:
        news_flags.append(_event_flag(df["period_end"], event.start, event.end))
    if news_flags:
        news_flag = pd.Series(np.logical_or.reduce(news_flags), index=df.index)
    else:
        news_flag = pd.Series(False, index=df.index)

    price_flag = (
        (df["drawdown_12m"] <= drawdown_threshold)
        | (df["return_6m"] <= drawdown_threshold / 2)
    )
    price_flag = price_flag.fillna(False)

    fundamental_flag = (
        (df["current_ratio"] < liquidity_threshold)
        & (df["net_margin"] < margin_threshold)
    ) | (df["altman_z_score"] <= altman_threshold)
    fundamental_flag = fundamental_flag.fillna(False)

    flags = {
        "news": news_flag,
        "price": price_flag,
        "fundamental": fundamental_flag,
    }

    df["distress_flag"] = 0
    reasons: List[List[str]] = [[] for _ in range(len(df))]
    for key, mask in flags.items():
        df.loc[mask, "distress_flag"] = 1
        for idx in df.index[mask]:
            reasons[idx].append(key)

    df["distress_reasons"] = ["|".join(reason) if reason else "" for reason in reasons]
    df.to_csv(PROCESSED_DATA_DIR / "sfix_financial_features_labeled.csv", index=False)
    return df


