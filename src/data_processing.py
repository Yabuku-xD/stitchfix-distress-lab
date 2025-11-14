"""A slightly overbuilt funnel that turns SEC company facts into tidy tables."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import METRIC_TAGS, RAW_DATA_DIR

META_COLUMNS = ["period_end", "fy", "fp", "form", "filed"]


def _select_first_available_tag(facts: dict, candidates: Iterable[str]) -> str | None:
    """Grab the first tag that actually exists. Nothing clever here."""
    for tag in candidates:
        if tag in facts:
            return tag
    return None


def _extract_metric_series(
    facts: dict,
    tag_candidates: Iterable[str],
    metric_name: str,
    preferred_units: tuple[str, ...] = ("USD",),
) -> pd.DataFrame | None:
    tag = _select_first_available_tag(facts, tag_candidates)
    if tag is None:
        return None

    tag_payload = facts[tag]
    units = tag_payload.get("units", {})
    unit_key = next((unit for unit in preferred_units if unit in units), None)
    if unit_key is None:
        # Fall back to any available unit
        if not units:
            return None
        unit_key = next(iter(units.keys()))

    entries = pd.DataFrame(units[unit_key])
    if entries.empty:
        return None

    entries = entries.rename(columns={"end": "period_end"})
    entries["period_end"] = pd.to_datetime(entries["period_end"])
    entries = (
        entries[META_COLUMNS + ["val"]]
        .sort_values(["period_end", "filed"])
        .drop_duplicates(subset=["period_end"], keep="last")
    )
    entries = entries.rename(columns={"val": metric_name})
    return entries


def build_financial_panel(
    company_facts: dict,
    *,
    metric_tags: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Convert the SEC company facts JSON into a merged metric panel."""

    # TODO: consider caching intermediate frames if this ever slows down.

    gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})
    if not gaap_facts:
        raise ValueError("No US GAAP facts present in the company facts payload.")

    metric_tags = metric_tags or METRIC_TAGS
    series_frames: list[pd.Series] = []
    metadata_frames: list[pd.DataFrame] = []
    for metric_name, candidates in metric_tags.items():
        frame = _extract_metric_series(gaap_facts, candidates, metric_name)
        if frame is None:
            continue
        metadata_frames.append(frame[META_COLUMNS])
        series = frame.set_index("period_end")[metric_name]
        series_frames.append(series)

    if not series_frames:
        raise ValueError("Failed to extract any metric series from company facts.")

    metrics_df = pd.concat(series_frames, axis=1)
    meta_df = (
        pd.concat(metadata_frames, ignore_index=True)
        .sort_values(["period_end", "filed"])
        .drop_duplicates(subset=["period_end"], keep="last")
        .set_index("period_end")
    )
    merged = meta_df.join(metrics_df, how="outer").reset_index()
    merged["period_end"] = pd.to_datetime(merged["period_end"])
    merged["frequency"] = merged["form"].astype(str).str.upper().apply(
        lambda form: "annual" if "10-K" in str(form) else "quarterly"
    )
    merged["period_start"] = merged.groupby("frequency")["period_end"].shift(1)
    merged = merged.sort_values("period_end").reset_index(drop=True)
    merged.to_csv(RAW_DATA_DIR / "sfix_sec_financial_panel.csv", index=False)
    return merged


def split_frequency_panels(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate annual and quarterly DataFrames."""

    annual = panel[panel["frequency"] == "annual"].copy()
    quarterly = panel[panel["frequency"] == "quarterly"].copy()
    # Keeping the resets so callers don't inherit weird index gaps.
    return annual.reset_index(drop=True), quarterly.reset_index(drop=True)


