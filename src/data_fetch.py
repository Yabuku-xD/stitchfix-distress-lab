"""Grabby little module for SEC facts + price history (warts and all)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from .config import (
    CIK,
    PRICE_HISTORY_START,
    RAW_DATA_DIR,
    SEC_USER_AGENT,
    TICKER,
)

SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def download_company_facts(
    cik: str = CIK,
    *,
    force_refresh: bool = False,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """
    Download the SEC company facts JSON for the given CIK.

    Yeah, it's just a glorified cache loader, but keeping the docstring
    reminds future-me where the file actually lives.
    """

    cache_file = cache_path or RAW_DATA_DIR / f"{cik}_companyfacts.json"
    if cache_file.exists() and not force_refresh:
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/json",
    }
    response = requests.get(
        SEC_COMPANY_FACTS_URL.format(cik=cik),
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _download_price_from_yahoo(
    ticker: str,
    start: str,
    end: str | None,
    auto_adjust: bool,
) -> pd.DataFrame:
    history = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if history.empty:
        raise ValueError("Empty response from Yahoo Finance.")
    return history


def _download_price_from_stooq(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        raise ValueError("Empty response from Stooq.")
    df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def download_price_history(
    ticker: str = TICKER,
    *,
    start: str = PRICE_HISTORY_START,
    end: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download historical OHLCV data, with a very janky fallback story."""

    cache_path = RAW_DATA_DIR / f"{ticker.lower()}_price_history.csv"
    try:
        history = _download_price_from_yahoo(ticker, start, end, auto_adjust)
    except Exception:
        # TODO: detect rate-limit vs legit outage so we can backoff politely.
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["date"])
        history = _download_price_from_stooq(ticker)

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [str(level_0).lower() for level_0, _ in history.columns]
    else:
        history.columns = [str(col).lower() for col in history.columns]

    if "date" not in history.columns:
        history.reset_index(inplace=True)
    if "Date" in history.columns:
        history.rename(columns={"Date": "date"}, inplace=True)
    elif "index" in history.columns:
        history.rename(columns={"index": "date"}, inplace=True)

    history["date"] = pd.to_datetime(history["date"])
    # Ensure canonical column casing (future-me: please don't remove this again).
    column_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "volume": "volume",
    }
    cleaned = history.rename(columns=column_map)
    if "adj_close" not in cleaned.columns and "close" in cleaned.columns:
        cleaned["adj_close"] = cleaned["close"]
    desired_cols = [col for col in column_map.values() if col in cleaned.columns]
    required_cols = {"date", "close"}
    if not required_cols.issubset(desired_cols):
        raise ValueError(
            f"Price history missing required columns {required_cols - set(desired_cols)}"
        )
    cleaned = (
        cleaned[desired_cols]
        .drop_duplicates("date")
        .sort_values("date")
    )
    cleaned.to_csv(cache_path, index=False)
    return cleaned.reset_index(drop=True)


