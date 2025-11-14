"""Feature engineering for Stitch Fix financial statements (aka spreadsheet glue)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict

from .config import PROCESSED_DATA_DIR


def _get_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Helper that quietly returns NaNs when the column vanished upstream."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def compute_financial_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidity, leverage, profitability, and efficiency ratios."""

    df = panel.copy()
    numeric_cols = df.select_dtypes(include=["number", "float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("period_end")
    df[numeric_cols] = df.groupby("frequency")[numeric_cols].ffill()

    total_current_assets = _get_series(df, "total_current_assets")
    total_current_liabilities = _get_series(df, "total_current_liabilities")
    total_liabilities = _get_series(df, "total_liabilities")
    total_assets = _get_series(df, "total_assets")
    equity = _get_series(df, "total_equity")
    cash = _get_series(df, "cash_and_equivalents")
    inventory = _get_series(df, "inventory")
    revenue = _get_series(df, "total_revenue")
    cost = _get_series(df, "cost_of_revenue")
    gross_profit = _get_series(df, "gross_profit")
    operating_income = _get_series(df, "operating_income")
    net_income = _get_series(df, "net_income")
    ocf = _get_series(df, "operating_cash_flow")
    capex = _get_series(df, "capital_expenditures")
    free_cash_flow = _get_series(df, "free_cash_flow")
    retained_earnings = _get_series(df, "retained_earnings")
    long_term_debt = _get_series(df, "long_term_debt")
    shares = _get_series(df, "shares_outstanding")
    sgna = _get_series(df, "selling_general_admin")
    rnd = _get_series(df, "research_and_development")
    interest_expense = _get_series(df, "interest_expense").abs()

    working_capital = total_current_assets - total_current_liabilities
    quick_assets = total_current_assets - inventory
    if free_cash_flow.isna().all():
        # TODO: capture capitalized software separately if SFIX discloses it later.
        free_cash_flow = ocf + capex

    df["working_capital"] = working_capital
    df["current_ratio"] = _safe_divide(total_current_assets, total_current_liabilities)
    df["quick_ratio"] = _safe_divide(quick_assets, total_current_liabilities)
    df["cash_ratio"] = _safe_divide(cash, total_current_liabilities)
    df["debt_to_equity"] = _safe_divide(total_liabilities, equity)
    df["debt_to_assets"] = _safe_divide(total_liabilities, total_assets)
    df["lt_debt_to_equity"] = _safe_divide(long_term_debt, equity)
    df["net_debt"] = total_liabilities - cash
    df["gross_margin"] = _safe_divide(gross_profit, revenue)
    df["operating_margin"] = _safe_divide(operating_income, revenue)
    df["net_margin"] = _safe_divide(net_income, revenue)
    df["ocf_margin"] = _safe_divide(ocf, revenue)
    df["fcf_margin"] = _safe_divide(free_cash_flow, revenue)
    df["sgna_ratio"] = _safe_divide(sgna, revenue)
    df["rnd_ratio"] = _safe_divide(rnd, revenue)
    df["capex_ratio"] = _safe_divide(capex, revenue)
    df["interest_coverage"] = _safe_divide(operating_income, interest_expense)
    df["asset_turnover"] = _safe_divide(revenue, total_assets)
    df["inventory_turnover"] = _safe_divide(cost, inventory)
    df["working_capital_turnover"] = _safe_divide(revenue, working_capital)
    df["roa"] = _safe_divide(net_income, total_assets)
    df["roe"] = _safe_divide(net_income, equity)
    df["fcf_to_debt"] = _safe_divide(free_cash_flow, total_liabilities)
    df["ocf_to_current_liabilities"] = _safe_divide(ocf, total_current_liabilities)
    df["cash_burn_rate"] = -ocf
    df["liquidity_runway_quarters"] = _safe_divide(cash, df["cash_burn_rate"]).abs()

    wc_over_assets = _safe_divide(working_capital, total_assets)
    re_over_assets = _safe_divide(retained_earnings, total_assets)
    ebit_over_assets = _safe_divide(operating_income, total_assets)
    sales_over_assets = _safe_divide(revenue, total_assets)
    equity_over_liabilities = _safe_divide(equity, total_liabilities)
    df["altman_z_score"] = (
        6.56 * wc_over_assets
        + 3.26 * re_over_assets
        + 6.72 * ebit_over_assets
        + 1.05 * equity_over_liabilities
        + sales_over_assets
    )

    df["revenue_growth_qoq"] = df.groupby("frequency")["total_revenue"].pct_change(
        fill_method=None
    )
    df["net_income_growth_qoq"] = df.groupby("frequency")["net_income"].pct_change(
        fill_method=None
    )

    quarterly_mask = df["frequency"] == "quarterly"
    annual_mask = df["frequency"] == "annual"
    df.loc[quarterly_mask, "revenue_growth_yoy"] = (
        df.loc[quarterly_mask, "total_revenue"].pct_change(periods=4, fill_method=None)
    )
    df.loc[annual_mask, "revenue_growth_yoy"] = (
        df.loc[annual_mask, "total_revenue"].pct_change(periods=1, fill_method=None)
    )

    df.loc[quarterly_mask, "net_income_growth_yoy"] = (
        df.loc[quarterly_mask, "net_income"].pct_change(periods=4, fill_method=None)
    )
    df.loc[annual_mask, "net_income_growth_yoy"] = (
        df.loc[annual_mask, "net_income"].pct_change(periods=1, fill_method=None)
    )

    df.to_csv(PROCESSED_DATA_DIR / "sfix_financial_features_raw.csv", index=False)
    return df


def _compute_price_indicators(price_history: pd.DataFrame) -> pd.DataFrame:
    prices = price_history.copy().sort_values("date")
    prices["return_3m"] = prices["close"].pct_change(63)
    prices["return_6m"] = prices["close"].pct_change(126)
    prices["return_12m"] = prices["close"].pct_change(252)
    prices["rolling_vol_3m"] = prices["close"].pct_change().rolling(63).std() * np.sqrt(252)
    prices["rolling_vol_6m"] = prices["close"].pct_change().rolling(126).std() * np.sqrt(252)
    prices["rolling_max_252"] = prices["close"].rolling(252).max()
    prices["drawdown_12m"] = prices["close"] / prices["rolling_max_252"] - 1
    prices = prices.drop(columns=["rolling_max_252"])
    return prices[
        [
            "date",
            "close",
            "return_3m",
            "return_6m",
            "return_12m",
            "rolling_vol_3m",
            "rolling_vol_6m",
            "drawdown_12m",
        ]
    ]


def merge_price_features(
    financial_df: pd.DataFrame,
    price_history: pd.DataFrame,
    benchmark_histories: Dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Merge rolling price-based indicators with financial + peer features."""

    base_prices = _compute_price_indicators(price_history)

    merged = pd.merge_asof(
        financial_df.sort_values("period_end"),
        base_prices.rename(columns={"date": "price_date"}),
        left_on="period_end",
        right_on="price_date",
        direction="backward",
    )

    merged["market_cap"] = merged["shares_outstanding"] * merged["close"]
    merged["enterprise_value"] = (
        merged["market_cap"] + merged["total_liabilities"] - merged["cash_and_equivalents"]
    )
    merged["ev_to_sales"] = _safe_divide(merged["enterprise_value"], merged["total_revenue"])

    benchmark_histories = benchmark_histories or {}
    benchmark_columns = []
    for label, history in benchmark_histories.items():
        prefix = label.lower()
        indicators = _compute_price_indicators(history)
        indicators = indicators.rename(
            columns={
                col: f"{prefix}_{col}" if col != "date" else f"{prefix}_price_date"
                for col in indicators.columns
            }
        )
        merged = pd.merge_asof(
            merged.sort_values("period_end"),
            indicators,
            left_on="period_end",
            right_on=f"{prefix}_price_date",
            direction="backward",
        )
        merged.drop(columns=[f"{prefix}_price_date"], inplace=True)
        benchmark_columns.extend(
            [
                f"{prefix}_return_3m",
                f"{prefix}_return_6m",
                f"{prefix}_return_12m",
                f"{prefix}_rolling_vol_3m",
                f"{prefix}_rolling_vol_6m",
                f"{prefix}_drawdown_12m",
                f"{prefix}_close",
            ]
        )

        for feature in ["return_3m", "return_6m", "return_12m", "drawdown_12m"]:
            bench_col = f"{prefix}_{feature}"
            if bench_col in merged.columns and feature in merged.columns:
                merged[f"excess_{feature}_{prefix}"] = merged[feature] - merged[bench_col]
        for feature in ["rolling_vol_3m", "rolling_vol_6m"]:
            bench_col = f"{prefix}_{feature}"
            if bench_col in merged.columns and feature in merged.columns:
                merged[f"vol_spread_{feature}_{prefix}"] = merged[feature] - merged[bench_col]
        bench_close = f"{prefix}_close"
        if bench_close in merged.columns and "close" in merged.columns:
            merged[f"relative_price_{prefix}"] = _safe_divide(merged["close"], merged[bench_close])

    if benchmark_columns:
        merged[benchmark_columns] = merged[benchmark_columns].ffill()

    merged.to_csv(PROCESSED_DATA_DIR / "sfix_financial_features_enriched.csv", index=False)
    return merged


