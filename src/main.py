"""Main orchestration script. Think of it as my personal Stitch Fix lab notebook."""

from __future__ import annotations

from pprint import pprint

import pandas as pd

from . import data_fetch
from . import data_processing
from . import eda
from . import features
from . import labeling
from . import modeling
from .config import (
    ALT_Z_THRESHOLD,
    BENCHMARK_TICKERS,
    CASH_RATIO_THRESHOLD,
    CURRENT_RATIO_THRESHOLD,
    PRIMARY_TARGET,
    PROCESSED_DATA_DIR,
    TARGET_LOOKAHEAD_PERIODS,
    WALK_FORWARD_MIN_TRAIN,
    WALK_FORWARD_TEST_WINDOW,
    WALK_FORWARD_STEP,
)


BASE_FEATURE_COLUMNS = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "debt_to_equity",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "revenue_growth_yoy",
    "ocf_margin",
    "fcf_margin",
    "liquidity_runway_quarters",
    "drawdown_12m",
    "return_6m",
    "asset_turnover",
    "altman_z_score",
]

BENCHMARK_FEATURE_TEMPLATES = [
    "excess_return_3m_{label}",
    "excess_return_6m_{label}",
    "excess_return_12m_{label}",
    "excess_drawdown_12m_{label}",
    "vol_spread_rolling_vol_3m_{label}",
    "vol_spread_rolling_vol_6m_{label}",
    "relative_price_{label}",
]

BENCHMARK_FEATURE_COLUMNS = [
    template.format(label=label)
    for label in BENCHMARK_TICKERS
    for template in BENCHMARK_FEATURE_TEMPLATES
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + BENCHMARK_FEATURE_COLUMNS


def build_dataset(lookahead_periods: int = TARGET_LOOKAHEAD_PERIODS) -> pd.DataFrame:
    company_facts = data_fetch.download_company_facts()
    panel = data_processing.build_financial_panel(company_facts)
    _, quarterly = data_processing.split_frequency_panels(panel)

    ratio_df = features.compute_financial_ratios(quarterly)
    price_history = data_fetch.download_price_history()

    benchmark_histories = {}
    for label, ticker in BENCHMARK_TICKERS.items():
        try:
            benchmark_histories[label] = data_fetch.download_price_history(ticker)
        except ValueError as exc:
            print(f"[WARN] Unable to fetch benchmark {ticker}: {exc}")
            # TODO: maybe swap in cached ETF data if this keeps happening mid-run.

    enriched = features.merge_price_features(ratio_df, price_history, benchmark_histories)
    labeled = labeling.label_distress_periods(enriched)

    final_dataset = labeled[labeled["frequency"] == "quarterly"].copy()
    final_dataset = final_dataset.sort_values("period_end").reset_index(drop=True)

    final_dataset["altman_distress_flag"] = (
        final_dataset["altman_z_score"] <= ALT_Z_THRESHOLD
    ).astype(int)
    final_dataset["liquidity_crunch_flag"] = (
        (final_dataset["current_ratio"] < CURRENT_RATIO_THRESHOLD)
        | (final_dataset["cash_ratio"] < CASH_RATIO_THRESHOLD)
    ).astype(int)

    base_flag_columns = {
        "distress_flag": final_dataset["distress_flag"].astype(int),
        "altman_distress_flag": final_dataset["altman_distress_flag"].astype(int),
        "liquidity_crunch_flag": final_dataset["liquidity_crunch_flag"].astype(int),
    }

    for name, series in base_flag_columns.items():
        final_dataset[name] = series
        target_col = f"target_{name}"
        if lookahead_periods > 0:
            final_dataset[target_col] = series.shift(-lookahead_periods)
        else:
            final_dataset[target_col] = series

    if PRIMARY_TARGET not in final_dataset.columns:
        raise ValueError(
            f"Primary target column '{PRIMARY_TARGET}' not found in dataset. "
            "Check TARGET_LOOKAHEAD_PERIODS or PRIMARY_TARGET settings."
        )

    final_dataset = final_dataset.dropna(subset=[PRIMARY_TARGET]).copy()
    final_dataset[PRIMARY_TARGET] = final_dataset[PRIMARY_TARGET].astype(int)

    target_columns = [col for col in final_dataset.columns if col.startswith("target_")]
    for col in target_columns:
        if col == PRIMARY_TARGET:
            continue
        final_dataset[col] = final_dataset[col].astype("Int64")

    final_dataset.to_csv(PROCESSED_DATA_DIR / "sfix_model_ready_dataset.csv", index=False)
    return final_dataset


def run_visualizations(dataset: pd.DataFrame) -> None:
    eda.plot_ratio_trends(
        dataset,
        ["current_ratio", "quick_ratio", "cash_ratio"],
        "Liquidity Ratios Over Time",
        "Ratio",
        "liquidity_ratios.png",
    )
    eda.plot_ratio_trends(
        dataset,
        ["gross_margin", "operating_margin", "net_margin"],
        "Margin Trends",
        "Margin",
        "margin_trends.png",
    )
    eda.plot_ratio_trends(
        dataset,
        ["debt_to_equity", "debt_to_assets", "lt_debt_to_equity"],
        "Leverage Ratios",
        "Ratio",
        "leverage_ratios.png",
    )
    eda.plot_correlation_heatmap(
        dataset,
        BASE_FEATURE_COLUMNS,
        "feature_correlation_core.png",
        title="Core Financial Feature Correlation",
    )
    eda.plot_correlation_heatmap(
        dataset,
        BENCHMARK_FEATURE_COLUMNS,
        "feature_correlation_benchmarks.png",
        title="Benchmark-Relative Feature Correlation",
    )


def main() -> None:
    dataset = build_dataset()
    run_visualizations(dataset)
    results = modeling.run_model_suite(
        dataset,
        FEATURE_COLUMNS,
        target_col=PRIMARY_TARGET,
        min_train=WALK_FORWARD_MIN_TRAIN,
        test_window=WALK_FORWARD_TEST_WINDOW,
        step=WALK_FORWARD_STEP,
    )

    print("Model training complete. Key metrics:")
    for name, result in results.items():
        eda.plot_calibration_curve(name, result.y_true, result.y_prob)
        print(f"\n{name}:")
        pprint(result.metrics)
        print(result.classification_report)


if __name__ == "__main__":
    main()


