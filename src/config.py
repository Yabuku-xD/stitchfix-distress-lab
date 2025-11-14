"""Config odds and ends for the Stitch Fix distress sandbox (yes, it's messy)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# Core metadata
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for path in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR):
    path.mkdir(parents=True, exist_ok=True)

TICKER = "SFIX"
CIK = "0001576942"
SEC_USER_AGENT = "StitchFixDistressResearch/1.0 (contact@example.com)"
PRICE_HISTORY_START = "2017-01-01"

# Label + modeling controls
# (Defaults are conservative so future filings don't torch the model unexpectedly.)
TARGET_LOOKAHEAD_PERIODS = 1  # predict distress one quarter ahead by default
PRIMARY_TARGET = "target_altman_distress_flag"
ALT_Z_THRESHOLD = 1.8
CURRENT_RATIO_THRESHOLD = 1.0
CASH_RATIO_THRESHOLD = 0.5

WALK_FORWARD_MIN_TRAIN = 8    # number of quarters in initial training window
WALK_FORWARD_TEST_WINDOW = 4  # quarters evaluated per walk-forward fold
WALK_FORWARD_STEP = 1         # slide window each quarter

# Market & peer benchmarking
# TODO: rotate peers occasionally if SFIX pivots again (Rent the Runway comeback?).
BENCHMARK_TICKERS = {
    "spy": "SPY",   # S&P 500 ETF as market proxy
    "xrt": "XRT",   # SPDR Retail ETF for sector exposure
    "etsy": "ETSY", # Tech-enabled retailer peer
}


# -----------------------------------------------------------------------------
# Mapping between canonical metrics and GAAP tags in the SEC company facts feed
# -----------------------------------------------------------------------------

METRIC_TAGS = {
    "total_revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_current_assets": [
        "AssetsCurrent",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
    ],
    "inventory": [
        "InventoryNet",
        "FinishedGoodsAndWorkInProcess",
    ],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
    ],
    "total_current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capital_expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "free_cash_flow": [
        "FreeCashFlow",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "research_and_development": [
        "ResearchAndDevelopmentExpense",
    ],
    "selling_general_admin": [
        "SellingGeneralAndAdministrativeExpense",
    ],
    "interest_expense": [
        "InterestExpense",
    ],
}


# -----------------------------------------------------------------------------
# Known distress events curated from press releases and news coverage.
# Dates are inclusive ISO strings and meant to approximate the fiscal period
# impacted by the event.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DistressEvent:
    start: str
    end: str
    label: str


DISTRESS_EVENTS: tuple[DistressEvent, ...] = (
    DistressEvent(
        start="2022-06-01",
        end="2022-09-30",
        label="June 2022 corporate layoffs & guidance cut",
    ),
    DistressEvent(
        start="2023-01-01",
        end="2023-04-30",
        label="Jan 2023 CEO transition and workforce reduction",
    ),
    DistressEvent(
        start="2024-05-01",
        end="2024-08-31",
        label="FY24 turnaround plan & continued losses",
    ),
)


