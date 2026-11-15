"""Generate deterministic Indian OTT subscriber telemetry for the Day 1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from rich.console import Console


SEED: Final[int] = 42
ROW_COUNT: Final[int] = 15_000
SNAPSHOT_DATE: Final[str] = "2026-06-03"
CONSOLE = Console()

PLAN_ECONOMICS: Final[dict[str, dict[str, int | float]]] = {
    "Mobile_79_Monthly": {"price": 79, "cycle": 30, "ad_supported": 1, "screens": 1, "ad_load": 7.5},
    "Super_149_Monthly": {"price": 149, "cycle": 30, "ad_supported": 1, "screens": 2, "ad_load": 6.0},
    "Premium_299_Monthly": {"price": 299, "cycle": 30, "ad_supported": 0, "screens": 4, "ad_load": 0.5},
    "Super_1099_Annual": {"price": 1099, "cycle": 365, "ad_supported": 1, "screens": 2, "ad_load": 5.5},
    "Premium_2199_Annual": {"price": 2199, "cycle": 365, "ad_supported": 0, "screens": 4, "ad_load": 0.5},
}
PLAN_NAMES: Final[list[str]] = list(PLAN_ECONOMICS)
LANGUAGES: Final[list[str]] = ["Hindi", "Telugu", "Tamil", "Malayalam", "Kannada", "Bengali"]
DRIVERS: Final[list[str]] = [
    "Paid_Tier_Ad_Fatigue",
    "IPL_Post_Tournament_Cliff",
    "UPI_AutoPay_Friction",
    "Mobile_Screen_Limit_Friction",
    "None",
]


def _raw_churn_rate(raw_dir: Path) -> float | None:
    """Read an optional Kaggle churn CSV and return its churn prevalence."""
    candidates = sorted(raw_dir.glob("*churn*.csv"))
    if not candidates:
        return None
    frame = pl.read_csv(candidates[0], infer_schema_length=1_000)
    churn_columns = [column for column in frame.columns if "churn" in column.lower()]
    if not churn_columns:
        return None
    values = frame.get_column(churn_columns[0]).cast(pl.String).str.to_lowercase()
    positive = values.is_in(["1", "true", "yes", "churned", "churn"]).mean()
    return float(positive) if positive is not None else None


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def synthesize(raw_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create the subscriber parquet and four flagship persona records."""
    rng = np.random.default_rng(SEED)
    source_rate = _raw_churn_rate(raw_dir)
    if source_rate is not None:
        CONSOLE.log("Loaded optional Kaggle churn signal from raw CSV (rate %.3f)", source_rate)

    driver = rng.choice(DRIVERS, ROW_COUNT, p=[0.16, 0.15, 0.13, 0.12, 0.44])
    plan = rng.choice(PLAN_NAMES, ROW_COUNT, p=[0.28, 0.25, 0.20, 0.14, 0.13])
    plan_index = {name: index for index, name in enumerate(PLAN_NAMES)}
    plan_prices = np.array([PLAN_ECONOMICS[name]["price"] for name in PLAN_NAMES])
    plan_cycles = np.array([PLAN_ECONOMICS[name]["cycle"] for name in PLAN_NAMES])
    plan_ads = np.array([PLAN_ECONOMICS[name]["ad_supported"] for name in PLAN_NAMES])
    plan_screens = np.array([PLAN_ECONOMICS[name]["screens"] for name in PLAN_NAMES])
    plan_loads = np.array([PLAN_ECONOMICS[name]["ad_load"] for name in PLAN_NAMES])
    indices = np.array([plan_index[name] for name in plan])

    fatigue = driver == "Paid_Tier_Ad_Fatigue"
    ipl = driver == "IPL_Post_Tournament_Cliff"
    upi = driver == "UPI_AutoPay_Friction"
    screen = driver == "Mobile_Screen_Limit_Friction"
    ad_supported = plan_ads[indices] == 1
    plan[fatigue & ~ad_supported] = "Super_149_Monthly"
    indices = np.array([plan_index[name] for name in plan])

    ad_supported = plan_ads[indices] == 1
    mid_roll_drop_rate = rng.beta(2.0, 8.0, ROW_COUNT) * ad_supported
    mid_roll_drop_rate[fatigue] = rng.uniform(0.45, 0.85, fatigue.sum())
    ad_skip_attempts = rng.poisson(1.0, ROW_COUNT) * ad_supported
    ad_skip_attempts[fatigue] = rng.integers(8, 26, fatigue.sum())

    ipl_cohort_signup = ipl.astype(np.int8)
    sports_watch_share = rng.beta(2.0, 8.0, ROW_COUNT)
    sports_watch_share[ipl] = rng.uniform(0.80, 0.99, ipl.sum())
    days_since_live_match = rng.integers(0, 31, ROW_COUNT)
    days_since_live_match[ipl] = rng.integers(3, 10, ipl.sum())

    payment_channels = ["UPI_AutoPay", "Sachet_UPI_Prepaid", "Paytm_PhonePe_Wallet", "Credit_Debit_Card"]
    payment_channel = rng.choice(payment_channels, ROW_COUNT, p=[0.55, 0.15, 0.15, 0.15]).astype(object)
    payment_channel[upi] = "UPI_AutoPay"
    mandate_statuses = ["Active", "Insufficient_Funds_Bounce", "Revoked_In_UPI_App"]
    mandate_status = rng.choice(mandate_statuses, ROW_COUNT, p=[0.90, 0.07, 0.03]).astype(object)
    mandate_status[upi] = rng.choice(["Insufficient_Funds_Bounce", "Revoked_In_UPI_App"], upi.sum())
    dunning_failures = rng.poisson(0.05, ROW_COUNT)
    dunning_failures[upi] = rng.integers(1, 4, upi.sum())

    concurrent_rejections = rng.poisson(0.25, ROW_COUNT)
    concurrent_rejections[screen] = rng.integers(3, 9, screen.sum())
    plan[screen] = "Mobile_79_Monthly"
    indices = np.array([plan_index[name] for name in plan])

    bounded_source_rate = None if source_rate is None else np.clip(source_rate, 0.01, 0.99)
    base_logit = -3.2 if bounded_source_rate is None else float(
        np.log(bounded_source_rate / (1 - bounded_source_rate))
    )
    log_odds = np.full(ROW_COUNT, base_logit)
    log_odds += np.where(fatigue, 2.4 + 2.1 * mid_roll_drop_rate + 0.04 * ad_skip_attempts, 0)
    log_odds += np.where(ipl, 2.8 + 1.5 * sports_watch_share + 0.10 * days_since_live_match, 0)
    log_odds += np.where(upi, 2.7 + 0.8 * dunning_failures, 0)
    log_odds += np.where(screen, 2.8 + 0.25 * concurrent_rejections, 0)
    churn_probability = _sigmoid(log_odds)
    churned = (rng.random(ROW_COUNT) < churn_probability).astype(np.int8)
    days_to_churn = np.where(churned == 1, rng.integers(1, 31, ROW_COUNT), 30).astype(np.int8)

    frame = pl.DataFrame(
        {
            "subscriber_id": [f"OTT{number:06d}" for number in range(1, ROW_COUNT + 1)],
            "snapshot_date": SNAPSHOT_DATE,
            "plan_tier": plan,
            "monthly_price_inr": plan_prices[indices],
            "billing_cycle_days": plan_cycles[indices],
            "ad_supported": plan_ads[indices],
            "max_screens": plan_screens[indices],
            "base_ad_load_mins_per_hr": plan_loads[indices],
            "preferred_language": rng.choice(LANGUAGES, ROW_COUNT),
            "mid_roll_drop_rate": np.round(mid_roll_drop_rate, 4),
            "ad_skip_attempts_7d": ad_skip_attempts,
            "ipl_cohort_signup": ipl_cohort_signup,
            "sports_watch_share": np.round(sports_watch_share, 4),
            "days_since_last_live_match": days_since_live_match,
            "payment_channel": payment_channel,
            "mandate_status": mandate_status,
            "consecutive_dunning_failures": dunning_failures,
            "concurrent_stream_rejections_30d": concurrent_rejections,
            "primary_churn_driver": driver,
            "churn_probability": np.round(churn_probability, 4),
            "churned_30d": churned,
            "days_to_churn": days_to_churn,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "ott_subscribers_2026.parquet"
    persona_path = output_dir / "flagship_personas.json"
    frame.write_parquet(parquet_path)
    personas = (
        frame.filter(pl.col("primary_churn_driver") != "None")
        .sort("churn_probability", descending=True)
        .unique("primary_churn_driver", keep="first", maintain_order=True)
        .sort("primary_churn_driver")
        .to_dicts()
    )
    persona_path.write_text(json.dumps(personas, indent=2), encoding="utf-8")
    CONSOLE.log(
        f"Wrote {frame.height} rows to {parquet_path} and {len(personas)} personas to {persona_path}"
    )
    return parquet_path, persona_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    synthesize(root / "data" / "raw", root / "data" / "processed")
