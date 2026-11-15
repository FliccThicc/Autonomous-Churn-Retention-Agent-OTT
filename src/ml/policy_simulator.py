"""Offline 90-day ROI simulation for Indian OTT retention policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "ott_subscribers_2026.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "policy_benchmark.json"
SEED = 42
HORIZON_DAYS = 90
CONSOLE = Console()

ARCHETYPES = [
    "Paid_Tier_Ad_Fatigue",
    "IPL_Post_Tournament_Cliff",
    "UPI_AutoPay_Friction",
    "Mobile_Screen_Limit_Friction",
]
AGENT_SAVE_RATES = {
    "Paid_Tier_Ad_Fatigue": 0.34,
    "IPL_Post_Tournament_Cliff": 0.42,
    "UPI_AutoPay_Friction": 0.38,
    "Mobile_Screen_Limit_Friction": 0.36,
}
AGENT_COSTS = {
    "Paid_Tier_Ad_Fatigue": 40.0,
    "IPL_Post_Tournament_Cliff": 0.0,
    "UPI_AutoPay_Friction": 15.0,
    "Mobile_Screen_Limit_Friction": 30.0,
}


def _diagnose(row: dict[str, Any]) -> str:
    if (
        row["mid_roll_drop_rate"] > 0.35
        and row["ad_supported"] == 1
        and row["ad_skip_attempts_7d"] > 0
    ):
        return "Paid_Tier_Ad_Fatigue"
    if (
        row["ipl_cohort_signup"] == 1
        and row["sports_watch_share"] > 0.75
        and row["days_since_last_live_match"] >= 3
    ):
        return "IPL_Post_Tournament_Cliff"
    if (
        row["consecutive_dunning_failures"] >= 1
        and row["mandate_status"] in {"Insufficient_Funds_Bounce", "Revoked_In_UPI_App"}
    ):
        return "UPI_AutoPay_Friction"
    if (
        row["plan_tier"] == "Mobile_79_Monthly"
        and row["concurrent_stream_rejections_30d"] >= 3
    ):
        return "Mobile_Screen_Limit_Friction"
    return "Organic_Low_Engagement"


def _remaining_ltv(row: dict[str, Any]) -> float:
    price = float(row.get("plan_price_inr", row["monthly_price_inr"]))
    cycle = float(row["billing_cycle_days"])
    horizon = max(float(row["days_to_churn"]), 1.0)
    probability = min(max(float(row["churn_probability"]), 0.0), 1.0)
    return price * horizon / cycle * (1.0 - probability)


def _policy_result(
    name: str,
    eligible: np.ndarray,
    churned: np.ndarray,
    value: np.ndarray,
    costs: np.ndarray,
    save_probability: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    saves = eligible & churned & (rng.random(len(churned)) < save_probability)
    spend = float(costs[eligible].sum())
    retained = float((~churned * value).sum() + (saves * value).sum() - spend)
    baseline = float((~churned * value).sum())
    incremental = retained - baseline
    roi = 0.0 if spend == 0 else incremental / spend * 100.0
    return {
        "name": name,
        "retained_inr": retained,
        "retention_roi_pct": roi,
        "spend_inr": spend,
        "saves": int(saves.sum()),
        "eligible": int(eligible.sum()),
        "save_mask": saves,
    }


def run_simulation() -> dict[str, Any]:
    """Run the deterministic holdout simulation and persist dashboard metrics."""
    frame = pl.read_parquet(DATA_PATH)
    target = frame["churned_30d"].to_numpy()
    _, validation_indices = train_test_split(
        np.arange(frame.height),
        test_size=3_000,
        stratify=target,
        random_state=SEED,
    )
    validation = frame[sorted(validation_indices)]
    rows = validation.to_dicts()
    row_count = len(rows)
    churned = np.array([bool(row["churned_30d"]) for row in rows])
    probability = np.array([float(row["churn_probability"]) for row in rows])
    value = np.array(
        [
            float(row["monthly_price_inr"]) * HORIZON_DAYS / float(row["billing_cycle_days"])
            for row in rows
        ]
    )
    archetypes = np.array([_diagnose(row) for row in rows])
    rng = np.random.default_rng(SEED)

    no_intervention = _policy_result(
        "No Intervention (Status Quo)",
        np.zeros(row_count, dtype=bool),
        churned,
        value,
        np.zeros(row_count),
        np.zeros(row_count),
        rng,
    )
    blanket_eligible = probability > 0.50
    blanket = _policy_result(
        "Naive Blanket Discount Baseline",
        blanket_eligible,
        churned,
        value,
        np.where(blanket_eligible, 50.0, 0.0),
        np.full(row_count, 0.12),
        rng,
    )

    agent_eligible = blanket_eligible & np.isin(archetypes, ARCHETYPES)
    agent_costs = np.zeros(row_count)
    agent_saves = np.zeros(row_count)
    for index, row in enumerate(rows):
        archetype = archetypes[index]
        if agent_eligible[index]:
            cap = 0.25 * _remaining_ltv(row)
            agent_costs[index] = min(AGENT_COSTS.get(archetype, 0.0), cap)
            agent_saves[index] = AGENT_SAVE_RATES.get(archetype, 0.0)
    agent = _policy_result(
        "Autonomous LangGraph Retention Agent",
        agent_eligible,
        churned,
        value,
        agent_costs,
        agent_saves,
        rng,
    )

    policies = [no_intervention, blanket, agent]
    breakdown: dict[str, dict[str, float | int]] = {}
    for archetype in ARCHETYPES:
        mask = archetypes == archetype
        at_risk = mask & churned & agent_eligible
        saved = agent["save_mask"] & mask
        margin = float((saved * value).sum() - agent_costs[mask].sum())
        breakdown[archetype] = {
            "subscriber_count": int(mask.sum()),
            "at_risk_count": int(at_risk.sum()),
            "save_rate": round(float(saved.sum()) / max(int(at_risk.sum()), 1), 4),
            "net_margin_inr": round(margin, 2),
        }

    metrics = {
        "holdout_rows": row_count,
        "horizon_days": HORIZON_DAYS,
        "net_revenue_retained_inr_lakhs": {
            result["name"]: round(result["retained_inr"] / 100_000, 4)
            for result in policies
        },
        "retention_roi_pct": {
            result["name"]: round(result["retention_roi_pct"], 2) for result in policies
        },
        "discount_leakage_prevented_inr": round(
            max(blanket["spend_inr"] - agent["spend_inr"], 0.0), 2
        ),
        "archetype_breakdown": breakdown,
    }
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CONSOLE.print(f"Policy benchmark written to {OUTPUT_PATH}")
    CONSOLE.print_json(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    run_simulation()
