"""Deterministic ML, LTV, and catalog retrieval tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.data.catalog_rag import query_indian_ott_catalog

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "data" / "processed" / "models" / "churn_bundle.joblib"


@lru_cache(maxsize=1)
def _load_bundle() -> Any:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle does not exist: {BUNDLE_PATH}")
    return joblib.load(BUNDLE_PATH)


def score_subscriber_ml(user_profile: dict) -> dict:
    """Score one subscriber with the persisted classifier, survival, and SHAP heads."""
    diagnostic = _load_bundle().score_and_explain(user_profile)
    return {
        "churn_probability": float(diagnostic["churn_probability"]),
        "expected_days_to_churn": float(diagnostic["expected_days_to_churn"]),
        "top_risk_drivers": list(diagnostic.get("top_risk_drivers", [])),
        "top_retention_anchors": list(diagnostic.get("top_retention_anchors", [])),
    }


def calculate_ltv_and_cap(
    user_profile: dict, churn_prob: float, expected_days: float
) -> dict:
    """Estimate remaining subscription value and apply the 25% offer cap."""
    price = float(
        user_profile.get("plan_price_inr", user_profile.get("monthly_price_inr", 0))
    )
    cycle_days = float(user_profile.get("billing_cycle_days", 30))
    if price < 0 or cycle_days <= 0:
        raise ValueError("Plan price must be non-negative and billing cycle must be positive")
    probability = min(max(float(churn_prob), 0.0), 1.0)
    horizon = max(float(expected_days), 0.0)
    predicted_ltv = price * (horizon / cycle_days) * (1.0 - probability)
    return {
        "predicted_remaining_ltv_inr": round(predicted_ltv, 2),
        "ltv_cap_inr": round(predicted_ltv * 0.25, 2),
    }


def fetch_rag_show_bridges(preferred_language: str, archetype: str) -> list[dict]:
    """Retrieve policy-aligned catalog bridges for the diagnosed archetype."""
    mood_queries = {
        "IPL_Post_Tournament_Cliff": "high-stakes thriller crime series with adrenaline after cricket",
        "Organic_Low_Engagement": "accessible regional feel-good drama with strong hooks",
        "Paid_Tier_Ad_Fatigue": "premium binge-worthy Indian series",
        "UPI_AutoPay_Friction": "popular Indian series to continue watching",
        "Mobile_Screen_Limit_Friction": "popular family Indian series for shared viewing",
    }
    return query_indian_ott_catalog(
        preferred_language,
        mood_queries.get(archetype, "popular Indian OTT series"),
        top_k=3,
    )
