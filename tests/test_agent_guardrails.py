"""Focused tests for the autonomous retention graph and financial guardrail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.graph import retention_graph

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = json.loads(
    (ROOT / "data" / "processed" / "flagship_personas.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("persona", PERSONAS, ids=lambda item: item["primary_churn_driver"])
def test_flagship_personas_route_and_pass_guardrail(persona: dict) -> None:
    result = retention_graph.invoke({"user_profile": persona})
    offer = result["candidate_offer"]
    assert offer.guardrail_passed
    assert offer.offer_cost_inr <= 0.25 * offer.predicted_remaining_ltv_inr + 0.01
    assert offer.copy_source == "Deterministic_Template"
    assert result["diagnosis"].archetype == persona["primary_churn_driver"]


def test_low_ltv_adversarial_profile_downgrades_without_overspend() -> None:
    profile = dict(PERSONAS[2])
    profile.update(
        {
            "monthly_price_inr": 79,
            "billing_cycle_days": 30,
            "mid_roll_drop_rate": 0.90,
            "ad_skip_attempts_7d": 20,
        }
    )
    result = retention_graph.invoke({"user_profile": profile})
    offer = result["candidate_offer"]
    assert result["diagnosis"].archetype == "Paid_Tier_Ad_Fatigue"
    assert offer.revision_count >= 1
    assert offer.offer_cost_inr == 0
    assert offer.offer_cost_inr <= offer.ltv_cap_inr
    assert "show" not in offer.title.lower()


def test_offline_copy_fallback_is_local_and_nonempty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = retention_graph.invoke({"user_profile": PERSONAS[0]})
    assert result["candidate_offer"].copy_source == "Deterministic_Template"
    assert result["candidate_offer"].localized_message
