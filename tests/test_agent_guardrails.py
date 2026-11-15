"""Focused tests for the autonomous retention graph and financial guardrail."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import graph as graph_module
from src.agent.graph import retention_graph

PERSONAS = json.loads(
    (ROOT / "data" / "processed" / "flagship_personas.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("persona", PERSONAS, ids=lambda item: item["primary_churn_driver"])
def test_flagship_personas_route_and_pass_guardrail(
    persona: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "_ollama_copy", lambda *args: None)
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
    monkeypatch.setattr(graph_module, "_ollama_copy", lambda *args: None)
    result = retention_graph.invoke({"user_profile": PERSONAS[0]})
    assert result["candidate_offer"].copy_source == "Deterministic_Template"
    assert result["candidate_offer"].localized_message


def test_ollama_copy_disables_thinking_and_cleans_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"response": "<think>private reasoning</think>Namaste! Your regional picks are ready."}'

    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(graph_module.urllib.request, "urlopen", fake_urlopen)
    result = retention_graph.invoke({"user_profile": PERSONAS[0]})
    offer = result["candidate_offer"]
    payload = captured["payload"]
    assert offer.copy_source == "Local_Ollama"
    assert offer.localized_message == "Namaste! Your regional picks are ready."
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "num_ctx": 4096,
        "num_predict": 180,
        "temperature": 0.4,
    }
    assert captured["timeout"] == 15.0
    print(f'copy_source = "{offer.copy_source}"')
