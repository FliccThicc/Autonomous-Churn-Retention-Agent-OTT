"""LangGraph state machine for autonomous, guardrailed retention decisions."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.schemas import AgentState, ChurnDiagnosis, RetentionOffer
from src.agent.tools import (
    calculate_ltv_and_cap,
    fetch_rag_show_bridges,
    score_subscriber_ml,
)

ROOT = Path(__file__).resolve().parents[2]


def _logs(state: AgentState) -> list[str]:
    return list(state.get("guardrail_logs", []))


def _load_dotenv() -> None:
    """Load simple KEY=VALUE entries from the project-local .env if present."""
    dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def diagnose_node(state: AgentState) -> dict:
    """Combine model drivers and telemetry into a canonical churn archetype."""
    profile = state["user_profile"]
    diagnostic = score_subscriber_ml(profile)
    if (
        float(profile.get("mid_roll_drop_rate", 0)) > 0.35
        and int(profile.get("ad_supported", 0)) == 1
        and int(profile.get("ad_skip_attempts_7d", 0)) > 0
    ):
        archetype = "Paid_Tier_Ad_Fatigue"
    elif (
        int(profile.get("ipl_cohort_signup", 0)) == 1
        and float(profile.get("sports_watch_share", 0)) > 0.75
        and float(profile.get("days_since_last_live_match", 0)) >= 3
    ):
        archetype = "IPL_Post_Tournament_Cliff"
    elif (
        int(profile.get("consecutive_dunning_failures", 0)) >= 1
        and profile.get("mandate_status")
        in {"Insufficient_Funds_Bounce", "Revoked_In_UPI_App"}
    ):
        archetype = "UPI_AutoPay_Friction"
    elif (
        profile.get("plan_tier") == "Mobile_79_Monthly"
        and int(profile.get("concurrent_stream_rejections_30d", 0)) >= 3
    ):
        archetype = "Mobile_Screen_Limit_Friction"
    else:
        archetype = "Organic_Low_Engagement"
    diagnosis = ChurnDiagnosis(
        archetype=archetype,
        churn_probability=diagnostic["churn_probability"],
        expected_days_to_churn=diagnostic["expected_days_to_churn"],
        top_risk_drivers=diagnostic["top_risk_drivers"],
        confidence_rationale=(
            f"Telemetry matched {archetype}; model probability is "
            f"{diagnostic['churn_probability']:.3f} with "
            f"{len(diagnostic['top_risk_drivers'])} positive SHAP drivers."
        ),
    )
    return {"ml_diagnostic": diagnostic, "diagnosis": diagnosis}


def _offer_policy(archetype: str, revision_count: int) -> dict[str, Any]:
    if revision_count > 0:
        if archetype == "Paid_Tier_Ad_Fatigue":
            return {
                "offer_code": "AD_FATIGUE_SERVICE_RECOVERY",
                "title": "Ad interruption service recovery",
                "channel": "InApp_Banner",
                "cost": 0.0,
            }
        return {
            "offer_code": "CONTENT_BRIDGE_NO_COST",
            "title": "Curated regional content bridge",
            "channel": "InApp_Banner",
            "cost": 0.0,
        }
    return {
        "Paid_Tier_Ad_Fatigue": {
            "offer_code": "AD_FREE_BOOST_7D",
            "title": "7-day Ad-Free Boost",
            "channel": "InApp_Banner",
            "cost": 40.0,
        },
        "IPL_Post_Tournament_Cliff": {
            "offer_code": "IPL_THRILLER_BRIDGE",
            "title": "Three-show regional thriller bridge",
            "channel": "WhatsApp_DeepLink",
            "cost": 0.0,
        },
        "UPI_AutoPay_Friction": {
            "offer_code": "UPI_SALARY_RETRY_SACHET",
            "title": "Salary-date AutoPay retry plus ₹15 sachet pass",
            "channel": "UPI_Smart_Dunning",
            "cost": 15.0,
        },
        "Mobile_Screen_Limit_Friction": {
            "offer_code": "MULTI_SCREEN_UPGRADE_CREDIT",
            "title": "₹30 multi-screen upgrade credit",
            "channel": "WhatsApp_DeepLink",
            "cost": 30.0,
        },
        "Organic_Low_Engagement": {
            "offer_code": "ORGANIC_CONTENT_BRIDGE",
            "title": "Personalized regional content bridge",
            "channel": "Push_Notification",
            "cost": 0.0,
        },
    }[archetype]


def propose_offer_node(state: AgentState) -> dict:
    """Select the policy action and attach only relevant catalog recommendations."""
    diagnosis = state["diagnosis"]
    profile = state["user_profile"]
    revision_count = int(state.get("revision_count", 0))
    policy = _offer_policy(diagnosis.archetype, revision_count)
    shows = (
        fetch_rag_show_bridges(profile.get("preferred_language", "Hindi"), diagnosis.archetype)
        if diagnosis.archetype in {"IPL_Post_Tournament_Cliff", "Organic_Low_Engagement"}
        else []
    )
    ltv = calculate_ltv_and_cap(
        profile, diagnosis.churn_probability, diagnosis.expected_days_to_churn
    )
    offer = RetentionOffer(
        offer_code=policy["offer_code"],
        title=policy["title"],
        channel=policy["channel"],
        offer_cost_inr=policy["cost"],
        predicted_remaining_ltv_inr=ltv["predicted_remaining_ltv_inr"],
        ltv_cap_inr=ltv["ltv_cap_inr"],
        guardrail_passed=False,
        revision_count=revision_count,
        recommended_shows=shows,
        copy_source="Deterministic_Template",
        localized_message="",
    )
    return {"candidate_offer": offer}


def ltv_guardrail_node(state: AgentState) -> dict:
    """Reject excessive or ad-fatigue-inappropriate offers before copy generation."""
    offer = state["candidate_offer"]
    diagnosis = state["diagnosis"]
    logs = _logs(state)
    if (
        diagnosis.archetype == "Paid_Tier_Ad_Fatigue"
        and offer.offer_cost_inr == 0
        and offer.recommended_shows
    ):
        logs.append("Rejected generic show-only spam for ad-fatigue diagnosis.")
        return {
            "candidate_offer": offer.model_copy(
                update={"guardrail_passed": False, "revision_count": offer.revision_count + 1}
            ),
            "revision_count": offer.revision_count + 1,
            "guardrail_logs": logs,
        }
    passed = offer.offer_cost_inr <= offer.ltv_cap_inr
    logs.append(
        f"LTV cap check: cost ₹{offer.offer_cost_inr:.2f} <= cap "
        f"₹{offer.ltv_cap_inr:.2f}: {'PASS' if passed else 'FAIL'}."
    )
    if not passed:
        logs.append("Offer downgraded through conditional loop.")
        next_revision = offer.revision_count + 1
        return {
            "candidate_offer": offer.model_copy(
                update={"guardrail_passed": False, "revision_count": next_revision}
            ),
            "revision_count": next_revision,
            "guardrail_logs": logs,
        }
    return {
        "candidate_offer": offer.model_copy(update={"guardrail_passed": True}),
        "guardrail_logs": logs,
    }


def _template_copy(profile: dict, offer: RetentionOffer, diagnosis: ChurnDiagnosis) -> str:
    name = profile.get("name", profile.get("subscriber_id", "there"))
    city = profile.get("city", "India")
    language = profile.get("preferred_language", "Hindi")
    if diagnosis.archetype == "UPI_AutoPay_Friction":
        return (
            f"Hi {name} in {city}, your {language} entertainment is safe. "
            "We will retry AutoPay around your salary date, or you can switch to "
            f"the ₹15 sachet pass using {offer.offer_code}."
        )
    if diagnosis.archetype == "Paid_Tier_Ad_Fatigue":
        return (
            f"Hi {name} in {city}, we heard your feedback on ad interruptions. "
            f"Your {offer.title} is ready in {language} via {offer.offer_code}."
        )
    if offer.recommended_shows:
        titles = ", ".join(show["series_title"] for show in offer.recommended_shows[:3])
        return (
            f"Hi {name} in {city}, keep the adrenaline going in {language} with {titles}. "
            f"Open your picks through {offer.offer_code}."
        )
    return (
        f"Hi {name} in {city}, we picked a better {language} viewing path for you. "
        f"Open {offer.offer_code} to continue watching."
    )


def _cloud_copy(profile: dict, offer: RetentionOffer, diagnosis: ChurnDiagnosis) -> str | None:
    """Use a configured REST provider without adding an SDK dependency."""
    _load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    prompt = (
        f"Write one concise {profile.get('preferred_language', 'Hindi')} OTT retention message "
        f"for {profile.get('name', profile.get('subscriber_id', 'the subscriber'))} in "
        f"{profile.get('city', 'India')}. Offer code: {offer.offer_code}. "
        f"Shows: {[show.get('series_title') for show in offer.recommended_shows]}. "
        f"Diagnosis: {diagnosis.archetype}. Do not invent discounts."
    )
    if os.getenv("GEMINI_API_KEY"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=4.0) as response:
            data = json.loads(response.read().decode())
        if os.getenv("GEMINI_API_KEY"):
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return data["choices"][0]["message"]["content"].strip()
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


def _ollama_copy(profile: dict, offer: RetentionOffer, diagnosis: ChurnDiagnosis) -> str | None:
    titles = ", ".join(
        str(show.get("series_title", "")).strip()
        for show in offer.recommended_shows
        if show.get("series_title")
    ) or "our latest regional favourites"
    prompt = (
        f"Write a friendly, 2-sentence customer retention message under 45 words in "
        f"{profile.get('preferred_language', 'Hindi')} for {profile.get('name', profile.get('subscriber_id', 'the subscriber'))}. "
        f"Offer: {offer.title}. Recommended TV series on our streaming catalog "
        f"(official show titles): {titles}. These are fictional/catalog TV series "
        "streaming on JioHotstar/SonyLIV. Write a friendly, 2-sentence customer "
        "retention message under 45 words inviting the subscriber to watch these "
        "shows. Do not invent discounts or discuss real-world events."
    )
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": 180,
            "temperature": 0.4,
        },
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            result = json.loads(response.read().decode())
        message = result.get("response")
        if not isinstance(message, str):
            return None
        message = re.sub(r"<think>.*?</think>", "", message, flags=re.IGNORECASE | re.DOTALL)
        message = re.sub(r"</?think>", "", message, flags=re.IGNORECASE).strip()
        invalid_markers = ("i cannot provide", "assassination", "sensitive historical")
        if len(message) > 350 or any(marker in message.lower() for marker in invalid_markers):
            return None
        return message or None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def generate_copy_node(state: AgentState) -> dict:
    """Run Cloud LLM -> Ollama -> deterministic template waterfall."""
    offer = state["candidate_offer"]
    diagnosis = state["diagnosis"]
    profile = state["user_profile"]
    copy = _cloud_copy(profile, offer, diagnosis)
    source = "Cloud_LLM"
    if copy is None:
        copy = _ollama_copy(profile, offer, diagnosis)
        source = "Local_Ollama"
    if copy is None:
        copy = _template_copy(profile, offer, diagnosis)
        source = "Deterministic_Template"
    final_offer = offer.model_copy(update={"copy_source": source, "localized_message": copy})
    return {
        "candidate_offer": final_offer,
        "final_decision": {
            "diagnosis": diagnosis.model_dump(),
            "offer": final_offer.model_dump(),
        },
    }


def _guardrail_route(state: AgentState) -> str:
    return "generate_copy" if state["candidate_offer"].guardrail_passed else "propose_offer"


def build_retention_graph():
    """Compile the autonomous retention graph with its conditional downgrade loop."""
    workflow = StateGraph(AgentState)
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("propose_offer", propose_offer_node)
    workflow.add_node("ltv_guardrail", ltv_guardrail_node)
    workflow.add_node("generate_copy", generate_copy_node)
    workflow.add_edge(START, "diagnose")
    workflow.add_edge("diagnose", "propose_offer")
    workflow.add_edge("propose_offer", "ltv_guardrail")
    workflow.add_conditional_edges(
        "ltv_guardrail",
        _guardrail_route,
        {"propose_offer": "propose_offer", "generate_copy": "generate_copy"},
    )
    workflow.add_edge("generate_copy", END)
    return workflow.compile()


retention_graph = build_retention_graph()
