"""Typed contracts for the autonomous retention agent."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Archetype = Literal[
    "Paid_Tier_Ad_Fatigue",
    "IPL_Post_Tournament_Cliff",
    "UPI_AutoPay_Friction",
    "Mobile_Screen_Limit_Friction",
    "Organic_Low_Engagement",
]
OfferChannel = Literal[
    "WhatsApp_DeepLink",
    "InApp_Banner",
    "Push_Notification",
    "UPI_Smart_Dunning",
]
CopySource = Literal["Cloud_LLM", "Local_Ollama", "Deterministic_Template"]


class ChurnDiagnosis(BaseModel):
    """Validated model diagnosis and root-cause explanation."""

    model_config = ConfigDict(extra="forbid")

    archetype: Archetype
    churn_probability: float = Field(ge=0, le=1)
    expected_days_to_churn: float = Field(ge=0)
    top_risk_drivers: list[dict]
    confidence_rationale: str


class RetentionOffer(BaseModel):
    """Validated offer, economics, guardrail result, and outbound copy."""

    model_config = ConfigDict(extra="forbid")

    offer_code: str
    title: str
    channel: OfferChannel
    offer_cost_inr: float = Field(ge=0)
    predicted_remaining_ltv_inr: float = Field(ge=0)
    ltv_cap_inr: float = Field(ge=0)
    guardrail_passed: bool
    revision_count: int = Field(ge=0)
    recommended_shows: list[dict]
    copy_source: CopySource
    localized_message: str


class AgentState(TypedDict, total=False):
    """LangGraph state shared by all retention nodes."""

    user_profile: dict
    ml_diagnostic: dict
    diagnosis: ChurnDiagnosis
    candidate_offer: RetentionOffer
    guardrail_logs: list[str]
    revision_count: int
    final_decision: dict
