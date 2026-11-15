"""Streamlit recruiter command center for the Indian OTT retention agent."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.agent.graph import retention_graph

ROOT = Path(__file__).resolve().parents[1]
PERSONAS_PATH = ROOT / "data" / "processed" / "flagship_personas.json"
DATA_PATH = ROOT / "data" / "processed" / "ott_subscribers_2026.parquet"
BENCHMARK_PATH = ROOT / "data" / "processed" / "policy_benchmark.json"

st.set_page_config(page_title="OTT Retention Command Center", page_icon="📺", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #eef2ff; }
    .trace { padding: .65rem 1rem; border: 1px solid #293454; border-radius: .6rem;
             background: #121a31; text-align: center; font-weight: 600; }
    .offer { padding: 1.2rem; border-radius: .8rem; background: linear-gradient(135deg,#162447,#213b68);
             border: 1px solid #3c65a5; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_personas() -> list[dict]:
    return json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_benchmark() -> dict:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _plan_options() -> list[str]:
    return [
        "Mobile_79_Monthly",
        "Super_149_Monthly",
        "Premium_299_Monthly",
        "Super_1099_Annual",
        "Premium_2199_Annual",
    ]


PLAN_ECONOMICS = {
    "Mobile_79_Monthly": {"monthly_price_inr": 79, "billing_cycle_days": 30, "ad_supported": 1, "max_screens": 1, "base_ad_load_mins_per_hr": 7.5},
    "Super_149_Monthly": {"monthly_price_inr": 149, "billing_cycle_days": 30, "ad_supported": 1, "max_screens": 2, "base_ad_load_mins_per_hr": 6.0},
    "Premium_299_Monthly": {"monthly_price_inr": 299, "billing_cycle_days": 30, "ad_supported": 0, "max_screens": 4, "base_ad_load_mins_per_hr": 0.5},
    "Super_1099_Annual": {"monthly_price_inr": 1099, "billing_cycle_days": 365, "ad_supported": 1, "max_screens": 2, "base_ad_load_mins_per_hr": 5.5},
    "Premium_2199_Annual": {"monthly_price_inr": 2199, "billing_cycle_days": 365, "ad_supported": 0, "max_screens": 4, "base_ad_load_mins_per_hr": 0.5},
}


st.title("📺 Indian OTT Retention Command Center")
st.caption("JioHotstar / SonyLIV autonomous engagement intelligence")
personas = load_personas()
with st.sidebar:
    st.header("Subscriber Controls")
    labels = [persona["primary_churn_driver"] for persona in personas]
    selected = st.selectbox("Flagship Persona", labels)
    selected_profile = next(persona for persona in personas if persona["primary_churn_driver"] == selected)
    if st.button("Sample Random High-Risk Subscriber"):
        selected_profile = (
            pl.read_parquet(DATA_PATH)
            .filter(pl.col("churn_probability") > 0.5)
            .sample(1, seed=42)
            .to_dicts()[0]
        )
    profile = dict(selected_profile)
    profile["mid_roll_drop_rate"] = st.slider(
        "Mid-roll drop rate", 0.0, 1.0, float(profile["mid_roll_drop_rate"]), 0.01
    )
    profile["sports_watch_share"] = st.slider(
        "Sports watch share", 0.0, 1.0, float(profile["sports_watch_share"]), 0.01
    )
    profile["consecutive_dunning_failures"] = st.slider(
        "Consecutive dunning failures", 0, 5, int(profile["consecutive_dunning_failures"])
    )
    profile["plan_tier"] = st.selectbox("Plan tier", _plan_options(), index=_plan_options().index(profile["plan_tier"]))
    profile.update(PLAN_ECONOMICS[profile["plan_tier"]])

tab_trace, tab_roi, tab_audit = st.tabs(
    ["Live Agent Decision Trace", "Executive Policy ROI Simulator", "Architecture & Guardrail Audit"]
)

with tab_trace:
    result = retention_graph.invoke({"user_profile": profile})
    diagnosis = result["diagnosis"]
    offer = result["candidate_offer"]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Subscriber Metrics")
        st.metric("Churn Probability", f"{diagnosis.churn_probability:.1%}")
        st.metric("Survival Urgency", f"{diagnosis.expected_days_to_churn:.1f} days")
        st.write(f"**Archetype:** `{diagnosis.archetype}`")
        drivers = result["ml_diagnostic"].get("top_risk_drivers", [])
        anchors = result["ml_diagnostic"].get("top_retention_anchors", [])
        chart_rows = [
            (item["feature"], item["shap_value"], "Risk Driver") for item in drivers
        ] + [(item["feature"], item["shap_value"], "Retention Anchor") for item in anchors]
        chart = go.Figure(
            go.Bar(
                x=[row[1] for row in chart_rows],
                y=[row[0] for row in chart_rows],
                orientation="h",
                marker_color=["#f97316" if row[2] == "Risk Driver" else "#22c55e" for row in chart_rows],
                text=[row[2] for row in chart_rows],
            )
        )
        chart.update_layout(title="TreeSHAP Drivers vs Anchors", template="plotly_dark", height=300)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("LangGraph Execution Trace")
        st.markdown(
            '<div class="trace">Diagnose → Propose → LTV Guardrail → Copy Generator</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Copy tier:** `{offer.copy_source}`")
        st.write(f"Offer cost: ₹{offer.offer_cost_inr:.2f}")
        st.write(
            f"LTV: ₹{offer.predicted_remaining_ltv_inr:.2f} · "
            f"Guardrail cap: ₹{offer.ltv_cap_inr:.2f}"
        )
        st.markdown(
            f'<div class="offer"><b>{offer.title}</b><br><br>{offer.localized_message}</div>',
            unsafe_allow_html=True,
        )
        if offer.recommended_shows:
            st.write("**RAG show bridges:**", ", ".join(show["series_title"] for show in offer.recommended_shows))
        st.write("**Guardrail audit:**")
        for log in result.get("guardrail_logs", []):
            st.caption(log)

with tab_roi:
    benchmark = load_benchmark()
    st.subheader("90-Day Policy Benchmark (₹ Lakhs)")
    revenue = benchmark["net_revenue_retained_inr_lakhs"]
    roi = benchmark["retention_roi_pct"]
    policies = list(revenue)
    figure = go.Figure()
    figure.add_bar(name="Net Revenue Retained (₹ Lakhs)", x=policies, y=[revenue[p] for p in policies])
    figure.add_bar(name="Retention ROI (%)", x=policies, y=[roi[p] for p in policies])
    figure.update_layout(barmode="group", template="plotly_dark", height=450)
    st.plotly_chart(figure, use_container_width=True)
    st.metric("Discount Leakage Prevented", f"₹{benchmark['discount_leakage_prevented_inr']:,.0f}")
    st.json(benchmark["archetype_breakdown"])

with tab_audit:
    st.subheader("Architecture & Guardrail Audit")
    cols = st.columns(3)
    cols[0].metric("ROC-AUC", "0.897")
    cols[1].metric("PR-AUC", "0.830")
    cols[2].metric("Brier Score", "0.111")
    st.info("RAG catalog: 480 series indexed with regional language boosting.")
    st.success("Hard policy: offer_cost_inr ≤ 0.25 × predicted_remaining_ltv_inr")
    st.write("Live guardrail logs")
    for log in result.get("guardrail_logs", []):
        st.code(log)
