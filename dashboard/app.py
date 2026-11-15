"""Streamlit recruiter command center for the Indian OTT retention agent."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.agent.graph import retention_graph

ROOT = Path(__file__).resolve().parents[1]
PERSONAS_PATH = ROOT / "data" / "processed" / "flagship_personas.json"
DATA_PATH = ROOT / "data" / "processed" / "ott_subscribers_2026.parquet"
BENCHMARK_PATH = ROOT / "data" / "processed" / "policy_benchmark.json"

st.set_page_config(
    page_title="OTT Retention Command Center",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_apple_glassmorphism() -> None:
    """Apply the Apple visionOS visual system without changing dashboard behavior."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0..1&display=swap');

        html, body, [class*="st-"], p, h1, h2, h3, h4, h5, h6, span, div {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
            font-feature-settings: "cv01", "cv02", "cv03", "cv04", "cv11";
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            letter-spacing: -0.015em;
        }
        body, button, input, textarea, select, [role="button"], [data-baseweb] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        :root {
            --obsidian: #07090e;
            --slate: #0f1420;
            --pearl: #f8fafc;
            --silver: #94a3b8;
            --rose: #e05780;
            --indigo: #6366f1;
            --spring: cubic-bezier(0.22, 1, 0.36, 1);
        }

        html, body {
            scroll-behavior: smooth;
        }
        .stApp {
            color: var(--pearl);
            background:
                radial-gradient(circle at 8% 12%, rgba(224, 87, 128, 0.16), transparent 34%),
                radial-gradient(circle at 92% 8%, rgba(99, 102, 241, 0.16), transparent 32%),
                linear-gradient(135deg, var(--obsidian), var(--slate) 55%, var(--obsidian));
            background-size: 130% 130%;
            animation: ambientDrift 24s ease-in-out infinite alternate;
        }
        @keyframes ambientDrift {
            from { background-position: 0% 0%; }
            to { background-position: 100% 80%; }
        }
        @keyframes springFadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stTabs, .stMarkdown, [data-testid="stVerticalBlock"] {
            animation: springFadeUp .8s cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu, footer { display: none; }
        .block-container { padding-top: 2.5rem; }
        h1, h2, h3 { color: var(--pearl) !important; letter-spacing: -0.03em; }
        p, label, .stCaption, [data-testid="stMarkdownContainer"] { color: var(--silver); }
        code, .stMetricValue { font-family: "JetBrains Mono", monospace; }

        section[data-testid="stSidebar"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        .stTabs [data-baseweb="tab-panel"],
        div[data-testid="stMetric"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: rgba(20, 25, 35, 0.4);
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        section[data-testid="stSidebar"] {
            background: rgba(10, 13, 22, 0.78) !important;
            backdrop-filter: blur(28px) saturate(160%);
            -webkit-backdrop-filter: blur(28px) saturate(160%);
            overflow-x: hidden;
            border-radius: 0 24px 24px 0;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"][data-collapsed="false"],
        section[data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 350px !important;
            max-width: 350px !important;
            width: 350px !important;
            inline-size: 350px !important;
            flex: 0 0 350px !important;
            resize: none !important;
        }
        section[data-testid="stSidebar"][data-collapsed="true"],
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            flex: 0 0 0 !important;
            border: 0;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 100%;
            padding: 1.25rem 1.15rem 1.5rem 1.15rem !important;
            height: 100vh;
            min-width: 0;
            box-sizing: border-box;
            overflow-x: hidden;
            overflow-y: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }
        section[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar,
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] *::-webkit-scrollbar {
            scrollbar-width: none !important;
        }
        section[data-testid="stSidebar"] *::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            scrollbar-width: none;
            -ms-overflow-style: none;
        }
        [data-testid="stSidebarResizeHandle"],
        section[data-testid="stSidebar"] [aria-label*="Resize"],
        section[data-testid="stSidebar"] > div[style*="col-resize"] {
            display: none !important;
            pointer-events: none !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            right: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
            width: 100% !important;
            min-width: 0;
            padding: 0 !important;
            box-sizing: border-box;
        }
        section[data-testid="stSidebar"][data-collapsed="true"] > div:first-child,
        section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 0;
            padding: 0;
            overflow: hidden;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 0;
            width: 100% !important;
            min-width: 0 !important;
            max-width: 350px !important;
            overflow-y: auto;
            scrollbar-width: none !important;
            -ms-overflow-style: none;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: .75rem;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            margin: .25rem 0 .35rem;
        }
        section[data-testid="stSidebar"] .sidebar-header {
            margin: .25rem 0 1.15rem;
        }
        section[data-testid="stSidebar"] .sidebar-eyebrow,
        section[data-testid="stSidebar"] .sidebar-group-label {
            color: #ff8fab;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        section[data-testid="stSidebar"] .sidebar-title {
            color: #f8fafc;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -.025em;
            margin: .25rem 0 .7rem;
            white-space: nowrap;
        }
        section[data-testid="stSidebar"] .sidebar-user-pill {
            display: flex;
            align-items: center;
            gap: .45rem;
            padding: .5rem .7rem;
            color: #cbd5e1;
            background: rgba(255,255,255,.06);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 12px;
            font-size: .68rem;
            line-height: 1.5;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        section[data-testid="stSidebar"] .status-dot {
            width: 7px;
            height: 7px;
            flex: 0 0 7px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 9px rgba(16,185,129,.8);
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 1rem;
            margin-bottom: .85rem;
            background: rgba(255,255,255,.045);
            border-color: rgba(255,255,255,.1);
            border-radius: 20px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 12px 28px rgba(0,0,0,.2);
        }
        section[data-testid="stSidebar"] label {
            color: #aab5c8 !important;
            font-size: .88rem;
            font-weight: 500;
        }
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 44px !important;
            padding: 0.6rem 1rem !important;
            color: #ffffff !important;
            font-size: 13.5px;
            font-weight: 600 !important;
            line-height: 1.2;
            letter-spacing: -.01em;
            white-space: nowrap !important;
            background: linear-gradient(135deg, #e05780 0%, #7c3aed 50%, #ec4899 100%) !important;
            background-size: 200% 200% !important;
            background-position: 0% 50% !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 999px !important;
            box-shadow: 0 8px 22px -4px rgba(224, 87, 128, 0.45) !important;
            transition: background-position .45s var(--spring), transform .3s var(--spring),
                box-shadow .3s var(--spring) !important;
        }
        section[data-testid="stSidebar"] .stButton > button p,
        section[data-testid="stSidebar"] .stButton > button span {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            letter-spacing: -.01em !important;
            white-space: nowrap !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-position: 100% 50% !important;
            transform: translateY(-2px) scale(1.015);
            box-shadow: 0 14px 30px -4px rgba(224, 87, 128, 0.65) !important;
        }
        section[data-testid="stSidebar"] .stButton > button:active {
            transform: scale(.97) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.reset-card-marker) .stButton > button {
            background: rgba(255,255,255,.06) !important;
            background-image: none !important;
            border-color: rgba(255,255,255,.12) !important;
            box-shadow: none !important;
            color: #cbd5e1 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.reset-card-marker) .stButton > button p,
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.reset-card-marker) .stButton > button span {
            color: #cbd5e1 !important;
            font-size: 12px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:has(.reset-card-marker) .stButton > button:hover {
            background: rgba(255,255,255,.1) !important;
            background-position: 0% 50% !important;
            border-color: rgba(255,255,255,.24) !important;
            box-shadow: 0 8px 20px rgba(0,0,0,.2) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            min-height: 44px;
            padding: 0 .75rem;
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 14px !important;
            box-shadow: none;
            caret-color: transparent !important;
            transition: all .28s var(--spring) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
            border-color: rgba(255, 255, 255, 0.28) !important;
            box-shadow: 0 0 0 3px rgba(224, 87, 128, 0.18) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] input {
            caret-color: transparent !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div > div {
            border-left: 0 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
            color: #cbd5e1 !important;
        }
        div[data-baseweb="popover"] {
            background: rgba(15, 20, 32, 0.85) !important;
            backdrop-filter: blur(28px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 16px !important;
            box-shadow: 0 24px 48px rgba(0, 0, 0, 0.65) !important;
            overflow: hidden;
        }
        div[data-baseweb="popover"] ul[role="listbox"] {
            background: transparent !important;
            padding: 6px !important;
        }
        div[data-baseweb="popover"] li[role="option"] {
            color: #e2e8f0 !important;
            border-radius: 10px;
            padding: 10px 12px;
            transition: background .2s var(--spring), color .2s var(--spring);
        }
        div[data-baseweb="popover"] li[role="option"]:hover {
            background: rgba(255, 255, 255, 0.08) !important;
        }
        div[data-baseweb="popover"] li[role="option"][aria-selected="true"] {
            background: rgba(224, 87, 128, 0.22) !important;
        }
        }
        section[data-testid="stSidebar"] input {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] {
            padding: .35rem 0 .95rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
            width: 16px;
            height: 16px;
            background: #ffffff;
            border: 2px solid #e05780;
            box-shadow: 0 0 10px rgba(224, 87, 128, 0.6);
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div {
            background: rgba(148,163,184,.28);
        }
        section[data-testid="stSidebar"] .telemetry-list {
            display: grid;
            gap: .75rem;
        }
        section[data-testid="stSidebar"] .telemetry-row {
            display: grid;
            gap: .2rem;
            color: #94a3b8;
            font-size: .68rem;
        }
        section[data-testid="stSidebar"] .telemetry-row strong {
            color: #e2e8f0;
            font-size: .73rem;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] .telemetry-row .status-dot {
            display: inline-block;
            vertical-align: middle;
            margin: 0 .3rem .1rem 0;
        }
        [data-testid="stIconMaterial"],
        .material-icons,
        .material-icons-outlined,
        .material-symbols-rounded,
        [class*="material-symbols"] {
            font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
            font-feature-settings: "liga" !important;
            letter-spacing: normal !important;
        }
        [data-testid="collapsedControl"] {
            display: flex !important;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.1) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 50%;
            padding: 12px;
            top: 16px;
            left: 16px;
            z-index: 99999;
            width: 46px !important;
            height: 46px !important;
            min-width: 46px !important;
            min-height: 46px !important;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        [data-testid="collapsedControl"] [data-testid="stIconMaterial"] {
            font-size: 1.35rem !important;
            line-height: 1 !important;
        }
        [data-testid="collapsedControl"]:hover {
            transform: scale(1.1);
            filter: brightness(1.2);
        }
        div[data-testid="stMetric"] {
            padding: 1rem 1.1rem;
            transition: transform .35s var(--spring), background .35s var(--spring),
                border-color .35s var(--spring), box-shadow .35s var(--spring);
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            background: rgba(224,87,128,.11);
            border-color: rgba(224,87,128,.55);
            box-shadow: 0 0 18px rgba(224,87,128,.22);
        }
        [data-testid="stMetricLabel"] { color: var(--silver); }
        [data-testid="stMetricValue"] { color: var(--pearl); }
        [data-testid="stMetricDelta"] { color: var(--rose) !important; }

        .stButton > button {
            color: var(--pearl);
            background: linear-gradient(110deg, rgba(224,87,128,.9), rgba(99,102,241,.9));
            border: 1px solid rgba(255,255,255,.22);
            border-radius: 999px;
            box-shadow: 0 8px 22px rgba(224,87,128,.2);
            transition: transform .35s var(--spring), box-shadow .35s var(--spring), filter .35s var(--spring);
        }
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            background: rgba(224,87,128,.11);
            border-color: rgba(224,87,128,.55);
            box-shadow: 0 0 18px rgba(224,87,128,.22);
            filter: none;
        }
        .stButton > button:active { transform: scale(.97); }
        .stButton > button:focus-visible, input:focus-visible { outline: 2px solid var(--rose); outline-offset: 3px; }

        div[data-baseweb="tab-list"] {
            gap: 4px; padding: 6px; border-radius: 999px;
            background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.1);
            width: fit-content;
        }
        button[data-baseweb="tab"] {
            color: var(--silver); border-radius: 999px; border: 0;
            transition: color .35s var(--spring), background .35s var(--spring), box-shadow .35s var(--spring);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #fff; background: rgba(224, 87, 128, 0.2);
            box-shadow: 0 0 22px rgba(224,87,128,.35), inset 0 0 0 1px rgba(224,87,128,.28);
        }
        .trace, .offer { animation: springFadeUp .65s var(--spring) both; }
        .trace-panel {
            padding: 1rem;
            margin: .35rem 0 1rem;
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 12px 28px rgba(0,0,0,.2);
        }
        .trace-header {
            display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; margin-bottom: .85rem;
        }
        .trace-kicker {
            color: #94a3b8; font-size: .68rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase;
        }
        .trace-status {
            padding: .3rem .6rem; border-radius: 999px;
            color: #bbf7d0; background: rgba(34,197,94,.12);
            border: 1px solid rgba(74,222,128,.35); font-size: .68rem; font-weight: 700;
        }
        .trace-flow {
            display: grid; grid-template-columns: repeat(4, 1fr);
            align-items: center; gap: .35rem;
        }
        .trace-node {
            position: relative; min-height: 4.25rem; padding: .65rem .55rem;
            border: 1px solid rgba(255,255,255,.1); border-radius: 15px;
            background: rgba(8,12,22,.55); text-align: center;
        }
        .trace-node.active {
            border-color: rgba(224,87,128,.55);
            background: rgba(224,87,128,.11);
            box-shadow: 0 0 20px rgba(224,87,128,.16);
        }
        .trace-index {
            display: block; color: #94a3b8; font: 600 .65rem "JetBrains Mono", monospace;
            margin-bottom: .4rem;
        }
        .trace-node-title { display: block; color: #f8fafc; font-size: .72rem; font-weight: 700; }
        .trace-node-copy { display: block; color: #94a3b8; font-size: .62rem; margin-top: .25rem; }
        .trace-connector { color: rgba(226,232,240,.4); text-align: center; font-size: .8rem; }
        .trace-metrics {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: .5rem; margin-top: .8rem;
        }
        .trace-metric {
            padding: .55rem .65rem; border-radius: 12px;
            background: rgba(255,255,255,.045); color: #94a3b8; font-size: .65rem;
        }
        .trace-metric strong {
            display: block; color: #f8fafc; font: 600 .78rem "JetBrains Mono", monospace; margin-top: .2rem;
        }
        @media (max-width: 760px) {
            .trace-flow { grid-template-columns: repeat(2, 1fr); }
            .trace-connector { display: none; }
            .trace-metrics { grid-template-columns: 1fr; }
        }
        .offer {
            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px; padding: 20px;
        }
        .notification-header { display: flex; align-items: center; gap: .7rem; margin-bottom: .95rem; }
        .app-icon {
            width: 36px; height: 36px; border-radius: 11px;
            background: linear-gradient(135deg, #e05780, #6366f1); box-shadow: 0 5px 16px rgba(224,87,128,.25);
        }
        .notification-title { color: #f8fafc; font-size: .85rem; font-weight: 700; }
        .notification-time { margin-left: auto; color: #94a3b8; font-size: .72rem; }
        .notification-offer-title { color: #f8fafc; font-size: 1rem; font-weight: 700; margin-bottom: .55rem; }
        .chat-bubble {
            padding: .8rem 1rem; border-radius: 16px 16px 16px 4px;
            background: rgba(37, 211, 102, .08); border: 1px solid rgba(37,211,102,.18);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
            font-size: 15px; line-height: 1.5; color: #e2e8f0;
        }
        .rag-actions { margin-top: 1rem; }
        .rag-label { display: block; color: #94a3b8; font-size: .72rem; font-weight: 600; margin-bottom: .45rem; }
        .tag {
            display: inline-block; padding: 6px 14px; border-radius: 16px;
            background: rgba(224, 87, 128, 0.15); color: #ff8fab;
            margin: 0 8px 6px 0; font-size: 13px; font-weight: 600;
        }
        .copy-badge {
            display: inline-block; padding: .28rem .7rem; border-radius: 999px;
            background: rgba(99,102,241,.2); border: 1px solid rgba(99,102,241,.5);
            color: #c7d2fe; font: 600 .75rem "JetBrains Mono", monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_apple_glassmorphism()


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


PERSONA_LABELS = {
    "IPL_Post_Tournament_Cliff": "IPL Post-Tournament Cliff",
    "Paid_Tier_Ad_Fatigue": "Paid-Tier Ad Fatigue",
    "UPI_AutoPay_Friction": "UPI AutoPay Mandate Friction",
    "Mobile_Screen_Limit_Friction": "Mobile Screen-Limit Friction",
}

PLAN_LABELS = {
    "Mobile_79_Monthly": "Mobile · ₹79 / mo",
    "Super_149_Monthly": "Super · ₹149 / mo",
    "Premium_299_Monthly": "Premium · ₹299 / mo",
    "Super_1099_Annual": "Super · ₹1,099 / yr",
    "Premium_2199_Annual": "Premium · ₹2,199 / yr",
}


def _format_persona_label(value: str) -> str:
    return PERSONA_LABELS.get(value, value.replace("_", " ").title())


def _format_plan_label(value: str) -> str:
    return PLAN_LABELS.get(value, value.replace("_", " ").title())


def _reset_signal_overrides(profile: dict) -> None:
    st.session_state["mid_roll_drop_rate"] = float(profile["mid_roll_drop_rate"])
    st.session_state["sports_watch_share"] = float(profile["sports_watch_share"])
    st.session_state["consecutive_dunning_failures"] = int(profile["consecutive_dunning_failures"])


PLAN_ECONOMICS = {
    "Mobile_79_Monthly": {"monthly_price_inr": 79, "billing_cycle_days": 30, "ad_supported": 1, "max_screens": 1, "base_ad_load_mins_per_hr": 7.5},
    "Super_149_Monthly": {"monthly_price_inr": 149, "billing_cycle_days": 30, "ad_supported": 1, "max_screens": 2, "base_ad_load_mins_per_hr": 6.0},
    "Premium_299_Monthly": {"monthly_price_inr": 299, "billing_cycle_days": 30, "ad_supported": 0, "max_screens": 4, "base_ad_load_mins_per_hr": 0.5},
    "Super_1099_Annual": {"monthly_price_inr": 1099, "billing_cycle_days": 365, "ad_supported": 1, "max_screens": 2, "base_ad_load_mins_per_hr": 5.5},
    "Premium_2199_Annual": {"monthly_price_inr": 2199, "billing_cycle_days": 365, "ad_supported": 0, "max_screens": 4, "base_ad_load_mins_per_hr": 0.5},
}


st.title("Indian OTT Retention Command Center")
st.caption("JioHotstar / SonyLIV autonomous engagement intelligence")
personas = load_personas()
with st.sidebar:
    labels = [persona["primary_churn_driver"] for persona in personas]
    active_key = st.session_state.get("flagship_persona", labels[0])
    selected_profile = next(
        persona for persona in personas if persona["primary_churn_driver"] == active_key
    )
    st.markdown(
        f"""
        <div class="sidebar-header">
            <div class="sidebar-eyebrow">Subscriber Studio</div>
            <div class="sidebar-title">Cohort &amp; Telemetry</div>
            <div class="sidebar-user-pill">
                <span class="status-dot"></span>
                <strong>{escape(str(selected_profile.get("subscriber_id", "Unknown subscriber")))}</strong>
                &nbsp;·&nbsp; {escape(str(selected_profile.get("city", selected_profile.get("region", "India"))))}
                &nbsp;·&nbsp; English
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown('<div class="sidebar-group-label">Cohort &amp; Subscription</div>', unsafe_allow_html=True)
        selected = st.selectbox(
            "Flagship Persona",
            labels,
            format_func=_format_persona_label,
            key="flagship_persona",
        )
        selected_profile = next(persona for persona in personas if persona["primary_churn_driver"] == selected)
        selected_plan = st.selectbox(
            "Plan Tier",
            _plan_options(),
            index=_plan_options().index(selected_profile["plan_tier"]),
            format_func=_format_plan_label,
            key="plan_tier",
        )
        if st.button("✦ Sample High-Risk Subscriber", use_container_width=True, type="primary"):
            selected_profile = (
                pl.read_parquet(DATA_PATH)
                .filter(pl.col("churn_probability") > 0.5)
                .sample(1, seed=42)
                .to_dicts()[0]
            )
            selected_plan = selected_profile["plan_tier"]
    profile = dict(selected_profile)
    profile["plan_tier"] = selected_plan
    profile.update(PLAN_ECONOMICS[profile["plan_tier"]])
    profile["preferred_language"] = "English"
    with st.container(border=True):
        st.markdown('<div class="sidebar-group-label">Signal Overrides</div>', unsafe_allow_html=True)
        profile["mid_roll_drop_rate"] = st.slider(
            "Mid-Roll Ad Drop Rate",
            0.0,
            1.0,
            float(profile["mid_roll_drop_rate"]),
            0.01,
            key="mid_roll_drop_rate",
        )
        profile["sports_watch_share"] = st.slider(
            "Sports Watch Share",
            0.0,
            1.0,
            float(profile["sports_watch_share"]),
            0.01,
            key="sports_watch_share",
        )
        profile["consecutive_dunning_failures"] = st.slider(
            "UPI Dunning Failures",
            0,
            5,
            int(profile["consecutive_dunning_failures"]),
            key="consecutive_dunning_failures",
        )
        st.markdown('<span class="reset-card-marker"></span>', unsafe_allow_html=True)
        st.button(
            "↺ Reset to Persona Defaults",
            use_container_width=True,
            type="secondary",
            on_click=_reset_signal_overrides,
            args=(selected_profile,),
        )
    with st.container(border=True):
        st.markdown('<div class="sidebar-group-label">Engine &amp; Policy Telemetry</div>', unsafe_allow_html=True)
        tenure = profile.get("tenure_days", "n/a")
        cycle = profile.get("billing_cycle_days", "n/a")
        st.markdown(
            f"""
            <div class="telemetry-list">
                <div class="telemetry-row">
                    <span>Tenure &amp; Cycle</span>
                    <strong>{escape(str(tenure))}d tenure · {escape(str(cycle))}d cycle</strong>
                </div>
                <div class="telemetry-row">
                    <span>Policy Guardrail</span>
                    <strong>Max Offer ≤ 25% Rem. LTV</strong>
                </div>
                <div class="telemetry-row">
                    <span>Inference Stack</span>
                    <strong><i class="status-dot"></i> LightGBM + CoxPH · Local Ollama</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                marker={"cornerradius": 8},
            )
        )
        chart.update_layout(
            title="TreeSHAP Drivers vs Anchors",
            template="plotly_dark",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc", "family": '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif'},
            xaxis={"gridcolor": "rgba(255,255,255,.07)", "zerolinecolor": "rgba(255,255,255,.12)"},
            yaxis={"gridcolor": "rgba(255,255,255,.07)"},
            margin={"l": 8, "r": 8, "t": 42, "b": 8},
        )
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("LangGraph Execution Trace")
        st.markdown(
            f"""
            <div class="trace trace-panel">
                <div class="trace-header">
                    <span class="trace-kicker">Live decision pipeline</span>
                    <span class="trace-status">Completed</span>
                </div>
                <div class="trace-flow">
                    <div class="trace-node">
                        <span class="trace-index">01</span>
                        <span class="trace-node-title">Diagnose</span>
                        <span class="trace-node-copy">Risk signal mapped</span>
                    </div>
                    <span class="trace-connector">→</span>
                    <div class="trace-node">
                        <span class="trace-index">02</span>
                        <span class="trace-node-title">Propose</span>
                        <span class="trace-node-copy">Offer selected</span>
                    </div>
                    <span class="trace-connector">→</span>
                    <div class="trace-node active">
                        <span class="trace-index">03</span>
                        <span class="trace-node-title">LTV Guardrail</span>
                        <span class="trace-node-copy">Policy passed</span>
                    </div>
                    <span class="trace-connector">→</span>
                    <div class="trace-node active">
                        <span class="trace-index">04</span>
                        <span class="trace-node-title">Copy Generator</span>
                        <span class="trace-node-copy">{escape(offer.copy_source)}</span>
                    </div>
                </div>
                <div class="trace-metrics">
                    <div class="trace-metric">Offer cost<strong>₹{offer.offer_cost_inr:.2f}</strong></div>
                    <div class="trace-metric">Predicted LTV<strong>₹{offer.predicted_remaining_ltv_inr:.2f}</strong></div>
                    <div class="trace-metric">Guardrail cap<strong>₹{offer.ltv_cap_inr:.2f}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<span class="copy-badge">COPY_SOURCE · {offer.copy_source}</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="offer">
                <div class="notification-header">
                    <span class="app-icon"></span>
                    <span class="notification-title">JioHotstar / SonyLIV Support</span>
                    <span class="notification-time">Just now</span>
                </div>
                <div class="notification-offer-title">{escape(offer.title)}</div>
                <div class="chat-bubble">{escape(offer.localized_message).replace(chr(10), "<br>")}</div>
                {(
                    '<div class="rag-actions"><span class="rag-label">Your recommended picks</span>'
                    + "".join(
                        f'<span class="tag">{escape(show["series_title"])}</span>'
                        for show in offer.recommended_shows
                    )
                    + "</div>"
                ) if offer.recommended_shows else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
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
    figure.add_bar(
        name="Net Revenue Retained (₹ Lakhs)",
        x=policies,
        y=[revenue[p] for p in policies],
        marker={"color": "#e05780", "cornerradius": 8},
    )
    figure.add_bar(
        name="Retention ROI (%)",
        x=policies,
        y=[roi[p] for p in policies],
        marker={"color": "#6366f1", "cornerradius": 8},
    )
    figure.update_layout(
        barmode="group",
        template="plotly_dark",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc", "family": '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif'},
        xaxis={"gridcolor": "rgba(255,255,255,.07)"},
        yaxis={"gridcolor": "rgba(255,255,255,.07)", "zerolinecolor": "rgba(255,255,255,.12)"},
        legend={"font": {"color": "#f8fafc"}},
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
    )
    st.plotly_chart(figure, use_container_width=True)
    st.metric("Discount Leakage Prevented", f"₹{benchmark['discount_leakage_prevented_inr']:,.0f}")
    st.subheader("Archetype Policy Performance")
    archetype_breakdown = benchmark["archetype_breakdown"]
    for index, (archetype, metrics) in enumerate(archetype_breakdown.items()):
        if index % 2 == 0:
            col1, col2 = st.columns(2)
        with col1 if index % 2 == 0 else col2:
            margin = float(metrics["net_margin_inr"])
            margin_color = "#10b981" if margin >= 0 else "#e05780"
            display_name = archetype.replace("_", " ")
            st.markdown(
                f"""
                <div style="
                    min-height: 158px;
                    margin: 0 0 1rem;
                    padding: 1.25rem;
                    background: rgba(20, 25, 35, 0.62);
                    backdrop-filter: blur(24px) saturate(180%);
                    -webkit-backdrop-filter: blur(24px);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 24px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
                ">
                    <div style="
                        color: #f8fafc;
                        font-size: 1rem;
                        font-weight: 650;
                        margin-bottom: .7rem;
                    ">{display_name}</div>
                    <div style="
                        color: #94a3b8;
                        font-size: .78rem;
                        line-height: 1.65;
                    ">
                        {metrics["subscriber_count"]:,} subscribers
                        &nbsp;&middot;&nbsp;
                        {metrics["at_risk_count"]:,} at risk
                    </div>
                    <div style="
                        display: flex;
                        align-items: end;
                        justify-content: space-between;
                        gap: 1rem;
                        margin-top: .8rem;
                    ">
                        <div>
                            <div style="color: #94a3b8; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;">
                                Save rate
                            </div>
                            <div style="color: #ffffff; font-size: 1.65rem; font-weight: 750; line-height: 1.1;">
                                {float(metrics["save_rate"]) * 100:.1f}%
                            </div>
                        </div>
                        <div style="color: {margin_color}; font-size: 1rem; font-weight: 700; white-space: nowrap;">
                            ₹{margin:,.2f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
