"""FastAPI endpoints for the OTT retention command center."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import _load_dotenv, retention_graph

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "ott_subscribers_2026.parquet"
PERSONAS_PATH = ROOT / "data" / "processed" / "flagship_personas.json"
BENCHMARK_PATH = ROOT / "data" / "processed" / "policy_benchmark.json"
BUNDLE_PATH = ROOT / "data" / "processed" / "models" / "churn_bundle.joblib"
CATALOG_PATH = ROOT / "data" / "processed" / "ott_catalog.duckdb"

app = FastAPI(title="Indian OTT Retention Agent API", version="1.0.0")


class AnalyzeRequest(BaseModel):
    user_profile: dict[str, Any] = Field(min_length=1)


class HealthResponse(BaseModel):
    status: str
    ml_bundle: bool
    rag_catalog: bool
    active_llm_tier: str


def _active_llm_tier() -> str:
    _load_dotenv()
    if os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "Cloud_LLM"
    try:
        request = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=0.5):
            return "Local_Ollama"
    except OSError:
        return "Deterministic_Template"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if BUNDLE_PATH.exists() and CATALOG_PATH.exists() else "degraded",
        ml_bundle=BUNDLE_PATH.exists(),
        rag_catalog=CATALOG_PATH.exists(),
        active_llm_tier=_active_llm_tier(),
    )


@app.get("/personas")
def personas() -> list[dict[str, Any]]:
    if not PERSONAS_PATH.exists():
        raise HTTPException(status_code=404, detail="Flagship personas are not available")
    return json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))


@app.get("/personas/random")
def random_persona() -> dict[str, Any]:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Subscriber parquet is not available")
    high_risk = pl.read_parquet(DATA_PATH).filter(pl.col("churn_probability") > 0.5)
    if high_risk.height == 0:
        raise HTTPException(status_code=404, detail="No high-risk subscriber found")
    return high_risk.sample(1, seed=42).to_dicts()[0]


@app.post("/retain/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        result = retention_graph.invoke({"user_profile": request.user_profile})
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "ml_diagnostic": result["ml_diagnostic"],
        "diagnosis": result["diagnosis"].model_dump(),
        "guardrail_logs": result.get("guardrail_logs", []),
        "revision_count": result["candidate_offer"].revision_count,
        "final_decision": result["final_decision"],
    }


@app.get("/benchmark")
def benchmark() -> dict[str, Any]:
    if not BENCHMARK_PATH.exists():
        raise HTTPException(status_code=404, detail="Policy benchmark is not available")
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
