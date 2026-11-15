# Copilot Instructions: Autonomous Indian OTT Churn & Retention Agent

## 1. Project Identity & Architecture

You are acting as a **Senior Staff Data Scientist and AI Engineer** building an end-to-end portfolio system: an **Autonomous Customer Engagement & Churn Retention Agent for an Indian OTT Streaming Platform** (modeled after JioHotstar and SonyLIV).

Follow a strict decoupled architecture across this monorepo:

- `data/raw/` — Immutable Kaggle source CSVs (never modify or commit).
- `data/processed/` — Engineered Parquet datasets, DuckDB stores, and `flagship_personas.json`.
- `src/data/` — Polars/DuckDB ETL, synthetic causal feature injection, and catalog embeddings.
- `src/ml/` — Predictive (`lightgbm`), Survival (`lifelines`), Uplift (`econml`), and Explainability (`shap`) pipelines.
- `src/agent/` — `langgraph` state machines, Pydantic schemas, LTV guardrail tools, and RAG retrievers.
- `src/api/` — Asynchronous `fastapi` endpoints streaming agent reasoning traces and SHAP profiles.
- `dashboard/` — Interactive Recruiter Command Center UI.
- `tests/` — `pytest` unit tests for causal invariants, LTV guardrails, and API contracts.

---

## 2. Environment, OS & Credit-Saving Rules (CRITICAL)

- **OS & Shell:** Windows (PowerShell). Use Windows-compatible path handling (`pathlib.Path`) and PowerShell syntax.
- **Strict `.venv` Isolation:** NEVER run global `pip` or global `python`. Always install packages and execute scripts via `uv` (`uv add ...`, `uv run ...`) or directly via `.venv\Scripts\python.exe`.
- **Copilot Credit Efficiency:**
  - Batch file edits and terminal verification steps together rather than making single-line incremental calls.
  - Never inspect raw CSVs or `.parquet` files directly in chat; inspect schemas/summaries using `duckdb` or `polars` one-liners.
  - **Anti-Loop Rule:** If a terminal command or test fails twice with the same error, stop executing commands, diagnose the root cause concisely, and ask the user before retrying.

---

## 3. Coding & Data Science Standards

- **DataFrames & SQL:** Default to `polars` and `duckdb` for ETL, feature engineering, and aggregations. Convert to `pandas`/`numpy` only at the model boundary (`lightgbm`, `lifelines`, `shap`).
- **Reproducibility:** Pin `seed=42` (`np.random.default_rng(42)`) across all synthetic data generators, train/validation splits, and ML estimators.
- **Type Safety & Validation:** Use Python 3.11+ type hints everywhere and `pydantic` v2 `BaseModel` schemas for all LangGraph states, tool inputs/outputs, and FastAPI payloads.
- **Logging:** Use `rich.console` or standard structured logging instead of raw `print()` statements in `src/` modules.

---

## 4. Indian OTT Domain Schema & Business Guardrails

All synthetic features, ML reason codes, and Agent policies must align with these 2026 Indian OTT mechanics:

### A. Subscription Tiers & Unit Economics

| Tier ID                 | Price (INR) | Cycle | Ad-Supported | Max Screens | Base Ad Load |
| :---------------------- | :---------: | :---: | :----------: | :---------: | :----------: |
| `Mobile_79_Monthly`   |    ₹79    |  30d  | Yes (`1`) |      1      | 7.5 mins/hr |
| `Super_149_Monthly`   |    ₹149    |  30d  | Yes (`1`) |      2      | 6.0 mins/hr |
| `Premium_299_Monthly` |    ₹299    |  30d  |  No (`0`)  |      4      | 0.5 mins/hr |
| `Super_1099_Annual`   |   ₹1,099   | 365d | Yes (`1`) |      2      | 5.5 mins/hr |
| `Premium_2199_Annual` |   ₹2,199   | 365d |  No (`0`)  |      4      | 0.5 mins/hr |

### B. The 4 Canonical Churn Archetypes & Intervention Mapping

1. **`Paid_Tier_Ad_Fatigue`**
   - *Signal:* High `mid_roll_drop_rate` (>0.35) and `ad_skip_attempts_7d` on ad-supported plans.
   - *Allowed Action:* Complimentary 7-day Ad-Free Boost or prorated upgrade discount to `Premium_299_Monthly`. Do NOT send generic show recommendations.
2. **`IPL_Post_Tournament_Cliff`**
   - *Signal:* `ipl_cohort_signup == 1`, `sports_watch_share > 0.75`, rising `days_since_last_live_match`.
   - *Allowed Action:* Cross-genre RAG content bridge matching high-adrenaline thrillers/crime series in the user's `preferred_language` (`Hindi`, `Telugu`, `Tamil`, `Malayalam`, etc.).
3. **`UPI_AutoPay_Friction`** (Involuntary Churn)
   - *Signal:* `consecutive_dunning_failures >= 1` with `Insufficient_Funds_Bounce` or `Revoked_In_UPI_App`.
   - *Allowed Action:* Smart dunning retry scheduled around Indian salary credit windows (1st–5th of month) or 1-click WhatsApp switch to a ₹79 sachet pass. Never treat as voluntary content dissatisfaction.
4. **`Mobile_Screen_Limit_Friction`**
   - *Signal:* `concurrent_stream_rejections_30d >= 3` on 1-screen `Mobile_79_Monthly` plans.
   - *Allowed Action:* Discounted multi-screen bridge offer to `Super_149_Monthly` or `Super_1099_Annual`.

### C. Hard Financial Guardrail

- Every retention offer proposed by the LangGraph agent MUST pass the deterministic LTV check:
  $$
  \text{Offer\_Cost\_INR} \le 0.25 \times \text{Predicted\_Remaining\_LTV\_INR}
  $$
- Any offer violating this cap must be rejected and downgraded by the agent's guardrail node before final output generation.
