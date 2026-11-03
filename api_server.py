"""
FastAPI Microservice Backend for the iFood GenAI Operations & Trust Sentinel.
Exposes REST endpoints for executive KPIs, ML metrics, Multi-Agent Orchestration,
and a local n8n Webhook Receiver endpoint.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database import get_kpi_summary, init_and_seed_db
from src.ml_pipeline import get_ml_pipeline
from src.multi_agent_orchestrator import MultiAgentOrchestrator, get_preset_scenarios
from src.schemas import OrchestrationVerdict


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_and_seed_db()
    get_ml_pipeline()
    yield


app = FastAPI(
    title="iFood GenAI Operations & Trust Sentinel API",
    description=(
        "Enterprise Multi-Agent GenAI, Multimodal Vision, Classical ML (Random Forest & Isolation Forest), "
        "RAG Policy Engine, and n8n Hyperautomation API."
    ),
    version="2.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = MultiAgentOrchestrator()


class OrchestrateRequest(BaseModel):
    workflow_type: str = Field(default="dispute_fraud", description="'dispute_fraud' or 'menu_audit'")
    case_input: dict[str, Any] = Field(default_factory=dict, description="Input payload for the case")
    scenario_key: str | None = Field(
        default=None,
        description="Optional preset key: 'dispute_high_risk', 'dispute_legit_damaged', or 'menu_physical_messy'",
    )
    image_path: str | None = None
    api_key: str | None = None
    n8n_webhook_url: str | None = None


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "ifood-genai-trust-sentinel",
        "version": "2.4.0",
        "agents_online": [
            "VisionMultimodalAgent",
            "MLFraudAnomalyAgent",
            "PolicyComplianceRAGAgent",
            "N8nActionDispatcherAgent",
        ],
    }


@app.get("/api/v1/kpis")
def api_get_kpis() -> dict[str, Any]:
    return get_kpi_summary()


@app.get("/api/v1/ml-metrics")
def api_get_ml_metrics() -> dict[str, Any]:
    pipeline = get_ml_pipeline()
    return pipeline.evaluate_models()


@app.post("/api/v1/orchestrate", response_model=OrchestrationVerdict)
def api_orchestrate(req: OrchestrateRequest) -> OrchestrationVerdict:
    presets = get_preset_scenarios()
    workflow_type = req.workflow_type
    case_input = dict(req.case_input)
    image_path = req.image_path

    if req.scenario_key and req.scenario_key in presets:
        preset = presets[req.scenario_key]
        workflow_type = preset["workflow_type"]
        if not case_input:
            case_input = dict(preset["case_input"])
        if not image_path:
            image_path = preset["image_path"]

    return orchestrator.run_pipeline(
        workflow_type=workflow_type,
        case_input=case_input,
        image_path=image_path,
        api_key=req.api_key,
        n8n_webhook_url=req.n8n_webhook_url,
    )


@app.post("/api/v1/n8n-webhook-receiver")
def api_n8n_webhook_receiver(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "received": True,
        "status": "PROCESSED_BY_SENTINEL_RECEIVER",
        "case_id": payload.get("case_id", "UNKNOWN"),
        "protocol_id": payload.get("protocol_id", "IFD-N8N-LOCAL"),
        "routed_branch": payload.get("expected_branch", payload.get("routed_branch", "INSTANT_AUTO_RESOLVE")),
        "timestamp": payload.get("timestamp"),
    }
