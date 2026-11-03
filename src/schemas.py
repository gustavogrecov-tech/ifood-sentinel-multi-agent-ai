"""
Pydantic v2 Data Contracts for the iFood GenAI Operations & Trust Sentinel.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class AgentStepTrace(BaseModel):
    agent_name: str = Field(..., description="Identifier of the specialized agent")
    role: str = Field(..., description="Functional role in the orchestration pipeline")
    status: str = Field(..., description="Execution status (COMPLETED, FALLBACK, WARNING)")
    latency_ms: float = Field(..., description="Execution duration in milliseconds")
    input_summary: str = Field(..., description="Short summary of agent inputs")
    output_data: dict[str, Any] = Field(default_factory=dict, description="Structured output payload")
    reasoning: str = Field(..., description="Chain-of-thought reasoning or decision rationale")


class VisionExtractionResult(BaseModel):
    media_type: str = Field(..., description="'dispute_photo' or 'menu_photo'")
    detected_items: list[dict[str, Any]] = Field(default_factory=list, description="Extracted items or dishes")
    visual_anomalies: list[str] = Field(default_factory=list, description="Detected visual anomalies or defects")
    packaging_integrity_score: float = Field(..., ge=0.0, le=100.0, description="Packaging integrity score 0..100")
    ocr_raw_text: str = Field(..., description="Extracted OCR text from receipt or menu")
    optimized_copywriting: list[dict[str, Any]] = Field(
        default_factory=list, description="SEO-optimized titles and descriptions for catalog items"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Vision confidence score 0..1")


class MLPredictionResult(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of fraud or anomaly 0..1")
    is_anomaly: bool = Field(..., description="Whether the case is flagged as anomalous/fraudulent")
    anomaly_score: float = Field(..., description="Raw or normalized anomaly score")
    risk_level: str = Field(..., description="'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    top_risk_factors: list[dict[str, Any]] = Field(
        default_factory=list, description="Feature attributions driving the risk score"
    )
    model_version: str = Field(default="ifood-rf-iforest-v2.4", description="Model identifier and version")


class RAGPolicyMatch(BaseModel):
    clause_id: str = Field(..., description="Official policy clause identifier, e.g., POL-REF-001")
    title: str = Field(..., description="Policy clause title")
    content: str = Field(..., description="Full policy normative text")
    similarity_score: float = Field(..., description="Cosine similarity score 0..1")
    recommended_sla_action: str = Field(..., description="Normative SLA action prescribed by the clause")


class N8nDispatchResult(BaseModel):
    dispatched: bool = Field(..., description="Whether the webhook payload was processed")
    webhook_url: str = Field(..., description="Target n8n webhook URL")
    http_status: int = Field(..., description="HTTP status code returned")
    protocol_id: str = Field(..., description="Generated protocol tracking ID")
    routed_branch: str = Field(
        ..., description="'CRITICAL_FRAUD_BLOCK', 'HITL_ESCALATION_QUEUE', or 'INSTANT_AUTO_RESOLVE'"
    )
    actions_taken: list[str] = Field(default_factory=list, description="Downstream actions triggered via n8n")
    response_time_ms: float = Field(..., description="Webhook round-trip or simulation latency in ms")
    mode: str = Field(..., description="'live_webhook' or 'simulated_fallback'")


class OrchestrationVerdict(BaseModel):
    case_id: str = Field(..., description="Unique case tracking identifier")
    timestamp: str = Field(..., description="ISO-8601 execution timestamp")
    workflow_type: str = Field(..., description="'dispute_fraud' or 'menu_audit'")
    final_decision: str = Field(..., description="Final decision code")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Combined ensemble confidence score")
    financial_impact_brl: float = Field(..., description="Estimated financial value protected or optimized in BRL")
    estimated_minutes_saved: float = Field(..., description="Estimated manual review time saved in minutes")
    vision_result: VisionExtractionResult
    ml_result: MLPredictionResult
    rag_matches: list[RAGPolicyMatch] = Field(default_factory=list)
    n8n_result: N8nDispatchResult
    agent_traces: list[AgentStepTrace] = Field(default_factory=list)
    executive_summary: str = Field(..., description="Executive decision summary in Portuguese/English")
