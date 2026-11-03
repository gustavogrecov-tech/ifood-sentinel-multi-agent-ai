"""
Comprehensive Unit & Integration Test Suite for iFood GenAI Operations & Trust Sentinel.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api_server import app
from src.database import (
    get_kpi_summary,
    get_logs_dataframe,
    get_menu_dataframe,
    get_orders_dataframe,
    init_and_seed_db,
)
from src.ml_pipeline import get_ml_pipeline
from src.multi_agent_orchestrator import MultiAgentOrchestrator, get_preset_scenarios
from src.rag_policy_engine import get_rag_engine
from src.synthetic_media import ensure_sample_images


def test_database_seeding_and_kpis() -> None:
    stats = init_and_seed_db()
    assert stats["orders_seeded"] >= 1500
    assert stats["menu_items_seeded"] >= 300

    orders_df = get_orders_dataframe()
    menu_df = get_menu_dataframe()
    logs_df = get_logs_dataframe()

    assert len(orders_df) >= 1500
    assert len(menu_df) >= 300
    assert len(logs_df) >= 3

    kpis = get_kpi_summary()
    assert kpis["total_orders_analyzed"] >= 1500
    assert kpis["fraud_prevented_brl"] > 10000.0
    assert kpis["total_hours_saved"] > 50.0


def test_ml_pipeline_performance_and_predictions() -> None:
    pipeline = get_ml_pipeline()
    metrics = pipeline.evaluate_models()

    assert metrics["roc_auc"] > 0.80
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["feature_importances"]) == 6
    assert "fpr" in metrics["roc_curve_data"]
    assert "tpr" in metrics["roc_curve_data"]

    high_risk_res = pipeline.predict_dispute_fraud(
        {
            "order_value_brl": 210.0,
            "delivery_time_min": 30.0,
            "customer_refund_rate_90d": 0.65,
            "account_age_days": 10,
            "distance_km": 3.5,
            "photo_evidence_score": 15.0,
        }
    )
    assert high_risk_res.fraud_probability >= 0.70
    assert high_risk_res.risk_level in ("HIGH", "CRITICAL")
    assert high_risk_res.is_anomaly is True

    low_risk_res = pipeline.predict_dispute_fraud(
        {
            "order_value_brl": 58.0,
            "delivery_time_min": 82.0,
            "customer_refund_rate_90d": 0.02,
            "account_age_days": 520,
            "distance_km": 4.0,
            "photo_evidence_score": 92.0,
        }
    )
    assert low_risk_res.fraud_probability < 0.40
    assert low_risk_res.risk_level in ("LOW", "MEDIUM")

    menu_res = pipeline.predict_menu_anomaly(
        [
            {
                "dish_name": "Combo Burger Erro Digitação",
                "description": "lanche simples",
                "price_brl": 450.00,
                "category_avg_price_brl": 42.90,
                "prep_time_min": 20,
            }
        ]
    )
    assert menu_res.is_anomaly is True
    assert menu_res.risk_level in ("HIGH", "CRITICAL")


def test_rag_policy_engine_search_and_custom_docs() -> None:
    rag = get_rag_engine()
    matches = rag.search_policies("bloqueio reembolso fraude recorrente embalagem íntegra", top_k=3)
    assert len(matches) == 3
    assert matches[0].similarity_score > 0.40
    assert matches[0].clause_id.startswith("POL-")

    added = rag.add_custom_document(
        title="Política Especial de Bebidas Lacradas",
        text="Bebidas entregues sem lacre térmico no gargalo dão direito a voucher imediato de R$ 15,00.",
    )
    assert added["clause_id"].startswith("POL-CUS-")

    custom_matches = rag.search_policies("bebidas sem lacre térmico no gargalo voucher", top_k=2)
    assert any(m.clause_id == added["clause_id"] for m in custom_matches)


def test_synthetic_media_generation() -> None:
    paths = ensure_sample_images()
    assert set(paths.keys()) == {"dispute_high_risk", "dispute_legit_damaged", "menu_physical_messy"}
    for file_path in paths.values():
        p = Path(file_path)
        assert p.exists()
        assert p.stat().st_size > 1000


def test_multi_agent_orchestrator_workflows() -> None:
    orchestrator = MultiAgentOrchestrator()
    presets = get_preset_scenarios()

    p1 = presets["dispute_high_risk"]
    v1 = orchestrator.run_pipeline(
        workflow_type=p1["workflow_type"],
        case_input=p1["case_input"],
        image_path=p1["image_path"],
    )
    assert v1.workflow_type == "dispute_fraud"
    assert v1.final_decision == "BLOCK_REFUND_FRAUD_SUSPECTED"
    assert v1.n8n_result.routed_branch in ("CRITICAL_FRAUD_BLOCK", "CRITICAL_RISK_AUTO_BLOCK")
    assert len(v1.agent_traces) == 4

    p2 = presets["dispute_legit_damaged"]
    v2 = orchestrator.run_pipeline(
        workflow_type=p2["workflow_type"],
        case_input=p2["case_input"],
        image_path=p2["image_path"],
    )
    assert v2.workflow_type == "dispute_fraud"
    assert v2.final_decision == "APPROVE_INSTANT_REFUND"
    assert v2.n8n_result.routed_branch in ("INSTANT_AUTO_RESOLVE", "LOW_RISK_INSTANT_AUTO_APPROVAL")

    p3 = presets["menu_physical_messy"]
    v3 = orchestrator.run_pipeline(
        workflow_type=p3["workflow_type"],
        case_input=p3["case_input"],
        image_path=p3["image_path"],
    )
    assert v3.workflow_type == "menu_audit"
    assert v3.final_decision == "QUARANTINE_MENU_ANOMALY_AND_OPTIMIZE"
    assert len(v3.vision_result.optimized_copywriting) == 6


def test_fastapi_endpoints() -> None:
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_kpis = client.get("/api/v1/kpis")
    assert r_kpis.status_code == 200
    assert r_kpis.json()["total_orders_analyzed"] >= 1500

    r_ml = client.get("/api/v1/ml-metrics")
    assert r_ml.status_code == 200
    assert r_ml.json()["roc_auc"] > 0.80

    r_orch = client.post(
        "/api/v1/orchestrate",
        json={"scenario_key": "dispute_high_risk"},
    )
    assert r_orch.status_code == 200
    body = r_orch.json()
    assert body["final_decision"] == "BLOCK_REFUND_FRAUD_SUSPECTED"
    assert body["n8n_result"]["dispatched"] is True

    r_hook = client.post(
        "/api/v1/n8n-webhook-receiver",
        json={"case_id": "SENTINEL-TEST-1", "protocol_id": "IFD-N8N-999"},
    )
    assert r_hook.status_code == 200
    assert r_hook.json()["received"] is True
