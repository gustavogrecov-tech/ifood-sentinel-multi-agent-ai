"""
Machine Learning Pipeline for iFood Operations & Trust Sentinel.
Implements:
1. RandomForestClassifier (n_estimators=120, max_depth=8, random_state=42) for Dispute/Refund Fraud Prediction
2. IsolationForest (contamination=0.08, random_state=42) for Menu Pricing & Margin Anomaly Detection
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.database import get_menu_dataframe, get_orders_dataframe, init_and_seed_db
from src.schemas import MLPredictionResult

DISPUTE_FEATURES = [
    "order_value_brl",
    "delivery_time_min",
    "customer_refund_rate_90d",
    "account_age_days",
    "distance_km",
    "photo_evidence_score",
]

MENU_FEATURES = [
    "price_brl",
    "category_avg_price_brl",
    "description_length",
    "prep_time_min",
]


class IFoodMLPipeline:
    """
    Dual-Engine ML Pipeline combining supervised Fraud Classification (Random Forest)
    and unsupervised Catalog Anomaly Detection (Isolation Forest).
    """

    def __init__(self) -> None:
        self.model_version = "ifood-rf-iforest-v2.4"
        self.rf_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        self.iforest_model = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=100,
        )
        self._is_trained = False
        self._metrics_cache: dict[str, Any] = {}
        self.train_models()

    def train_models(self) -> None:
        init_and_seed_db()
        orders_df = get_orders_dataframe()
        menu_df = get_menu_dataframe()

        X = orders_df[DISPUTE_FEATURES].astype(float)
        y = orders_df["is_fraud_ground_truth"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_test)
        y_prob = self.rf_model.predict_proba(X_test)[:, 1]

        roc_auc = float(roc_auc_score(y_test, y_prob))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        step = max(1, len(fpr) // 45)
        fpr_list = [round(float(v), 4) for v in fpr[::step]]
        tpr_list = [round(float(v), 4) for v in tpr[::step]]
        if fpr_list[-1] != 1.0:
            fpr_list.append(1.0)
            tpr_list.append(1.0)

        importances = [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in sorted(
                zip(DISPUTE_FEATURES, self.rf_model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        X_menu = menu_df[MENU_FEATURES].astype(float)
        self.iforest_model.fit(X_menu)
        raw_scores = self.iforest_model.decision_function(X_menu)
        iforest_preds = self.iforest_model.predict(X_menu)

        normal_scores = [round(float(s), 4) for s, p in zip(raw_scores, iforest_preds) if p == 1]
        anomaly_scores = [round(float(s), 4) for s, p in zip(raw_scores, iforest_preds) if p == -1]

        self._metrics_cache = {
            "model_version": self.model_version,
            "roc_auc": round(roc_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "feature_importances": importances,
            "roc_curve_data": {
                "fpr": fpr_list,
                "tpr": tpr_list,
            },
            "anomaly_distribution": {
                "normal_scores": normal_scores,
                "anomaly_scores": anomaly_scores,
                "total_catalog_items": len(menu_df),
                "detected_anomalies": int(np.sum(iforest_preds == -1)),
            },
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        self._is_trained = True

    def evaluate_models(self) -> dict[str, Any]:
        if not self._is_trained:
            self.train_models()
        return self._metrics_cache

    def predict_dispute_fraud(self, features: dict[str, Any]) -> MLPredictionResult:
        if not self._is_trained:
            self.train_models()

        row = {
            "order_value_brl": float(features.get("order_value_brl", 95.0)),
            "delivery_time_min": float(features.get("delivery_time_min", 40.0)),
            "customer_refund_rate_90d": float(features.get("customer_refund_rate_90d", 0.08)),
            "account_age_days": float(features.get("account_age_days", 240)),
            "distance_km": float(features.get("distance_km", 4.5)),
            "photo_evidence_score": float(features.get("photo_evidence_score", 75.0)),
        }
        X_input = pd.DataFrame([row])[DISPUTE_FEATURES]
        prob = float(self.rf_model.predict_proba(X_input)[0, 1])

        factors: list[dict[str, Any]] = []
        if row["customer_refund_rate_90d"] >= 0.30:
            factors.append(
                {
                    "feature": "customer_refund_rate_90d",
                    "value": f"{row['customer_refund_rate_90d']*100:.1f}%",
                    "impact": "HIGH_RISK",
                    "detail": "Taxa de reembolso em 90 dias muito acima do limite de segurança (15%).",
                }
            )
        else:
            factors.append(
                {
                    "feature": "customer_refund_rate_90d",
                    "value": f"{row['customer_refund_rate_90d']*100:.1f}%",
                    "impact": "LOW_RISK",
                    "detail": "Histórico de reembolso do cliente dentro da normalidade.",
                }
            )

        if row["photo_evidence_score"] <= 45.0:
            factors.append(
                {
                    "feature": "photo_evidence_score",
                    "value": f"{row['photo_evidence_score']:.1f}/100",
                    "impact": "HIGH_RISK",
                    "detail": "Evidência visual de avaria fraca ou embalagem íntegra contradizendo a reclamação.",
                }
            )
        else:
            factors.append(
                {
                    "feature": "photo_evidence_score",
                    "value": f"{row['photo_evidence_score']:.1f}/100",
                    "impact": "PROTECTIVE",
                    "detail": "Evidência visual confirma avaria real na embalagem/produto.",
                }
            )

        if row["account_age_days"] <= 30:
            factors.append(
                {
                    "feature": "account_age_days",
                    "value": f"{int(row['account_age_days'])} dias",
                    "impact": "MEDIUM_RISK",
                    "detail": "Conta recém-criada (< 30 dias) com pedido de alto valor.",
                }
            )

        if row["order_value_brl"] >= 150.0:
            factors.append(
                {
                    "feature": "order_value_brl",
                    "value": f"R$ {row['order_value_brl']:.2f}",
                    "impact": "HIGH_EXPOSURE",
                    "detail": "Ticket elevado aumenta exposição financeira ao risco de chargeback/reembolso.",
                }
            )

        if prob >= 0.80:
            risk_level = "CRITICAL"
        elif prob >= 0.55:
            risk_level = "HIGH"
        elif prob >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return MLPredictionResult(
            fraud_probability=round(prob, 4),
            is_anomaly=bool(prob >= 0.50),
            anomaly_score=round(prob * 100.0, 2),
            risk_level=risk_level,
            top_risk_factors=factors,
            model_version=self.model_version,
        )

    def predict_menu_anomaly(self, items: list[dict[str, Any]]) -> MLPredictionResult:
        if not self._is_trained:
            self.train_models()

        if not items:
            return MLPredictionResult(
                fraud_probability=0.05,
                is_anomaly=False,
                anomaly_score=5.0,
                risk_level="LOW",
                top_risk_factors=[],
                model_version=self.model_version,
            )

        rows = []
        factors: list[dict[str, Any]] = []
        max_deviation_ratio = 1.0

        for item in items:
            price = float(item.get("price_brl", 45.0))
            cat_avg = float(item.get("category_avg_price_brl", 48.0))
            desc = str(item.get("description", item.get("raw_description", "")))
            prep_time = int(item.get("prep_time_min", 25))
            dish_name = str(item.get("dish_name", item.get("name", "Item do Cardápio")))

            rows.append(
                {
                    "price_brl": price,
                    "category_avg_price_brl": cat_avg,
                    "description_length": len(desc),
                    "prep_time_min": prep_time,
                }
            )

            ratio = price / max(1.0, cat_avg)
            inv_ratio = cat_avg / max(0.5, price)
            dev = max(ratio, inv_ratio)
            if dev > max_deviation_ratio:
                max_deviation_ratio = dev

            if price >= cat_avg * 3.0 or price <= 5.0:
                factors.append(
                    {
                        "feature": "price_brl_outlier",
                        "value": f"{dish_name}: R$ {price:.2f} (Média: R$ {cat_avg:.2f})",
                        "impact": "CRITICAL_ANOMALY",
                        "detail": f"Desvio severo de preço ({ratio:.1f}x da categoria) — provável erro de digitação ou abuso.",
                    }
                )
            if len(desc.strip()) < 22:
                factors.append(
                    {
                        "feature": "description_length",
                        "value": f"{dish_name} ({len(desc.strip())} caracteres)",
                        "impact": "SEO_DEFECT",
                        "detail": "Descrição muito curta prejudica conversão no app iFood (< 22 caracteres).",
                    }
                )

        X_batch = pd.DataFrame(rows)[MENU_FEATURES]
        preds = self.iforest_model.predict(X_batch)
        decision_scores = self.iforest_model.decision_function(X_batch)

        has_iforest_anomaly = bool(np.any(preds == -1)) or max_deviation_ratio >= 3.0
        min_decision = float(np.min(decision_scores))

        base_prob = float(np.clip(0.5 - min_decision * 1.8, 0.05, 0.98))
        if max_deviation_ratio >= 4.0:
            base_prob = max(base_prob, 0.88)
        elif max_deviation_ratio >= 2.5:
            base_prob = max(base_prob, 0.68)

        if base_prob >= 0.80:
            risk_level = "CRITICAL"
        elif base_prob >= 0.55:
            risk_level = "HIGH"
        elif base_prob >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return MLPredictionResult(
            fraud_probability=round(base_prob, 4),
            is_anomaly=has_iforest_anomaly,
            anomaly_score=round(base_prob * 100.0, 2),
            risk_level=risk_level,
            top_risk_factors=factors[:6],
            model_version=self.model_version,
        )


_ML_PIPELINE_SINGLETON: IFoodMLPipeline | None = None


def get_ml_pipeline() -> IFoodMLPipeline:
    global _ML_PIPELINE_SINGLETON
    if _ML_PIPELINE_SINGLETON is None:
        _ML_PIPELINE_SINGLETON = IFoodMLPipeline()
    return _ML_PIPELINE_SINGLETON
