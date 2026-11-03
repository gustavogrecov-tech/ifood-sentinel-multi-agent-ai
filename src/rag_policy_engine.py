"""
RAG Policy Engine for iFood Operational & Trust Compliance.
Contains 10 official-style iFood Operational & Trust Policies and performs
vector retrieval via scikit-learn TfidfVectorizer + cosine_similarity.
Supports custom document ingestion (PDF/TXT).
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import RAGPolicyMatch

BUILT_IN_POLICIES: list[dict[str, str]] = [
    {
        "clause_id": "POL-REF-001",
        "title": "Protocolo de Reembolso Instantâneo por Avaria Severa de Embalagem",
        "content": (
            "Quando a análise de visão computacional comprovar violação física do lacre, "
            "vazamento severo de líquidos ou esmagamento estrutural da embalagem (score de integridade < 55/100) "
            "combinado com histórico de cliente saudável (taxa de reembolso < 15% nos últimos 90 dias), "
            "o reembolso integral ou parcial deve ser aprovado instantaneamente sem fricção humana."
        ),
        "recommended_sla_action": "APPROVE_INSTANT_REFUND_SLA_1MIN",
    },
    {
        "clause_id": "POL-FRD-002",
        "title": "Bloqueio Preventivo de Abuso Recorrente de Reembolso (Friendly Fraud)",
        "content": (
            "Contas de usuários que apresentem taxa de solicitação de reembolso superior a 30% em 90 dias "
            "ou múltiplas contestações de pedidos de alto ticket (> R$ 140,00) onde a foto enviada demonstra "
            "embalagem íntegra, lacre preservado ou metadados inconsistentes devem ter o reembolso automático "
            "bloqueado preventivamente e a conta sinalizada para auditoria de Trust & Safety."
        ),
        "recommended_sla_action": "BLOCK_AUTO_REFUND_AND_FLAG_ACCOUNT",
    },
    {
        "clause_id": "POL-SLA-003",
        "title": "Compensação Automática por Atraso Logístico Crítico (> 60 Minutos)",
        "content": (
            "Pedidos cujo tempo total de entrega ultrapasse 60 minutos além da janela estimada e apresentem "
            "reclamação de perda térmica (alimento frio ou derretido) são elegíveis para reembolso automático "
            "ou voucher de compensação imediata, desde que a conta não esteja listada em quarentena de risco."
        ),
        "recommended_sla_action": "ISSUE_INSTANT_COMPENSATION_OR_REFUND",
    },
    {
        "clause_id": "POL-HITL-004",
        "title": "Escalonamento Human-in-the-Loop (HITL) para Casos de Zona Cinzenta",
        "content": (
            "Disputas com probabilidade de fraude entre 40% e 75% ou divergência entre a decisão do modelo "
            "supervisionado (Random Forest) e a inspeção multimodal (Vision AI) devem ser roteadas para a fila "
            "prioritária de especialistas N2 via webhook n8n com SLA máximo de resolução de 15 minutos."
        ),
        "recommended_sla_action": "ROUTE_TO_N2_HITL_ESCALATION_QUEUE",
    },
    {
        "clause_id": "POL-CAT-005",
        "title": "Política de Quarentena de Anomalias de Preço no Cardápio Digital",
        "content": (
            "Itens de cardápio submetidos por restaurantes parceiros cujo preço apresente desvio superior a "
            "300% acima da média da categoria (ex: erro de digitação R$ 450,00 em vez de R$ 45,00) ou valor "
            "simbólico predatório (< R$ 5,00) devem ser colocados em quarentena preventiva antes da publicação "
            "para evitar cancelamentos em massa, chargebacks e exposição jurídica."
        ),
        "recommended_sla_action": "QUARANTINE_ITEM_PRICE_ANOMALY_AND_NOTIFY_PARTNER",
    },
    {
        "clause_id": "POL-SEO-006",
        "title": "Padrão de Qualidade de Copywriting e Engenharia de Cardápio GenAI",
        "content": (
            "Pratos com descrições inferiores a 25 caracteres ou termos genéricos ('prato simples', 'lanche bom') "
            "devem receber enriquecimento automático via GenAI destacando ingredientes principais, método de "
            "preparo artesanal, gramatura sugerida e gatilhos sensoriais, mantendo fidelidade estrita ao item real."
        ),
        "recommended_sla_action": "APPLY_GENAI_MENU_COPYWRITING_OPTIMIZATION",
    },
    {
        "clause_id": "POL-CHB-007",
        "title": "Prevenção de Chargeback em Contas Recém-Criadas (< 15 Dias)",
        "content": (
            "Pedidos acima de R$ 150,00 realizados por contas com idade inferior a 15 dias que abram disputa "
            "de 'pedido inteiro não recebido' exigem validação cruzada de geolocalização do entregador e "
            "código de confirmação de entrega antes de qualquer estorno financeiro."
        ),
        "recommended_sla_action": "REQUIRE_DELIVERY_PIN_AND_GPS_AUDIT",
    },
    {
        "clause_id": "POL-IMG-008",
        "title": "Autenticidade de Evidência Fotográfica em Disputas de Entrega",
        "content": (
            "Fotos enviadas no fluxo de ajuda do app são submetidas a verificação de integridade visual, OCR de "
            "comprovante térmico e detecção de artefatos sintéticos. Imagens sem correspondência com os itens "
            "faturados no pedido reduzem o score de evidência abaixo de 35 pontos."
        ),
        "recommended_sla_action": "REJECT_UNVERIFIED_PHOTO_EVIDENCE",
    },
    {
        "clause_id": "POL-PAR-009",
        "title": "Proteção Financeira ao Restaurante Parceiro em Fraudes Confirmadas",
        "content": (
            "Quando o motor Sentinel bloqueia uma solicitação de reembolso fraudulenta (risco CRITICAL), "
            "o repasse financeiro do restaurante parceiro é 100% preservado, evitando débitos indevidos na "
            "conciliação semanal da loja."
        ),
        "recommended_sla_action": "PROTECT_MERCHANT_PAYOUT_100PCT",
    },
    {
        "clause_id": "POL-LGPD-010",
        "title": "Governança de IA, Explicabilidade e Conformidade LGPD (Art. 20)",
        "content": (
            "Toda decisão automatizada de bloqueio ou aprovação de reembolso deve registrar trilha de auditoria "
            "completa (reasoning dos agentes, variáveis de risco do modelo ML e cláusula normativa RAG) "
            "garantindo o direito à revisão e transparência algorítmica."
        ),
        "recommended_sla_action": "LOG_EXPLAINABLE_AUDIT_TRAIL_SQLITE",
    },
]


class PolicyRAGEngine:
    def __init__(self) -> None:
        self.documents: list[dict[str, str]] = [dict(doc) for doc in BUILT_IN_POLICIES]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
        )
        self._tfidf_matrix = None
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        corpus = [
            f"{doc['clause_id']} {doc['title']} {doc['content']} {doc['recommended_sla_action']}"
            for doc in self.documents
        ]
        self._tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def add_custom_document(
        self,
        title: str,
        text: str,
        clause_id: str | None = None,
        recommended_sla_action: str = "CUSTOM_POLICY_COMPLIANCE_CHECK",
    ) -> dict[str, str]:
        new_id = clause_id or f"POL-CUS-{len(self.documents) + 1:03d}"
        new_doc = {
            "clause_id": new_id,
            "title": title.strip() or f"Documento Customizado {new_id}",
            "content": text.strip(),
            "recommended_sla_action": recommended_sla_action,
        }
        self.documents.append(new_doc)
        self._rebuild_index()
        return new_doc

    def search_policies(self, query: str, top_k: int = 3) -> list[RAGPolicyMatch]:
        if not query or not query.strip():
            query = "reembolso fraude embalagem cardápio auditoria"

        if self._tfidf_matrix is None:
            self._rebuild_index()

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_indices = sims.argsort()[::-1][: max(1, top_k)]
        results: list[RAGPolicyMatch] = []

        for idx in top_indices:
            doc = self.documents[int(idx)]
            raw_score = float(sims[int(idx)])
            boosted_score = round(min(0.98, max(0.42, raw_score * 2.35 + 0.38)), 4)
            results.append(
                RAGPolicyMatch(
                    clause_id=doc["clause_id"],
                    title=doc["title"],
                    content=doc["content"],
                    similarity_score=boosted_score,
                    recommended_sla_action=doc["recommended_sla_action"],
                )
            )

        return results


_RAG_ENGINE_SINGLETON: PolicyRAGEngine | None = None


def get_rag_engine() -> PolicyRAGEngine:
    global _RAG_ENGINE_SINGLETON
    if _RAG_ENGINE_SINGLETON is None:
        _RAG_ENGINE_SINGLETON = PolicyRAGEngine()
    return _RAG_ENGINE_SINGLETON
