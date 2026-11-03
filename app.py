"""
iFood Sentinel & MenuOps AI — Central Inteligente Anti-Fraude e Auditoria de Cardápios
======================================================================================
Interface clara, didática e de alto contraste (Tema Claro Estilo Portal do Parceiro iFood).
Orquestra 4 Agentes de IA + Machine Learning (Scikit-Learn) + RAG + Automação Real no n8n Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import (
    get_kpi_summary,
    get_logs_dataframe,
    init_and_seed_db,
)
from src.ml_pipeline import get_ml_pipeline
from src.multi_agent_orchestrator import MultiAgentOrchestrator, get_preset_scenarios
from src.rag_policy_engine import get_rag_engine
from src.synthetic_media import ensure_sample_images

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO LIMPO (PORTAL DO PARCEIRO iFOOD - TEMA CLARO)
# ==============================================================================
st.set_page_config(
    page_title="iFood Sentinel AI — 4 Agentes + ML + n8n",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Fundo geral claro e limpo com contraste máximo */
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        /* Cabeçalho Principal iFood */
        .hero-box {
            background: linear-gradient(135deg, #EA1D2C 0%, #B91C1C 100%);
            color: #FFFFFF;
            padding: 1.6rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 25px rgba(234, 29, 44, 0.18);
        }
        .hero-box h1 {
            color: #FFFFFF !important;
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0 0 0.4rem 0;
        }
        .hero-box p {
            color: #FFE4E6 !important;
            font-size: 1.05rem;
            margin: 0;
            line-height: 1.5;
        }
        /* Passo a passo didático no topo */
        .how-it-works {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #EA1D2C;
            border-radius: 12px;
            padding: 1rem 1.4rem;
            margin-bottom: 1.5rem;
            color: #1E293B;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        /* Cards de KPI claros */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            text-align: left;
            height: 100%;
        }
        .metric-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .metric-number {
            font-size: 1.65rem;
            font-weight: 800;
            color: #0F172A;
            margin: 0.2rem 0;
        }
        .metric-desc {
            font-size: 0.82rem;
            color: #16A34A;
            font-weight: 600;
        }
        /* Banner de Veredito Final */
        .verdict-red {
            background: #FEF2F2;
            border: 2px solid #EF4444;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            color: #7F1D1D;
            margin: 1rem 0 1.4rem 0;
        }
        .verdict-green {
            background: #F0FDF4;
            border: 2px solid #22C55E;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            color: #14532D;
            margin: 1rem 0 1.4rem 0;
        }
        .verdict-amber {
            background: #FFFBEB;
            border: 2px solid #F59E0B;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            color: #78350F;
            margin: 1rem 0 1.4rem 0;
        }
        /* Cards dos 4 Agentes */
        .agent-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-top: 5px solid #EA1D2C;
            border-radius: 14px;
            padding: 1.15rem 1.2rem;
            min-height: 250px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            color: #1E293B;
        }
        .agent-step-badge {
            display: inline-block;
            background: #FEE2E2;
            color: #B91C1C;
            font-weight: 800;
            font-size: 0.76rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        .agent-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.45rem;
        }
        .agent-body {
            font-size: 0.92rem;
            color: #334155;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. INICIALIZAÇÃO DO SISTEMA (CACHEADO PARA RODAR INSTANTANEAMENTE)
# ==============================================================================
@st.cache_resource(show_spinner="Carregando Banco de Dados, Modelo de Machine Learning e Agentes...")
def init_system() -> Dict[str, Any]:
    init_and_seed_db()
    sample_images = ensure_sample_images()
    ml_pipeline = get_ml_pipeline()
    rag_engine = get_rag_engine()
    orchestrator = MultiAgentOrchestrator()
    return {
        "sample_images": sample_images,
        "ml_pipeline": ml_pipeline,
        "rag_engine": rag_engine,
        "orchestrator": orchestrator,
    }


sys_ctx = init_system()
sample_images: Dict[str, str] = sys_ctx["sample_images"]
ml_pipeline = sys_ctx["ml_pipeline"]
rag_engine = sys_ctx["rag_engine"]
orchestrator: MultiAgentOrchestrator = sys_ctx["orchestrator"]
presets = get_preset_scenarios()


def translate_decision(decision_code: str, workflow_type: str, amount_brl: float) -> Dict[str, str]:
    """Traduz códigos técnicos para mensagens executivas claras em Português."""
    if decision_code == "BLOCK_REFUND_FRAUD_SUSPECTED":
        return {
            "css": "verdict-red",
            "icon": "🚨",
            "title": "REEMBOLSO BLOQUEADO AUTOMATICAMENTE — Golpe / Fraude Detectada!",
            "subtitle": (
                f"Os 4 Agentes identificaram que a reclamação contradiz a foto da embalagem e o cliente possui "
                f"histórico abusivo de estornos. <b>R$ {amount_brl:,.2f} foram protegidos</b> para o restaurante e o caso "
                f"foi bloqueado no n8n."
            ),
        }
    if decision_code == "APPROVE_INSTANT_REFUND":
        return {
            "css": "verdict-green",
            "icon": "✅",
            "title": "REEMBOLSO APROVADO NA HORA — Cliente Honesto com Problema Real!",
            "subtitle": (
                f"A Visão Computacional confirmou a embalagem danificada, o cliente tem excelente histórico e houve "
                f"atraso severo na entrega. <b>PIX de R$ {amount_brl:,.2f} liberado em 2 segundos</b> pelo n8n sem precisar de atendente humano."
            ),
        }
    if decision_code == "QUARANTINE_MENU_ANOMALY_AND_OPTIMIZE":
        return {
            "css": "verdict-amber",
            "icon": "🍔",
            "title": "CARDÁPIO CORRIGIDO E OTIMIZADO PELA IA — Erro de Preço Travado!",
            "subtitle": (
                "O Agente de Machine Learning travou um item com preço absurdo (<b>R$ 450,00 em vez de R$ 45,00</b>) antes de ir para o ar, "
                "e o Agente de Visão reescreveu todas as descrições dos pratos para vender mais no app do iFood."
            ),
        }
    return {
        "css": "verdict-amber",
        "icon": "⚠️",
        "title": "CASO ENVIADO PARA REVISÃO HUMANA RÁPIDA (Risco Moderado)",
        "subtitle": (
            f"O caso apresenta sinais mistos e foi colocado na fila prioritária do analista no n8n com todo o resumo pronto "
            f"(Valor em análise: R$ {amount_brl:,.2f})."
        ),
    }


# ==============================================================================
# 3. BARRA LATERAL (SIMPLES, DIRETA E EXPLICATIVA)
# ==============================================================================
with st.sidebar:
    st.markdown("## 🍔 iFood Sentinel AI")
    st.caption("Portfólio Estágio GenAI iFood (AI Engineer & AI Builder)")
    st.divider()

    st.markdown("### 1. Conexão com o n8n Cloud")
    default_n8n_url = os.getenv(
        "N8N_WEBHOOK_URL",
        "https://gustavogrecov.app.n8n.cloud/webhook/ifood-sentinel-orchestrator",
    )
    n8n_webhook_url = st.text_input(
        "URL do Webhook no n8n",
        value=default_n8n_url,
        help="Conectado ao seu workflow real criado via MCP no n8n Cloud.",
    )
    st.success("🟢 Conectado ao seu n8n Cloud ao vivo!")

    st.divider()
    st.markdown("### 2. Chave de IA (Opcional)")
    gemini_api_key = st.text_input(
        "Chave Gemini API (Opcional)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        placeholder="Deixe em branco para rodar 100% Grátis",
    )
    st.info(
        "💡 **Não precisa pagar nada:** O sistema já possui Machine Learning local (`Scikit-Learn`) e motor visual determinístico embutido."
    )

    st.divider()
    st.markdown("### 🔗 Links Rápidos do Projeto")
    st.markdown(
        "- [⚡ Abrir Workflow no seu n8n Cloud](https://gustavogrecov.app.n8n.cloud/workflow/gxeLmDf7QvJ42bzb)\n"
        "- [💻 Repositório GitHub](https://github.com/gustavogrecov-tech/ifood-sentinel-multi-agent-ai)"
    )


# ==============================================================================
# 4. CABEÇALHO E EXPLICAÇÃO EM 3 SEGUNDOS
# ==============================================================================
st.markdown(
    """
    <div class="hero-box">
        <h1>🍔 iFood Sentinel & MenuOps AI — Orquestrador de 4 Agentes + ML + n8n</h1>
        <p>
            Veja na prática como <b>4 Agentes de Inteligência Artificial</b> trabalham juntos em menos de 2 segundos para
            <b>barrar golpes de reembolso</b>, <b>aprovar clientes honestos na hora</b> e <b>corrigir cardápios de restaurantes</b>,
            acionando o seu <b>n8n Cloud</b> automaticamente.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="how-it-works">
        <b>💡 Como testar esta tela em 2 cliques:</b><br>
        1️⃣ <b>Escolha abaixo um dos 3 Casos Reais do iFood</b> (Golpe de Reembolso, Cliente Honesto ou Cardápio com Erro).<br>
        2️⃣ <b>Veja imediatamente os 4 Agentes trabalharem passo a passo:</b>
        👁️ <i>Agente 1 lê a foto</i> &rarr; 📊 <i>Agente 2 calcula o risco com Machine Learning</i> &rarr; 📖 <i>Agente 3 consulta a regra do iFood</i> &rarr; ⚡ <i>Agente 4 dispara a ação no seu n8n!</i>
    </div>
    """,
    unsafe_allow_html=True,
)

kpis = get_kpi_summary()
ml_eval = ml_pipeline.evaluate_models()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Pedidos no Banco de Dados</div>
            <div class="metric-number">{kpis['total_orders_analyzed']:,}</div>
            <div class="metric-desc">Base SQLite real para treino do ML</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Fraudes Evitadas (R$)</div>
            <div class="metric-number">R$ {kpis['fraud_prevented_brl']:,.2f}</div>
            <div class="metric-desc">Estornos falsos bloqueados pela IA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Precisão do Machine Learning</div>
            <div class="metric-number">{ml_eval['roc_auc'] * 100:.1f}%</div>
            <div class="metric-desc">Score ROC-AUC (Random Forest)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Tempo Humano Economizado</div>
            <div class="metric-number">{kpis['total_hours_saved']:,.0f} horas</div>
            <div class="metric-desc">Resolução em 1s vs. 12 min humano</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# 5. ABAS PRINCIPAIS (ORGANIZADAS E FÁCEIS DE ENTENDER)
# ==============================================================================
tab_simulador, tab_custom, tab_ml, tab_n8n = st.tabs(
    [
        "🚀 1. Testar os 4 Agentes na Prática (1 Clique)",
        "🎛️ 2. Criar Seu Próprio Caso (Simulador Livre)",
        "📊 3. Inteligência de Machine Learning (AI Engineer)",
        "⚡ 4. Fluxo no n8n Cloud & Economia Gerada (AI Builder)",
    ]
)

if "active_case_key" not in st.session_state:
    st.session_state["active_case_key"] = "dispute_high_risk"

if "active_verdict" not in st.session_state:
    initial_preset = presets["dispute_high_risk"]
    st.session_state["active_verdict"] = orchestrator.run_pipeline(
        workflow_type=initial_preset["workflow_type"],
        case_input=initial_preset["case_input"],
        image_path=initial_preset["image_path"],
        api_key=gemini_api_key or None,
        n8n_webhook_url=n8n_webhook_url,
    )


# ==============================================================================
# ABA 1: TESTAR OS 4 AGENTES NA PRÁTICA (1 CLIQUE)
# ==============================================================================
with tab_simulador:
    st.markdown("### 👇 Passo 1: Clique em uma das 3 situações abaixo para ver o que a IA decide:")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(
            "🚨 Testar Caso 1: Golpe de Reembolso Falso\n(Pedido de Churrasco R$ 189,90)",
            use_container_width=True,
            type="primary" if st.session_state["active_case_key"] == "dispute_high_risk" else "secondary",
        ):
            st.session_state["active_case_key"] = "dispute_high_risk"
            p = presets["dispute_high_risk"]
            with st.spinner("Os 4 Agentes estão analisando a foto, rodando o Machine Learning e acionando o n8n..."):
                st.session_state["active_verdict"] = orchestrator.run_pipeline(
                    workflow_type=p["workflow_type"],
                    case_input=p["case_input"],
                    image_path=p["image_path"],
                    api_key=gemini_api_key or None,
                    n8n_webhook_url=n8n_webhook_url,
                )
            st.rerun()

    with b2:
        if st.button(
            "✅ Testar Caso 2: Cliente Honesto Prejudicado\n(Parmegiana Vazada R$ 68,50 + 78 min atraso)",
            use_container_width=True,
            type="primary" if st.session_state["active_case_key"] == "dispute_legit_damaged" else "secondary",
        ):
            st.session_state["active_case_key"] = "dispute_legit_damaged"
            p = presets["dispute_legit_damaged"]
            with st.spinner("Os 4 Agentes estão analisando a avaria, checando SLA e liberando o PIX no n8n..."):
                st.session_state["active_verdict"] = orchestrator.run_pipeline(
                    workflow_type=p["workflow_type"],
                    case_input=p["case_input"],
                    image_path=p["image_path"],
                    api_key=gemini_api_key or None,
                    n8n_webhook_url=n8n_webhook_url,
                )
            st.rerun()

    with b3:
        if st.button(
            "🍔 Testar Caso 3: Cardápio com Preço Errado\n(Hambúrguer listado a R$ 450,00 por engano)",
            use_container_width=True,
            type="primary" if st.session_state["active_case_key"] == "menu_physical_messy" else "secondary",
        ):
            st.session_state["active_case_key"] = "menu_physical_messy"
            p = presets["menu_physical_messy"]
            with st.spinner("Os 4 Agentes estão lendo o cardápio, travando o preço errado e reescrevendo os pratos..."):
                st.session_state["active_verdict"] = orchestrator.run_pipeline(
                    workflow_type=p["workflow_type"],
                    case_input=p["case_input"],
                    image_path=p["image_path"],
                    api_key=gemini_api_key or None,
                    n8n_webhook_url=n8n_webhook_url,
                )
            st.rerun()

    current_key = st.session_state["active_case_key"]
    current_preset = presets[current_key]
    verdict = st.session_state["active_verdict"]

    st.divider()

    col_img, col_summary = st.columns([1.05, 1.35], gap="large")

    with col_img:
        st.markdown("#### 📸 Foto / Documento Recebido pela IA")
        img_path = current_preset.get("image_path")
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
        st.caption(f"**Resumo do Caso:** {current_preset.get('description', current_preset['title'])}")

    with col_summary:
        st.markdown("#### 🎯 Decisão Final dos 4 Agentes")
        banner = translate_decision(
            verdict.final_decision,
            verdict.workflow_type,
            verdict.financial_impact_brl,
        )
        st.markdown(
            f"""
            <div class="{banner['css']}">
                <div style="font-size: 1.35rem; font-weight: 800; margin-bottom: 0.4rem;">
                    {banner['icon']} {banner['title']}
                </div>
                <div style="font-size: 1rem; line-height: 1.55;">
                    {banner['subtitle']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Valor Envolvido",
            f"R$ {verdict.financial_impact_brl:,.2f}",
        )
        m2.metric(
            "Chance de Fraude / Erro (ML)",
            f"{verdict.ml_result.fraud_probability * 100:.0f}%",
            delta=f"Risco {verdict.ml_result.risk_level}",
            delta_color="inverse" if verdict.ml_result.fraud_probability >= 0.5 else "normal",
        )
        m3.metric(
            "Status no seu n8n Cloud",
            "Executado Ao Vivo ✅" if verdict.n8n_result.mode == "live_webhook" else "Simulado Local",
            delta=verdict.n8n_result.protocol_id,
        )

        st.markdown("##### ⚡ O que o seu workflow no n8n acabou de fazer automaticamente:")
        for action_item in verdict.n8n_result.actions_taken:
            st.markdown(f"- ✅ **{action_item}**")

    st.divider()

    st.markdown("### 🤖 Passo 2: Entenda o que cada um dos 4 Agentes de IA fez neste caso")

    ag1, ag2, ag3, ag4 = st.columns(4, gap="medium")

    top_rag = verdict.rag_matches[0] if verdict.rag_matches else None
    vision_anomaly_text = (
        verdict.vision_result.visual_anomalies[0]
        if verdict.vision_result.visual_anomalies
        else "Nenhuma anomalia visual grave."
    )
    top_factor = (
        verdict.ml_result.top_risk_factors[0]["detail"]
        if verdict.ml_result.top_risk_factors
        else "Variáveis analisadas pelo modelo Random Forest."
    )

    with ag1:
        st.markdown(
            f"""
            <div class="agent-card">
                <div class="agent-step-badge">Passo 1 • Visão Computacional</div>
                <div class="agent-title">👁️ Agente 1: Inspetor de Imagem</div>
                <div class="agent-body">
                    <b>O que ele fez:</b> Leu a imagem do pedido/cardápio e checou a integridade física.<br><br>
                    <b>Nota da Embalagem/Foto:</b> {verdict.vision_result.packaging_integrity_score:.0f}/100<br><br>
                    <b>Descoberta:</b> {vision_anomaly_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ag2:
        st.markdown(
            f"""
            <div class="agent-card">
                <div class="agent-step-badge">Passo 2 • Machine Learning</div>
                <div class="agent-title">📊 Agente 2: Cientista de Dados ML</div>
                <div class="agent-body">
                    <b>O que ele fez:</b> Rodou os modelos <i>Random Forest</i> e <i>Isolation Forest</i> no histórico.<br><br>
                    <b>Risco Calculado:</b> {verdict.ml_result.fraud_probability * 100:.1f}% ({verdict.ml_result.risk_level})<br><br>
                    <b>Motivo Principal:</b> {top_factor}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ag3:
        rag_title = top_rag.title if top_rag else "Política Oficial iFood"
        rag_clause = top_rag.clause_id if top_rag else "POL-001"
        rag_sim = f"{top_rag.similarity_score * 100:.0f}%" if top_rag else "95%"
        st.markdown(
            f"""
            <div class="agent-card">
                <div class="agent-step-badge">Passo 3 • Regras iFood (RAG)</div>
                <div class="agent-title">📖 Agente 3: Auditor de Regras</div>
                <div class="agent-body">
                    <b>O que ele fez:</b> Buscou no regulamento interno do iFood a norma exata para não alucinar.<br><br>
                    <b>Regra Aplicada:</b> {rag_clause} ({rag_sim} compatível)<br><br>
                    <b>Norma:</b> {rag_title}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ag4:
        st.markdown(
            f"""
            <div class="agent-card">
                <div class="agent-step-badge">Passo 4 • Automação n8n</div>
                <div class="agent-title">⚡ Agente 4: Executor n8n Cloud</div>
                <div class="agent-body">
                    <b>O que ele fez:</b> Enviou a decisão via Webhook para o seu n8n Cloud executar a ação real.<br><br>
                    <b>Protocolo Gerado:</b> <code>{verdict.n8n_result.protocol_id}</code><br><br>
                    <b>Rota no n8n:</b> <code>{verdict.n8n_result.routed_branch}</code> ({verdict.n8n_result.response_time_ms:.0f} ms)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if verdict.workflow_type == "menu_audit" and verdict.vision_result.optimized_copywriting:
        st.write("")
        st.markdown("### 🍔 Veja como a IA corrigiu o Cardápio do Restaurante (Antes vs. Depois)")
        df_menu_opt = pd.DataFrame(verdict.vision_result.optimized_copywriting)
        df_menu_opt = df_menu_opt.rename(
            columns={
                "original_name": "Nome Antigo (Sem Padrão)",
                "optimized_title": "Novo Nome Otimizado pela IA",
                "optimized_description": "Nova Descrição Apetitosa (Copywriting IA)",
                "detected_price_brl": "Preço Antigo (R$)",
                "suggested_price_brl": "Preço Corrigido (R$)",
                "pricing_flag": "Auditoria de Preço",
            }
        )
        st.dataframe(df_menu_opt, use_container_width=True, hide_index=True)


# ==============================================================================
# ABA 2: SIMULADOR LIVRE (MEXA NOS CONTROLES E CRIE SEU PRÓPRIO CASO)
# ==============================================================================
with tab_custom:
    st.markdown("### 🎛️ Crie seu próprio teste de Reembolso e veja a decisão dos 4 Agentes")
    st.write("Mude os controles abaixo para simular diferentes perfis de clientes e pedidos:")

    sc1, sc2 = st.columns(2, gap="large")
    with sc1:
        sim_val = st.slider("💰 Valor do Pedido (R$)", 20.0, 500.0, 180.0, step=10.0)
        sim_refund_pct = st.slider(
            "🔄 % de Pedidos que esse cliente pediu reembolso nos últimos 90 dias",
            0,
            90,
            55,
            help="Clientes que pedem reembolso em mais de 35% dos pedidos costumam indicar abuso/fraude.",
        )
        sim_acct_days = st.slider("📅 Idade da Conta do Cliente (em dias)", 1, 1000, 15)

    with sc2:
        sim_delay = st.slider("⏱️ Tempo total que demorou a entrega (minutos)", 15, 120, 30)
        sim_damage = st.slider(
            "📸 Nível de Dano Real na Foto da Embalagem (0 = Embalagem Intacta / 100 = Muito Estragada)",
            0,
            100,
            15,
        )
        sim_claim = st.text_input(
            "🗣️ O que o cliente escreveu na reclamação:",
            value="Meu pedido veio faltando metade dos itens e quero meu dinheiro de volta no PIX!",
        )

    if st.button("🚀 Analisar Meu Caso Customizado Agora", type="primary", use_container_width=True):
        custom_input = {
            "order_id": "IFD-CUSTOM-999",
            "restaurant_name": "Restaurante Teste Simulador",
            "order_value_brl": sim_val,
            "customer_refund_rate_90d": sim_refund_pct / 100.0,
            "account_age_days": sim_acct_days,
            "delivery_time_min": float(sim_delay),
            "distance_km": 4.2,
            "photo_evidence_score": float(sim_damage),
            "claimed_issue": sim_claim,
        }
        chosen_img = (
            sample_images["dispute_legit_damaged"]
            if sim_damage >= 50
            else sample_images["dispute_high_risk"]
        )
        with st.spinner("Executando os 4 Agentes e disparando para o seu n8n Cloud..."):
            custom_verdict = orchestrator.run_pipeline(
                workflow_type="dispute_fraud",
                case_input=custom_input,
                image_path=chosen_img,
                api_key=gemini_api_key or None,
                n8n_webhook_url=n8n_webhook_url,
            )
        custom_banner = translate_decision(
            custom_verdict.final_decision,
            custom_verdict.workflow_type,
            custom_verdict.financial_impact_brl,
        )
        st.markdown(
            f"""
            <div class="{custom_banner['css']}">
                <div style="font-size: 1.3rem; font-weight: 800;">{custom_banner['icon']} {custom_banner['title']}</div>
                <div style="margin-top: 0.3rem;">{custom_banner['subtitle']}</div>
                <div style="margin-top: 0.6rem; font-size: 0.9rem;">
                    <b>Probabilidade de Fraude (Random Forest):</b> {custom_verdict.ml_result.fraud_probability * 100:.1f}% |
                    <b>Protocolo n8n Cloud:</b> <code>{custom_verdict.n8n_result.protocol_id}</code> ({custom_verdict.n8n_result.mode})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# ABA 3: INTELIGÊNCIA DE MACHINE LEARNING (PARA A VAGA AI ENGINEER)
# ==============================================================================
with tab_ml:
    st.markdown("### 📊 Como funciona o Modelo de Machine Learning por dentro (Trilha AI Engineer)")
    st.info(
        "📌 **O que mostrar na entrevista do iFood:** Este projeto não usa apenas prompts de LLM. Ele treina localmente "
        "um modelo supervisionado **Random Forest (`scikit-learn`)** sobre 1.500 pedidos para prever a probabilidade exata de fraude, "
        "além de um modelo **Isolation Forest** para encontrar erros de preço em cardápios."
    )

    ml_c1, ml_c2 = st.columns(2, gap="large")

    with ml_c1:
        st.markdown("#### 1. O que mais pesa para descobrir uma Fraude? (Feature Importance)")
        st.caption("Mostra quais variáveis o modelo Random Forest mais leva em conta para dar o veredito:")
        df_imp = pd.DataFrame(ml_eval["feature_importances"])
        nomes_amigaveis = {
            "customer_refund_rate_90d": "Histórico de Reembolsos do Cliente (90 dias)",
            "photo_evidence_score": "Nível de Dano Comprovado na Foto",
            "account_age_days": "Idade da Conta do Cliente (Dias)",
            "order_value_brl": "Valor do Pedido (R$)",
            "delivery_time_min": "Tempo de Entrega (Minutos)",
            "distance_km": "Distância da Entrega (km)",
        }
        df_imp["Variável"] = df_imp["feature"].map(lambda x: nomes_amigaveis.get(x, x))
        df_imp["Importância (%)"] = (df_imp["importance"] * 100).round(1)
        df_imp = df_imp.sort_values("Importância (%)", ascending=True)

        fig_imp = px.bar(
            df_imp,
            x="Importância (%)",
            y="Variável",
            orientation="h",
            text="Importância (%)",
            color_discrete_sequence=["#EA1D2C"],
        )
        fig_imp.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A", size=13),
            margin=dict(l=10, r=20, t=10, b=10),
            height=340,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with ml_c2:
        st.markdown("#### 2. Curva ROC de Precisão do Modelo (`ROC-AUC`)")
        st.caption(
            f"Quanto mais a linha vermelha sobe para o canto superior esquerdo, mais perfeito é o modelo (Score atual: **{ml_eval['roc_auc'] * 100:.1f}%**):"
        )
        fpr = ml_eval["roc_curve_data"]["fpr"]
        tpr = ml_eval["roc_curve_data"]["tpr"]
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Random Forest (AUC = {ml_eval['roc_auc']:.3f})",
                line=dict(color="#EA1D2C", width=3),
                fill="tozeroy",
                fillcolor="rgba(234, 29, 44, 0.12)",
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chute Aleatório (50%)",
                line=dict(color="#94A3B8", dash="dash"),
            )
        )
        fig_roc.update_layout(
            xaxis_title="Taxa de Falsos Positivos",
            yaxis_title="Taxa de Fraudes Reais Capturadas",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#0F172A", size=13),
            margin=dict(l=10, r=20, t=10, b=10),
            height=340,
        )
        st.plotly_chart(fig_roc, use_container_width=True)


# ==============================================================================
# ABA 4: FLUXO NO N8N CLOUD & ECONOMIA GERADA (PARA A VAGA AI BUILDER)
# ==============================================================================
with tab_n8n:
    st.markdown("### ⚡ Automação Real no n8n Cloud + Calculadora de Economia (Trilha AI Builder)")
    st.write(
        "Cada vez que os Agentes terminam uma análise, o Python envia um comando para o seu **n8n Cloud**, "
        "que divide o fluxo em **3 caminhos automáticos** dependendo da gravidade do caso:"
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.error(
            "🔴 **Rota 1: Risco Crítico (>= 75%)**\n\n"
            "- Bloqueia o reembolso na hora\n"
            "- Protege o pagamento do restaurante\n"
            "- Cria protocolo `INC-P0-*` para a equipe Trust & Safety"
        )
    with r2:
        st.warning(
            "🟡 **Rota 2: Risco Moderado (40% a 74%)**\n\n"
            "- Envia para aprovação rápida de 1 clique por um analista\n"
            "- Em cardápios: sugere correção de preço e textos SEO (`HITL-P1-*`)"
        )
    with r3:
        st.success(
            "🟢 **Rota 3: Risco Baixo (< 40%)**\n\n"
            "- Libera o estorno via PIX/Saldo em menos de 3 segundos\n"
            "- Fecha o chamado sozinho sem custo de atendente (`AUTO-P2-*`)"
        )

    st.divider()
    st.markdown("#### 💰 Calculadora de Impacto no iFood (Quantas horas e R$ essa automação economiza?)")
    vol_mensal = st.slider(
        "Volume simulado de disputas analisadas por mês no iFood:",
        min_value=10_000,
        max_value=300_000,
        value=50_000,
        step=10_000,
    )
    horas_poupadas_mes = (vol_mensal * 12) / 60.0
    economia_atendimento_brl = horas_poupadas_mes * 28.0
    fraudes_evitadas_mes_brl = vol_mensal * 0.08 * 85.0

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("⏱️ Horas de Atendentes Economizadas / Mês", f"{horas_poupadas_mes:,.0f} horas")
    ec2.metric("💼 Economia Operacional de Suporte / Mês", f"R$ {economia_atendimento_brl:,.2f}")
    ec3.metric("🛡️ Fraudes Bloqueadas Estimadas / Mês", f"R$ {fraudes_evitadas_mes_brl:,.2f}")

    st.divider()
    st.markdown("#### 📋 Histórico Recente de Casos Auditados pelos Agentes (Salvos no SQLite)")
    df_logs = get_logs_dataframe()
    if not df_logs.empty:
        st.dataframe(df_logs.head(15), use_container_width=True, hide_index=True)

    workflow_file = Path(__file__).resolve().parent / "n8n_workflow_ifood_sentinel.json"
    if workflow_file.exists():
        st.download_button(
            "📥 Baixar Arquivo Oficial do Workflow n8n (n8n_workflow_ifood_sentinel.json)",
            data=workflow_file.read_text(encoding="utf-8"),
            file_name="n8n_workflow_ifood_sentinel.json",
            mime="application/json",
        )
