# 🍔 iFood Sentinel & MenuOps AI Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit Portal](https://img.shields.io/badge/Streamlit-iFood_Portal_UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI Microservice](https://img.shields.io/badge/FastAPI-REST_%26_Webhooks-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn ML](https://img.shields.io/badge/Scikit--Learn-RF_%2B_IsolationForest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![n8n Workflow MCP](https://img.shields.io/badge/n8n-MCP_Workflow_Automation-EA4B71?style=for-the-badge&logo=n8n&logoColor=white)](https://n8n.io/)

> **Central Inteligente Multi-Agente (GenAI + Machine Learning Clássico + RAG + Automação n8n via MCP)** construída para o Portfólio do **Programa de Estágio GenAI do iFood** (Trilhas **AI Engineer** e **AI Builder**).

---

## 💡 O Problema que o Sistema Resolve

Todos os dias, plataformas de delivery recebem milhares de pedidos de reembolso e cadastros de cardápios com erros. Em vez de um analista humano levar **12 minutos** por chamado, **4 Agentes de IA trabalham juntos em menos de 2 segundos** para:

1. **👁️ Agente 1 (Visão Computacional):** Analisa a foto da embalagem ou do cardápio físico.
2. **📊 Agente 2 (Machine Learning — `Scikit-Learn`):** Executa um modelo supervisionado **Random Forest** (`ROC-AUC > 95%`) para medir a chance de fraude em reembolsos e um modelo **Isolation Forest** para travar preços errados em cardápios (ex: hambúrguer cadastrado a `R$ 450,00` por erro de digitação).
3. **📖 Agente 3 (Auditor de Regras — `RAG`):** Busca a cláusula exata no Manual de Políticas do iFood (`TF-IDF + Cosine Similarity`).
4. **⚡ Agente 4 (Executor `n8n Cloud`):** Dispara o veredito via Webhook para um workflow real criado via **MCP no n8n**, que bloqueia fraudes (`INC-P0-*`), escala casos moderados (`HITL-P1-*`) ou libera o estorno via PIX na hora (`AUTO-P2-*`).

---

## 🏗️ Arquitetura do Sistema

```mermaid
flowchart LR
    UI["🖥️ Interface Streamlit\n(app.py)"] --> ORCH["🧠 Orquestrador Multi-Agente\n(Python / FastAPI)"]
    ORCH --> A1["👁️ Agente 1\nVisão Computacional"]
    A1 --> A2["📊 Agente 2\nRandom Forest + Isolation Forest"]
    A2 --> A3["📖 Agente 3\nRAG Políticas iFood"]
    A3 --> A4["⚡ Agente 4\nWebhook n8n Cloud"]
    A4 <--> N8N["🔗 Workflow n8n Cloud\n(3 Rotas de Risco)"]
```

---

## 🚀 Como Rodar Localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para rodar os testes automatizados (`6/6 PASSED`):

```bash
pytest -v
```
