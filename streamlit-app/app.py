"""
RAG Patterns Explorer — Interactive Streamlit App

Showcases 4 RAG patterns with a step-by-step visual pipeline:
  1. Naive RAG     — pure vector similarity
  2. Hybrid RAG    — vector + BM25 + metadata filter + reranking
  3. GraphRAG      — knowledge graph traversal
  4. AgenticRAG    — ReAct loop with tool use
"""

import os, json, ast, operator, time
import streamlit as st
import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
#  Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Patterns Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
#  Custom CSS — built on top of Streamlit's dark base theme
#  (theme colours set in .streamlit/config.toml)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }

/* ── App background gradient ── */
.stApp {
    background: radial-gradient(ellipse at top left, #130d2e 0%, #0a0a18 50%, #071420 100%);
}

/* ── Sidebar — clearly differentiated from main bg ── */
section[data-testid="stSidebar"] {
    background-color: #1a1a3e !important;
    border-right: 2px solid rgba(129,140,248,0.3) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(99,102,241,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8b9dc3 !important;
    border-bottom: none !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.18) !important;
    color: #c4b5fd !important;
}

/* ── Inputs — ensure readable text on dark bg ── */
input, textarea, select,
.stTextInput input,
.stTextArea textarea {
    color: #f1f5f9 !important;
    background-color: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
input::placeholder, textarea::placeholder { color: #4b5563 !important; }
input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}

/* ── Selectbox dropdown ── */
[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
details summary {
    color: #a5b4fc !important;
    font-size: 13px !important;
}

/* ─────────────────────────────────────────────
   Custom HTML component styles
───────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(139,92,246,0.07));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-desc { color: #94a3b8; font-size: 14px; line-height: 1.65; margin: 0; }

.pattern-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.badge-naive  { background: rgba(59,130,246,0.15);  color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.badge-hybrid { background: rgba(139,92,246,0.15); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }
.badge-graph  { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.badge-agent  { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }

/* Pipeline */
.pipeline-wrap {
    display: flex; align-items: center; gap: 0;
    overflow-x: auto; padding: 14px 0 6px; scrollbar-width: none;
}
.pipeline-wrap::-webkit-scrollbar { display: none; }
.p-step {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 9px;
    padding: 8px 15px;
    font-size: 12px; font-weight: 600;
    white-space: nowrap; flex-shrink: 0;
}
.p-step.naive  { color: #93c5fd; border-color: rgba(59,130,246,0.4);  background: rgba(59,130,246,0.1); }
.p-step.hybrid { color: #c4b5fd; border-color: rgba(139,92,246,0.4); background: rgba(139,92,246,0.1); }
.p-step.graph  { color: #6ee7b7; border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.1); }
.p-step.agent  { color: #fcd34d; border-color: rgba(245,158,11,0.4);  background: rgba(245,158,11,0.1); }
.p-arrow { color: #4a5568; font-size: 18px; padding: 0 5px; flex-shrink: 0; }

/* Result cards */
.result-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 22px;
    margin-top: 18px;
}
.step-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 600; color: #cbd5e1;
    margin: 14px 0 8px;
}
.doc-card {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #3b82f6;
    border-radius: 0 9px 9px 0;
    padding: 11px 15px; margin: 6px 0;
    font-size: 13px; color: #e2e8f0; line-height: 1.6;
}
.doc-card.hybrid { border-left-color: #8b5cf6; }
.doc-card.graph  { border-left-color: #10b981; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #d1fae5; }
.doc-card.agent  { border-left-color: #f59e0b; }
.doc-meta { font-size: 11px; color: #94a3b8; margin-top: 5px; }

.answer-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.05));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 18px 22px; margin-top: 14px;
    color: #f1f5f9; font-size: 15px; line-height: 1.75;
}
.answer-label {
    font-size: 10px; font-weight: 700; color: #818cf8;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}
.agent-tool-call {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 9px; padding: 10px 14px; margin: 7px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #fcd34d;
}
.agent-tool-result {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 9px; padding: 10px 14px; margin: 3px 0 10px;
    font-size: 13px; color: #a7f3d0;
}
.no-api-warning {
    background: rgba(245,158,11,0.09);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px; padding: 12px 16px;
    color: #fde68a; font-size: 13px; margin: 10px 0;
}
.sidebar-logo {
    font-size: 26px; font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.section-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #8b9dc3; margin: 18px 0 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  Unified knowledge base (shared across all patterns)
# ─────────────────────────────────────────────────────────────────
DOCS = [
    {"id": "d0",  "text": "OpenAI was founded in 2015 as a nonprofit AI research company. In 2019 it became 'capped-profit'. It created GPT-4, DALL-E 3, Sora, and Whisper. Microsoft has invested approximately $13 billion.", "metadata": {"category": "company", "year": 2015}},
    {"id": "d1",  "text": "GPT-4 is OpenAI's flagship large language model, released in March 2023. It significantly outperforms GPT-3.5 on reasoning, coding, and instruction following. Available via API and in ChatGPT.", "metadata": {"category": "model", "year": 2023}},
    {"id": "d2",  "text": "Anthropic was co-founded in 2021 by Dario Amodei and former OpenAI colleagues. It created the Claude model family. Google invested ~$300M. The company focuses on AI safety and Constitutional AI.", "metadata": {"category": "company", "year": 2021}},
    {"id": "d3",  "text": "Claude 3 is Anthropic's model family (Haiku, Sonnet, Opus) released in March 2024. Opus outperforms GPT-4 on many benchmarks. Known for 200K token context and reduced hallucinations.", "metadata": {"category": "model", "year": 2024}},
    {"id": "d4",  "text": "Google DeepMind was formed by merging Google Brain and DeepMind in 2023. It created Gemini, AlphaFold (protein structure), and AlphaCode. Led by CEO Demis Hassabis.", "metadata": {"category": "company", "year": 2023}},
    {"id": "d5",  "text": "Gemini is Google DeepMind's multimodal LLM released December 2023. Comes in Ultra, Pro, and Nano variants. Ultra surpasses GPT-4 on MMLU. Powers Google products including Workspace and Search.", "metadata": {"category": "model", "year": 2023}},
    {"id": "d6",  "text": "Microsoft invested $10B in OpenAI in 2023 and integrated GPT models across Azure OpenAI Service, Bing Copilot, Office 365, and GitHub Copilot. This is their primary AI strategy.", "metadata": {"category": "investment", "year": 2023}},
    {"id": "d7",  "text": "The transformer architecture was introduced in 'Attention Is All You Need' (2017) by Vaswani et al. at Google. It is the foundation of all modern LLMs including GPT, Claude, and Gemini.", "metadata": {"category": "research", "year": 2017}},
    {"id": "d8",  "text": "Meta AI open-sourced LLaMA (2023) and Llama 3 (2024). These models powered the open-source AI ecosystem. Llama 3.1 405B competes with GPT-4 class models.", "metadata": {"category": "model", "year": 2024}},
    {"id": "d9",  "text": "RAG (Retrieval-Augmented Generation) was introduced by Lewis et al. at Facebook AI Research in 2020. It combines LLM generation with non-parametric retrieval to reduce hallucinations.", "metadata": {"category": "research", "year": 2020}},
    {"id": "d10", "text": "Sam Altman is the CEO of OpenAI. He was briefly ousted in November 2023 before being reinstated within days. He was president of Y Combinator from 2014 to 2019.", "metadata": {"category": "person", "year": 2023}},
    {"id": "d11", "text": "Mistral AI is a French startup founded in 2023. It released Mistral 7B and Mixtral 8x7B (mixture-of-experts) as open-weight models, becoming a leading open-source alternative.", "metadata": {"category": "company", "year": 2023}},
]

# Knowledge graph for GraphRAG
GRAPH_ENTITIES = [
    ("OpenAI",        {"type": "company", "desc": "Created GPT-4, ChatGPT, DALL-E, Sora"}),
    ("Anthropic",     {"type": "company", "desc": "Created Claude model family, AI safety focus"}),
    ("Google DeepMind",{"type":"company", "desc": "Created Gemini, AlphaFold; formed by merging Brain + DeepMind"}),
    ("Meta AI",       {"type": "company", "desc": "Open-sourced LLaMA 2 and Llama 3 models"}),
    ("Microsoft",     {"type": "company", "desc": "Azure, GitHub, Office; invested $13B in OpenAI"}),
    ("Mistral AI",    {"type": "company", "desc": "French startup, open-weight Mixtral models"}),
    ("Sam Altman",    {"type": "person",  "desc": "CEO of OpenAI, ex-president of Y Combinator"}),
    ("Dario Amodei",  {"type": "person",  "desc": "CEO of Anthropic, former VP Research at OpenAI"}),
    ("Demis Hassabis",{"type": "person",  "desc": "CEO of Google DeepMind, co-founder of DeepMind"}),
    ("GPT-4",         {"type": "model",   "desc": "OpenAI flagship LLM, released March 2023"}),
    ("Claude 3",      {"type": "model",   "desc": "Anthropic model family: Haiku/Sonnet/Opus, released 2024"}),
    ("Gemini",        {"type": "model",   "desc": "Google DeepMind multimodal LLM, released Dec 2023"}),
    ("LLaMA",         {"type": "model",   "desc": "Meta's open-weight model family"}),
    ("ChatGPT",       {"type": "product", "desc": "Consumer chat interface for GPT models by OpenAI"}),
    ("Azure OpenAI",  {"type": "product", "desc": "Microsoft enterprise API for OpenAI models"}),
    ("GitHub Copilot",{"type": "product", "desc": "AI coding assistant by GitHub/Microsoft using GPT-4"}),
]

GRAPH_EDGES = [
    ("Sam Altman",     "OpenAI",         "CEO_of",        "CEO since 2019 (briefly ousted Nov 2023)"),
    ("Dario Amodei",   "Anthropic",      "CEO_of",        "Co-founded and leads Anthropic"),
    ("Dario Amodei",   "OpenAI",         "formerly_at",   "VP of Research before leaving in 2021"),
    ("Demis Hassabis", "Google DeepMind","CEO_of",        "CEO since Google acquired DeepMind in 2014"),
    ("Microsoft",      "OpenAI",         "invested_in",   "~$13B investment across multiple rounds"),
    ("Google DeepMind","Anthropic",      "invested_in",   "~$300M strategic investment in 2023"),
    ("OpenAI",         "GPT-4",          "created",       "Released March 2023"),
    ("Anthropic",      "Claude 3",       "created",       "Released March 2024"),
    ("Google DeepMind","Gemini",         "created",       "Released December 2023"),
    ("Meta AI",        "LLaMA",          "created",       "LLaMA (2023), Llama 3 (2024) open-sourced"),
    ("OpenAI",         "ChatGPT",        "created",       "Launched Nov 2022, uses GPT models"),
    ("Microsoft",      "Azure OpenAI",   "created",       "Enterprise access to OpenAI models via Azure"),
    ("OpenAI",         "Azure OpenAI",   "partners_with", "OpenAI API available through Azure"),
    ("Microsoft",      "GitHub Copilot", "created",       "AI coding assistant built on OpenAI's Codex/GPT-4"),
]


# ─────────────────────────────────────────────────────────────────
#  Cached model & index loaders
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading reranker…")
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner="Building vector index…")
def build_vector_index():
    embedder = load_embedder()
    client = chromadb.Client()
    col = client.create_collection("rag_explorer")
    texts = [d["text"] for d in DOCS]
    col.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCS],
        metadatas=[d["metadata"] for d in DOCS]
    )
    return col

@st.cache_resource(show_spinner="Building BM25 index…")
def build_bm25():
    texts = [d["text"] for d in DOCS]
    return BM25Okapi([t.lower().split() for t in texts])

@st.cache_resource(show_spinner="Building knowledge graph…")
def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from(GRAPH_ENTITIES)
    for src, tgt, rel, detail in GRAPH_EDGES:
        G.add_edge(src, tgt, relation=rel, detail=detail)
    return G


# ─────────────────────────────────────────────────────────────────
#  LLM client
# ─────────────────────────────────────────────────────────────────

def get_llm_client(provider, api_key, model):
    """Returns an OpenAI-compatible client for the chosen provider."""
    if provider == "Groq":
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), model
    if provider == "Ollama":
        return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"), model
    return OpenAI(api_key=api_key), model

def llm_generate(prompt, provider, api_key, model):
    """Call the LLM and return the response string (or an error message)."""
    try:
        client, mdl = get_llm_client(provider, api_key, model)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": "You are a precise, factual assistant. Answer using ONLY the provided context. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"


# ─────────────────────────────────────────────────────────────────
#  Retrieval logic
# ─────────────────────────────────────────────────────────────────

def naive_retrieve(query, top_k=3):
    embedder = load_embedder()
    collection = build_vector_index()
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    ids = results["ids"][0]
    docs = [next(d for d in DOCS if d["id"] == id_) for id_ in ids]
    distances = results["distances"][0]
    return docs, distances


def hybrid_retrieve(query, category_filter=None, year_filter=None, top_k=3):
    embedder = load_embedder()
    reranker = load_reranker()
    collection = build_vector_index()
    bm25 = build_bm25()

    steps = {}

    # Stage 1: metadata filter
    allowed = [i for i, d in enumerate(DOCS)
               if (not category_filter or d["metadata"]["category"] == category_filter)
               and (not year_filter or d["metadata"]["year"] >= year_filter)]
    steps["filter"] = f"{len(allowed)} / {len(DOCS)} docs after filtering"

    if not allowed:
        return [], steps

    # Stage 2: vector search
    where = {}
    if category_filter: where["category"] = category_filter
    q_emb = embedder.encode([query]).tolist()
    vr = collection.query(
        query_embeddings=q_emb,
        n_results=min(6, len(allowed)),
        where=where if where else None
    )
    vector_ids = vr["ids"][0]
    steps["vector"] = f"Top vector IDs: {vector_ids[:4]}"

    # Stage 3: BM25 on allowed subset
    subset_texts = [DOCS[i]["text"].lower().split() for i in allowed]
    bm25_sub = BM25Okapi(subset_texts)
    bm25_scores = bm25_sub.get_scores(query.lower().split())
    top_bm25_local = sorted(range(len(allowed)), key=lambda i: bm25_scores[i], reverse=True)[:6]
    top_bm25_global = [allowed[i] for i in top_bm25_local]
    steps["bm25"] = f"Top BM25 doc indices: {[DOCS[i]['id'] for i in top_bm25_global[:4]]}"

    # Stage 4: RRF fusion
    scores = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(top_bm25_global):
        doc_id = DOCS[idx]["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (60 + rank + 1)
    fused_ids = sorted(scores, key=scores.get, reverse=True)[:6]
    steps["rrf"] = f"Fused ranking: {fused_ids}"

    # Stage 5: cross-encoder reranking
    candidates = [next(d for d in DOCS if d["id"] == id_) for id_ in fused_ids if any(d["id"] == id_ for d in DOCS)]
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    steps["rerank"] = f"Rerank scores: {[round(s, 3) for _, s in ranked[:top_k]]}"

    return [d for d, _ in ranked[:top_k]], steps


def graph_retrieve(query):
    G = build_graph()
    query_lower = query.lower()
    seeds = [node for node in G.nodes if node.lower() in query_lower]

    if not seeds:
        return "", [], []

    visited = set()
    lines = []

    def collect(node):
        if node in visited: return
        visited.add(node)
        data = G.nodes[node]
        lines.append(f"[{data['type'].upper()}] {node}: {data['desc']}")
        for _, tgt, ed in G.out_edges(node, data=True):
            lines.append(f"  → {node} --[{ed['relation']}]--> {tgt}: {ed['detail']}")
        for src, _, ed in G.in_edges(node, data=True):
            lines.append(f"  ← {src} --[{ed['relation']}]--> {node}: {ed['detail']}")

    for e in seeds:
        collect(e)
        for n in list(G.successors(e)) + list(G.predecessors(e)):
            collect(n)

    return "\n".join(lines), seeds, lines


# Agent tools
def agent_search_docs(query):
    embedder = load_embedder()
    collection = build_vector_index()
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=2)
    return "\n---\n".join(results["documents"][0])

AGENT_ENTITIES = {
    "openai": "OpenAI: Founded 2015, created GPT-4, ChatGPT, DALL-E. Microsoft invested ~$13B.",
    "anthropic": "Anthropic: Founded 2021 by Dario Amodei. Created Claude 3. Google invested ~$300M.",
    "google deepmind": "Google DeepMind: Created Gemini (Dec 2023), AlphaFold. CEO: Demis Hassabis.",
    "meta ai": "Meta AI: Open-sourced LLaMA 2 (2023) and Llama 3 (2024). Competes with GPT-4 class models.",
    "microsoft": "Microsoft: Azure OpenAI Service, GitHub Copilot, Bing. Invested $13B in OpenAI.",
    "sam altman": "Sam Altman: CEO of OpenAI. Briefly ousted Nov 2023. Former YC president (2014-2019).",
    "gpt-4": "GPT-4: OpenAI's flagship LLM. Released March 2023. Powers ChatGPT Plus and Azure OpenAI.",
    "claude": "Claude 3: Anthropic's model family. Opus/Sonnet/Haiku variants. 200K context. 2024.",
    "gemini": "Gemini: Google DeepMind's multimodal LLM. Ultra/Pro/Nano. Released December 2023.",
}

def agent_get_entity(name):
    result = AGENT_ENTITIES.get(name.lower().strip())
    return result or f"No entity found for '{name}'. Available: {list(AGENT_ENTITIES.keys())}"

def agent_calculate(expression):
    allowed = set("0123456789 +-*/().")
    if not all(c in allowed for c in expression):
        return "Error: only arithmetic allowed."
    try:
        ops = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}
        def ev(node):
            if isinstance(node, ast.Constant): return node.value
            if isinstance(node, ast.BinOp): return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp): return ops[type(node.op)](ev(node.operand))
            raise ValueError()
        result = ev(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result:.4f}"
    except Exception as e:
        return f"Error: {e}"

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the knowledge base for information about AI companies, models, or research.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "get_entity",
        "description": "Look up a specific named entity (OpenAI, Anthropic, Google DeepMind, Meta AI, Microsoft, Sam Altman, GPT-4, Claude, Gemini).",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a math expression for numeric reasoning.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    }},
]

def agent_dispatch(name, args):
    if name == "search_docs": return agent_search_docs(args["query"])
    if name == "get_entity":  return agent_get_entity(args["name"])
    if name == "calculate":   return agent_calculate(args["expression"])
    return f"Unknown tool: {name}"

def run_agent(question, provider, api_key, model, max_iter=5):
    """Run the ReAct agent loop and return (answer, list_of_steps)."""
    tool_steps = []
    try:
        client, mdl = get_llm_client(provider, api_key, model)
        messages = [
            {"role": "system", "content": "You are a research assistant for AI industry questions. Use your tools to gather facts, then synthesize a clear answer. For comparisons, look up each entity separately."},
            {"role": "user", "content": question}
        ]
        for _ in range(max_iter):
            resp = client.chat.completions.create(
                model=mdl, messages=messages,
                tools=AGENT_TOOLS, tool_choice="auto"
            )
            msg = resp.choices[0].message
            if resp.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = agent_dispatch(name, args)
                    tool_steps.append({"tool": name, "args": args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                return msg.content, tool_steps
        return "Agent reached max iterations.", tool_steps
    except Exception as e:
        return f"⚠️ Error: {str(e)}", tool_steps


# ─────────────────────────────────────────────────────────────────
#  HTML helpers
# ─────────────────────────────────────────────────────────────────

def pipeline_html(steps, pattern_class):
    """Render a horizontal pipeline with connected boxes."""
    parts = []
    for i, label in enumerate(steps):
        parts.append(f'<div class="p-step {pattern_class}">{label}</div>')
        if i < len(steps) - 1:
            parts.append('<span class="p-arrow">→</span>')
    return f'<div class="pipeline-wrap">{"".join(parts)}</div>'

def doc_card_html(text, meta=None, style_class=""):
    meta_html = f'<div class="doc-meta">{meta}</div>' if meta else ""
    return f'<div class="doc-card {style_class}">{text}{meta_html}</div>'

def answer_html(text):
    return f'<div class="answer-box"><div class="answer-label">💬 Answer</div>{text}</div>'

def step_header_html(icon, label):
    return f'<div class="step-header"><span class="step-icon">{icon}</span>{label}</div>'


# ─────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 24px 0 16px 0">
        <div class="sidebar-logo">🧠 RAG Explorer</div>
        <div class="sidebar-sub">Interactive pattern showcase</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">LLM Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox("Provider", ["OpenAI", "Groq", "Ollama"], label_visibility="collapsed")

    default_models = {"OpenAI": "gpt-4o-mini", "Groq": "llama-3.1-8b-instant", "Ollama": "llama3.2"}
    api_key_label = "API Key" if provider != "Ollama" else "Ollama URL (unused)"
    api_key = st.text_input(
        api_key_label,
        value=os.getenv("OPENAI_API_KEY", "") if provider == "OpenAI"
              else os.getenv("GROQ_API_KEY", "") if provider == "Groq" else "ollama",
        type="password",
        label_visibility="collapsed",
        placeholder=f"Paste {provider} API key…"
    )
    model = st.text_input("Model", value=default_models[provider], label_visibility="collapsed")

    has_key = bool(api_key and api_key not in ("sk-...", "gsk_...", ""))

    if has_key:
        st.markdown('<div style="color: #34d399; font-size: 12px; margin: 8px 0;">● LLM ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #f87171; font-size: 12px; margin: 8px 0;">● No API key — retrieval only</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin: 20px 0"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 12px; color: #94a3b8; line-height: 1.8">
        <b style="color:#c4cde8">4 Patterns covered:</b><br>
        🔵 Naive RAG — vector similarity<br>
        🟣 Hybrid RAG — vector + BM25 + rerank<br>
        🟢 GraphRAG — entity graph traversal<br>
        🟠 AgenticRAG — ReAct + tools
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  Main content — tabs per pattern
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="hero-title">RAG Patterns Explorer</div>
    <div class="hero-desc">
        Pick a pattern, ask a question, and watch the retrieval pipeline run step by step.
        Each tab shows a different retrieval strategy — same question, different approach.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔵  Naive RAG", "🟣  Hybrid RAG", "🟢  GraphRAG", "🟠  Agentic RAG"])

# ── Helpers for example query buttons ────────────────────────────
def example_buttons(examples, key_prefix):
    """Render example query buttons and return selected query or None."""
    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        if col.button(ex, key=f"{key_prefix}_{i}"):
            return ex
    return None


# ══════════════════════════════════════════════════════════════════
#  TAB 1 — Naive RAG
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="pattern-badge badge-naive">Naive RAG</span>', unsafe_allow_html=True)
    st.markdown("**Pure vector similarity** — embed the query, find the closest chunks, generate an answer. The simplest possible RAG pipeline.")

    st.markdown(pipeline_html(
        ["Embed Query", "ChromaDB Search", "Top-K Docs", "LLM Generate", "Answer"],
        "naive"
    ), unsafe_allow_html=True)

    st.markdown("**Try an example:**")
    NAIVE_EXAMPLES = ["Who founded OpenAI?", "What is the transformer architecture?", "Which models did Meta release?"]
    selected = example_buttons(NAIVE_EXAMPLES, "naive")

    query_naive = st.text_input("Your question", value=selected or "", placeholder="Ask anything about AI companies or models…", key="q_naive")

    if st.button("▶  Run Naive RAG", key="run_naive"):
        if not query_naive.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving…"):
                docs, distances = naive_retrieve(query_naive)

            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown(step_header_html("✅", f"Retrieved {len(docs)} documents"), unsafe_allow_html=True)

            for doc, dist in zip(docs, distances):
                similarity = round(1 - dist, 3) if dist <= 1 else round(1 / (1 + dist), 3)
                meta = f"Category: {doc['metadata']['category']} | Year: {doc['metadata']['year']} | Similarity: {similarity}"
                st.markdown(doc_card_html(doc["text"], meta, ""), unsafe_allow_html=True)

            if has_key:
                context = "\n".join(f"{i+1}. {d['text']}" for i, d in enumerate(docs))
                with st.spinner("Generating answer…"):
                    answer = llm_generate(f"Context:\n{context}\n\nQuestion: {query_naive}", provider, api_key, model)
                st.markdown(answer_html(answer), unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-api-warning">🔑 Add an API key in the sidebar to see the LLM-generated answer.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — Hybrid RAG
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="pattern-badge badge-hybrid">Hybrid RAG</span>', unsafe_allow_html=True)
    st.markdown("**Vector + BM25 + metadata filtering + cross-encoder reranking.** Each layer fixes a different failure mode of naive RAG.")

    st.markdown(pipeline_html(
        ["Metadata Filter", "Vector Search", "BM25 Search", "RRF Fusion", "Rerank", "LLM Generate", "Answer"],
        "hybrid"
    ), unsafe_allow_html=True)

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        cat_options = ["(none)", "company", "model", "research", "investment", "person"]
        cat_filter = st.selectbox("Category filter", cat_options)
        cat_filter = None if cat_filter == "(none)" else cat_filter
    with col_filter2:
        year_options = {"(none)": None, "After 2020": 2020, "After 2022": 2022, "After 2023": 2023}
        year_label = st.selectbox("Year filter", list(year_options.keys()))
        year_filter = year_options[year_label]

    st.markdown("**Try an example:**")
    HYBRID_EXAMPLES = ["Latest AI models in 2024", "What research introduced transformers?", "Open source model from Meta"]
    selected_h = example_buttons(HYBRID_EXAMPLES, "hybrid")

    query_hybrid = st.text_input("Your question", value=selected_h or "", placeholder="Ask with optional category/year filters…", key="q_hybrid")

    if st.button("▶  Run Hybrid RAG", key="run_hybrid"):
        if not query_hybrid.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running hybrid pipeline…"):
                docs, steps = hybrid_retrieve(query_hybrid, cat_filter, year_filter)

            st.markdown('<div class="result-section">', unsafe_allow_html=True)

            with st.expander("🔍 Pipeline internals", expanded=False):
                for stage, detail in steps.items():
                    st.markdown(f"**{stage}:** `{detail}`")

            if not docs:
                st.warning("No documents matched the filters. Try relaxing the category or year filter.")
            else:
                st.markdown(step_header_html("✅", f"Final {len(docs)} docs after reranking"), unsafe_allow_html=True)
                for doc in docs:
                    meta = f"Category: {doc['metadata']['category']} | Year: {doc['metadata']['year']}"
                    st.markdown(doc_card_html(doc["text"], meta, "hybrid"), unsafe_allow_html=True)

                if has_key:
                    context = "\n".join(f"{i+1}. {d['text']}" for i, d in enumerate(docs))
                    with st.spinner("Generating answer…"):
                        answer = llm_generate(f"Context:\n{context}\n\nQuestion: {query_hybrid}", provider, api_key, model)
                    st.markdown(answer_html(answer), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-api-warning">🔑 Add an API key to generate the final answer.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 3 — GraphRAG
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="pattern-badge badge-graph">GraphRAG</span>', unsafe_allow_html=True)
    st.markdown("**Knowledge graph traversal.** Extract entities from the query, walk the graph, collect relationship context. Best for 'who', 'what invested in', 'how are X and Y connected' questions.")

    st.markdown(pipeline_html(
        ["Extract Entities", "Find Nodes", "Traverse Edges", "Subgraph Context", "LLM Generate", "Answer"],
        "graph"
    ), unsafe_allow_html=True)

    st.markdown("**Try an example:**")
    GRAPH_EXAMPLES = ["What has Microsoft invested in?", "Who leads Google DeepMind?", "What did Dario Amodei do before Anthropic?"]
    selected_g = example_buttons(GRAPH_EXAMPLES, "graph")

    query_graph = st.text_input("Your question", value=selected_g or "", placeholder="Ask about companies, people, models, or their relationships…", key="q_graph")

    if st.button("▶  Run GraphRAG", key="run_graph"):
        if not query_graph.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Traversing knowledge graph…"):
                context_str, seeds, lines = graph_retrieve(query_graph)

            st.markdown('<div class="result-section">', unsafe_allow_html=True)

            if not seeds:
                st.warning("No entities found in the graph for this query. Try mentioning a company, person, or model by name (e.g., 'OpenAI', 'Sam Altman', 'GPT-4').")
            else:
                st.markdown(step_header_html("🔍", f"Entities found: **{', '.join(seeds)}**"), unsafe_allow_html=True)
                st.markdown(step_header_html("🕸️", f"Traversed {len(lines)} graph facts"), unsafe_allow_html=True)

                with st.expander("📊 Subgraph context", expanded=True):
                    for line in lines[:20]:
                        st.markdown(doc_card_html(line, style_class="graph"), unsafe_allow_html=True)
                    if len(lines) > 20:
                        st.caption(f"… and {len(lines) - 20} more facts")

                if has_key:
                    with st.spinner("Generating answer from graph context…"):
                        answer = llm_generate(
                            f"Knowledge Graph Context:\n{context_str}\n\nQuestion: {query_graph}",
                            provider, api_key, model
                        )
                    st.markdown(answer_html(answer), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-api-warning">🔑 Add an API key to generate the final answer.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 4 — Agentic RAG
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="pattern-badge badge-agent">Agentic RAG</span>', unsafe_allow_html=True)
    st.markdown("**ReAct agent with tools.** The LLM plans its own retrieval — picking tools, observing results, iterating until it has enough to answer. Best for multi-step or ambiguous questions.")

    st.markdown(pipeline_html(
        ["Question", "Agent Thinks", "Tool Call", "Observe", "Iterate?", "Final Answer"],
        "agent"
    ), unsafe_allow_html=True)

    st.markdown('<div style="font-size:12px; color: #94a3b8; margin: 4px 0 16px">Available tools: <code>search_docs</code> · <code>get_entity</code> · <code>calculate</code></div>', unsafe_allow_html=True)

    st.markdown("**Try an example:**")
    AGENT_EXAMPLES = ["Compare OpenAI and Anthropic", "Who leads the major AI labs?", "What models did Google release and when?"]
    selected_a = example_buttons(AGENT_EXAMPLES, "agent")

    query_agent = st.text_input("Your question", value=selected_a or "", placeholder="Ask a multi-step question — the agent will plan its own retrieval…", key="q_agent")

    if st.button("▶  Run Agentic RAG", key="run_agent"):
        if not query_agent.strip():
            st.warning("Enter a question first.")
        elif not has_key and provider != "Ollama":
            st.error("Agentic RAG requires an LLM API key — the agent loop needs to call the LLM to plan tool use.")
        else:
            with st.spinner("Agent is thinking…"):
                answer, tool_steps = run_agent(query_agent, provider, api_key, model)

            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown(step_header_html("🤖", f"Agent used {len(tool_steps)} tool call(s)"), unsafe_allow_html=True)

            for i, step in enumerate(tool_steps, 1):
                st.markdown(f'<div class="agent-tool-call">Step {i}: {step["tool"]}({json.dumps(step["args"])})</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-tool-result">{step["result"][:300]}{"…" if len(step["result"]) > 300 else ""}</div>', unsafe_allow_html=True)

            st.markdown(answer_html(answer), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
