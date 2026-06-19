"""
Hybrid RAG — vector search + BM25 keyword search + metadata filtering + reranking.

The Problem with Naive RAG:
  Pure vector similarity misses exact keyword matches (names, codes, product IDs)
  and has no way to filter by structured metadata (date, source, category).

The Fix — three upgrades layered on top of each other:
  1. Metadata filtering  → narrow the search space before any ranking
  2. Hybrid search       → run vector search AND BM25 in parallel, fuse scores
  3. Reranking           → use a cross-encoder to re-score the fused candidates

This combination is what most production RAG systems use.
"""

import os
import math
from openai import OpenAI
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
#  LLM abstraction — same pattern as 01-naive-rag
# ─────────────────────────────────────────────────────────────────

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        ), "llama-3.1-8b-instant"
    if provider == "ollama":
        return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"), \
               os.getenv("OLLAMA_MODEL", "llama3.2")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────
#  Knowledge base — documents WITH metadata
# ─────────────────────────────────────────────────────────────────

# Each document now carries structured metadata alongside the text.
# This is the foundation for metadata filtering — you can pre-filter
# before running any expensive embedding or BM25 computation.
DOCUMENTS = [
    {
        "id": "doc_0",
        "text": "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. "
                "Neil Armstrong and Buzz Aldrin walked on the lunar surface.",
        "metadata": {"category": "missions", "year": 1969, "source": "nasa.gov"}
    },
    {
        "id": "doc_1",
        "text": "The James Webb Space Telescope (JWST) launched on December 25, 2021. "
                "It uses infrared imaging to observe early universe galaxies and exoplanet atmospheres.",
        "metadata": {"category": "telescopes", "year": 2021, "source": "nasa.gov"}
    },
    {
        "id": "doc_2",
        "text": "SpaceX's Falcon 9 is a partially reusable rocket. The first stage booster lands "
                "autonomously after launch and is refurbished for reuse.",
        "metadata": {"category": "rockets", "year": 2015, "source": "spacex.com"}
    },
    {
        "id": "doc_3",
        "text": "NASA's Artemis program aims to return humans to the Moon by 2026 and establish "
                "a sustainable lunar presence as a stepping stone to Mars.",
        "metadata": {"category": "missions", "year": 2022, "source": "nasa.gov"}
    },
    {
        "id": "doc_4",
        "text": "The Hubble Space Telescope has been operating since 1990, producing iconic images "
                "of nebulae and distant galaxies in visible and ultraviolet light.",
        "metadata": {"category": "telescopes", "year": 1990, "source": "nasa.gov"}
    },
    {
        "id": "doc_5",
        "text": "SpaceX's Starship is designed for full reusability — both the Super Heavy booster "
                "and the Starship upper stage return and land after flight.",
        "metadata": {"category": "rockets", "year": 2023, "source": "spacex.com"}
    },
    {
        "id": "doc_6",
        "text": "The Mars Perseverance Rover landed in Jezero Crater in February 2021. "
                "It is searching for signs of ancient microbial life and collecting rock samples.",
        "metadata": {"category": "missions", "year": 2021, "source": "nasa.gov"}
    },
    {
        "id": "doc_7",
        "text": "The Nancy Grace Roman Space Telescope is NASA's next flagship telescope, "
                "designed to survey wide fields of the sky for dark energy and exoplanets.",
        "metadata": {"category": "telescopes", "year": 2026, "source": "nasa.gov"}
    },
]


# ─────────────────────────────────────────────────────────────────
#  Build the two search indexes
# ─────────────────────────────────────────────────────────────────

# --- Embedding model (vector search) ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Cross-encoder (reranking) ---
# A cross-encoder takes (query, document) pairs and outputs a relevance score.
# It's slower than bi-encoders but far more accurate — used as a final step.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- ChromaDB (vector store) ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("space_hybrid")

texts = [d["text"] for d in DOCUMENTS]
doc_embeddings = embedding_model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=doc_embeddings,
    ids=[d["id"] for d in DOCUMENTS],
    metadatas=[d["metadata"] for d in DOCUMENTS]
)

# --- BM25 (keyword search) ---
# BM25 (Best Match 25) is a classic probabilistic ranking function.
# It scores documents based on term frequency and inverse document frequency.
# It excels at exact keyword matches that embeddings often miss.
tokenized_corpus = [doc.lower().split() for doc in texts]
bm25_index = BM25Okapi(tokenized_corpus)

print(f"✅ Indexed {len(DOCUMENTS)} documents (vector + BM25).\n")


# ─────────────────────────────────────────────────────────────────
#  Retrieval — 3-stage pipeline
# ─────────────────────────────────────────────────────────────────

def filter_by_metadata(category: str | None = None, min_year: int | None = None) -> list[int]:
    """
    Stage 1: Metadata filtering.

    Returns the indices of documents that match the given filters.
    Filtering BEFORE vector/keyword search narrows the corpus, making
    retrieval faster and more precise — especially important at scale.
    """
    indices = []
    for i, doc in enumerate(DOCUMENTS):
        meta = doc["metadata"]
        if category and meta["category"] != category:
            continue
        if min_year and meta["year"] < min_year:
            continue
        indices.append(i)
    return indices


def reciprocal_rank_fusion(
    vector_ids: list[str],
    bm25_indices: list[int],
    k: int = 60
) -> list[str]:
    """
    Stage 2b: Score fusion using Reciprocal Rank Fusion (RRF).

    RRF combines rankings from multiple retrieval systems without needing
    to normalize their raw scores. The formula is: score = 1 / (k + rank).
    Results ranked high in EITHER system get boosted — catching what the
    other system missed.
    """
    scores: dict[str, float] = {}

    # Score from vector ranking
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Score from BM25 ranking
    for rank, doc_idx in enumerate(bm25_indices):
        doc_id = DOCUMENTS[doc_idx]["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Sort by combined score, highest first
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_retrieve(
    query: str,
    category: str | None = None,
    min_year: int | None = None,
    top_k: int = 3,
    candidate_k: int = 6
) -> list[dict]:
    """
    Full hybrid retrieval:
      1. Filter by metadata  → narrow the candidate pool
      2. Vector search       → semantic similarity
      3. BM25 search         → keyword matching
      4. RRF fusion          → combine the two rankings
      5. Cross-encoder rerank → final high-precision scoring

    candidate_k: how many candidates to gather before reranking.
                 More candidates = more recall, but slower reranking.
    """

    # ── Stage 1: Metadata filter ──────────────────────────────────
    allowed_indices = filter_by_metadata(category=category, min_year=min_year)
    allowed_ids = {DOCUMENTS[i]["id"] for i in allowed_indices}
    allowed_texts = [DOCUMENTS[i]["text"] for i in allowed_indices]

    if not allowed_ids:
        print("  ⚠️  No documents match the metadata filters.")
        return []

    # ── Stage 2a: Vector search (within filtered set) ─────────────
    query_embedding = embedding_model.encode([query]).tolist()
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(candidate_k, len(allowed_ids)),
        where={"category": category} if category else None  # ChromaDB metadata filter
    )
    vector_ids = vector_results["documents"][0] and vector_results["ids"][0] or []
    # Keep only IDs that passed our metadata filter
    vector_ids = [id_ for id_ in vector_ids if id_ in allowed_ids]

    # ── Stage 2b: BM25 keyword search (within filtered set) ───────
    tokenized_query = query.lower().split()
    # Score all filtered documents with BM25
    bm25_filtered = BM25Okapi([DOCUMENTS[i]["text"].lower().split() for i in allowed_indices])
    bm25_scores = bm25_filtered.get_scores(tokenized_query)
    # Get indices sorted by score (within the filtered subset)
    top_bm25_local = sorted(range(len(allowed_indices)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k]
    # Map back to original DOCUMENTS indices
    top_bm25_global = [allowed_indices[i] for i in top_bm25_local]

    # ── Stage 3: RRF fusion ────────────────────────────────────────
    fused_ids = reciprocal_rank_fusion(vector_ids, top_bm25_global)[:candidate_k]

    # ── Stage 4: Cross-encoder reranking ──────────────────────────
    # Fetch the actual text for each candidate
    candidates = [
        next(d for d in DOCUMENTS if d["id"] == doc_id)
        for doc_id in fused_ids
        if any(d["id"] == doc_id for d in DOCUMENTS)
    ]

    # Cross-encoder scores each (query, document) pair independently
    # This is more accurate than embedding similarity but O(n) slower
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    # Sort candidates by reranker score, take top_k
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────
#  Generate
# ─────────────────────────────────────────────────────────────────

def generate(query: str, context_docs: list[dict]) -> str:
    client, model = get_llm()

    context = "\n".join(
        f"{i+1}. [{doc['metadata']['category']} | {doc['metadata']['year']}] {doc['text']}"
        for i, doc in enumerate(context_docs)
    )

    messages = [
        {
            "role": "system",
            "content": "You are a factual assistant. Answer using ONLY the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
#  Run example queries
# ─────────────────────────────────────────────────────────────────

examples = [
    {
        "query": "Tell me about reusable rockets",
        "filters": {"category": "rockets"},          # only look at rocket documents
        "description": "Metadata filter: rockets only"
    },
    {
        "query": "What telescope launched most recently?",
        "filters": {"category": "telescopes", "min_year": 2020},  # telescopes after 2020
        "description": "Metadata filter: telescopes after 2020"
    },
    {
        "query": "NASA Mars mission 2021",
        "filters": {"category": "missions"},
        "description": "BM25 shines: exact keywords (NASA, Mars, 2021)"
    },
]

for ex in examples:
    print(f"❓ {ex['query']}")
    print(f"   Filter: {ex['description']}")

    docs = hybrid_retrieve(
        query=ex["query"],
        category=ex["filters"].get("category"),
        min_year=ex["filters"].get("min_year"),
    )

    print(f"   Retrieved {len(docs)} docs after reranking:")
    for doc in docs:
        print(f"     • [{doc['metadata']['category']} | {doc['metadata']['year']}] {doc['text'][:80]}...")

    answer = generate(ex["query"], docs)
    print(f"   💬 {answer}\n")
