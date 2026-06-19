# Hybrid RAG

Combines **vector search + BM25 keyword search + metadata filtering + cross-encoder reranking** for production-grade retrieval precision.

## The Problem with Naive RAG

Pure vector similarity works well for "what does this mean?" queries but fails on:
- **Exact matches** — product names, error codes, person names (BM25 handles these better)
- **Filtered retrieval** — "only from 2023 docs" or "category: legal" (needs metadata)
- **Noisy top-k** — the 3rd result might be irrelevant; reranking fixes this

## The Pattern

```
Query
  │
  ├─→ Metadata filter  ──→ narrow candidate pool
  │
  ├─→ Vector search    ──┐
  │                      ├─→ RRF fusion  ──→ Cross-encoder rerank  ──→ Top-k  ──→ LLM
  └─→ BM25 search     ──┘
```

### Stage 1: Metadata Filtering
Pre-filter the corpus by structured fields (date, source, category) **before** any embedding computation. This reduces noise and speeds up retrieval.

### Stage 2: Hybrid Search
Run **two retrieval systems in parallel**:
- **Vector search** (ChromaDB) — catches semantically similar content
- **BM25** (rank_bm25) — catches exact keyword matches that vectors miss

### Stage 3: RRF Fusion
**Reciprocal Rank Fusion** merges the two ranked lists without needing to normalize scores. Formula: `score = 1 / (k + rank)`. A result ranked high in either list gets a strong combined score.

### Stage 4: Cross-Encoder Reranking
A **cross-encoder** reads each `(query, document)` pair together (unlike bi-encoders that embed them separately) and outputs a precise relevance score. Slower, but dramatically more accurate as the final step.

## When to Use It

✅ Production systems where recall AND precision both matter  
✅ Corpora with structured metadata (dates, categories, sources)  
✅ Mixed query types — some semantic, some keyword-heavy  
✅ When naive RAG is missing relevant documents or returning irrelevant ones  

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## What to Look At in the Code

1. **`DOCUMENTS`** — each entry has `text` + `metadata`
2. **`filter_by_metadata()`** — Stage 1: pre-filtering
3. **`reciprocal_rank_fusion()`** — Stage 3: merging vector + BM25 rankings
4. **`hybrid_retrieve()`** — the full 4-stage pipeline end-to-end
5. The third example query — how BM25 outperforms vectors on exact keyword queries

## Dependencies

- `chromadb` — vector database
- `sentence-transformers` — bi-encoder for embeddings + cross-encoder for reranking
- `rank-bm25` — BM25 keyword search
- `openai` — LLM client
