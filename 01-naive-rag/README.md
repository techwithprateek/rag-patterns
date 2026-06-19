# Naive RAG

The simplest RAG pattern. Embed your documents, store them in a vector database, and retrieve by similarity at query time.

## The Pattern

```
Documents → chunk → embed → vector DB
Query     → embed → similarity search → top-k chunks → LLM → Answer
```

## When to Use It

✅ You're building a prototype or demo  
✅ Your corpus is small and clean (< ~50k chunks)  
✅ Semantic similarity is a good proxy for what you need (prose, narratives, Q&A)  

## When It Breaks Down

❌ **Wrong retrieval** — embeddings miss exact terms: names, codes, IDs  
❌ **No metadata awareness** — can't filter by date, source, or category  
❌ **No relevance ranking** — the 3rd-closest vector might be irrelevant noise  
❌ **Scales poorly** — full vector scan gets expensive with millions of chunks  

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API key
python main.py
```

## What to Look At in the Code

1. **Indexing** — how documents are embedded and stored in ChromaDB
2. **Retrieval** — the `retrieve()` function: just cosine similarity, nothing else
3. **Generation** — the system prompt that constrains the LLM to only use context
4. **"Who invented the telephone?"** — a query not in the knowledge base, showing graceful degradation

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Full pipeline: index → retrieve → generate |
| `requirements.txt` | Dependencies |
| `.env.example` | Environment variable template |

## Dependencies

- `chromadb` — local vector database
- `sentence-transformers` — local embedding model (no API key needed)
- `openai` — LLM client (also works with Groq/Ollama)
