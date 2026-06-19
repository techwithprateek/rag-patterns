# GraphRAG

Uses a **knowledge graph** as the retrieval backbone instead of a vector database.

## The Problem GraphRAG Solves

Vector search answers "what text is similar to my query?" — but some questions are fundamentally about **relationships**:

- *"Who invested in OpenAI?"* → Traversing edges is trivial; embedding similarity is fragile
- *"What products did Microsoft acquire?"* → A graph traversal, not a similarity search
- *"What's the connection between X and Y?"* → Multi-hop reasoning over relationships

Text embeddings flatten all this relational structure into a single dense vector. Graphs preserve it explicitly.

## The Pattern

```
Documents / structured data
  │
  └─→ Entity + Relationship extraction  ──→  Knowledge Graph (nodes + edges)

Query
  │
  ├─→ Entity extraction  ──→  Find matching nodes
  │
  └─→ Graph traversal (N hops)  ──→  Subgraph context  ──→  LLM  ──→  Answer
```

### Graph Structure
- **Nodes** = entities (companies, people, products, concepts) with descriptions
- **Edges** = typed, directed relationships (`invested_in`, `acquired`, `CEO_of`, `created`)

### Retrieval via Traversal
1. **Entity extraction** — find which nodes the query mentions
2. **Seed traversal** — collect the entity's own attributes + all direct edges
3. **Multi-hop expansion** — optionally walk to neighbors of neighbors for richer context

## When to Use It

✅ Your data has clear entities with named relationships  
✅ Queries ask "who", "what is connected to", "what did X do with Y"  
✅ You need explainable retrieval (you can show the exact path taken)  
✅ Knowledge needs to be updated frequently without re-embedding everything  

## How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## What to Look At in the Code

1. **Graph construction** — how nodes and typed edges are defined
2. **`extract_entities()`** — simple substring matching (NER in production)
3. **`retrieve_from_graph()`** — outgoing + incoming edge traversal
4. **`max_hops` parameter** — trade-off between context richness and noise
5. The last query — Dario Amodei's history requires traversing from a person to two companies

## Dependencies

- `networkx` — graph construction and traversal
- `openai` — LLM client
