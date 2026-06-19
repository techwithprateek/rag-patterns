"""
GraphRAG — retrieval using a knowledge graph instead of (or alongside) a vector DB.

Why a graph?
  Documents store text. Graphs store *relationships* — who owns what, what caused what,
  which thing is part of which system. When your questions are about connections
  ("Who invested in OpenAI?", "What products did Microsoft acquire?"), traversing a
  graph gives you exactly the right context without keyword or semantic guesswork.

Pattern:
  1. Build a knowledge graph from entities and their relationships
  2. Extract entities from the query (simple keyword matching here; NER in production)
  3. Find matching nodes in the graph
  4. Traverse 1-2 hops to collect related context
  5. Format the subgraph as text and pass to the LLM
"""

import os
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
#  LLM abstraction
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
#  Knowledge Graph — entities and relationships in the AI industry
# ─────────────────────────────────────────────────────────────────

# We use networkx's directed graph: nodes are entities, edges are relationships.
# In a production GraphRAG system, this graph would be extracted automatically
# from documents using an LLM or NER pipeline. Here we build it manually.
G = nx.DiGraph()

# ── Nodes — each entity has a type and a short description ───────
entities = [
    ("OpenAI",          {"type": "company",  "desc": "AI research company, creator of GPT-4, DALL-E, and Sora"}),
    ("Microsoft",       {"type": "company",  "desc": "Technology giant, owner of Azure, Office, and GitHub"}),
    ("Google",          {"type": "company",  "desc": "Technology company, owner of Search, YouTube, and DeepMind"}),
    ("DeepMind",        {"type": "company",  "desc": "AI research lab, creator of AlphaFold and Gemini"}),
    ("Anthropic",       {"type": "company",  "desc": "AI safety company, creator of the Claude model family"}),
    ("Sam Altman",      {"type": "person",   "desc": "CEO of OpenAI, former president of Y Combinator"}),
    ("Ilya Sutskever",  {"type": "person",   "desc": "Co-founder of OpenAI and SSI, key architect of GPT models"}),
    ("Demis Hassabis",  {"type": "person",   "desc": "CEO and co-founder of DeepMind"}),
    ("Dario Amodei",    {"type": "person",   "desc": "CEO and co-founder of Anthropic, former VP Research at OpenAI"}),
    ("GPT-4",           {"type": "product",  "desc": "Large language model by OpenAI, released in 2023"}),
    ("Claude",          {"type": "product",  "desc": "LLM family by Anthropic, focused on safety and helpfulness"}),
    ("Gemini",          {"type": "product",  "desc": "Multimodal LLM by Google DeepMind, released in 2023"}),
    ("AlphaFold",       {"type": "product",  "desc": "AI system by DeepMind that predicts protein structures"}),
    ("Azure",           {"type": "product",  "desc": "Microsoft's cloud platform, hosts OpenAI models via Azure OpenAI"}),
    ("GitHub",          {"type": "product",  "desc": "Code hosting platform owned by Microsoft"}),
    ("Y Combinator",    {"type": "org",      "desc": "Startup accelerator that funded companies like OpenAI, Dropbox, Airbnb"}),
]

G.add_nodes_from(entities)

# ── Edges — directional relationships between entities ────────────
# Format: (source, target, {"relation": "...", "detail": "..."})
relationships = [
    ("Sam Altman",     "OpenAI",       {"relation": "CEO_of",       "detail": "CEO since 2019 (with a brief ouster in 2023)"}),
    ("Ilya Sutskever", "OpenAI",       {"relation": "co_founded",   "detail": "Co-founded OpenAI in 2015 alongside Sam Altman"}),
    ("Dario Amodei",   "Anthropic",    {"relation": "co_founded",   "detail": "Founded Anthropic in 2021 after leaving OpenAI"}),
    ("Dario Amodei",   "OpenAI",       {"relation": "former_VP_at", "detail": "Was VP of Research at OpenAI before leaving"}),
    ("Demis Hassabis", "DeepMind",     {"relation": "co_founded",   "detail": "Co-founded DeepMind in 2010, acquired by Google in 2014"}),
    ("Microsoft",      "OpenAI",       {"relation": "invested_in",  "detail": "Invested ~$13B in OpenAI across multiple rounds"}),
    ("Google",         "Anthropic",    {"relation": "invested_in",  "detail": "Invested ~$300M in Anthropic in 2023"}),
    ("Google",         "DeepMind",     {"relation": "acquired",     "detail": "Acquired DeepMind in 2014 for ~$500M"}),
    ("Microsoft",      "GitHub",       {"relation": "acquired",     "detail": "Acquired GitHub in 2018 for $7.5B"}),
    ("OpenAI",         "GPT-4",        {"relation": "created",      "detail": "Released GPT-4 in March 2023"}),
    ("Anthropic",      "Claude",       {"relation": "created",      "detail": "Claude 3 family released in 2024"}),
    ("DeepMind",       "Gemini",       {"relation": "created",      "detail": "Gemini 1.0 released December 2023 as GPT-4 competitor"}),
    ("DeepMind",       "AlphaFold",    {"relation": "created",      "detail": "AlphaFold 2 solved protein structure prediction in 2020"}),
    ("Microsoft",      "Azure",        {"relation": "owns",         "detail": "Azure hosts OpenAI's models via Azure OpenAI Service"}),
    ("OpenAI",         "Azure",        {"relation": "partners_with","detail": "OpenAI's API and models are available through Azure"}),
    ("Sam Altman",     "Y Combinator", {"relation": "led",          "detail": "Was president of Y Combinator from 2014 to 2019"}),
]

G.add_edges_from(relationships)

print(f"✅ Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n")


# ─────────────────────────────────────────────────────────────────
#  Step 1 — Entity extraction from the query
# ─────────────────────────────────────────────────────────────────

def extract_entities(query: str) -> list[str]:
    """
    Find which graph nodes are mentioned in the query.

    This is simple case-insensitive substring matching — good enough for demos.
    In production you'd use an LLM or a NER model to extract entities, which
    handles synonyms, abbreviations, and entities not spelled out exactly.
    """
    query_lower = query.lower()
    return [node for node in G.nodes if node.lower() in query_lower]


# ─────────────────────────────────────────────────────────────────
#  Step 2 — Graph traversal to collect context
# ─────────────────────────────────────────────────────────────────

def retrieve_from_graph(query: str, max_hops: int = 2) -> str:
    """
    Retrieves context by traversing the graph from query entities.

    For each entity found in the query:
      - Include the entity's own description
      - Walk outgoing edges (what this entity does / relates to)
      - Walk incoming edges (what points to this entity)
      - Optionally go 1 more hop for richer context

    max_hops=1 gives direct neighbors; max_hops=2 also includes neighbors' neighbors.
    More hops = richer context but also more noise.
    """
    seed_entities = extract_entities(query)

    if not seed_entities:
        return "No matching entities found in the knowledge graph."

    print(f"   Entities found: {seed_entities}")

    context_lines = []
    visited = set()

    def collect_node(node: str):
        """Collect a node's description and all its edges as text."""
        if node in visited:
            return
        visited.add(node)

        data = G.nodes[node]
        context_lines.append(f"[{data['type'].upper()}] {node}: {data['desc']}")

        # Outgoing edges: what this entity does/relates to
        for _, target, edge_data in G.out_edges(node, data=True):
            context_lines.append(
                f"  → {node} --[{edge_data['relation']}]--> {target}: {edge_data['detail']}"
            )

        # Incoming edges: what points to this entity
        for source, _, edge_data in G.in_edges(node, data=True):
            context_lines.append(
                f"  ← {source} --[{edge_data['relation']}]--> {node}: {edge_data['detail']}"
            )

    # Collect seed entities and their neighbors up to max_hops
    for entity in seed_entities:
        collect_node(entity)

        if max_hops >= 2:
            # Also collect 1-hop neighbors for richer context
            for neighbor in list(G.successors(entity)) + list(G.predecessors(entity)):
                collect_node(neighbor)

    return "\n".join(context_lines)


# ─────────────────────────────────────────────────────────────────
#  Step 3 — Generate answer from graph context
# ─────────────────────────────────────────────────────────────────

def generate(query: str, graph_context: str) -> str:
    client, model = get_llm()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to a knowledge graph about the AI industry. "
                "Answer the question using ONLY the graph context provided. "
                "Be concise and factual."
            )
        },
        {
            "role": "user",
            "content": f"Knowledge Graph Context:\n{graph_context}\n\nQuestion: {query}"
        }
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
#  Run example queries
# ─────────────────────────────────────────────────────────────────

queries = [
    "Who co-founded OpenAI?",
    "What has Microsoft invested in or acquired?",
    "What is the relationship between Google and DeepMind?",
    "What did Dario Amodei do before Anthropic?",
]

for query in queries:
    print(f"❓ {query}")

    context = retrieve_from_graph(query)
    print(f"   Graph context snippet:\n     {context[:200]}...")

    answer = generate(query, context)
    print(f"   💬 {answer}\n")
